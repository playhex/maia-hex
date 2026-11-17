#!/usr/bin/env python3
"""
Export a train.py checkpoint to ONNX, for inference in JS (onnxruntime-node / onnxruntime-web).

Writes model.onnx and model.json (input encoding, to reproduce in JS).
Then checks ONNX outputs against PyTorch on a few validation batches.

ONNX inputs (B = batch size, S = board size, N = S*S):
    planes   float32 [B, 6, S, S]   see ENCODING below, board canonicalized for the player to move
    scalars  float32 [B, 9]
    legal    bool    [B, N+2]       empty cells, swap (only at ply 1 if allowed), resign (always)

ONNX outputs:
    policy       float32 [B, N+2]   probabilities: cells (row*S+col, canonical), swap, resign
    win          float32 [B]        probability that the player to move wins
    think_mu     float32 [B]        mean of log(think seconds)
    think_sigma  float32 [B]        std of log(think seconds)

Usage:
    python export_onnx.py data/model.pt    # writes web/public/model.onnx, used by the demo page
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from train import HexMaia, HexSamples

ENCODING = {
    'canonical': (
        'If blue (player 1) is to move, transpose the board (cell row*S+col -> col*S+row), so the player '
        'to move always connects top <-> bottom. Transpose the predicted cell back. '
        'After a swap, the swapped stone is at the transposed cell of the first move, with blue color.'
    ),
    'planes': [
        'stones of the player to move (1/0)',
        'stones of the opponent (1/0)',
        'all ones (lets convolutions see board edges)',
        'swap rule: all ones if the swap rule is enabled for this game (at every move), else zeros',
        'last move: 1 on the last placed stone (canonical coords; swapped stone after a swap; none after a pass or at ply 0)',
        'color of the player to move: all zeros for red, all ones for blue',
    ],
    'scalars': [
        'rating: (rating - 1500) / 400, 0 if unknown',
        'rating confidence: clip((350 - deviation) / 250, 0, 1), 0 if unknown',
        'clock: log1p(max(seconds remaining for the player to move, 0)) / 10, 0 if unknown',
        'has clock: 1 if clock known (fischer), else 0',
        'initial time: log1p(initial seconds) / 10',
        'increment: log1p(increment seconds) / 5, 0 if not fischer',
        'vs AI: 1/0',
        'ranked: 1/0',
        'ply / 60 (number of moves played so far, passes and swap included)',
    ],
    'policy': 'index < N: cell row*S+col (canonical), N: swap, N+1: resign',
    'time': 'think seconds ~ LogNormal(think_mu, think_sigma): median exp(mu), sample exp(mu + sigma * randn)',
}


class Exported(nn.Module):
    """Wraps HexMaia with masking and softmax, so JS gets probabilities directly."""

    def __init__(self, model: HexMaia):
        super().__init__()
        self.model = model

    def forward(self, planes, scalars, legal):
        policy, value, mu, log_sigma = self.model(planes, scalars)
        policy = policy.masked_fill(~legal, -1e9).softmax(1)
        return policy, value.sigmoid(), mu, log_sigma.exp()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('checkpoint')
    parser.add_argument('-o', '--output', default='web/public/model.onnx')
    parser.add_argument('--dataset', default='data/dataset-11.npz', help='For the PyTorch vs ONNX check')
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    size = ckpt['boardsize']
    model = HexMaia(size, ckpt['args']['channels'], ckpt['args']['blocks'])
    model.load_state_dict(ckpt['model'])
    wrapped = Exported(model).eval()

    n = size * size
    dummy = (torch.zeros(2, 6, size, size), torch.zeros(2, 9), torch.ones(2, n + 2, dtype=torch.bool))
    torch.onnx.export(
        wrapped, dummy, args.output,
        input_names=['planes', 'scalars', 'legal'],
        output_names=['policy', 'win', 'think_mu', 'think_sigma'],
        dynamic_axes={k: {0: 'batch'} for k in ['planes', 'scalars', 'legal', 'policy', 'win', 'think_mu', 'think_sigma']},
        opset_version=17,
        dynamo=False,
    )

    meta = {'boardsize': size, 'channels': ckpt['args']['channels'], 'blocks': ckpt['args']['blocks'], **ENCODING}
    meta_path = Path(args.output).with_suffix('.json')
    meta_path.write_text(json.dumps(meta, indent=2) + '\n')
    print(f'Written {args.output} ({Path(args.output).stat().st_size / 1e6:.1f} MB) and {meta_path}')

    # Check ONNX == PyTorch on real validation samples
    import onnxruntime as ort

    data = dict(np.load(args.dataset))
    val_idx = np.flatnonzero(data['samples']['game'] % 50 == 0)
    ds = HexSamples(data, val_idx, augment=False)
    sess = ort.InferenceSession(args.output, providers=['CPUExecutionProvider'])

    max_diff = 0.0
    for start in range(0, 2048, 512):
        b = ds[np.arange(start, start + 512)]
        with torch.no_grad():
            ref = wrapped(b['planes'], b['scalars'], b['legal'])
        out = sess.run(None, {'planes': b['planes'].numpy(), 'scalars': b['scalars'].numpy(), 'legal': b['legal'].numpy()})
        max_diff = max(max_diff, *(float(np.abs(r.numpy() - o).max()) for r, o in zip(ref, out)))

    print(f'ONNX vs PyTorch max abs diff: {max_diff:.2e}')
    if max_diff > 1e-3:
        raise SystemExit('ONNX output differs from PyTorch')


if __name__ == '__main__':
    main()
