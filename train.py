#!/usr/bin/env python3
"""
Train a rating-conditioned Hex policy (Maia-like) on dataset.py output.

Heads:
    policy  123 logits: 121 cells, swap, resign  (illegal actions masked)
    value   P(player to move wins the game)      (auxiliary, helps resign decisions)
    time    Gaussian over log(think seconds)     (mean + log std)

Inputs are canonicalized: the board is transposed when blue is to move, so the
player to move always connects top <-> bottom. 180 degree rotation is used as
data augmentation (it preserves sides).

Usage:
    python train.py data/dataset-11.npz --steps 300          # smoke test
    python train.py data/dataset-11.npz --epochs 5 -o data/model-pretrain.pt          # pretrain
    python train.py data/dataset-11.npz --init data/model-pretrain.pt --max-rd 150 \
        --epochs 3 --lr 3e-4 -o data/model.pt                                        # fine-tune
"""

import argparse
import math
import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, SequentialSampler

from hexboard import RESIGN, SWAP

N_SCALARS = 9


def transpose_index(size: int) -> np.ndarray:
    idx = np.arange(size * size)
    return (idx % size) * size + idx // size


class HexSamples(Dataset):
    """Indexed by a list of sample indexes, returns a full batch (boards rebuilt with numpy)."""

    def __init__(self, data, indexes: np.ndarray, augment: bool):
        self.size = int(data['boardsize'])
        self.moves = data['moves']
        self.offsets = data['game_offsets']
        self.winners = data['game_winners']
        self.samples = data['samples']
        self.indexes = indexes
        self.augment = augment
        self.T = transpose_index(self.size)
        self.n_cells = self.size * self.size

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, batch_idx):
        s = self.samples[self.indexes[batch_idx]]
        B, N, size = len(s), self.n_cells, self.size

        planes = np.zeros((B, 6, N), dtype=np.float32)
        legal = np.zeros((B, N + 2), dtype=bool)
        target = np.zeros(B, dtype=np.int64)
        value = np.full(B, -1.0, dtype=np.float32)
        rotate = np.random.rand(B) < 0.5 if self.augment else np.zeros(B, dtype=bool)

        for i, smp in enumerate(s):
            off = self.offsets[smp['game']]
            ply = int(smp['ply'])
            mv = self.moves[off:off + ply]
            mover = ply % 2

            board = np.full(N, -1, dtype=np.int8)
            red, blue = mv[0::2], mv[1::2]
            board[red[red >= 0]] = 0
            board[blue[blue >= 0]] = 1
            if ply >= 2 and mv[1] == SWAP:
                board[mv[0]] = -1
                board[self.T[mv[0]]] = 1

            # Last move cell, -1 if none or pass (careful: SWAP is also -1)
            last = -1
            if ply > 0:
                last = int(self.T[mv[0]]) if mv[-1] == SWAP else max(int(mv[-1]), -1)

            tgt = int(smp['target'])

            # Canonicalize: player to move connects top-bottom
            if mover == 1:
                board = board[self.T]
                last = int(self.T[last]) if last >= 0 else -1
                tgt = int(self.T[tgt]) if tgt >= 0 else tgt
            if rotate[i]:
                board = board[::-1]
                last = N - 1 - last if last >= 0 else -1
                tgt = N - 1 - tgt if tgt >= 0 else tgt

            swap_legal = ply == 1 and bool(smp['swap_allowed'])

            planes[i, 0] = board == mover
            planes[i, 1] = (board != mover) & (board >= 0)
            planes[i, 2] = 1.0
            planes[i, 3] = float(bool(smp['swap_allowed']))  # swap rule, whole game (changes first move too)
            if last >= 0:
                planes[i, 4, last] = 1.0
            planes[i, 5] = float(mover)  # color still matters (red moves first)

            legal[i, :N] = board < 0
            legal[i, N] = swap_legal
            legal[i, N + 1] = True  # resign
            target[i] = N if tgt == SWAP else N + 1 if tgt == RESIGN else tgt

            w = self.winners[smp['game']]
            if w >= 0:
                value[i] = float(w == mover)

        rated = ~np.isnan(s['rating'])
        rating = np.where(rated, (s['rating'] - 1500) / 400, 0)
        conf = np.where(rated, np.clip((350 - s['deviation']) / 250, 0, 1), 0)
        has_clock = ~np.isnan(s['clock'])
        clock = np.where(has_clock, np.log1p(np.maximum(s['clock'], 0)) / 10, 0)

        scalars = np.stack([
            rating, conf,
            clock, has_clock,
            np.log1p(s['initial']) / 10, np.log1p(s['increment']) / 5,
            s['vs_ai'], s['ranked'],
            s['ply'] / 60,
        ], axis=1).astype(np.float32)

        think = s['think']
        time_ok = np.isfinite(think) & (think > 0.05) & (think < 3 * 86400)
        log_think = np.where(time_ok, np.log(np.where(time_ok, think, 1)), 0).astype(np.float32)

        return {
            'planes': torch.from_numpy(planes.reshape(B, 6, size, size)),
            'scalars': torch.from_numpy(scalars),
            'legal': torch.from_numpy(legal),
            'target': torch.from_numpy(target),
            'value': torch.from_numpy(value),
            'log_think': torch.from_numpy(log_think),
            'time_ok': torch.from_numpy(time_ok),
        }


class ResBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)
        self.cond = nn.Linear(N_SCALARS, c)  # rating & context conditioning in every block

    def forward(self, x, scalars):
        h = F.relu(self.bn1(self.conv1(x)) + self.cond(scalars)[:, :, None, None])
        h = self.bn2(self.conv2(h))
        return F.relu(x + h)


class HexMaia(nn.Module):
    def __init__(self, size=11, channels=64, blocks=6):
        super().__init__()
        self.size = size
        self.stem = nn.Sequential(nn.Conv2d(6, channels, 3, padding=1, bias=False), nn.BatchNorm2d(channels), nn.ReLU())
        self.blocks = nn.ModuleList(ResBlock(channels) for _ in range(blocks))
        self.policy_conv = nn.Conv2d(channels, 1, 1)
        self.global_mlp = nn.Sequential(nn.Linear(2 * channels + N_SCALARS, 128), nn.ReLU())
        self.special = nn.Linear(128, 2)  # swap, resign
        self.value = nn.Linear(128, 1)
        self.time = nn.Linear(128, 2)     # mu, log_sigma of log(think)

    def forward(self, planes, scalars):
        x = self.stem(planes)
        for block in self.blocks:
            x = block(x, scalars)
        cells = self.policy_conv(x).flatten(1)
        pooled = torch.cat([x.mean((2, 3)), x.amax((2, 3)), scalars], 1)
        g = self.global_mlp(pooled)
        policy = torch.cat([cells, self.special(g)], 1)
        mu, log_sigma = self.time(g).unbind(1)
        return policy, self.value(g).squeeze(1), mu, log_sigma.clamp(-3, 3)


def compute_losses(model, batch, device):
    b = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    policy, value, mu, log_sigma = model(b['planes'], b['scalars'])
    policy = policy.float().masked_fill(~b['legal'], -1e9)

    loss_policy = F.cross_entropy(policy, b['target'])

    vmask = b['value'] >= 0
    loss_value = F.binary_cross_entropy_with_logits(value.float()[vmask], b['value'][vmask]) if vmask.any() else policy.new_zeros(())

    tmask = b['time_ok']
    z = (b['log_think'][tmask] - mu.float()[tmask]) / log_sigma.float()[tmask].exp()
    loss_time = (0.5 * z ** 2 + log_sigma.float()[tmask]).mean() if tmask.any() else policy.new_zeros(())

    with torch.no_grad():
        pred = policy.argmax(1)
        n_cells = policy.shape[1] - 2
        is_resign = b['target'] == n_cells + 1
        stats = {
            'acc': (pred == b['target']).float().mean().item(),
            'resign_recall': (pred[is_resign] == n_cells + 1).float().mean().item() if is_resign.any() else float('nan'),
            'time_mae_log': (b['log_think'][tmask] - mu.float()[tmask]).abs().mean().item() if tmask.any() else float('nan'),
        }

    loss = loss_policy + 0.5 * loss_value + 0.2 * loss_time
    return loss, {'loss': loss.item(), 'policy': loss_policy.item(), 'value': loss_value.item(), 'time': loss_time.item(), **stats}


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device('cuda')
        print(f'Using GPU: {torch.cuda.get_device_name(0)}')
        return dev

    if torch.version.cuda is None:
        reason = 'this PyTorch build is CPU-only (reinstall torch from a CUDA index, e.g. https://download.pytorch.org/whl/cu128)'
    else:
        reason = 'CUDA not available (driver issue? check `nvidia-smi`, a reboot may be needed after a driver update)'
    warnings.warn(f'No GPU, training on CPU: {reason}', stacklevel=2)
    return torch.device('cpu')


def evaluate(model, loader, device, max_batches, autocast):
    model.eval()
    totals, n = {}, 0
    with torch.no_grad(), autocast():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            _, stats = compute_losses(model, batch, device)
            for k, v in stats.items():
                if not math.isnan(v):
                    totals[k] = totals.get(k, 0) + v
            n += 1
    model.train()
    return {k: v / n for k, v in totals.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset')
    parser.add_argument('--epochs', type=float, default=1)
    parser.add_argument('--steps', type=int, help='Stop after N steps (overrides epochs)')
    parser.add_argument('--batch', type=int, default=512)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--channels', type=int, default=64)
    parser.add_argument('--blocks', type=int, default=6)
    parser.add_argument('--cpus', type=int, default=max(1, (os.cpu_count() or 1) - 4),
                        help='Max CPU cores to use (default: all but 4, to keep the machine usable)')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--init', help='Start from this checkpoint (model size is taken from it)')
    parser.add_argument('--max-rd', type=float, help='Only train on samples with deviation < MAX_RD (fine-tuning)')
    parser.add_argument('--eval-every', type=int, default=1000)
    parser.add_argument('--eval-batches', type=int, default=50)
    parser.add_argument('-o', '--output', default='data/model.pt')
    args = parser.parse_args()

    # Restrict this process and its data loader workers (inherited) to the first N cores
    if hasattr(os, 'sched_setaffinity'):
        os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:args.cpus]))
    torch.set_num_threads(min(torch.get_num_threads(), args.cpus))

    device = pick_device()
    data = dict(np.load(args.dataset))
    samples = data['samples']

    keep = np.ones(len(samples), dtype=bool)
    if args.max_rd:
        keep &= samples['deviation'] < args.max_rd
    is_val = samples['game'] % 50 == 0  # split by game, ~2% validation
    train_idx = np.flatnonzero(keep & ~is_val)
    val_idx = np.flatnonzero(keep & is_val)
    print(f'train samples: {len(train_idx)}, val samples: {len(val_idx)}')

    train_ds = HexSamples(data, train_idx, augment=True)
    val_ds = HexSamples(data, val_idx, augment=False)
    loader_kw = dict(batch_size=None, num_workers=args.workers, pin_memory=device.type == 'cuda',
                     persistent_workers=args.workers > 0)
    train_loader = DataLoader(train_ds, sampler=BatchSampler(RandomSampler(train_ds), args.batch, drop_last=True), **loader_kw)
    val_loader = DataLoader(val_ds, sampler=BatchSampler(SequentialSampler(val_ds), args.batch, drop_last=False), **loader_kw)

    init = None
    if args.init:
        init = torch.load(args.init, map_location='cpu')
        if init['boardsize'] != int(data['boardsize']):
            raise SystemExit(f"--init checkpoint is for boardsize {init['boardsize']}, dataset is {int(data['boardsize'])}")
        args.channels, args.blocks = init['args']['channels'], init['args']['blocks']

    model = HexMaia(int(data['boardsize']), args.channels, args.blocks)
    if init:
        model.load_state_dict(init['model'])
        print(f'Initialized from {args.init}')
    model = model.to(device)
    print(f'model params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')

    steps_per_epoch = len(train_ds) // args.batch
    total_steps = args.steps or int(args.epochs * steps_per_epoch)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.05)

    def autocast():
        return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda')

    step, t0, running = 0, time.time(), {}
    while step < total_steps:
        for batch in train_loader:
            with autocast():
                loss, stats = compute_losses(model, batch, device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1

            for k, v in stats.items():
                if not math.isnan(v):
                    running[k] = 0.98 * running.get(k, v) + 0.02 * v

            if step % 50 == 0 or step == total_steps:
                rate = step * args.batch / (time.time() - t0)
                eta_epoch = steps_per_epoch * args.batch / rate / 60
                print(f'step {step}/{total_steps}  ' + '  '.join(f'{k} {v:.3f}' for k, v in running.items())
                      + f'  | {rate:.0f} samples/s, ~{eta_epoch:.0f} min/epoch', flush=True)

            if step % args.eval_every == 0 or step == total_steps:
                ev = evaluate(model, val_loader, device, args.eval_batches, autocast)
                print('  val: ' + '  '.join(f'{k} {v:.3f}' for k, v in ev.items()), flush=True)

            if step >= total_steps:
                break

    torch.save({'model': model.state_dict(), 'args': vars(args), 'boardsize': int(data['boardsize'])}, args.output)
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
