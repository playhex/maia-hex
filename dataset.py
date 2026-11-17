#!/usr/bin/env python3
"""
Build a compact training dataset from export_games.py output.

Positions are NOT stored: we store all game moves once, plus one small record
per human move. Boards are rebuilt on the fly at training time (see train.py).

Output (.npz):
    moves         int16 [total_moves]   cell index (row*size+col), -1 swap, -2 pass
    game_offsets  int64 [n_games+1]     game g moves = moves[offsets[g]:offsets[g+1]]
    game_ids      int32 [n_games]       PlayHex game id
    game_winners  int8  [n_games]       0 red, 1 blue, -1 unknown
    game_outcomes int8  [n_games]       see OUTCOMES, -1 unknown
    samples       structured [n_samples], one per human move (see SAMPLE_DTYPE)

Usage:
    python dataset.py data/games.jsonl.gz -o data/dataset-11.npz --boardsize 11
"""

import argparse
import gzip
import json
import sys
from datetime import datetime

import numpy as np

from hexboard import PASS, RESIGN, SWAP, has_connection, parse_move, replay

# game_outcomes encoding
OUTCOMES = {'path': 0, 'resign': 1, 'time': 2, 'forfeit': 3}

SAMPLE_DTYPE = np.dtype([
    ('game', np.int32),        # index in game_offsets
    ('ply', np.int16),         # move index in game, also number of moves before it
    ('target', np.int16),      # played cell index, -1 swap, -3 resign (ply = len(moves))
    ('rating', np.float32),    # Glicko-2 overall before game, NaN if never rated
    ('deviation', np.float32), # NaN if never rated
    ('think', np.float32),     # seconds spent on this move, NaN if unknown
    ('clock', np.float32),     # seconds remaining on mover's clock before move, NaN if unknown
    ('initial', np.float32),   # time control initial time (s)
    ('increment', np.float32), # time control increment (s), 0 for byo-yomi
    ('vs_ai', np.bool_),
    ('ranked', np.bool_),
    ('swap_allowed', np.bool_),
    ('player', np.int32),      # PlayHex player id (for per-player splits/stats)
])


def ts(s: str) -> float:
    return datetime.fromisoformat(s.replace('Z', '+00:00')).timestamp()


def process_game(g: dict, size: int):
    """Returns (encoded moves, list of sample tuples) or raises ValueError."""
    moves = np.array([parse_move(m, size) for m in g['moves']], dtype=np.int16)
    replay(moves, size)  # validates legality

    tc = g['timeControl']
    opts = tc.get('options', {})
    fischer = tc.get('family') == 'fischer'
    initial = opts.get('initialTime', 0) / 1000
    increment = opts.get('timeIncrement', 0) / 1000 if fischer else 0.0
    max_time = (opts.get('maxTime') or float('inf')) / 1000 if fischer else float('inf')

    stamps = [ts(t) for t in g['moveTimestamps']] if len(g['moveTimestamps']) == len(moves) else None
    start = ts(g['startedAt']) if g['startedAt'] else None

    clocks = [initial, initial]
    samples = []
    prev = start

    for ply, m in enumerate(moves):
        p = ply % 2
        think = np.nan
        if stamps is not None and prev is not None:
            think = stamps[ply] - prev
        clock_before = clocks[p] if fischer else np.nan

        if fischer and not np.isnan(think):
            clocks[p] = min(max_time, clocks[p] - think + increment)

        if stamps is not None:
            prev = stamps[ply]

        player = g['players'][p]
        if player['isBot'] or m == PASS:
            continue

        samples.append((
            0, ply, m,
            np.nan if player['rating'] is None else player['rating'],
            np.nan if player['deviation'] is None else player['deviation'],
            think, clock_before, initial, increment,
            g['opponentType'] == 'ai', g['ranked'], g['swapRule'],
            player['id'],
        ))

    # Resignation: one extra sample on the final position, played by the loser instead of a move.
    # Only when the loser resigned on their own turn (~99% of resignations), the model acts on its turn.
    if g['outcome'] == 'resign' and g['winner'] is not None:
        loser = 1 - g['winner']
        player = g['players'][loser]
        ply = len(moves)

        if ply % 2 == loser and not player['isBot']:
            think = np.nan
            if g['endedAt'] and prev is not None:
                think = ts(g['endedAt']) - prev

            samples.append((
                0, ply, RESIGN,
                np.nan if player['rating'] is None else player['rating'],
                np.nan if player['deviation'] is None else player['deviation'],
                think, clocks[loser] if fischer else np.nan, initial, increment,
                g['opponentType'] == 'ai', g['ranked'], g['swapRule'],
                player['id'],
            ))

    return moves, samples


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input')
    parser.add_argument('-o', '--output', default='data/dataset-11.npz')
    parser.add_argument('--boardsize', type=int, default=11)
    args = parser.parse_args()

    size = args.boardsize
    all_moves, offsets, game_ids, game_winners, game_outcomes, samples = [], [0], [], [], [], []
    skipped, checked, check_failed = 0, 0, 0

    with gzip.open(args.input, 'rt', encoding='utf-8') as f:
        for line in f:
            g = json.loads(line)
            if g['boardsize'] != size:
                continue

            try:
                moves, game_samples = process_game(g, size)
            except (ValueError, KeyError, TypeError):
                skipped += 1
                continue

            # Games ended by path must have the winner connected, else data is corrupted (legacy games)
            if g['outcome'] == 'path' and g['winner'] is not None:
                checked += 1
                if not has_connection(replay(moves, size), size, g['winner']):
                    check_failed += 1
                    continue

            if not game_samples:
                continue

            gi = len(game_ids)
            game_ids.append(g['id'])
            game_winners.append(-1 if g['winner'] is None else g['winner'])
            game_outcomes.append(OUTCOMES.get(g['outcome'], -1))
            all_moves.append(moves)
            offsets.append(offsets[-1] + len(moves))
            samples.extend((gi, *s[1:]) for s in game_samples)

            if gi % 20000 == 0:
                print(f'\r{gi} games, {len(samples)} samples', end='', file=sys.stderr)

    samples = np.array(samples, dtype=SAMPLE_DTYPE)
    np.savez(
        args.output,
        boardsize=np.int32(size),
        moves=np.concatenate(all_moves).astype(np.int16),
        game_offsets=np.array(offsets, dtype=np.int64),
        game_ids=np.array(game_ids, dtype=np.int32),
        game_winners=np.array(game_winners, dtype=np.int8),
        game_outcomes=np.array(game_outcomes, dtype=np.int8),
        samples=samples,
    )

    rated = ~np.isnan(samples['rating']) & (samples['deviation'] < 350)
    print(f'\r{len(game_ids)} games, {len(samples)} samples, skipped {skipped} invalid games', file=sys.stderr)
    print(f'replay check: dropped {check_failed}/{checked} path-ended games with winner not connected', file=sys.stderr)
    print(f'rated samples: {rated.sum()}  (RD<150: {(samples["deviation"] < 150).sum()}, RD<100: {(samples["deviation"] < 100).sum()})', file=sys.stderr)
    print(f'swap targets: {(samples["target"] == SWAP).sum()}, resign targets: {(samples["target"] == RESIGN).sum()}, think known: {(~np.isnan(samples["think"])).sum()}', file=sys.stderr)
    print(f'Written {args.output}', file=sys.stderr)


if __name__ == '__main__':
    main()
