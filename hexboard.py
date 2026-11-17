"""
Board helpers shared by dataset.py and train.py.

Conventions (same as PlayHex engine):
- Move "c5" -> col = 'c' - 'a' = 2, row = 5 - 1 = 4. Cell index = row * size + col.
- Player 0 (red) plays first and connects top (row 0) to bottom (row size-1).
- Player 1 (blue) connects left (col 0) to right (col size-1).
- "swap-pieces": the first stone is removed and its transpose (row <-> col) is
  placed with player 1's color. Player indexes keep alternating afterwards.
"""

import numpy as np

SWAP = -1
PASS = -2
RESIGN = -3  # only used as a training target, never in move lists


def parse_move(move: str, size: int) -> int:
    """Returns cell index, SWAP, or PASS. Raises ValueError on unknown/out of board moves."""
    if move == 'swap-pieces':
        return SWAP
    if move == 'pass':
        return PASS

    i = 0
    while i < len(move) and move[i].isalpha():
        i += 1
    letters, number = move[:i], move[i:]

    if len(letters) == 1:
        col = ord(letters) - 97
    elif len(letters) == 2:
        col = ord(letters[1]) - 97 + 26 * (ord(letters[0]) - 97 + 1)
    else:
        raise ValueError(move)

    row = int(number) - 1
    if not (0 <= row < size and 0 <= col < size):
        raise ValueError(move)

    return row * size + col


def transpose_cell(cell: int, size: int) -> int:
    row, col = divmod(cell, size)
    return col * size + row


def replay(moves: np.ndarray, size: int) -> np.ndarray:
    """
    Replay encoded moves, returns board of shape (size*size,) with
    -1 empty, 0 red, 1 blue. Raises ValueError on illegal sequence.
    """
    board = np.full(size * size, -1, dtype=np.int8)
    for ply, m in enumerate(moves):
        player = ply % 2
        if m == PASS:
            continue
        if m == SWAP:
            if ply != 1 or moves[0] < 0:
                raise ValueError('illegal swap')
            board[moves[0]] = -1
            board[transpose_cell(int(moves[0]), size)] = 1
            continue
        if board[m] != -1:
            raise ValueError('occupied cell')
        board[m] = player

    return board


def has_connection(board: np.ndarray, size: int, player: int) -> bool:
    """Red: top to bottom, Blue: left to right."""
    grid = board.reshape(size, size)
    if player == 1:
        grid = grid.T

    stack = [(0, c) for c in range(size) if grid[0, c] == player]
    seen = set(stack)
    while stack:
        r, c = stack.pop()
        if r == size - 1:
            return True
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size and (nr, nc) not in seen and grid[nr, nc] == player:
                seen.add((nr, nc))
                stack.append((nr, nc))

    return False
