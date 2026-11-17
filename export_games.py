#!/usr/bin/env python3
"""
Export PlayHex games for the maia-hex POC.

One JSON line per game (gzipped), with both players' Glicko-2 "overall" rating
as it was just before the game started. Expansion into per-move training
samples (position, move, think time, clock) is done later, at training time.

Usage:
    DATABASE_URL="mysql://user:pass@host:port/hex" python export_games.py -o data/games.jsonl.gz
    # or reads DATABASE_URL from ../hex/.env (last uncommented mysql:// line)
"""

import argparse
import gzip
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote, urlparse

import pymysql
import pymysql.cursors

QUERY = """
select
    g.id, g.publicId, g.boardsize, g.swapRule, g.ranked, g.opponentType,
    g.timeControlType, g.moves, g.moveTimestamps, g.winner, g.outcome,
    coalesce(g.startedAt, g.createdAt) as startedAt, g.endedAt,
    gtp.`order`, gtp.playerId, p.isBot, p.isGuest,
    r.rating, r.deviation, r.createdAt as ratingAt
from game g
join game_to_player gtp on gtp.gameId = g.id
join player p on p.id = gtp.playerId
left join lateral (
    select rating, deviation, createdAt
    from rating r
    where r.playerId = gtp.playerId
      and r.category = 'overall'
      and r.createdAt < coalesce(g.startedAt, g.createdAt)
    order by r.createdAt desc
    limit 1
) r on true
where g.state = 'ended'
  and g.moves <> ''
  {extra}
order by g.id, gtp.`order`
"""


def find_database_url() -> str:
    url = os.environ.get('DATABASE_URL')
    if url:
        return url

    env_file = Path(__file__).resolve().parent.parent / 'hex' / '.env'
    if env_file.exists():
        found = None
        for line in env_file.read_text().splitlines():
            m = re.match(r'^\s*DATABASE_URL\s*=\s*"?(mysql://[^"\s]+)"?', line)
            if m:
                found = m.group(1)
        if found:
            return found

    sys.exit('DATABASE_URL not set and not found in ../hex/.env')


def connect(url: str):
    u = urlparse(url)
    return pymysql.connect(
        host=u.hostname or 'localhost',
        port=u.port or 3306,
        user=unquote(u.username or 'root'),
        password=unquote(u.password or ''),
        database=u.path.lstrip('/'),
        charset='utf8mb4',
        # Server-side cursor: stream rows instead of loading everything in memory
        cursorclass=pymysql.cursors.SSDictCursor,
    )


def iso(d: datetime | None) -> str | None:
    return d.isoformat(timespec='milliseconds') + 'Z' if d else None


def player_entry(row: dict) -> dict:
    return {
        'id': row['playerId'],
        'isBot': bool(row['isBot']),
        'isGuest': bool(row['isGuest']),
        # None when player has no rating row before this game
        'rating': row['rating'],
        'deviation': row['deviation'],
        'ratingAt': iso(row['ratingAt']),
    }


def build_game(rows: list[dict]) -> dict:
    g = rows[0]
    players = [None, None]
    for row in rows:
        players[row['order']] = player_entry(row)

    return {
        'id': g['id'],
        'publicId': g['publicId'],
        'boardsize': g['boardsize'],
        'swapRule': bool(g['swapRule']),
        'ranked': bool(g['ranked']),
        'opponentType': g['opponentType'],
        'timeControl': json.loads(g['timeControlType']),
        'startedAt': iso(g['startedAt']),
        'endedAt': iso(g['endedAt']),
        # Moves like "d4 swap-pieces e5 pass ...", players[0] plays even indexes
        'moves': g['moves'].split(' '),
        # ISO timestamps, one per move
        'moveTimestamps': g['moveTimestamps'].split(' ') if g['moveTimestamps'] else [],
        'winner': g['winner'],
        'outcome': g['outcome'],
        'players': players,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-o', '--output', default='data/games.jsonl.gz')
    parser.add_argument('--boardsize', type=int, help='Only export this board size (e.g. 11)')
    parser.add_argument('--pvp-only', action='store_true', help='Exclude games against AI')
    args = parser.parse_args()

    extra, params = [], []
    if args.boardsize:
        extra.append('and g.boardsize = %s')
        params.append(args.boardsize)
    if args.pvp_only:
        extra.append("and g.opponentType = 'player'")

    query = QUERY.format(extra='\n  '.join(extra))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    conn = connect(find_database_url())
    n_games, n_moves = 0, 0

    def write(game_rows: list[dict]):
        nonlocal n_games, n_moves
        game = build_game(game_rows)
        f.write(json.dumps(game, separators=(',', ':')) + '\n')
        n_games += 1
        n_moves += len(game['moves'])
        if n_games % 10000 == 0:
            print(f'\r{n_games} games, {n_moves} moves', end='', file=sys.stderr)

    with gzip.open(out, 'wt', encoding='utf-8') as f, conn.cursor() as cur:
        cur.execute(query, params)
        current: list[dict] = []

        # Rows are ordered by game id: one game = 2 consecutive rows (or 1 if a player is missing)
        for row in cur:
            if current and current[0]['id'] != row['id']:
                write(current)
                current = []
            current.append(row)

        if current:
            write(current)

    print(f'\r{n_games} games, {n_moves} moves\nWritten {out}', file=sys.stderr)


if __name__ == '__main__':
    main()
