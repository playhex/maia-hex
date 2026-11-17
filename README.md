# maia-hex (POC)

Predict, for a given Hex position and player rating, which move a human of that
rating would play and how long they would think. Inspired by Maia Chess / Maia-2.

## Export data

```bash
pip install -r requirements.txt
python export_games.py -o data/games.jsonl.gz              # all ended games
python export_games.py -o data/games-11.jsonl.gz --boardsize 11 --pvp-only
```

`DATABASE_URL` is read from the environment, or from `../hex/.env`.

One JSON line per game: board size, time control, moves, move timestamps, and
for both players `isBot`, `isGuest`, and their Glicko-2 `overall` rating and
deviation **as of just before the game** (`null` if never rated).

- `players[0]` plays moves at even indexes.
- Moves are like `d4`, `swap-pieces`, `pass`.
- Think time of move `i` = `moveTimestamps[i] - moveTimestamps[i-1]` (or `startedAt` for move 0).
- Bot moves are included (games vs AI): filter them out with `players[k].isBot`.

## Build dataset (11x11)

```bash
python dataset.py data/games.jsonl.gz -o data/dataset-11.npz --boardsize 11
```

Compact format (~350 MB): moves of every game + one record per human move
(rating, deviation, think time, clock before move, time control...). Boards are
rebuilt on the fly when training. Bot moves are not samples, but are replayed
to build positions. Human resignations are an extra sample on the final
position (target `RESIGN`), so the model learns when a player of a given rating resigns.

## Train

```bash
python train.py data/dataset-11.npz --steps 300                                   # smoke test
python train.py data/dataset-11.npz --epochs 5 -o data/model-pretrain.pt          # pretrain on all human moves (~1h on RTX A1000)
python train.py data/dataset-11.npz --init data/model-pretrain.pt --max-rd 150 \
    --epochs 3 --lr 3e-4 -o data/model.pt                                        # fine-tune on established ratings
```

Uses the GPU if available, otherwise warns and trains on CPU. Uses all CPU cores
but 4 (`--cpus N` to change).
Policy over 121 cells + swap + resign, auxiliary value head (who wins),
time head (Gaussian over log seconds of thinking).

## Export to ONNX

```bash
python export_onnx.py data/model.pt    # -> web/public/model.onnx
```

Writes `model.onnx` (~2 MB) and `model.json`, which describes the input encoding
to reproduce in JS. Inputs: `planes`, `scalars`, `legal` mask. Outputs: move
probabilities (cells, swap, resign), win probability, and think time
(log-normal `think_mu`, `think_sigma`). Then checks that ONNX gives the same
outputs as PyTorch.

Requires `onnx`, `onnxscript` (export) and `onnxruntime` (check).

## Demo page

Sources in `web/`, built site in `docs/` (committed, served by GitHub Pages).

```bash
cd web
npm install
npm run dev      # http://localhost:5173
npm run build    # -> ../docs
```

GitHub Pages: Settings > Pages > Deploy from a branch, folder `/docs`.
Rebuild and commit `docs/` after changing the page or the model.

Hex board (`@playhex/pixi-board`) with ONNX inference in the browser (`onnxruntime-web`, WebAssembly).
Play against the model, or watch it play against itself.

- One rating (newcomer, or 1200 to 2000) for how the model plays and what the analysis shows.
- Policy overlay for that rating, or a mix of levels (weak green, average blue, strong red).
- Win probability, resign and swap probabilities, predicted think time.
- The model waits for its sampled think time, at normal speed, x2, x5, or instantly.
  A Fischer clock is simulated.
