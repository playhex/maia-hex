# Notes

What we learned while building this POC.

## Data

- Few established ratings: only ~8.7k ranked PvP games. Most of the ~930k `rating` rows are initial ratings (1500, RD 350) with no game.
- 8.3M human moves on 11x11, but only ~1M with a rating, and ~300k with RD < 150.
- Ratings with RD < 150 are only reliable from 1200 to 2000: below 1200 there are 18 players, above 2000 there are about 10. Outside this range the model extrapolates.
- Samples with RD < 150, by rating (11x11):

  | Rating    | Samples | Players |
  |-----------|--------:|--------:|
  | 800-1000  |       0 |       0 |
  | 1000-1200 |   4,600 |      18 |
  | 1200-1400 |  23,500 |      80 |
  | 1400-1600 |  80,600 |     151 |
  | 1600-1800 | 103,600 |     112 |
  | 1800-2000 |  40,500 |      40 |
  | 2000-2200 |  41,700 |      11 |
  | 2200-2400 |   6,100 |       3 |

- Rating deviation of rated samples: peak at 70-80 (7.2%), the most common low RD, so it is the default (70) in the demo. Another peak at 250-270 (~10%) = uncertain ratings, new players.

  | Deviation | Share of samples |
  |-----------|-----------------:|
  | 60-70     | 4.5% |
  | 70-80     | 7.2% |
  | 80-90     | 1.8% |
  | 90-130    | 1.6-2.2% per 10 |
  | 130-150   | 4.5% per 10 |
  | 250-270   | ~10% per 10 |
  | 290-300   | 6.9% |

- Real beginners are mostly guests and unrated players. "Newcomer" (unknown rating) plays more like a beginner than a low rating does.
- PlayHex bot ratings are not updated, so they cannot be used to estimate guest ratings.
- Beginners play lost games to the end, strong players resign (lost games ending by connection: 44% at 1000, 0% at 2000+).
- Without swap, the first move is in the center 51% of the time, versus 23% with swap. Only 2% of games have no swap.
- 23 legacy AI games (Dec 2023) have a winner that is not connected: replay and check games ended by `path`.
- Rating "as of the game" = last `overall` rating row before `startedAt`.

## Export / SQL

- A `LATERAL` subquery for the rating, with pagination (`id > X limit N`), is too slow because it is recomputed each batch. Use one streaming query instead (`SSDictCursor`).
- `pkill -f <pattern>` can kill the shell that runs it if the pattern matches its own command line. Kill by PID instead.

## Encoding pitfalls

- `-1` was used both for "no last move" and for `SWAP`, which caused IndexError. Keep sentinels distinct.
- The swap input must be "swap rule enabled" during the whole game, not "swap legal now" (ply 1 only). Otherwise the first player can't know about the swap. Changing it requires retraining.
- Canonicalize the board: transpose it when blue is to move. Transpose predicted cells back.
- Swap = remove the first stone and place its transpose with blue.
- Never store positions: store moves and rebuild boards on the fly. This takes ~340 MB instead of tens of GB.

## Training

- One rating-conditioned network (Maia-2 style), with the rating injected in every block. Pretraining already learns the rating effect.
- Fine-tuning on RD < 150 brought nothing (~+0.1%): those samples were already in pretraining.
- Accuracy plateaus around 48% (60% on rated players). Training 2x longer would not help, and the train/val gap shows the model starts memorizing games.
- An auxiliary value head (who wins) helps learn when to resign.
- Think time: a Gaussian over log(seconds) gives a median and a range, and lets us sample realistic times.
- Data loading is not the bottleneck (~80k samples/s per worker), the network is. CPU: ~200 samples/s. RTX A1000 laptop: ~11k samples/s, ~12 min/epoch.
- Limit CPU usage (`--cpus`, `taskset` on running processes) to keep the machine usable.

## Web demo

- `onnxruntime-web/wasm` (15 MB) is enough. The default import pulls the WebGPU build (29 MB).
- ONNX export (`torch.onnx.export(..., dynamo=False)`) matches PyTorch to ~1e-6. Wrapping the softmax and legal mask in the export saves work on the JS side.
- Temperature 0 must be handled separately (`p^(1/0)`): otherwise the last action, resign, was always chosen.
- Two rating sliders (model vs analysis) confused users. Use a single rating for both.
- Sampling makes play uneven across games: what makes a low rating weak is a few blunders, not every move.
- Vite `base: './'` and `outDir: '../docs'` give a static site that works on GitHub Pages under `/<repo>/`.
