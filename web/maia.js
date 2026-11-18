/**
 * Encodes a Hex position like train.py (HexSamples) and runs the ONNX model.
 * See ../export_onnx.py ENCODING, or model.json written next to model.onnx.
 */
// WebAssembly-only build (no WebGPU): smaller, enough for this tiny model. Vite bundles its .wasm file.
import * as ort from 'onnxruntime-web/wasm';

export const SWAP = -1;
export const PASS = -2;

/** Inference blocks the main thread: give the browser a turn at least this often (ms) */
const YIELD_EVERY = 30;

export class Maia {
    constructor(session, size = 11) {
        this.session = session;
        this.size = size;
        this.n = size * size;
        this.T = Array.from({ length: this.n }, (_, i) => (i % size) * size + Math.floor(i / size));
        this.lastYield = performance.now();
    }

    /**
     * Lets the browser handle events and timers when inference has been running for a while,
     * so the page stays responsive during long computations (search).
     */
    async yieldIfBusy() {
        if (performance.now() - this.lastYield < YIELD_EVERY) return;
        await new Promise(resolve => setTimeout(resolve, 0));
        this.lastYield = performance.now();
    }

    static async load(url, size = 11) {
        ort.env.wasm.numThreads = 1; // multithreading would need cross-origin isolation headers
        const session = await ort.InferenceSession.create(url, { executionProviders: ['wasm'] });

        return new Maia(session, size);
    }

    parseMove(move) {
        if (move === 'swap-pieces') return SWAP;
        if (move === 'pass') return PASS;
        const col = move.charCodeAt(0) - 97;
        const row = parseInt(move.slice(1), 10) - 1;

        return row * this.size + col;
    }

    cellToMove(cell) {
        return String.fromCharCode(97 + (cell % this.size)) + (Math.floor(cell / this.size) + 1);
    }

    /**
     * @param {string[]} moves like ['f6', 'swap-pieces', 'c3']
     * @returns {{ board: Int8Array, last: number }} -1 empty, 0 red, 1 blue; last placed stone cell or -1
     */
    buildBoard(moves) {
        const cells = moves.map(m => this.parseMove(m));
        const board = new Int8Array(this.n).fill(-1);

        cells.forEach((m, ply) => {
            if (m === PASS) return;
            if (m === SWAP) {
                board[cells[0]] = -1;
                board[this.T[cells[0]]] = 1;
                return;
            }
            board[m] = ply % 2;
        });

        let last = -1;
        if (cells.length > 0) {
            const m = cells[cells.length - 1];
            last = m === SWAP ? this.T[cells[0]] : m >= 0 ? m : -1;
        }

        return { board, last };
    }

    /**
     * Red connects top to bottom, blue left to right.
     */
    hasConnection(board, player) {
        const S = this.size;
        const at = (r, c) => board[player === 0 ? r * S + c : c * S + r]; // blue: transposed
        const stack = [];
        const seen = new Set();

        for (let c = 0; c < S; ++c) {
            if (at(0, c) === player) {
                stack.push([0, c]);
                seen.add(c);
            }
        }

        while (stack.length) {
            const [r, c] = stack.pop();
            if (r === S - 1) return true;
            for (const [dr, dc] of [[-1, 0], [1, 0], [0, -1], [0, 1], [-1, 1], [1, -1]]) {
                const nr = r + dr, nc = c + dc;
                if (nr < 0 || nc < 0 || nr >= S || nc >= S || at(nr, nc) !== player) continue;
                const key = nr * S + nc;
                if (seen.has(key)) continue;
                seen.add(key);
                stack.push([nr, nc]);
            }
        }

        return false;
    }

    /**
     * @param ctx {{ rating: ?number, deviation: number, clock: ?number, initial: number,
     *               increment: number, vsAi: boolean, ranked: boolean, swapAllowed: boolean }}
     *        rating null = unknown player, clock null = unknown clock (in seconds)
     */
    async predict(moves, ctx) {
        await this.yieldIfBusy();

        const N = this.n, S = this.size, A = N + 2;
        const ply = moves.length;
        const mover = ply % 2;
        const { board, last } = this.buildBoard(moves);
        const swapLegal = ply === 1 && ctx.swapAllowed;

        // Batch of 2: the position and its 180° rotation (same position for the model), averaged.
        const B = 2;
        const planes = new Float32Array(B * 6 * N);
        const legal = new Uint8Array(B * A);
        const scalars = new Float32Array(B * 9);
        const toOriginal = []; // canonical cell -> original cell, per batch item

        for (let b = 0; b < B; ++b) {
            const rotate = b === 1;
            const map = Array.from({ length: N }, (_, i) => {
                const j = rotate ? N - 1 - i : i;
                return mover === 1 ? this.T[j] : j;
            });
            toOriginal.push(map);

            const p = planes.subarray(b * 6 * N, (b + 1) * 6 * N);
            for (let i = 0; i < N; ++i) {
                const v = board[map[i]];
                p[i] = v === mover ? 1 : 0;
                p[N + i] = v >= 0 && v !== mover ? 1 : 0;
                p[2 * N + i] = 1;
                p[3 * N + i] = ctx.swapAllowed ? 1 : 0; // swap rule, whole game
                p[4 * N + i] = map[i] === last ? 1 : 0;
                p[5 * N + i] = mover;
                legal[b * A + i] = v < 0 ? 1 : 0;
            }
            legal[b * A + N] = swapLegal ? 1 : 0;
            legal[b * A + N + 1] = 1; // resign

            const hasRating = ctx.rating !== null;
            const hasClock = ctx.clock !== null;
            scalars.set([
                hasRating ? (ctx.rating - 1500) / 400 : 0,
                hasRating ? Math.min(1, Math.max(0, (350 - ctx.deviation) / 250)) : 0,
                hasClock ? Math.log1p(Math.max(ctx.clock, 0)) / 10 : 0,
                hasClock ? 1 : 0,
                Math.log1p(ctx.initial) / 10,
                Math.log1p(ctx.increment) / 5,
                ctx.vsAi ? 1 : 0,
                ctx.ranked ? 1 : 0,
                ply / 60,
            ], b * 9);
        }

        const out = await this.session.run({
            planes: new ort.Tensor('float32', planes, [B, 6, S, S]),
            scalars: new ort.Tensor('float32', scalars, [B, 9]),
            legal: new ort.Tensor('bool', legal, [B, A]),
        });

        const policy = out.policy.data;
        const cells = new Float32Array(N);
        let swap = 0, resign = 0;
        for (let b = 0; b < B; ++b) {
            for (let i = 0; i < N; ++i) {
                cells[toOriginal[b][i]] += policy[b * A + i] / B;
            }
            swap += policy[b * A + N] / B;
            resign += policy[b * A + N + 1] / B;
        }

        const mean = t => Array.from(t.data).reduce((a, x) => a + x, 0) / B;

        return {
            cells, // probability per original cell index (row * size + col)
            swap,
            resign,
            win: mean(out.win), // probability that the player to move wins
            thinkMu: mean(out.think_mu),
            thinkSigma: mean(out.think_sigma),
        };
    }

    /**
     * Sample a move from prediction: a move string, 'swap-pieces' or 'resign'.
     */
    sample(pred, temperature = 1) {
        const actions = [...pred.cells, pred.swap, pred.resign].map(p => Math.pow(p, 1 / temperature));
        const total = actions.reduce((a, x) => a + x, 0);
        let r = Math.random() * total;
        let i = 0;
        for (; i < actions.length - 1; ++i) {
            r -= actions[i];
            if (r <= 0) break;
        }

        return this.actionToMove(i);
    }

    best(pred) {
        const actions = [...pred.cells, pred.swap, pred.resign];

        return this.actionToMove(actions.indexOf(Math.max(...actions)));
    }

    actionToMove(i) {
        if (i === this.n) return 'swap-pieces';
        if (i === this.n + 1) return 'resign';

        return this.cellToMove(i);
    }

    /**
     * Think time sample in seconds, from the predicted log-normal.
     */
    sampleThink(pred) {
        const u = 1 - Math.random(), v = Math.random();
        const z = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);

        return Math.exp(pred.thinkMu + pred.thinkSigma * z);
    }
}
