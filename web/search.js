/**
 * Small human-like search (PUCT, AlphaZero style) on top of the Maia policy.
 *
 * - Priors: the human policy for the player to move (resign excluded).
 * - Evaluation: the value head (who wins if humans keep playing), or a win when a move connects.
 * - Few playouts, and the move is sampled from visit counts: stays close to human moves,
 *   but avoids tactical blunders a player of that level would have seen.
 */

/** Playouts at rating 2000 by default, halved every 150 points below */
export const DEFAULT_MAX_PLAYOUTS = 64;
const HALVING = 150;
const C_PUCT = 1.5;

/** Moves considered: most likely ones covering this probability mass, at most MAX_CANDIDATES, none below MIN_PRIOR */
const CANDIDATES_MASS = 0.95;
const MAX_CANDIDATES = 20;
const MIN_PRIOR = 0.005;

/**
 * Number of playouts for a move: depends on rating (strong players read more),
 * and on the sampled think time (a quick move is not searched much).
 *
 * @param {?number} rating null for newcomer: no search
 * @param {number} think sampled think time (s)
 * @param {number} median median predicted think time for this position (s)
 */
export function playoutsFor(rating, think, median, maxPlayouts = DEFAULT_MAX_PLAYOUTS) {
    if (rating === null || maxPlayouts <= 0) return 0;

    const base = maxPlayouts * Math.pow(2, (rating - 2000) / HALVING);
    // sqrt: think time matters, but less than rating. 4x faster than usual -> half the search
    const timeFactor = Math.min(2, Math.max(0.25, Math.sqrt(think / median)));
    const n = Math.round(base * timeFactor);

    return n < 2 ? 0 : n;
}

class Node {
    /**
     * @param {string[]} moves position
     * @param pred Maia prediction for this position
     */
    constructor(maia, moves, pred) {
        this.moves = moves;
        this.value = pred.win; // for the player to move
        this.visits = 0;

        // Edges: only moves a human would consider (resign excluded), renormalized.
        // The value head never saw positions after very unlikely moves and overrates them:
        // without this, search exploits its errors and plays odd moves.
        const candidates = [...pred.cells, pred.swap]
            .map((p, i) => ({ p, i }))
            .filter(c => c.p >= MIN_PRIOR)
            .sort((a, b) => b.p - a.p);
        this.edges = [];
        let mass = 0;
        for (const c of candidates) {
            if (this.edges.length >= MAX_CANDIDATES || mass >= CANDIDATES_MASS) break;
            this.edges.push({ move: maia.actionToMove(c.i), prior: c.p, visits: 0, valueSum: 0, child: null, terminal: false });
            mass += c.p;
        }
        if (this.edges.length === 0) {
            const c = [...pred.cells, pred.swap].reduce((best, p, i) => (p > best.p ? { p, i } : best), { p: -1, i: 0 });
            this.edges.push({ move: maia.actionToMove(c.i), prior: 1, visits: 0, valueSum: 0, child: null, terminal: false });
            mass = 1;
        }
        for (const e of this.edges) e.prior /= mass;
    }

    select() {
        const sqrtN = Math.sqrt(this.visits + 1);
        // Unvisited moves: a bit pessimistic compared to the position value (first play urgency)
        const fpu = Math.max(0, this.value - 0.1);
        let best = null, bestScore = -Infinity;

        for (const e of this.edges) {
            const q = e.visits > 0 ? e.valueSum / e.visits : fpu;
            const score = q + C_PUCT * e.prior * sqrtN / (1 + e.visits);
            if (score > bestScore) {
                bestScore = score;
                best = e;
            }
        }

        return best;
    }
}

/**
 * @param maia Maia instance
 * @param {string[]} moves current position
 * @param {(player: number) => object} contextFor Maia context for a player (rating, clock...)
 * @param {number} playouts
 * @param rootPred prediction already computed for the root
 * @returns {{ move: string, visits: number }[]} root moves with visit counts
 */
export async function search(maia, moves, contextFor, playouts, rootPred) {
    const root = new Node(maia, moves, rootPred);

    for (let i = 0; i < playouts; ++i) {
        let node = root;
        const path = [];
        let value; // for the player who made the last move of the path

        for (;;) {
            const edge = node.select();
            path.push({ node, edge });

            if (edge.terminal) {
                value = 1;
                break;
            }

            if (!edge.child) {
                const childMoves = [...node.moves, edge.move];
                const { board } = maia.buildBoard(childMoves);

                if (maia.hasConnection(board, node.moves.length % 2)) {
                    edge.terminal = true;
                    value = 1;
                    break;
                }

                const player = childMoves.length % 2;
                const pred = await maia.predict(childMoves, contextFor(player));
                edge.child = new Node(maia, childMoves, pred);
                value = 1 - edge.child.value;
                break;
            }

            node = edge.child;
        }

        // Backpropagate, value flips at each level (from the point of view of each node's mover)
        for (let k = path.length - 1; k >= 0; --k) {
            const { node: n, edge } = path[k];
            edge.visits++;
            edge.valueSum += value;
            n.visits++;
            value = 1 - value;
        }
    }

    return root.edges.filter(e => e.visits > 0).map(e => ({ move: e.move, visits: e.visits }));
}

/**
 * Choose among searched moves: sample from visits^(1/T), most visited when T = 0.
 */
export function pickFromVisits(results, temperature) {
    if (!(temperature > 0)) {
        return results.reduce((a, b) => (b.visits > a.visits ? b : a)).move;
    }

    const weights = results.map(r => Math.pow(r.visits, 1 / temperature));
    let x = Math.random() * weights.reduce((a, w) => a + w, 0);
    for (let i = 0; i < results.length; ++i) {
        x -= weights[i];
        if (x <= 0) return results[i].move;
    }

    return results[results.length - 1].move;
}
