import { BoardEntity, GameView, Hex, PlayingGameFacade, PolicyOverlayFacade } from '@playhex/pixi-board';
import { Graphics } from 'pixi.js';
import { Maia } from './maia.js';
import { pickFromVisits, playoutsFor, search } from './search.js';

const $ = id => document.getElementById(id);
const PLAYER_NAMES = ['Red', 'Blue'];

let maia = null;
let gameView = null;
let facade = null;
let overlay = null;

/** Settings fixed for the current game (applied on "Nouvelle partie") */
let game = null;

/** Current game state */
let clocks = [0, 0];      // game seconds remaining, at start of current turn for the mover
let turnSpent = 0;        // game seconds already spent on current turn (before last resume)
let turnSince = 0;        // performance.now() when the current turn (re)started running
let turnRate = 1;         // game seconds per real second for the running turn
let paused = false;
let gameOver = false;
let turnToken = 0;        // invalidates async predictions / scheduled moves of previous turns
let pending = null;       // { move, think, timer } model move waiting to be played
let lastPred = null;      // prediction shown (policy rating)
let lastPlayPred = null;  // prediction used for the model's own move (model rating)
let pausedPlan = null;    // model move drawn before pause, replayed on resume
let history = [];         // { player, move, think, prob, clocksBefore }

// ---------- settings ----------

const checked = id => $(id).checked;
const radio = name => document.querySelector(`input[name=${name}]:checked`).value;
const speed = () => Number(radio('speed')); // 0 = instant

const modelPlays = p => {
    const v = $('model-plays').value;

    return v === 'both' || v === String(p);
};

const mover = () => facade.getMoves().length % 2;

/** Leftmost slider position: newcomer, i.e unrated player (rating unknown to the model) */
const NEWCOMER = 1150;
const COMPARE_RATINGS = [null, 1300, 1600, 1900];

const ratingLabel = r => r === null ? 'Newcomer' : String(r);

/** Rating of a slider, null for newcomer */
function readRating(name) {
    const v = Number($(`rating-${name}-range`).value);

    return v <= NEWCOMER ? null : v;
}

/** Mix overlay: one RGB channel per level, a move liked by all levels is white */
const MIX_LEVELS = [
    { rating: 1300, channel: 1 }, // weak: green
    { rating: 1600, channel: 2 }, // average: blue
    { rating: 1900, channel: 0 }, // strong: red
];

/** Colored disc on a cell, for the mix overlay */
class MixMark extends BoardEntity {
    constructor() {
        super();
        this.color = 0xffffff;
        this.opacity = 0;
    }

    draw() {
        const g = new Graphics();
        g.circle(0, 0, Hex.INNER_RADIUS * 0.55);
        g.fill({ color: this.color, alpha: this.opacity });

        return g;
    }

    set(color, opacity) {
        this.color = color;
        this.opacity = opacity;
        this.redraw();
    }
}

let mixMarks = [];        // MixMark per cell index
let comparePreds = null;  // { rating: prediction } for COMPARE_RATINGS, current position

/**
 * Rating the model plays with as player p: one per color when it plays both sides.
 */
function playRating(p) {
    if ($('model-plays').value === 'both') return readRating(p);

    return readRating('model');
}

/**
 * Rating of the displayed analysis (policy overlay, think time...): same as the model rating,
 * or the rating of the player to move when the model plays both sides.
 */
function viewRating() {
    if ($('model-plays').value === 'both') return readRating(mover());

    return readRating('model');
}

/** Show only the rating fields relevant to who the model plays */
function updateRatingBlocks() {
    const mode = $('model-plays').value;
    $('model-rating-block').hidden = mode === 'both';
    $('both-ratings-block').hidden = mode !== 'both';
    $('model-rating-label').textContent = mode === 'none' ? 'Rating' : 'Model rating';
}

function context(p, rating) {
    return {
        rating,
        deviation: Number($('deviation').value),
        clock: game.useClock ? clocks[p] : null,
        initial: game.initial,
        increment: game.increment,
        vsAi: checked('vs-ai'),
        ranked: checked('ranked'),
        swapAllowed: game.swapAllowed,
    };
}

// ---------- formatting ----------

function fmtDuration(s) {
    if (!isFinite(s)) return '–';
    if (s < 10) return `${s.toFixed(1)} s`;
    if (s < 60) return `${Math.round(s)} s`;
    const m = Math.floor(s / 60);
    const r = Math.round(s % 60);

    return `${m}m ${String(r).padStart(2, '0')}s`;
}

function fmtClock(s) {
    s = Math.max(0, s);
    const m = Math.floor(s / 60);
    const sec = s % 60;

    return s < 20 ? `${m}:${sec.toFixed(1).padStart(4, '0')}` : `${m}:${String(Math.floor(sec)).padStart(2, '0')}`;
}

const pct = p => `${(p * 100).toFixed(p < 0.1 ? 1 : 0)}%`;

const moveLabel = m => m === 'swap-pieces' ? 'swap' : m === 'resign' ? 'resign' : m;

// ---------- clocks ----------

function spentNow() {
    if (paused || gameOver) return turnSpent;

    return turnSpent + (performance.now() - turnSince) / 1000 * turnRate;
}

function renderClocks() {
    if (!game) return;
    const m = mover();

    for (const p of [0, 1]) {
        const el = $(`clock-${p}`);
        el.classList.toggle('active', p === m && !gameOver);
        const remaining = p === m ? clocks[p] - spentNow() : clocks[p];
        el.querySelector('.time').textContent = game.useClock ? fmtClock(remaining) : '–';
        const rating = playRating(p);
        $(`who-${p}`).textContent = `${PLAYER_NAMES[p]} · ${modelPlays(p) ? 'model' : 'you'}`
            + (modelPlays(p) ? ` ${ratingLabel(rating)}` : '');
    }

    if (game.useClock && !gameOver && !paused && clocks[m] - spentNow() <= 0) {
        endGame(1 - m, 'on time');
    }
}

setInterval(renderClocks, 100);

// ---------- game flow ----------

/** Resolves once the browser has painted a frame */
const nextPaint = () => new Promise(resolve => requestAnimationFrame(() => setTimeout(resolve, 0)));

function cancelPending() {
    if (pending) clearTimeout(pending.timer);
    pending = null;
}

async function startTurn() {
    cancelPending();
    const token = ++turnToken;
    turnSpent = 0;
    turnSince = performance.now();
    turnRate = 1;

    if (gameOver) return;

    // Inference runs on the main thread: let the board render the last move first
    await nextPaint();
    if (token !== turnToken) return;

    comparePreds = null; // previous position, would draw on occupied cells
    const p = mover();
    const moves = facade.getMoves();
    const viewPred = await maia.predict(moves, context(p, viewRating()));
    const playPred = modelPlays(p) ? await maia.predict(moves, context(p, playRating(p))) : null;
    if (token !== turnToken) return;

    lastPred = viewPred;
    lastPlayPred = playPred;
    renderAnalysis();
    renderCompare(token);

    if (playPred && !paused) {
        scheduleModelMove(playPred, token);
    }
}

function chooseModelMove(pred) {
    const p = { ...pred };
    if (!checked('allow-resign')) p.resign = 0;

    const temperature = Number($('temperature').value);
    if (!(temperature > 0)) return maia.best(p);

    return maia.sample(p, temperature);
}

const maxPlayouts = () => Number($('max-playouts').value);

/**
 * Clock management: past this share of the remaining clock, a human stops thinking and hurries to play.
 * Hurrying takes a random reaction time (log-normal): with a few seconds left, it can be too late.
 */
const HURRY_SHARE = 0.5;
const HURRY_TIME = { thinkMu: Math.log(1.5), thinkSigma: 0.5 };

/**
 * Think time for player p: the predicted human time, cut short when it would burn too much clock.
 * Flags only when a long think (hard position) meets a short clock: P(flag) = P(think > c/2) * P(reaction >= c/2).
 */
function sampleThink(pred, p) {
    const think = maia.sampleThink(pred);
    if (!game.useClock) return think;

    const budget = clocks[p] * HURRY_SHARE;

    return think <= budget ? think : budget + maia.sampleThink(HURRY_TIME);
}

/** Shows playouts at 2000 and for the current model rating (at median think time) */
function renderPlayouts() {
    const max = maxPlayouts();
    if (max === 0) {
        $('playouts-value').textContent = 'off';
        return;
    }
    const rating = $('model-plays').value === 'both' ? null : readRating('model');
    const current = rating === null || rating === 2000 ? '' : ` · ${playoutsFor(rating, 1, 1, max)} at ${ratingLabel(rating)}`;
    $('playouts-value').textContent = `${max} at 2000${current}`;
}

async function scheduleModelMove(pred, token, plan = null) {
    const p = mover();
    let think = plan?.think ?? sampleThink(pred, p);
    let move = plan?.move ?? null;
    let playouts = plan?.playouts ?? 0;

    if (!move) {
        move = chooseModelMove(pred);

        // Search, unless resigning: more playouts for strong ratings and long thinks
        if (move !== 'resign') {
            playouts = playoutsFor(playRating(p), think, Math.exp(pred.thinkMu), maxPlayouts());
            if (playouts > 0) {
                const results = await search(maia, facade.getMoves(), q => context(q, playRating(q)), playouts, pred);
                if (token !== turnToken || paused || pending) return;
                move = pickFromVisits(results, Number($('temperature').value));
            }
        }
    }

    // Model would lose on time: think until flag
    if (game.useClock && think >= clocks[p]) {
        think = clocks[p] + 0.05;
    }

    const s = speed();
    const remaining = Math.max(0, think - turnSpent);
    turnRate = s === 0 ? 0 : s;
    turnSince = performance.now();

    if (s === 0) {
        // Instant: clock still charged with the sampled think time
        pending = { move, think, playouts, timer: setTimeout(() => token === turnToken && playModelMove(move, think, playouts), 0) };
        return;
    }

    pending = {
        move,
        think,
        playouts,
        timer: setTimeout(() => token === turnToken && playModelMove(move, think, playouts), remaining / s * 1000),
    };
}

function playModelMove(move, think, playouts = 0) {
    pending = null;
    const p = mover();
    if (game.useClock && think >= clocks[p]) {
        turnSpent = clocks[p];
        endGame(1 - p, 'on time');
        return;
    }
    applyMove(move, think, lastPlayPred, playouts);
}

/**
 * @param pred prediction the move is logged against (its probability is shown in history)
 */
function applyMove(move, think, pred = lastPred, playouts = 0) {
    const p = mover();
    const prob = !pred ? null
        : move === 'swap-pieces' ? pred.swap
        : move === 'resign' ? pred.resign
        : pred.cells[maia.parseMove(move)];

    history.push({ player: p, move, think, prob, playouts, clocksBefore: [...clocks] });

    if (move === 'resign') {
        turnSpent = think;
        renderLog();
        endGame(1 - p, 'by resignation');
        return;
    }

    if (game.useClock) {
        clocks[p] = clocks[p] - think + game.increment;
    }

    facade.addMove(move);
    renderLog();

    const { board } = maia.buildBoard(facade.getMoves());
    if (maia.hasConnection(board, p)) {
        endGame(p, 'by connection');
        return;
    }

    startTurn();
}

function endGame(winner, how) {
    if (gameOver) return;
    turnSpent = spentNow();
    gameOver = true;
    cancelPending();
    turnToken++;
    $('banner').textContent = `${PLAYER_NAMES[winner]} wins ${how}`;
    $('banner').style.color = winner === 0 ? 'var(--red)' : 'var(--blue)';
    renderClocks();
}

function onHexClicked(cell) {
    if (!game || gameOver || paused || modelPlays(mover())) return;

    const move = facade.guessMove(cell);
    if (move !== 'swap-pieces' && facade.getStoneAt(move) !== null) return;

    applyMove(move, spentNow());
}

// ---------- rendering ----------

function renderAnalysis() {
    const pred = lastPred;
    const p = mover();
    if (!pred || !overlay) return;

    const hide = !checked('show-policy') || (checked('hide-for-human') && !modelPlays(p) && !gameOver);
    overlay.setShowNumbers(checked('show-numbers'));
    overlay.setShowBestMark(checked('show-best'));
    const mix = radio('overlay-mode') === 'mix';
    $('mix-legend').hidden = !mix;
    renderMix(!hide && mix);
    if (hide || mix) {
        overlay.clear();
    } else {
        const S = maia.size;
        const grid = Array.from({ length: S }, (_, r) => Array.from(pred.cells.slice(r * S, (r + 1) * S)));
        overlay.apply(grid, p === 0 ? 'black' : 'white');
    }

    const rating = viewRating();
    $('pred-who').textContent = `${PLAYER_NAMES[p]}${rating === null ? ', newcomer' : ` at ${rating}`}`;

    const winRed = p === 0 ? pred.win : 1 - pred.win;
    $('win-bar').style.width = `${winRed * 100}%`;
    $('win-red').textContent = `Red ${pct(winRed)}`;
    $('win-blue').textContent = `Blue ${pct(1 - winRed)}`;

    $('think-median').textContent = fmtDuration(Math.exp(pred.thinkMu));
    $('think-range').textContent = `${fmtDuration(Math.exp(pred.thinkMu - 1.2816 * pred.thinkSigma))} – ${fmtDuration(Math.exp(pred.thinkMu + 1.2816 * pred.thinkSigma))}`;
    $('p-resign').textContent = pct(pred.resign);
    $('p-swap').textContent = game.swapAllowed && facade.getMoves().length === 1 ? pct(pred.swap) : '–';

    const actions = [...pred.cells, pred.swap, pred.resign]
        .map((prob, i) => ({ prob, move: maia.actionToMove(i) }))
        .sort((a, b) => b.prob - a.prob)
        .slice(0, 8);
    $('top-moves').innerHTML = actions
        .map(a => `<tr><td>${moveLabel(a.move)}</td><td class="num">${pct(a.prob)}</td></tr>`)
        .join('');
}

/**
 * Top moves of the current position for a few ratings side by side.
 */
async function renderCompare(token) {
    const p = mover();
    const moves = facade.getMoves();
    const preds = [];
    for (const rating of COMPARE_RATINGS) {
        preds.push(await maia.predict(moves, context(p, rating)));
    }
    if (token !== turnToken) return;

    comparePreds = Object.fromEntries(COMPARE_RATINGS.map((r, k) => [r, preds[k]]));
    renderAnalysis();

    const tops = preds.map(pred => [...pred.cells, pred.swap, pred.resign]
        .map((prob, i) => ({ prob, move: maia.actionToMove(i) }))
        .sort((a, b) => b.prob - a.prob)
        .slice(0, 4));
    const current = viewRating();
    const cls = r => r === current ? ' class="current"' : '';

    $('compare').innerHTML = `<tr>${COMPARE_RATINGS.map(r => `<th${cls(r)}>${ratingLabel(r)}</th>`).join('')}</tr>`
        + [0, 1, 2, 3].map(i => `<tr>${tops.map((top, k) => `<td${cls(COMPARE_RATINGS[k])}>${moveLabel(top[i].move)} <span class="p">${pct(top[i].prob)}</span></td>`).join('')}</tr>`).join('')
        + `<tr>${preds.map((pred, k) => `<td${cls(COMPARE_RATINGS[k])}><span class="p" title="Median think time">⏱ ${fmtDuration(Math.exp(pred.thinkMu))}</span></td>`).join('')}</tr>`;
}

/**
 * Mix overlay: each level's probabilities relative to its favorite move, in its own color channel.
 */
function renderMix(visible) {
    const preds = comparePreds && MIX_LEVELS.map(l => comparePreds[l.rating]);
    if (!visible || !preds || preds.some(p => !p)) {
        for (const m of mixMarks) if (m.opacity) m.set(0, 0);
        return;
    }

    const maxes = preds.map(p => Math.max(...p.cells));
    for (let i = 0; i < mixMarks.length; ++i) {
        const rgb = [0, 0, 0];
        MIX_LEVELS.forEach((l, k) => { rgb[l.channel] = maxes[k] > 0 ? preds[k].cells[i] / maxes[k] : 0; });
        const m = Math.max(...rgb);

        if (m < 0.06) {
            if (mixMarks[i].opacity) mixMarks[i].set(0, 0);
            continue;
        }

        // Hue from channel ratios (brightest channel at 255), intensity as opacity
        const [r, g, b] = rgb.map(v => Math.round(v / m * 255));
        mixMarks[i].set((r << 16) | (g << 8) | b, 0.2 + 0.8 * m);
    }
}

function renderLog() {
    $('log-body').innerHTML = history.map((h, i) => `<tr>
        <td>${i + 1}.</td>
        <td><span class="dot ${h.player === 0 ? 'r' : 'b'}"></span>${moveLabel(h.move)}</td>
        <td class="num">${fmtDuration(h.think)}</td>
        <td class="num muted" title="Search playouts">${h.playouts ? `${h.playouts} po` : ''}</td>
        <td class="num muted" title="Probability the model gave to this move">${h.prob === null ? '' : pct(h.prob)}</td>
    </tr>`).join('');
    $('log').scrollTop = $('log').scrollHeight;
}

// ---------- game setup ----------

async function newGame() {
    cancelPending();
    turnToken++;

    game = {
        swapAllowed: checked('swap-allowed'),
        useClock: checked('use-clock'),
        initial: Number($('initial').value) * 60,
        increment: checked('use-clock') ? Number($('increment').value) : 0,
    };
    clocks = [game.initial, game.initial];
    history = [];
    gameOver = false;
    paused = false;
    lastPred = null;
    $('pause').textContent = 'Pause';
    $('banner').textContent = '';
    renderLog();

    await buildBoard([]);
    startTurn();
}

/**
 * (Re)creates the board view with given moves already played.
 */
async function buildBoard(moves) {
    if (gameView) gameView.destroy();
    gameView = new GameView(maia.size, { displayCoords: checked('show-coords') });
    facade = new PlayingGameFacade(gameView, game.swapAllowed, moves);
    overlay = new PolicyOverlayFacade(gameView);
    comparePreds = null;
    mixMarks = [];
    for (let i = 0; i < maia.size * maia.size; ++i) {
        const mark = new MixMark().setCoords({ row: Math.floor(i / maia.size), col: i % maia.size });
        gameView.addEntity(mark, 'mix-overlay');
        mixMarks.push(mark);
    }
    gameView.on('hexClicked', onHexClicked);
    await gameView.mount($('board'));
}

/**
 * Swap rule changed: applies to the current game.
 * Model input changes at every move. Before move 2, swap legality changes too, so the board is rebuilt.
 */
async function onSwapRuleChanged() {
    if (!game) return;
    game.swapAllowed = checked('swap-allowed');
    const moves = facade.getMoves();

    if (moves.length <= 1) {
        cancelPending();
        turnToken++;
        await buildBoard(moves);
    }

    if (!gameOver) startTurn();
    else repredict();
}

function undo() {
    if (!history.length) return;
    cancelPending();

    // Undo until it is a human turn (unless the model plays both sides)
    do {
        const h = history.pop();
        if (h.move !== 'resign') facade.undoLastMove();
        clocks = h.clocksBefore;
    } while (history.length && modelPlays(mover()) && $('model-plays').value !== 'both');

    gameOver = false;
    $('banner').textContent = '';
    renderLog();
    startTurn();
}

function togglePause() {
    if (gameOver) return;

    if (!paused) {
        turnSpent = spentNow();
        paused = true;
        pausedPlan = pending && { move: pending.move, think: pending.think, playouts: pending.playouts };
        cancelPending();
        $('pause').textContent = 'Resume';
        return;
    }

    paused = false;
    turnSince = performance.now();
    turnRate = 1;
    $('pause').textContent = 'Pause';
    if (modelPlays(mover()) && lastPlayPred) scheduleModelMove(lastPlayPred, turnToken, pausedPlan);
    pausedPlan = null;
}

// ---------- wiring ----------

function bindRating(name) {
    const range = $(`rating-${name}-range`);
    const show = () => { $(`rating-${name}`).textContent = ratingLabel(readRating(name)); };
    range.addEventListener('input', () => { show(); repredict(); });
    show();
}

let repredictTimer = null;

/**
 * Context or rating changed: recompute predictions.
 * If the model is thinking, it redraws its move with the new rating (time already spent is kept).
 */
function repredict() {
    clearTimeout(repredictTimer);
    repredictTimer = setTimeout(async () => {
        if (!game) return; // also after game over: analysis of the final position
        const token = turnToken;
        const p = mover();
        const moves = facade.getMoves();
        const pred = await maia.predict(moves, context(p, viewRating()));
        const playPred = modelPlays(p) && !gameOver ? await maia.predict(moves, context(p, playRating(p))) : null;
        if (token !== turnToken) return;

        lastPred = pred;
        renderAnalysis();
        renderCompare(token);

        if (playPred) {
            lastPlayPred = playPred;
            pausedPlan = null;
            if (pending) {
                turnSpent = spentNow();
                cancelPending();
                scheduleModelMove(playPred, token);
            }
        }
    }, 100);
}

function wire() {
    for (const name of ['model', 0, 1]) bindRating(name);
    for (const radio of document.querySelectorAll('input[name=preset]')) {
        radio.addEventListener('input', () => {
            $('rating-model-range').value = radio.value;
            $('rating-model').textContent = ratingLabel(readRating('model'));
            repredict();
        });
    }

    // Touching a rating: show it on the board (mix overlay uses fixed ratings)
    const showSingle = () => { document.querySelector('input[name=overlay-mode][value=single]').checked = true; };
    for (const el of [...document.querySelectorAll('.rating-range'), ...document.querySelectorAll('input[name=preset]')]) {
        el.addEventListener('input', showSingle);
    }

    // Select the matching preset when the rating is set by hand
    $('rating-model-range').addEventListener('input', () => {
        for (const radio of document.querySelectorAll('input[name=preset]')) radio.checked = radio.value === $('rating-model-range').value;
    });
    updateRatingBlocks();
    for (const id of ['deviation', 'ranked', 'vs-ai']) {
        $(id).addEventListener('input', repredict);
    }
    const showTemperature = () => {
        const t = Number($('temperature').value);
        $('temperature-value').textContent = t > 0 ? t.toFixed(2) : 'best move';
    };
    $('temperature').addEventListener('input', showTemperature);
    showTemperature();
    for (const id of ['show-policy', 'show-numbers', 'show-best', 'hide-for-human']) {
        $(id).addEventListener('input', renderAnalysis);
    }
    for (const r of document.querySelectorAll('input[name=overlay-mode]')) {
        r.addEventListener('input', renderAnalysis);
    }
    $('show-coords').addEventListener('input', () => gameView?.setDisplayCoords(checked('show-coords')));
    $('model-plays').addEventListener('input', () => { updateRatingBlocks(); startTurn(); });
    $('new-game').addEventListener('click', newGame);
    $('swap-allowed').addEventListener('input', onSwapRuleChanged);
    for (const el of [$('max-playouts'), $('rating-model-range'), $('model-plays'), ...document.querySelectorAll('input[name=preset]')]) {
        el.addEventListener('input', renderPlayouts);
    }
    renderPlayouts();
    $('undo').addEventListener('click', undo);
    $('pause').addEventListener('click', togglePause);
    $('play-model').addEventListener('click', () => {
        if (!game || gameOver || !lastPred) return;
        cancelPending();
        // Model side: plays with its rating. Human side: plays with the policy rating
        const pred = modelPlays(mover()) && lastPlayPred ? lastPlayPred : lastPred;
        applyMove(chooseModelMove(pred), spentNow(), pred);
    });
    $('resign').addEventListener('click', () => {
        if (!game || gameOver) return;
        const m = mover();
        const human = !modelPlays(m) ? m : !modelPlays(1 - m) ? 1 - m : m;
        if (human === m) {
            applyMove('resign', spentNow());
        } else {
            endGame(1 - human, 'by resignation');
        }
    });
}

async function init() {
    wire();

    try {
        maia = await Maia.load('model.onnx');
    } catch (e) {
        console.error(e);
        $('status').textContent = 'Model not found (model.onnx): run export_onnx.py, or load a file:';
        $('model-file').style.display = 'flex';
        await new Promise(resolve => {
            $('model-file').querySelector('input').addEventListener('change', async ev => {
                const buf = new Uint8Array(await ev.target.files[0].arrayBuffer());
                maia = await Maia.load(buf);
                $('model-file').style.display = 'none';
                resolve();
            });
        });
    }

    $('status').textContent = 'Click a cell to play. Click the first stone to swap.';
    await newGame();
}

init();
