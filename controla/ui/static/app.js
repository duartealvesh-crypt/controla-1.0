/* Controla — interface unique, sans framework ni étape de build.
 *
 * Règle de cette couche : elle N'INVENTE AUCUN CHIFFRE. Tout ce qui s'affiche
 * vient d'un instantané, d'un manifeste ou du masque d'actions légales servi
 * par le moteur. Si un nombre manque à l'écran, il manque en amont — recalculer
 * ici ferait diverger l'écran de la partie, ce que tout le projet s'attache à
 * empêcher ailleurs.
 */

'use strict';

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

const state = {
  ctx: null,       // /api/context, chargé une fois
  game: null,      // partie affichée dans le replay
  frames: [],      // un instantané de plateau par tour, 0 = état initial
  frame: 0,
  timer: null,
  session: null,   // partie jouable en cours
  pending: null,   // action choisie, en attente d'une cible sur la carte
};

const FACTION_GLYPH = ['◆', '■', '▲', '●'];

/* Glyphe d'un bâtiment : TOUJOURS celui que sert le moteur. Le dériver du
 * libellé collisionnerait — « Camp militaire » et « Champs » commencent tous
 * deux par C — et deux bâtiments deviendraient indiscernables sur la carte. */
function glyphOf(key, world) {
  const table = (world && world.glyphs) || (state.ctx && state.ctx.glyphs) || {};
  return table[key] || (key ? key[0].toUpperCase() : '?');
}

/* Sprites pixel art (provisoires, cf. controla/ui/make_sprites.py), un par clé
 * du catalogue. Une clé sans sprite — terrain ou bâtiment d'un mod — garde son
 * glyphe : la carte reste lisible sans asset. */
const SPRITES = new Set(['plain', 'river', 'forest', 'city', 'lab', 'camp', 'field']);

function iconOf(key, world) {
  return SPRITES.has(key)
    ? el('img', { class: 'sprite', src: `/static/sprites/${key}.png`, alt: glyphOf(key, world) })
    : el('span', { class: 'glyph' }, glyphOf(key, world));
}

/* Légende commune au replay et au mode jouable : terrains puis bâtiments. */
function legendIcons(world, terrainKeys, buildingKeys, labels) {
  const entry = (key, icon) => el('span', { class: 'dim legend-item' },
    icon, labels[key] || key);
  return [
    ...terrainKeys.map((key) => entry(key, SPRITES.has(key)
      ? iconOf(key, world) : el('span', { class: `swatch t-${key}` }))),
    ...buildingKeys.map((key) => entry(key, iconOf(key, world))),
  ];
}

function setStatus(text, kind = '') {
  const node = $('#status');
  node.textContent = text || '';
  node.className = kind;
}

async function api(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = response.statusText;
    try { detail = (await response.json()).detail || detail; } catch { /* corps non JSON */ }
    throw new Error(detail);
  }
  return response.json();
}

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === 'class') node.className = value;
    else if (key === 'html') node.innerHTML = value;
    else if (key.startsWith('on')) node.addEventListener(key.slice(2), value);
    else if (value !== null && value !== undefined) node.setAttribute(key, value);
  }
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

const num = (value, digits = 1) =>
  value === null || value === undefined ? '—' : Number(value).toFixed(digits);
const pct = (value) =>
  value === null || value === undefined ? '—' : `${(value * 100).toFixed(0)} %`;

/* ── Navigation ───────────────────────────────────────────────────────── */

function show(view) {
  $$('#tabs button').forEach((b) => b.classList.toggle('on', b.dataset.view === view));
  $$('.view').forEach((s) => s.classList.toggle('on', s.id === `view-${view}`));
  if (view === 'models') loadModels();
  if (view === 'library') loadLibrary();
}

$$('#tabs button').forEach((b) => b.addEventListener('click', () => show(b.dataset.view)));
document.addEventListener('click', (event) => {
  const goto = event.target.closest('[data-goto]');
  if (goto) { event.preventDefault(); show(goto.dataset.goto); }
});

/* ── Bibliothèque ─────────────────────────────────────────────────────── */

async function loadLibrary() {
  try {
    const data = await api('/api/runs');
    $('#root-path').textContent = data.root;

    const runs = $('#runs-list');
    runs.replaceChildren();
    if (!data.runs.length) {
      runs.append(el('p', { class: 'hint' },
        'Aucune campagne. Lancez par exemple : ',
        el('code', {}, 'python -m controla.cli tournament --seeds 1,2,3')));
    }
    for (const run of data.runs) {
      runs.append(el('div', { class: 'card', onclick: () => openCampaign(run.path) },
        el('div', { class: 'title' }, run.run),
        el('div', { class: 'meta' },
          `${run.games} parties · ${run.seats.join(', ')}`,
          el('br'),
          run.leader ? `meneur ${run.leader}` : '',
          run.domination !== null && run.domination !== undefined
            ? ` · domination ${pct(run.domination)}` : '')));
    }

    const loose = $('#loose-list');
    loose.replaceChildren();
    if (!data.loose.length) {
      loose.append(el('p', { class: 'hint' },
        'Aucune partie isolée. ',
        el('code', {}, 'python -m controla.cli play --snapshots --out runs/demo.json')));
    }
    for (const entry of data.loose) {
      loose.append(el('div', { class: 'card', onclick: () => openGame(entry.path) },
        el('div', { class: 'title' }, entry.name),
        el('div', { class: 'meta' }, entry.path)));
    }
    setStatus('');
  } catch (error) {
    setStatus(`bibliothèque : ${error.message}`, 'bad');
  }
}

/* ── Replay ───────────────────────────────────────────────────────────── */

async function openGame(path) {
  show('replay');
  setStatus('chargement…');
  try {
    state.game = await api(`/api/game?path=${encodeURIComponent(path)}`);
    buildFrames();
    $('#replay-empty').classList.add('hidden');
    $('#replay-body').classList.remove('hidden');
    $('#rp-title').textContent =
      `seed ${state.game.seed} · ${(state.game.seats || []).join(' / ')}`;
    $('#rp-slider').max = String(Math.max(0, state.frames.length - 1));
    renderLegend($('#rp-legend'));
    gotoFrame(0);
    setStatus('');
  } catch (error) {
    setStatus(`partie : ${error.message}`, 'bad');
  }
}

/* Une image par tour. L'indice 0 est l'état initial, l'indice N la fin du
 * tour N — donc `frames[i]` et `history[i-1]` parlent bien du même tour. */
function buildFrames() {
  const game = state.game;
  const frames = [];
  if (game.world && game.world.initial_board) {
    frames.push({ turn: 0, board: game.world.initial_board, report: null });
  }
  for (const report of game.history || []) {
    if (report.board) frames.push({ turn: report.turn, board: report.board, report });
    else frames.push({ turn: report.turn, board: null, report });
  }
  state.frames = frames;
}

function gotoFrame(index) {
  const frames = state.frames;
  if (!frames.length) return;
  state.frame = Math.max(0, Math.min(index, frames.length - 1));
  const frame = frames[state.frame];
  $('#rp-slider').value = String(state.frame);
  $('#rp-turn').textContent =
    `tour ${frame.turn}/${(state.game.world || {}).max_turns || state.game.turns_played}`;

  const world = state.game.world;
  if (world && frame.board) {
    renderMap($('#rp-map'), world, frame.board);
    renderFactions($('#rp-factions'), world, frame.board, state.game.ranking);
    renderCharts($('#rp-charts'));
  } else {
    $('#rp-map').replaceChildren();
    $('#rp-map').append(el('p', { class: 'hint' },
      'Partie enregistrée sans instantanés : pas de carte ni de courbes. ' +
      'Rejouez-la avec --snapshots pour les obtenir.'));
    $('#rp-factions').replaceChildren();
    $('#rp-charts').replaceChildren();
  }
  renderEvents($('#rp-events'), frame.report);
  // Les actions affichées à l'image N ont été décidées au tour N−1 (elles ne
  // s'appliquent qu'à la barrière, puis la passe fait passer au tour N). Le
  // raisonnement montré à côté doit donc être celui du tour N−1 — sans quoi on
  // lit une justification à côté d'actions qu'elle n'a pas produites.
  renderReasoning($('#rp-reasoning'), frame.turn - 1);
}

function names(game) {
  const world = game.world;
  if (world && world.factions) return world.factions.map((f) => f.name);
  const out = [];
  for (const entry of game.ranking || []) out[entry.faction_id] = entry.name;
  return out;
}

/* ── Rendu de la carte ────────────────────────────────────────────────── */

function renderMap(container, world, board) {
  const size = world.size;
  container.replaceChildren();
  container.style.gridTemplateColumns = `repeat(${size}, 1fr)`;

  const atWar = new Set();
  for (const faction of board.factions || []) {
    for (const other of faction.at_war_with || []) atWar.add(faction.id);
  }

  for (let i = 0; i < size * size; i += 1) {
    const owner = board.owner[i];
    const buildingIndex = board.building[i];
    const terrain = world.terrain_keys[world.terrain[i]];
    const x = i % size;
    const y = Math.floor(i / size);
    const classes = ['tile', `t-${terrain}`];
    if (owner >= 0) {
      classes.push('owned', `o${owner}`);
      // Frontière : un côté dont la case voisine (ou le bord) n'est pas à nous.
      const other = (nx, ny) => nx < 0 || ny < 0 || nx >= size || ny >= size
        || board.owner[ny * size + nx] !== owner;
      if (other(x, y - 1)) classes.push('bn');
      if (other(x + 1, y)) classes.push('be');
      if (other(x, y + 1)) classes.push('bs');
      if (other(x - 1, y)) classes.push('bw');
    }
    if (owner >= 0 && atWar.has(owner)) classes.push('war');

    const buildingKey = buildingIndex >= 0 ? world.building_keys[buildingIndex] : null;
    const ownerName = owner >= 0 ? (world.factions[owner] || {}).name : 'neutre';
    container.append(el('div', {
      class: classes.join(' '),
      'data-x': x, 'data-y': y,
      title: `(${x},${y}) · ${world.labels[terrain] || terrain} · ${ownerName}` +
             (buildingKey ? ` · ${world.labels[buildingKey] || buildingKey}` : ''),
    }, buildingKey ? iconOf(buildingKey, world) : null));
  }
}

function renderLegend(container) {
  const ctx = state.ctx;
  container.replaceChildren();
  const world = (state.game && state.game.world) || null;
  const factions = world ? world.factions
    : ctx.labels && [...Array(ctx.num_factions).keys()].map((i) => ({ id: i, name: `F${i}` }));
  for (const faction of factions || []) {
    container.append(el('span', {},
      el('span', { class: 'swatch', style: `background: var(--f${faction.id})` }),
      faction.name));
  }
  container.append(...legendIcons(world, ctx.terrain_keys || [], ctx.building_keys, ctx.labels));
  container.append(el('span', { class: 'dim' }, '┄ frontière en pointillés blancs = en guerre'));
}

function renderFactions(container, world, board, ranking) {
  container.replaceChildren();
  const ranks = {};
  for (const entry of ranking || []) ranks[entry.faction_id] = entry.rank;
  const rows = [...(board.factions || [])].sort(
    (a, b) => (b.score || 0) - (a.score || 0));

  for (const faction of rows) {
    const name = (world.factions[faction.id] || {}).name || `F${faction.id}`;
    const buildings = Object.entries(faction.buildings || {})
      .map(([key, count]) => `${glyphOf(key, world)}${count}`)
      .join(' ') || '—';
    container.append(el('div', {
      class: `faction f-border-${faction.id}${faction.alive ? '' : ' dead'}`,
    },
      el('div', { class: 'name' },
        el('span', { class: `f-text-${faction.id}` }, FACTION_GLYPH[faction.id]),
        name,
        el('span', { class: 'grow' }),
        el('span', { class: 'mono' }, `score ${num(faction.score)}`)),
      el('div', { class: 'stats' },
        `pop ${num(faction.population)}  cases ${faction.tiles}  ` +
        `rech ${num(faction.research)} (T${faction.research_tier})  ` +
        `force ${num(faction.force)}`),
      el('div', { class: 'stats' },
        `nour ${num(faction.food)}  bois ${num(faction.wood)}  mét ${num(faction.metal)}  ` +
        `faim ${pct(faction.hunger)}  bât ${buildings}`),
      el('div', { class: 'stats' },
        `parole ${faction.reputation.honored}/${faction.reputation.total || 0}` +
        (faction.alive ? '' : `  · ÉLIMINÉ T${faction.eliminated_turn}`))));
  }
}

/* ── Événements ───────────────────────────────────────────────────────── */

/* La cible d'une action, en clair. Trois formes coexistent dans les
 * événements : un identifiant de faction, une case [x, y], et pour une
 * négociation le triplet [partenaire, engagement, victime]. */
function describeTarget(target, who) {
  if (target === null || target === undefined) return '';
  const tinted = (id) => el('span', { class: `f-text-${id}` }, who(id));
  if (typeof target === 'number') return [' → ', tinted(target)];
  if (Array.isArray(target) && target.length === 2) return ` → case (${target[0]},${target[1]})`;
  if (Array.isArray(target) && target.length === 3) {
    const [partner, commitment, victim] = target;
    const parts = [' → ', tinted(partner)];
    if (commitment && commitment !== 'none') {
      parts.push(` [${commitment.replace('_', ' ')}`);
      if (typeof victim === 'number' && victim >= 0) parts.push(' contre ', tinted(victim));
      parts.push(']');
    }
    return parts;
  }
  return ` → ${JSON.stringify(target)}`;
}

function renderEvents(container, report) {
  container.replaceChildren();
  if (!report) {
    container.append(el('div', { class: 'empty' }, 'État initial — rien ne s\'est encore passé.'));
    return;
  }
  const who = (id) => {
    const list = names(state.game);
    return id === null || id === undefined ? '?' : (list[id] || `F${id}`);
  };
  const tinted = (id, text) => el('span', { class: `f-text-${id}` }, text);

  const rows = [];
  for (const event of report.events || []) {
    const actor = event.faction;
    const cause = event.cause ? el('span', { class: 'cause' }, ` — ${event.cause}`) : '';
    switch (event.kind) {
      case 'divine_action': {
        const effects = (event.effects || []).map((e) => e.effect).join(', ');
        rows.push(['✦ ', tinted(actor, who(actor)),
          ` : ${state.ctx.labels[event.action] || event.action}`,
          describeTarget(event.target, who),
          effects ? el('span', { class: 'cause' }, `  [${effects}]`) : '']);
        break;
      }
      case 'expansion':
        rows.push(['▸ ', tinted(actor, who(actor)), ` s'étend en (${event.tile})`, cause]);
        break;
      case 'build':
        rows.push(['⌂ ', tinted(actor, who(actor)),
          ` bâtit ${state.ctx.labels[event.building] || event.building} en (${event.tile})`, cause]);
        break;
      case 'trade':
        rows.push(['⇄ Commerce ', ...(event.factions || []).map((f) => tinted(f, who(f) + ' ')),
          `: ${JSON.stringify(event.moved)}`, cause]);
        break;
      case 'war_launched':
        rows.push(['⚔ ', tinted(actor, who(actor)), ' lance l\'assaut sur ',
          tinted(event.target, who(event.target)), ` en (${event.tile})`,
          event.surprise ? ' ⚡SURPRISE' : '', cause]);
        break;
      case 'research_tier':
        rows.push(['✧ ', tinted(actor, who(actor)), ` atteint le palier ${event.tier}`]);
        break;
      case 'negotiation_delivered':
        rows.push(['✉ ', tinted(actor, who(actor)), ' → ', tinted(event.target, who(event.target)),
          ` « ${event.pledge} »`, event.argument
            ? el('span', { class: 'cause' }, ` « ${event.argument} »`) : '']);
        break;
      case 'commitment_formed':
        rows.push(['🤝 engagement ', event.commitment, ' : ',
          ...(event.parties || []).map((f) => tinted(f, who(f) + ' ')), cause]);
        break;
      case 'commitment_honoured':
        rows.push(['✔ parole tenue : ', ...(event.parties || []).map((f) => tinted(f, who(f) + ' ')), cause]);
        break;
      case 'commitment_betrayed':
        rows.push(['✘ ', tinted(actor, who(actor)), ' rompt son engagement', cause]);
        break;
      case 'elimination':
        rows.push(['☠ ', tinted(actor, who(actor)), ' est éliminé', cause]);
        break;
      case 'disease': case 'natural_disaster': case 'discovery':
        rows.push(['🌍 ', tinted(actor, who(actor)), ' : ', event.cause || event.kind]);
        break;
      case 'drift': {
        const moved = Object.entries(event.moved || {})
          .map(([k, v]) => `${state.ctx.labels[k] || k}${v >= 0 ? '+' : ''}${Number(v).toFixed(3)}`)
          .join(', ');
        rows.push(['↝ ', tinted(actor, who(actor)), ` dérive : ${moved}`]);
        break;
      }
      default:
        break;   // effect_resolved, attack_aborted… : bruit de mécanique
    }
  }

  for (const battle of report.battles || []) {
    const won = battle.attacker_won;
    rows.push(['⚔ ', tinted(battle.attacker, who(battle.attacker)), ' vs ',
      tinted(battle.defender, who(battle.defender)), ' : ',
      el('strong', { class: won ? 'f-text-2' : 'f-text-0' }, won ? 'VICTOIRE' : 'DÉFAITE'),
      el('span', { class: 'cause' },
        ` p=${num(battle.win_probability, 2)} · ${num(battle.attacker_force, 0)}` +
        ` vs ${num(battle.defender_force, 0)} · pertes ${num(battle.attacker_losses)}` +
        ` / ${num(battle.defender_losses)}` +
        (battle.annexed ? ` · annexe (${battle.annexed})` : ''))]);
  }

  if (!rows.length) {
    container.append(el('div', { class: 'empty' }, 'Rien de notable ce tour-ci.'));
    return;
  }
  for (const row of rows) container.append(el('div', { class: 'row' }, row));
}

function renderReasoning(container, decisionTurn) {
  container.replaceChildren();
  const reports = (state.game || {}).policy_reports || {};
  if (!Object.keys(reports).length) {
    container.append(el('div', { class: 'empty' },
      'Partie sans siège LLM — pas de raisonnement à montrer.'));
    return;
  }
  if (decisionTurn < 0) {
    container.append(el('div', { class: 'empty' },
      'État initial. Les décisions du tour 0 apparaissent à l\'image suivante, ' +
      'à côté des actions qu\'elles ont produites.'));
    return;
  }
  let found = 0;
  for (const [agent, report] of Object.entries(reports)) {
    for (const entry of report.reasonings || []) {
      if (entry.turn !== decisionTurn) continue;
      found += 1;
      const badges = [];
      if (report.memory) badges.push(el('span', { class: 'badge ok' }, 'mémoire'));
      if (entry.status === 'truncated') {
        badges.push(el('span', { class: 'badge warn' }, 'tronqué (max_tokens)'));
      } else if (entry.status === 'blank') {
        badges.push(el('span', { class: 'badge warn' }, 'réponse vide'));
      } else if (entry.status === 'filtered') {
        badges.push(el('span', { class: 'badge warn' }, 'filtré par le fournisseur'));
      } else if (entry.status !== 'ok') {
        badges.push(el('span', { class: 'badge bad' }, entry.status));
      }
      const meta = [
        entry.plan && entry.plan.length ? `${entry.plan.length} coups` : 'plan vide',
        entry.attempts > 1 ? `${entry.attempts} essais` : null,
        entry.finish_reason ? `fin : ${entry.finish_reason}` : null,
      ].filter(Boolean).join(' · ');
      const native = entry.native_reasoning
        ? el('details', { class: 'native' },
            el('summary', { class: 'dim' },
              `raisonnement interne (${entry.native_reasoning.length.toLocaleString('fr-FR')} car.)`),
            el('pre', {}, entry.native_reasoning))
        : '';
      container.append(el('div', { class: 'row' },
        el('details', {},
          el('summary', {},
            el('span', { class: `f-text-${entry.faction_id}` }, report.model || agent),
            ' ', ...badges, el('span', { class: 'dim' }, ` ${meta}`)),
          el('pre', {}, entry.reasoning || '(aucun raisonnement rendu)'),
          native)));
    }
  }
  if (!found) {
    container.append(el('div', { class: 'empty' }, 'Aucune décision LLM enregistrée pour ce tour.'));
  }
}

/* ── Courbes ──────────────────────────────────────────────────────────── */

function renderCharts(container) {
  container.replaceChildren();
  const frames = state.frames.filter((f) => f.board);
  if (frames.length < 2) return;
  const world = state.game.world;
  const series = [
    ['score', 'Score'],
    ['population', 'Population'],
    ['tiles', 'Territoire'],
    ['research', 'Recherche'],
  ];
  for (const [key, caption] of series) {
    container.append(lineChart(caption, frames, world, key));
  }
}

function lineChart(caption, frames, world, key) {
  const width = 320;
  const height = 90;
  const pad = 4;
  const count = world.factions.length;

  let max = 0;
  for (const frame of frames) {
    for (const faction of frame.board.factions) max = Math.max(max, faction[key] || 0);
  }
  max = max || 1;

  const paths = [];
  for (let id = 0; id < count; id += 1) {
    const points = frames.map((frame, i) => {
      const faction = frame.board.factions.find((f) => f.id === id) || {};
      const x = pad + (i / (frames.length - 1)) * (width - 2 * pad);
      const y = height - pad - ((faction[key] || 0) / max) * (height - 2 * pad);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(' ');
    paths.push(`<polyline fill="none" stroke="var(--f${id})" stroke-width="1.6" points="${points}"/>`);
  }
  const cursorX = pad + (Math.min(state.frame, frames.length - 1) / (frames.length - 1)) * (width - 2 * pad);
  paths.push(`<line x1="${cursorX}" y1="0" x2="${cursorX}" y2="${height}" ` +
             `stroke="var(--dim)" stroke-width="1" stroke-dasharray="2 2"/>`);

  return el('div', { class: 'chart' },
    el('div', { class: 'caption' }, `${caption} — max ${num(max)}`),
    el('div', {
      html: `<svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">${paths.join('')}</svg>`,
    }));
}

/* ── Contrôles du replay ──────────────────────────────────────────────── */

$('#rp-slider').addEventListener('input', (e) => gotoFrame(Number(e.target.value)));
$('#rp-first').addEventListener('click', () => gotoFrame(0));
$('#rp-prev').addEventListener('click', () => gotoFrame(state.frame - 1));
$('#rp-next').addEventListener('click', () => gotoFrame(state.frame + 1));
$('#rp-last').addEventListener('click', () => gotoFrame(state.frames.length - 1));
$('#rp-play').addEventListener('click', () => {
  if (state.timer) {
    clearInterval(state.timer); state.timer = null; $('#rp-play').textContent = '▶';
    return;
  }
  $('#rp-play').textContent = '⏸';
  state.timer = setInterval(() => {
    if (state.frame >= state.frames.length - 1) {
      clearInterval(state.timer); state.timer = null; $('#rp-play').textContent = '▶';
      return;
    }
    gotoFrame(state.frame + 1);
  }, 900);
});
document.addEventListener('keydown', (event) => {
  if (!$('#view-replay').classList.contains('on')) return;
  if (event.target.matches('input, textarea, select')) return;
  if (event.key === 'ArrowRight') gotoFrame(state.frame + 1);
  if (event.key === 'ArrowLeft') gotoFrame(state.frame - 1);
});

/* ── Campagne ─────────────────────────────────────────────────────────── */

async function openCampaign(path) {
  show('campaign');
  setStatus('chargement…');
  try {
    const data = await api(`/api/run?path=${encodeURIComponent(path)}`);
    renderCampaign(data.manifest, data.games);
    $('#campaign-empty').classList.add('hidden');
    $('#campaign-body').classList.remove('hidden');
    setStatus('');
  } catch (error) {
    setStatus(`campagne : ${error.message}`, 'bad');
  }
}

function verdictBadge(value) {
  if (value === null || value === undefined) return el('span', { class: 'badge na' }, '—');
  return el('span', { class: `badge ${value ? 'ok' : 'bad'}` }, value ? 'OK' : 'NON');
}

function renderCampaign(manifest, games) {
  const body = $('#campaign-body');
  body.replaceChildren();
  const spec = manifest.spec || {};

  body.append(el('h2', {}, `${manifest.run} — ${manifest.games} parties`));
  body.append(el('p', { class: 'hint' },
    `${(spec.seeds || []).length} seeds × ${spec.rotations} rotations × ` +
    `${spec.repetitions} répétitions · ${num(manifest.wall_seconds, 1)} s`));

  /* Classement */
  body.append(el('h3', {}, 'Classement — l\'ordinal μ − 3σ fait foi'));
  const rank = el('table', {}, el('thead', {}, el('tr', {},
    el('th', {}, '#'), el('th', {}, 'participant'),
    el('th', { class: 'num' }, 'μ−3σ'), el('th', { class: 'num' }, 'μ'),
    el('th', { class: 'num' }, 'σ'), el('th', { class: 'num' }, 'parties'),
    el('th', { class: 'num' }, 'victoires'), el('th', { class: 'num' }, 'rang moyen'))));
  const rankBody = el('tbody');
  (manifest.leaderboard || []).forEach((entry, i) => {
    rankBody.append(el('tr', {},
      el('td', {}, String(i + 1)), el('td', {}, entry.name),
      el('td', { class: 'num' }, num(entry.ordinal, 2)),
      el('td', { class: 'num' }, num(entry.mu, 2)),
      el('td', { class: 'num' }, num(entry.sigma, 2)),
      el('td', { class: 'num' }, entry.games),
      el('td', { class: 'num' }, entry.wins),
      el('td', { class: 'num' }, num(entry.mean_rank, 2))));
  });
  rank.append(rankBody);
  body.append(rank);

  /* Les cinq métriques §12 */
  body.append(el('h3', {}, 'Les cinq métriques (§12)'));
  const metrics = el('table', {}, el('thead', {}, el('tr', {},
    el('th', {}, 'participant'), el('th', { class: 'num' }, 'parole'),
    el('th', { class: 'num' }, 'exactitude'), el('th', { class: 'num' }, 'focus'),
    el('th', { class: 'num' }, 'engagement'), el('th', { class: 'num' }, 'alliances/partie'),
    el('th', { class: 'num' }, 'survie'), el('th', { class: 'num' }, 'score moyen'))));
  const metricsBody = el('tbody');
  for (const [name, report] of Object.entries(manifest.metrics || {})) {
    metricsBody.append(el('tr', {},
      el('td', {}, name),
      el('td', { class: 'num' }, num(report.parole.rate, 2)),
      el('td', { class: 'num' }, num(report.exactitude.rate, 2)),
      el('td', { class: 'num' }, num(report.coherence.focus, 2)),
      el('td', { class: 'num' }, num(report.coherence.engagement, 2)),
      el('td', { class: 'num' }, num(report.alliances.formed_per_game, 2)),
      el('td', { class: 'num' }, num(report.survie.survival_rate, 2)),
      el('td', { class: 'num' }, num(report.score.mean, 1))));
  }
  metrics.append(metricsBody);
  body.append(metrics);

  /* Consommation des sièges LLM — la matière de l'étude mémoire */
  const llmRows = Object.entries(manifest.metrics || {})
    .filter(([, report]) => report.exactitude && report.exactitude.llm);
  if (llmRows.length) {
    body.append(el('h3', {}, 'Sièges LLM — consommation et santé'));
    const table = el('table', {}, el('thead', {}, el('tr', {},
      el('th', {}, 'participant'), el('th', {}, 'mémoire'),
      el('th', { class: 'num' }, 'appels'), el('th', { class: 'num' }, 'réponse'),
      el('th', { class: 'num' }, 'tokens entrée/partie'),
      el('th', { class: 'num' }, 'cache'), el('th', { class: 'num' }, 'raisonnement'),
      el('th', { class: 'num' }, 'tours muets'), el('th', { class: 'num' }, 'dont tronqués'),
      el('th', { class: 'num' }, 'dont filtrés'), el('th', { class: 'num' }, 'dont vides'))));
    const tbody = el('tbody');
    for (const [name, report] of llmRows) {
      const llm = report.exactitude.llm;
      const memory = llm.memory_seats
        ? el('span', { class: 'badge ok' }, 'oui')
        : el('span', { class: 'badge na' }, 'non');
      tbody.append(el('tr', {},
        el('td', {}, name), el('td', {}, memory),
        el('td', { class: 'num' }, llm.calls ?? '—'),
        el('td', { class: 'num' }, pct(llm.response_rate)),
        el('td', { class: 'num' }, num(llm.prompt_tokens_per_game, 0)),
        el('td', { class: 'num' }, pct(llm.cache_hit_rate)),
        el('td', { class: 'num' }, llm.reasoning_tokens ?? '—'),
        el('td', { class: 'num' }, llm.mute_turns ?? 0),
        el('td', { class: 'num' }, llm.truncated_turns ?? 0),
        el('td', { class: 'num' }, llm.filtered_turns ?? 0),
        el('td', { class: 'num' }, llm.blank_turns ?? 0)));
    }
    table.append(tbody);
    body.append(table);
    body.append(el('p', { class: 'hint' },
      'Un tour tronqué est un plafond max_tokens trop serré, pas un modèle ' +
      'défaillant : relever max_tokens pour ce modèle dans models.yaml. Un tour ' +
      'filtré a été coupé par le filtre de contenu du fournisseur — il tombe ' +
      'surtout quand le modèle planifie une attaque. Le taux ' +
      'de cache est la part de l\'entrée servie par le cache du fournisseur.'));
  }

  /* Critères §13 */
  const criteria = manifest.criteria;
  if (criteria) {
    body.append(el('h3', {}, 'Les trois critères de calibration (§13)'));
    const table = el('table', {}, el('thead', {}, el('tr', {},
      el('th', {}, 'critère'), el('th', {}, 'verdict'), el('th', {}, 'mesure'))));
    table.append(el('tbody', {},
      el('tr', {}, el('td', {}, '(a) horizon'), el('td', {}, verdictBadge(criteria.verdict.a_horizon)),
        el('td', { class: 'mono dim' },
          `tour moyen ${num(criteria.horizon.mean_turns, 2)} · ` +
          `${pct(criteria.horizon.reached_max)} au bout · ` +
          `${num(criteria.horizon.eliminations_per_game, 2)} élimination/partie`)),
      el('tr', {}, el('td', {}, '(b) équilibre'), el('td', {}, verdictBadge(criteria.verdict.b_balance)),
        el('td', { class: 'mono dim' },
          `domination ${pct(criteria.balance.domination)} · ` +
          `plancher ${pct(criteria.balance.floor)} · ` +
          `étalement ${num(criteria.balance.spread, 2)}`)),
      el('tr', {}, el('td', {}, '(c) skill'), el('td', {}, verdictBadge(criteria.verdict.c_skill)),
        el('td', { class: 'mono dim' },
          `écart au hasard ${num(criteria.skill.over_chance, 1)} · ` +
          `amplitude ${num(criteria.skill.separation, 1)}`))));
    body.append(table);
    body.append(el('p', { class: 'hint' },
      'Le critère (b) ne se lit que sur un panel construit pour l\'équilibre : ' +
      'un panel contenant noop ou random_legal a une domination mécaniquement ≥ 50 %.'));
  }

  /* Parties */
  body.append(el('h3', {}, `Parties (${games.length})`));
  const list = el('div', { class: 'cards' });
  for (const game of games) {
    list.append(el('div', { class: 'card', onclick: () => openGame(game.path) },
      el('div', { class: 'title' }, game.name),
      el('div', { class: 'meta' }, 'ouvrir le replay')));
  }
  body.append(list);
}

/* ── Partie jouable ───────────────────────────────────────────────────── */

function renderSeatPicker() {
  const container = $('#pl-seats');
  container.replaceChildren();
  const scripted = state.ctx.seats.scripted || [];
  const models = state.ctx.seats.models || [];
  const defaults = ['human', 'cooperator', 'expansionist', 'turtle'];

  for (let seat = 0; seat < state.ctx.num_factions; seat += 1) {
    const select = el('select', { id: `pl-seat-${seat}` });
    select.append(el('option', { value: 'human' }, '— vous —'));
    const scriptedGroup = el('optgroup', { label: 'archétypes scriptés' });
    for (const name of scripted) scriptedGroup.append(el('option', { value: name }, name));
    select.append(scriptedGroup);
    if (models.length) {
      const modelGroup = el('optgroup', { label: 'modèles (appels réels)' });
      for (const name of models) modelGroup.append(el('option', { value: name }, name));
      select.append(modelGroup);
    }
    select.value = defaults[seat] || scripted[0] || 'human';
    container.append(el('label', {}, `Siège ${seat}`, select));
  }
}

$('#pl-start').addEventListener('click', async () => {
  const seats = [];
  for (let seat = 0; seat < state.ctx.num_factions; seat += 1) {
    seats.push($(`#pl-seat-${seat}`).value);
  }
  if (seats.filter((s) => s === 'human').length !== 1) {
    setStatus('exactement un siège doit être « — vous — »', 'bad');
    return;
  }
  setStatus('démarrage…');
  try {
    const snapshot = await api('/api/session', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ seed: Number($('#pl-seed').value), seats }),
    });
    state.session = snapshot;
    $('#play-setup').classList.add('hidden');
    $('#play-body').classList.remove('hidden');
    renderSession();
    setStatus('');
  } catch (error) {
    setStatus(`démarrage : ${error.message}`, 'bad');
  }
});

$('#pl-quit').addEventListener('click', async () => {
  if (state.session) {
    await fetch(`/api/session/${state.session.id}`, { method: 'DELETE' });
  }
  state.session = null;
  state.pending = null;
  $('#play-body').classList.add('hidden');
  $('#play-setup').classList.remove('hidden');
});

$('#pl-save').addEventListener('click', async () => {
  if (!state.session) return;
  try {
    const saved = await api(`/api/session/${state.session.id}/save`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({}),
    });
    setStatus(`enregistré : ${saved.path}`, 'ok');
  } catch (error) {
    setStatus(`sauvegarde : ${error.message}`, 'bad');
  }
});

$('#pl-end').addEventListener('click', () => {
  const view = state.session && state.session.view;
  if (!view) return;
  const end = view.legal_actions.find((a) => a.action === 'end_turn');
  submitAction(end ? end.index : null);
});

async function submitAction(index, argument = '') {
  if (index === null || index === undefined || !state.session) return;
  state.pending = null;
  setStatus('résolution…');
  try {
    state.session = await api(`/api/session/${state.session.id}/act`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ index, argument }),
    });
    renderSession();
    setStatus('');
  } catch (error) {
    setStatus(`coup refusé : ${error.message}`, 'bad');
  }
}

function renderSession() {
  const session = state.session;
  const world = sessionWorld();
  $('#pl-turn').textContent = `tour ${session.turn}/${session.max_turns}`;

  renderMap($('#pl-map'), world, session.board);
  renderLegendFor($('#pl-legend'), world);
  renderEventsForSession($('#pl-events'));

  const view = session.view;
  $('#pl-points').textContent = view
    ? `${view.self.intervention_points} points d'intervention`
    : (session.finished ? 'partie terminée' : '…');
  $('#pl-end').disabled = !view;

  renderSelf($('#pl-self'), session, view);
  renderRivals($('#pl-rivals'), session, view);
  renderLegalActions($('#pl-actions'), view);

  if (session.finished) {
    $('#pl-actions').replaceChildren(el('div', { class: 'hint' }, 'Partie terminée.'));
  }
  attachMapTargeting();
}

/* Le monde d'une partie jouable, avec les factions renommées « peuple (siège) ».
 * Les deux informations comptent et disent des choses différentes : le peuple
 * est ce que montre la carte, le siège est QUI le tient — et deux sièges
 * peuvent porter le même archétype, auquel cas le nom du peuple est la seule
 * façon de les distinguer. */
function sessionWorld() {
  const world = state.session.world;
  const seats = state.session.seats || [];
  return {
    ...world,
    factions: (world.factions || []).map((faction) => ({
      ...faction,
      name: faction.id === state.session.human_seat
        ? `${faction.name} (vous)`
        : `${faction.name} (${seats[faction.id] || '?'})`,
    })),
  };
}

function renderLegendFor(container, world) {
  container.replaceChildren();
  for (const faction of world.factions) {
    container.append(el('span', {},
      el('span', { class: 'swatch', style: `background: var(--f${faction.id})` }),
      faction.name));
  }
  container.append(...legendIcons(world, world.terrain_keys || [],
    world.building_keys || [], world.labels));
}

function renderEventsForSession(container) {
  const history = state.session.history || [];
  const last = history[history.length - 1] || null;
  const saved = state.game;
  // `renderEvents` lit `state.game` pour les noms : on lui prête une partie
  // minimale le temps du rendu plutôt que de dupliquer la mise en forme.
  state.game = { world: sessionWorld(), ranking: [] };
  renderEvents(container, last);
  state.game = saved;
}

function renderSelf(container, session, view) {
  container.replaceChildren();
  const board = (session.board.factions || []).find((f) => f.id === session.human_seat);
  if (!board) return;
  const traits = Object.entries(board.traits || {})
    .map(([k, v]) => `${(state.ctx.labels[k] || k).slice(0, 4)}${v >= 0 ? '+' : ''}${v.toFixed(2)}`)
    .join('  ');
  container.append(el('div', { class: `faction f-border-${board.id}` },
    el('div', { class: 'name' }, 'Votre peuple',
      el('span', { class: 'grow' }),
      el('span', { class: 'mono' }, `score ${num(board.score)}`)),
    el('div', { class: 'stats' },
      `pop ${num(board.population)}  cases ${board.tiles}  force ${num(board.force)}  ` +
      `rech ${num(board.research)} (T${board.research_tier})`),
    el('div', { class: 'stats' },
      `nour ${num(board.food)}  bois ${num(board.wood)}  mét ${num(board.metal)}  faim ${pct(board.hunger)}`),
    el('div', { class: 'stats' }, traits)));

  if (view && (view.inbox || []).length) {
    const inbox = el('div', { class: 'feed' });
    for (const message of view.inbox) {
      inbox.append(el('div', { class: 'row' },
        el('span', { class: `f-text-${message.from}` }, message.from_name),
        ` propose « ${message.pledge} »`,
        message.argument ? el('div', { class: 'cause' }, `« ${message.argument} »`) : ''));
    }
    container.append(el('h3', {}, 'Messages reçus'), inbox);
  }
}

function renderRivals(container, session, view) {
  container.replaceChildren();
  if (!view) return;
  container.append(el('h3', {}, 'Rivaux'));
  for (const rival of view.rivals) {
    const buildings = Object.values(rival.buildings || {}).reduce((a, b) => a + b, 0);
    container.append(el('div', { class: `faction f-border-${rival.id}${rival.alive ? '' : ' dead'}` },
      el('div', { class: 'name' }, rival.name,
        rival.at_war_with_me ? el('span', { class: 'badge bad' }, 'EN GUERRE') : ''),
      el('div', { class: 'stats' },
        `${rival.tiles} cases  ${buildings} bâtiments  ` +
        `commerce ${rival.trade_relation >= 0 ? '+' : ''}${num(rival.trade_relation, 2)}`),
      el('div', { class: 'stats' },
        rival.reputation.total
          ? `parole ${rival.reputation.honored}/${rival.reputation.total}`
          : 'ne s\'est jamais engagé')));
  }
}

/* Les coups légaux, groupés par action. Les actions à cible géographique
 * passent par la carte : quarante boutons de coordonnées seraient illisibles
 * là où une case surlignée se lit d'un coup d'œil. */
function renderLegalActions(container, view) {
  container.replaceChildren();
  if (!view) {
    container.append(el('div', { class: 'hint' }, 'Les autres dieux jouent…'));
    return;
  }
  const groups = new Map();
  for (const item of view.legal_actions) {
    if (!groups.has(item.action)) groups.set(item.action, []);
    groups.get(item.action).push(item);
  }
  const specs = Object.fromEntries(state.ctx.actions.map((a) => [a.key, a]));

  for (const [key, items] of groups) {
    if (key === 'end_turn') continue;
    const spec = specs[key] || {};
    const isTile = Array.isArray(items[0].target) && items[0].target.length === 2;
    const group = el('div', { class: 'action-group' },
      el('div', { class: 'head' },
        el('strong', {}, spec.label || key),
        el('span', { class: 'cost' }, `${items[0].cost} pt${items[0].cost > 1 ? 's' : ''}`)),
      spec.note ? el('div', { class: 'note' }, spec.note) : '');

    if (isTile) {
      const active = state.pending && state.pending.action === key;
      group.append(el('button', {
        class: active ? 'primary' : '',
        onclick: () => {
          state.pending = active ? null : { action: key, items };
          renderSession();
        },
      }, active ? `Choisissez une case (${items.length} possibles)` : 'Cibler une case…'));
    } else if (key === 'negotiate') {
      group.append(renderNegotiate(items));
    } else {
      const targets = el('div', { class: 'targets' });
      for (const item of items) {
        const label = item.label.includes('→') ? item.label.split('→')[1].trim() : 'jouer';
        targets.append(el('button', { onclick: () => submitAction(item.index) }, label));
      }
      group.append(targets);
    }
    container.append(group);
  }
}

function renderNegotiate(items) {
  const select = el('select', {});
  for (const item of items) {
    const label = item.label.includes('→') ? item.label.split('→')[1].trim() : String(item.index);
    select.append(el('option', { value: String(item.index) }, label));
  }
  const argument = el('textarea', {
    rows: '2', placeholder: 'Votre argumentaire — il est lu par le dieu visé.',
    style: 'width:100%; margin-top:6px;',
  });
  return el('div', {}, select, argument,
    el('button', {
      style: 'margin-top:6px',
      onclick: () => submitAction(Number(select.value), argument.value),
    }, 'Envoyer'));
}

/* Ciblage à la carte : surligne les cases jouables et intercepte le clic. */
function attachMapTargeting() {
  const map = $('#pl-map');
  if (!state.pending) return;
  const legal = new Map();
  for (const item of state.pending.items) legal.set(`${item.target[0]},${item.target[1]}`, item.index);
  for (const tile of $$('.tile', map)) {
    const key = `${tile.dataset.x},${tile.dataset.y}`;
    if (!legal.has(key)) { tile.style.opacity = '0.35'; continue; }
    tile.style.cursor = 'pointer';
    tile.style.outline = '2px solid #fff';
    tile.addEventListener('click', () => submitAction(legal.get(key)), { once: true });
  }
}

/* ── Modèles ──────────────────────────────────────────────────────────── */

async function loadModels() {
  const body = $('#models-body');
  body.replaceChildren(el('p', { class: 'hint' }, 'Interrogation des fournisseurs…'));
  try {
    const data = await api('/api/models');
    body.replaceChildren();

    for (const [provider, message] of Object.entries(data.errors || {})) {
      body.append(el('p', { class: 'hint' },
        el('span', { class: 'badge bad' }, provider), ' ', message));
    }

    const table = el('table', {}, el('thead', {}, el('tr', {},
      el('th', {}, 'modèle'), el('th', {}, 'fournisseur'), el('th', {}, 'état'),
      el('th', { class: 'num' }, '$/Mtok in'), el('th', { class: 'num' }, '$/Mtok out'),
      el('th', { class: 'num' }, 'contexte'), el('th', { class: 'num' }, '48 parties'),
      el('th', {}, 'remarques'))));
    const tbody = el('tbody');
    for (const report of data.models) {
      const problems = report.problems || [];
      const warnings = report.warnings || [];
      const badge = problems.length
        ? el('span', { class: 'badge bad' }, 'à corriger')
        : (warnings.length ? el('span', { class: 'badge warn' }, 'invérifiable')
                           : el('span', { class: 'badge ok' }, 'ok'));
      tbody.append(el('tr', {},
        el('td', { class: 'mono' }, report.model),
        el('td', {}, report.provider || '—'),
        el('td', {}, badge),
        el('td', { class: 'num' }, num(report.price_in, 3)),
        el('td', { class: 'num' }, num(report.price_out, 3)),
        el('td', { class: 'num' }, report.context_length || '—'),
        el('td', { class: 'num' },
          report.cost_48_games === null || report.cost_48_games === undefined
            ? '—' : `$${num(report.cost_48_games, 2)}`),
        el('td', { class: 'dim' }, [...problems, ...warnings].join(' · ') || '—')));
    }
    table.append(tbody);
    body.append(table);
    body.append(el('p', { class: 'hint' },
      'Le coût est une extrapolation des 45 000 tokens par siège et par partie ' +
      'mesurés en T2 (docs/FINDINGS.md n° 5), pour UN siège sur 48 parties. ' +
      'Il ignore le cache et la verbosité propre au modèle : il sert à séparer ' +
      'les centimes des centaines de dollars, pas à facturer.'));
  } catch (error) {
    body.replaceChildren(el('p', { class: 'hint' }, `Erreur : ${error.message}`));
  }
}

/* ── Démarrage ────────────────────────────────────────────────────────── */

(async function boot() {
  try {
    state.ctx = await api('/api/context');
    renderSeatPicker();
    await loadLibrary();
  } catch (error) {
    setStatus(`démarrage : ${error.message}`, 'bad');
  }
})();
