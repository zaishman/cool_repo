/* ============================================================
   Debate Judge Assistant — frontend

   Talks to the Flask app in app.py:
     GET  /health   -> {status:"ok"}          (connection indicator)
     POST /analyze  -> full analysis JSON     (file + position)

   The response shape comes straight from wiring.analyze_full_submission:
     position, transcript, sentences, acoustics, contentions[],
     filler_words, structure, summary, speaker_points, refutations,
     video_analysis
   Nothing here mutates the backend payload — client-only fields (date,
   label, filename) live under a single "_client" key so they can never
   collide with a backend field.
   ============================================================ */

// Normally Flask serves this page, so the API is same-origin and relative URLs
// just work. If index.html is opened directly as a file:// page instead, probe
// the usual dev ports so both ways of running the app still work.
const IS_FILE = location.protocol === 'file:';
const FALLBACK_ORIGINS = [
  'http://127.0.0.1:5002',
  'http://localhost:5002',
  'http://127.0.0.1:5000',
];

let apiBase = IS_FILE ? null : '';

async function resolveApiBase() {
  if (!IS_FILE) return '';
  if (apiBase !== null) return apiBase;
  for (const origin of FALLBACK_ORIGINS) {
    try {
      const res = await fetch(`${origin}/health`, { cache: 'no-store' });
      if (res.ok) { apiBase = origin; return apiBase; }
    } catch { /* try the next one */ }
  }
  return null;
}

const STORAGE_KEY = 'debate_reports';
const SPEAKER_POINTS_MAX = 82;   // calculate_speaker_points clamps to 0..82
const ARGUMENTATION_MAX = 10;    // SUMMARY_PROMPT asks for 1-10

// Mirrors SPEECH_RULES in transcript_pipline/structure_rules.py — shown so you
// know what the structure checker will hold the speech to before you submit.
const POSITIONS = [
  { id: 'PROP1', rules: '1–2 new contentions · POIs & heckles allowed · no refutation expected' },
  { id: 'OPP1',  rules: '1–2 new contentions · POIs & heckles allowed · refutation expected' },
  { id: 'PROP2', rules: 'No new contentions · POIs & heckles allowed · refutation expected' },
  { id: 'OPP2',  rules: 'No new contentions · POIs & heckles allowed · refutation expected' },
  { id: 'PROP3', rules: 'Summary only · no new points, no POIs or heckles' },
  { id: 'OPP3',  rules: 'Summary only · no new points, no POIs or heckles' },
];

const PIPELINE_STAGES = [
  'Uploading recording',
  'Transcribing speech',
  'Measuring vocal delivery',
  'Extracting arguments & refutations',
  'Analyzing video frames',
  'Scoring',
];

// Rough elapsed-second marks used only to animate the stage list. These are
// estimates, not live backend events — the note in the UI says so.
const STAGE_MARKS = [0, 6, 40, 60, 150, 240];

const $ = (id) => document.getElementById(id);

/* ---------------- state ---------------- */
const state = {
  position: 'PROP1',
  label: '',
  pendingFile: null,      // kept so "Try again" can resubmit without re-picking
  inFlight: null,         // AbortController while an analysis is running
  mediaStream: null,
  mediaRecorder: null,
  recordedChunks: [],
  recordExt: 'webm',
  timers: [],
};

/* ---------------- view switching ---------------- */
function showView(viewId) {
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active-view'));
  $(viewId).classList.add('active-view');
  document.body.classList.toggle('on-landing', viewId === 'landing-view');
  window.scrollTo(0, 0);
}

document.querySelectorAll('[data-goto]').forEach(btn => {
  btn.addEventListener('click', () => {
    const target = btn.dataset.goto;
    if (target === 'library-view') renderLibrary();
    showView(target);
  });
});

/* ---------------- escaping ----------------
   Everything below builds HTML strings, and the payload contains
   model-written prose plus a raw transcript. Escape all of it. */
function esc(value) {
  if (value === null || value === undefined) return '';
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

/* ---------------- number helpers ---------------- */
const isNum = (n) => typeof n === 'number' && Number.isFinite(n);
const num = (n, digits = 1) => (isNum(n) ? n.toFixed(digits) : '—');
const pct = (ratio) => (isNum(ratio) ? `${Math.round(ratio * 100)}%` : '—');
const clamp01 = (n) => Math.max(0, Math.min(1, n));

function severity(value, goodAt, warnAt) {
  if (!isNum(value)) return '';
  if (value >= goodAt) return 'good';
  if (value >= warnAt) return 'warning';
  return 'critical';
}

function fmtDuration(ms) {
  const total = Math.max(0, Math.round(ms / 1000));
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, '0')}`;
}

function fmtBytes(bytes) {
  if (!isNum(bytes)) return '';
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/* ---------------- backend health ---------------- */
async function checkBackend() {
  const pill = $('backend-status');
  const text = $('backend-status-text');
  try {
    const base = await resolveApiBase();
    if (base === null) throw new Error('no backend reachable');
    const res = await fetch(`${base}/health`, { cache: 'no-store' });
    if (!res.ok) throw new Error(String(res.status));
    pill.className = 'status-pill status-pill--ok';
    text.textContent = 'Backend connected';
    $('landing-hint').textContent = 'Backend connected.';
    return true;
  } catch {
    if (IS_FILE) apiBase = null;   // re-probe, so a restarted server is found again
    pill.className = 'status-pill status-pill--down';
    text.textContent = 'Backend offline';
    $('landing-hint').textContent = 'Backend not reachable — start it with:  python3 app.py';
    return false;
  }
}

checkBackend();
setInterval(checkBackend, 15000);

/* ---------------- landing ---------------- */
$('get-started-btn').addEventListener('click', () => {
  renderLibrary();
  showView('library-view');
});

/* ---------------- setup view ---------------- */
function buildPositionPicker() {
  const picker = $('position-picker');
  const select = $('position-select');

  picker.innerHTML = POSITIONS.map(p => `
    <button type="button" class="segmented-btn" role="radio"
            aria-checked="${p.id === state.position}" data-position="${esc(p.id)}">${esc(p.id)}</button>
  `).join('');

  // Mirrored into a real <select> so the choice is still reachable for
  // assistive tech and form autofill.
  select.innerHTML = POSITIONS.map(p => `<option value="${esc(p.id)}">${esc(p.id)}</option>`).join('');

  picker.querySelectorAll('.segmented-btn').forEach(btn => {
    btn.addEventListener('click', () => setPosition(btn.dataset.position));
  });
  select.addEventListener('change', () => setPosition(select.value));

  setPosition(state.position);
}

function setPosition(id) {
  const match = POSITIONS.find(p => p.id === id);
  if (!match) return;
  state.position = id;
  $('position-select').value = id;
  $('position-rules').textContent = match.rules;
  $('position-picker').querySelectorAll('.segmented-btn').forEach(btn => {
    btn.setAttribute('aria-checked', String(btn.dataset.position === id));
  });
}

function openSetup() {
  state.pendingFile = null;
  $('label-input').value = state.label;
  showView('setup-view');
}

$('new-analysis-btn').addEventListener('click', openSetup);
$('empty-new-btn').addEventListener('click', openSetup);
$('label-input').addEventListener('input', (e) => { state.label = e.target.value; });

/* ---------------- upload flow ---------------- */
const fileInput = $('file-input');
const dropzone = $('dropzone');

dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
});

['dragenter', 'dragover'].forEach(evt => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add('is-dragover');
  });
});
['dragleave', 'drop'].forEach(evt => {
  dropzone.addEventListener(evt, () => dropzone.classList.remove('is-dragover'));
});

dropzone.addEventListener('drop', (e) => {
  e.preventDefault();
  const file = e.dataTransfer.files[0];
  if (file) processSpeech(file, file.name);
});

fileInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) return;
  processSpeech(file, file.name);
  fileInput.value = '';   // so picking the same file twice still fires "change"
});

/* ---------------- record flow ---------------- */
function pickRecordingType() {
  // Prefer mp4: OpenCV (used by video_pipline) reads it far more reliably than
  // VP8/VP9 webm. Fall back through webm for browsers without mp4 recording.
  const candidates = [
    { mime: 'video/mp4', ext: 'mp4' },
    { mime: 'video/webm;codecs=vp8,opus', ext: 'webm' },
    { mime: 'video/webm', ext: 'webm' },
  ];
  for (const c of candidates) {
    if (window.MediaRecorder && MediaRecorder.isTypeSupported(c.mime)) return c;
  }
  return { mime: '', ext: 'webm' };
}

$('choose-record-btn').addEventListener('click', async () => {
  const position = POSITIONS.find(p => p.id === state.position);
  $('record-position-note').textContent = `${state.position} — ${position.rules}`;
  $('record-error').hidden = true;
  $('start-record-btn').hidden = false;
  $('stop-record-btn').hidden = true;
  $('rec-badge').hidden = true;
  showView('record-view');

  try {
    // Both tracks: the pipeline analyzes audio and video.
    state.mediaStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
    $('live-preview').srcObject = state.mediaStream;
  } catch (err) {
    $('start-record-btn').disabled = true;
    $('record-error').hidden = false;
    $('record-error').textContent =
      `Could not access camera and microphone: ${err.message}. Check your browser permissions, or upload a file instead.`;
  }
});

$('start-record-btn').addEventListener('click', () => {
  if (!state.mediaStream) return;

  const { mime, ext } = pickRecordingType();
  state.recordExt = ext;
  state.recordedChunks = [];
  state.mediaRecorder = new MediaRecorder(state.mediaStream, mime ? { mimeType: mime } : undefined);

  // MediaRecorder fires "dataavailable" with chunks of the recording so far —
  // collect them all and combine into one file at the end.
  state.mediaRecorder.ondataavailable = (e) => {
    if (e.data.size > 0) state.recordedChunks.push(e.data);
  };

  state.mediaRecorder.onstop = () => {
    stopRecordTimer();
    const blob = new Blob(state.recordedChunks, { type: state.mediaRecorder.mimeType || 'video/webm' });
    stopStream();
    processSpeech(blob, `speech.${state.recordExt}`);
  };

  state.mediaRecorder.start();
  startRecordTimer();
  $('start-record-btn').hidden = true;
  $('stop-record-btn').hidden = false;
  $('rec-badge').hidden = false;
});

$('stop-record-btn').addEventListener('click', () => {
  if (state.mediaRecorder && state.mediaRecorder.state !== 'inactive') state.mediaRecorder.stop();
  $('stop-record-btn').hidden = true;
});

$('record-cancel-btn').addEventListener('click', () => {
  if (state.mediaRecorder && state.mediaRecorder.state !== 'inactive') {
    state.mediaRecorder.onstop = null;      // cancelling must not submit
    state.mediaRecorder.stop();
  }
  stopRecordTimer();
  stopStream();
  showView('setup-view');
});

let recordStart = 0;
let recordTimerId = null;

function startRecordTimer() {
  recordStart = Date.now();
  $('rec-timer').textContent = '0:00';
  recordTimerId = setInterval(() => {
    $('rec-timer').textContent = fmtDuration(Date.now() - recordStart);
  }, 250);
}

function stopRecordTimer() {
  clearInterval(recordTimerId);
  recordTimerId = null;
  $('rec-badge').hidden = true;
}

function stopStream() {
  if (state.mediaStream) {
    state.mediaStream.getTracks().forEach(t => t.stop());   // turns off camera/mic
    state.mediaStream = null;
  }
  $('live-preview').srcObject = null;
}

/* ---------------- analysis ---------------- */
async function processSpeech(fileOrBlob, fileName) {
  if (state.inFlight) return;   // one at a time: the pipeline writes fixed temp filenames

  state.pendingFile = { blob: fileOrBlob, name: fileName };

  const position = state.position;
  const label = state.label.trim();
  const ext = (fileName.split('.').pop() || 'webm').toLowerCase();
  // Unique name so concurrent-ish uploads can't overwrite each other in uploads/.
  const uploadName = `speech-${Date.now()}.${ext}`;

  showView('processing-view');
  $('processing-meta').textContent =
    [fileName, fmtBytes(fileOrBlob.size)].filter(Boolean).join(' · ');
  startProgress();

  const formData = new FormData();
  formData.append('file', fileOrBlob, uploadName);
  formData.append('position', position);

  const controller = new AbortController();
  state.inFlight = controller;
  const startedAt = Date.now();

  try {
    const base = await resolveApiBase();
    if (base === null) {
      throw new Error('Could not reach the backend. Start it with:  python3 app.py');
    }

    const res = await fetch(`${base}/analyze`, {
      method: 'POST',
      body: formData,
      signal: controller.signal,
    });

    const raw = await res.text();
    let payload;
    try {
      payload = JSON.parse(raw);
    } catch {
      throw new Error(`Server returned a non-JSON response (HTTP ${res.status}):\n${raw.slice(0, 600)}`);
    }

    if (!res.ok || payload.error) {
      throw new Error(payload.error || `Analysis failed with HTTP ${res.status}`);
    }

    payload._client = {
      id: `r-${startedAt}`,
      analyzedAt: new Date(startedAt).toISOString(),
      label,
      fileName,
      elapsedMs: Date.now() - startedAt,
    };

    finishProgress();
    saveReport(payload);
    showReport(payload);
  } catch (err) {
    if (err.name === 'AbortError') return;   // cancel already moved the view
    showError(err.message);
  } finally {
    stopProgress();
    state.inFlight = null;
  }
}

$('processing-cancel-btn').addEventListener('click', () => {
  if (state.inFlight) state.inFlight.abort();
  stopProgress();
  state.inFlight = null;
  showView('setup-view');
});

$('error-retry-btn').addEventListener('click', () => {
  if (state.pendingFile) {
    processSpeech(state.pendingFile.blob, state.pendingFile.name);
  } else {
    openSetup();
  }
});

function showError(message) {
  $('error-detail').textContent = message;
  showView('error-view');
}

/* ---------------- processing progress ---------------- */
function startProgress() {
  const list = $('processing-steps');
  list.innerHTML = PIPELINE_STAGES
    .map((s, i) => `<li data-step="${i}">${esc(s)}</li>`).join('');

  const begin = Date.now();
  setStage(0);

  const tick = setInterval(() => {
    const elapsed = (Date.now() - begin) / 1000;
    // Never mark the final stage done on a timer — only the response does that.
    let stage = 0;
    for (let i = 0; i < STAGE_MARKS.length; i++) {
      if (elapsed >= STAGE_MARKS[i]) stage = i;
    }
    setStage(Math.min(stage, PIPELINE_STAGES.length - 1));
    $('processing-status').textContent = `${PIPELINE_STAGES[stage]}…`;
    $('processing-meta').textContent =
      [state.pendingFile?.name, fmtBytes(state.pendingFile?.blob?.size), `${fmtDuration(Date.now() - begin)} elapsed`]
        .filter(Boolean).join(' · ');
  }, 500);

  state.timers.push(tick);
}

function setStage(index) {
  $('processing-steps').querySelectorAll('li').forEach((li, i) => {
    li.classList.toggle('is-done', i < index);
    li.classList.toggle('is-active', i === index);
  });
}

function finishProgress() {
  $('processing-steps').querySelectorAll('li').forEach(li => {
    li.classList.remove('is-active');
    li.classList.add('is-done');
  });
}

function stopProgress() {
  state.timers.forEach(clearInterval);
  state.timers = [];
}

/* ---------------- storage ---------------- */
function getAllReports() {
  try {
    const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function saveReport(report) {
  const reports = getAllReports();
  reports.unshift(report);   // newest first
  // Full payloads (transcript + per-contention explanations) add up; drop the
  // oldest until it fits rather than losing the report that just ran.
  while (reports.length) {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(reports));
      return;
    } catch {
      if (reports.length === 1) return;   // single report too big to store — keep it in memory only
      reports.pop();
    }
  }
}

function deleteReport(id) {
  const remaining = getAllReports().filter(r => r._client?.id !== id);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(remaining));
  renderLibrary();
}

/* ---------------- library ---------------- */
function renderLibrary() {
  const reports = getAllReports();
  const listEl = $('reports-list');
  const emptyEl = $('empty-state');

  $('clear-all-btn').hidden = reports.length === 0;
  $('library-sub').textContent = reports.length
    ? `${reports.length} saved ${reports.length === 1 ? 'report' : 'reports'} · stored in this browser`
    : '';

  if (reports.length === 0) {
    listEl.innerHTML = '';
    emptyEl.hidden = false;
    return;
  }
  emptyEl.hidden = true;

  listEl.innerHTML = reports.map(r => {
    const meta = r._client || {};
    const title = meta.label || r.position || 'Speech';
    const when = meta.analyzedAt
      ? new Date(meta.analyzedAt).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })
      : 'Date unknown';
    const sub = [r.position, when, meta.fileName].filter(Boolean).join(' · ');

    return `
      <div class="report-card" role="button" tabindex="0" data-id="${esc(meta.id)}">
        <div class="report-card-main">
          <div class="report-card-title">${esc(title)}</div>
          <div class="report-card-meta">${esc(sub)}</div>
        </div>
        <div class="report-card-score">
          <div class="report-card-score-value">${isNum(r.speaker_points) ? r.speaker_points : '—'}</div>
          <div class="report-card-score-label">of ${SPEAKER_POINTS_MAX} pts</div>
        </div>
        <button class="card-delete" data-delete="${esc(meta.id)}" title="Delete report" aria-label="Delete report">×</button>
      </div>
    `;
  }).join('');

  listEl.querySelectorAll('.report-card').forEach(card => {
    const open = () => {
      const report = getAllReports().find(r => r._client?.id === card.dataset.id);
      if (report) showReport(report);
    };
    card.addEventListener('click', open);
    card.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(); }
    });
  });

  listEl.querySelectorAll('[data-delete]').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      deleteReport(btn.dataset.delete);
    });
  });
}

$('clear-all-btn').addEventListener('click', () => {
  if (!confirm('Delete all saved reports? This cannot be undone.')) return;
  localStorage.removeItem(STORAGE_KEY);
  renderLibrary();
});

/* ============================================================
   REPORT RENDERING
   ============================================================ */

/* A meter's fill carries the value; the track is a lighter step of the same
   hue. The numeric value is always printed beside the label, so the color is
   never the only channel. */
function meter(label, value, max, sevClass) {
  const width = isNum(value) && max ? clamp01(value / max) * 100 : 0;
  const shown = isNum(value) ? `${num(value, Number.isInteger(value) ? 0 : 1)} / ${max}` : '—';
  return `
    <div class="meter ${sevClass ? `meter--${sevClass}` : ''}">
      <div class="meter-head">
        <span class="meter-label">${esc(label)}</span>
        <span class="meter-value">${esc(shown)}</span>
      </div>
      <div class="meter-track" role="img" aria-label="${esc(`${label}: ${shown}`)}">
        <div class="meter-fill" style="width:${width.toFixed(1)}%"></div>
      </div>
    </div>
  `;
}

const STATUS_ICONS = { good: '●', warning: '▲', critical: '■' };

function statusPill(sevClass, text) {
  return `<span class="status status--${sevClass}">
    <span class="status-icon" aria-hidden="true">${STATUS_ICONS[sevClass] || '●'}</span>${esc(text)}
  </span>`;
}

function tile(label, value, max, sub) {
  return `
    <div class="tile">
      <div class="tile-label">${esc(label)}</div>
      <div class="tile-value">${esc(value)}${max ? `<span class="tile-max"> / ${esc(max)}</span>` : ''}</div>
      ${sub ? `<div class="tile-sub">${esc(sub)}</div>` : ''}
    </div>
  `;
}

function areiRow(key, value) {
  const empty = value === null || value === undefined || String(value).trim() === '';
  return `
    <div class="arei-row">
      <div class="arei-key">${esc(key)}</div>
      <div class="arei-val ${empty ? 'arei-val--empty' : ''}">${empty ? 'Not stated' : esc(value)}</div>
    </div>
  `;
}

function showReport(report) {
  const meta = report._client || {};
  function showReport(report) {
  const meta = report._client || {};
  const annotatedVideo = $('annotated-video');
  if (report.annotated_video_url) {
    annotatedVideo.src = report.annotated_video_url;
    annotatedVideo.hidden = false;
  } else {
    annotatedVideo.hidden = true;
  }
  const acoustics = report.acoustics || {};
  const projection = acoustics.projection || {};
  const summary = report.summary || {};
  const structure = report.structure || {};
  const fillers = report.filler_words || {};
  const video = report.video_analysis;
  const contentions = Array.isArray(report.contentions) ? report.contentions : [];
  const refutations = report.refutations?.refutations || [];

  const when = meta.analyzedAt
    ? new Date(meta.analyzedAt).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })
    : '';

  /* --- header --- */
  const head = `
    <div class="report-head">
      <div class="report-head-badges">
        <span class="badge">${esc(report.position || '—')}</span>
        ${structure.complaint !== undefined
          ? statusPill(structure.complaint ? 'good' : 'critical',
              structure.complaint ? 'Format compliant' : `${(structure.violations || []).length} format issue${(structure.violations || []).length === 1 ? '' : 's'}`)
          : ''}
      </div>
      <h2>${esc(meta.label || 'Speech report')}</h2>
      <div class="report-head-meta">${esc([when, meta.fileName,
        isNum(meta.elapsedMs) ? `analyzed in ${fmtDuration(meta.elapsedMs)}` : ''].filter(Boolean).join(' · '))}</div>
    </div>
  `;

  /* --- hero: exactly one per view --- */
  const hero = `
    <div class="card">
      <div class="card-title">Speaker points</div>
      <div class="hero">
        <span class="hero-value">${isNum(report.speaker_points) ? report.speaker_points : '—'}</span>
        <span class="hero-max">/ ${SPEAKER_POINTS_MAX}</span>
      </div>
      <div class="hero-label">Combined from delivery, argument validity, evidence use, refutation strength and format compliance.</div>
    </div>
  `;

  /* --- KPI row --- */
  const tiles = `
    <div class="tile-row">
      ${tile('Argumentation', isNum(summary.overall_argumentation_score) ? summary.overall_argumentation_score : '—',
             isNum(summary.overall_argumentation_score) ? ARGUMENTATION_MAX : '', 'Overall reasoning quality')}
      ${tile('Clarity', num(acoustics.clarity, 1), 100, 'Transcription confidence')}
      ${tile('Contentions', contentions.length, '', `${contentions.filter(c => c.evidence?.has_evidence).length} with evidence`)}
      ${tile('Refutations', refutations.length, '', isNum(report.refutations?.refutation_count) ? 'Detected in speech' : '')}
      ${video ? tile('Video confidence', num(video.confidence_score, 1), 100, 'Eye contact + posture') : ''}
      ${tile('Filler words', isNum(fillers.total_fillers) ? fillers.total_fillers : '—', '', 'Total across speech')}
    </div>
  `;

  /* --- delivery --- */
  const fillerBreakdown = Object.entries(fillers.breakdown || {})
    .filter(([, count]) => count > 0)
    .sort((a, b) => b[1] - a[1]);

  const delivery = `
    <div class="card">
      <div class="card-title">Vocal delivery</div>
      ${meter('Clarity', acoustics.clarity, 100, severity(acoustics.clarity, 80, 60))}
      <div class="kv-grid" style="margin-top:18px">
        <div><div class="kv-key">Projection</div><div class="kv-val">${esc(projection.projection_label || '—')}</div></div>
        <div><div class="kv-key">Energy</div><div class="kv-val">${num(acoustics.energy_mean_db, 1)} dB</div></div>
        <div><div class="kv-key">Mean pitch</div><div class="kv-val">${num(acoustics.pitch_mean_hz, 0)} Hz</div></div>
        <div><div class="kv-key">Pitch variation</div><div class="kv-val">${num(acoustics.pitch_std_hz, 0)} Hz</div></div>
        <div><div class="kv-key">Pitch range</div><div class="kv-val">${num(acoustics.pitch_range_hz, 0)} Hz</div></div>
        <div><div class="kv-key">Jitter</div><div class="kv-val">${num(acoustics.jitter, 4)}</div></div>
        <div><div class="kv-key">Shimmer</div><div class="kv-val">${num(acoustics.shimmer, 4)}</div></div>
        ${acoustics.emotion_label ? `<div><div class="kv-key">Tone</div><div class="kv-val">${esc(acoustics.emotion_label)}</div></div>` : ''}
      </div>
      ${fillerBreakdown.length ? `
        <div class="explain">
          <strong>Filler words (${esc(fillers.total_fillers)}):</strong>
          ${fillerBreakdown.map(([word, count]) => `${esc(word)} ×${esc(count)}`).join(' · ')}
        </div>` : `<div class="explain">No filler words detected.</div>`}
    </div>
  `;

  /* --- argumentation summary --- */
  const strengths = Array.isArray(summary.strengths) ? summary.strengths : [];
  const weaknesses = Array.isArray(summary.weaknesses) ? summary.weaknesses : [];

  const argumentation = `
    <div class="card">
      <div class="card-title">Argumentation summary</div>
      ${meter('Overall argumentation', summary.overall_argumentation_score, ARGUMENTATION_MAX,
              severity(summary.overall_argumentation_score, 7, 5))}
      <div style="margin-top:20px">
        <div class="list-group">
          <div class="list-group-label"><span class="status-icon status--good" style="color:var(--good)" aria-hidden="true">●</span>Strengths</div>
          ${strengths.length
            ? `<ul class="bullet-list">${strengths.map(s => `<li>${esc(s)}</li>`).join('')}</ul>`
            : `<p class="empty-note">None identified.</p>`}
        </div>
        <div class="list-group">
          <div class="list-group-label"><span class="status-icon" style="color:var(--critical)" aria-hidden="true">■</span>Weaknesses</div>
          ${weaknesses.length
            ? `<ul class="bullet-list">${weaknesses.map(w => `<li>${esc(w)}</li>`).join('')}</ul>`
            : `<p class="empty-note">None identified.</p>`}
        </div>
      </div>
    </div>
  `;

  /* --- structure compliance --- */
  const violations = Array.isArray(structure.violations) ? structure.violations : [];
  const structureCard = `
    <div class="card">
      <div class="card-title">Format compliance — ${esc(structure.position || report.position || '')}</div>
      ${structure.complaint
        ? `${statusPill('good', 'Meets the rules for this speech slot')}`
        : `${statusPill('critical', `${violations.length} issue${violations.length === 1 ? '' : 's'}`)}
           <ul class="bullet-list" style="margin-top:12px">${violations.map(v => `<li>${esc(v)}</li>`).join('')}</ul>`}
    </div>
  `;

  /* --- contentions --- */
  const contentionCards = contentions.map((c, i) => {
    const arei = c.arei || {};
    const validity = c.validity || {};
    const evidence = c.evidence || {};
    const e2r = validity.evidence_supports_reasoning;
    const r2a = validity.reasoning_supports_assertion;

    return `
      <div class="card">
        <div class="card-title">Contention ${i + 1}</div>
        <div class="claim">${arei.assertion ? esc(arei.assertion) : '<span class="arei-val--empty">No clear assertion extracted</span>'}</div>
        ${areiRow('Reasoning', arei.reasoning)}
        ${areiRow('Evidence', arei.evidence)}
        ${areiRow('Impact', arei.impact)}
        <div class="meter-pair">
          ${meter('Evidence → reasoning', e2r, 5, severity(e2r, 4, 3))}
          ${meter('Reasoning → assertion', r2a, 5, severity(r2a, 4, 3))}
        </div>
        <div style="margin-top:14px">
          ${statusPill(evidence.has_evidence ? 'good' : 'warning',
                       evidence.has_evidence ? 'Evidence detected' : 'No evidence detected')}
          ${isNum(evidence.semantic_confidence)
            ? `<span class="tile-sub" style="margin-left:8px">match ${num(evidence.semantic_confidence, 2)}</span>` : ''}
        </div>
        ${validity.brief_explaination ? `<div class="explain">${esc(validity.brief_explaination)}</div>` : ''}
      </div>
    `;
  }).join('');

  /* --- refutations --- */
  const refutationCards = refutations.map((r, i) => `
    <div class="card">
      <div class="card-title">Refutation ${i + 1}</div>
      <div class="claim">${esc(r.point_being_refuted || 'Unidentified point')}</div>
      ${meter('Refutation strength', r.refutation_strength, 5, severity(r.refutation_strength, 4, 3))}
      ${r.brief_explanation ? `<div class="explain">${esc(r.brief_explanation)}</div>` : ''}
    </div>
  `).join('');

  /* --- video --- */
  const videoCard = video ? `
    <div class="card">
      <div class="card-title">Video delivery</div>
      ${video.warning ? `${statusPill('warning', esc(video.warning))}<div style="height:14px"></div>` : ''}
      ${meter('Eye contact', isNum(video.eye_contact_ratio) ? video.eye_contact_ratio * 100 : null, 100,
              severity(video.eye_contact_ratio * 100, 60, 35))}
      ${meter('Posture stability', isNum(video.posture_stability) ? video.posture_stability * 100 : null, 100,
              severity(video.posture_stability * 100, 70, 45))}
      ${meter('Confidence score', video.confidence_score, 100, severity(video.confidence_score, 65, 40))}
      <div class="explain">Eye contact ${pct(video.eye_contact_ratio)} of detected frames · posture stability ${pct(video.posture_stability)}. Confidence weights eye contact 60% and posture 40%.</div>
    </div>
  ` : '';

  /* --- transcript + raw payload --- */
  const wordCount = report.transcript ? report.transcript.trim().split(/\s+/).filter(Boolean).length : 0;
  const disclosures = `
    <details class="disclosure">
      <summary>Transcript${wordCount ? ` — ${wordCount} words` : ''}</summary>
      <div class="disclosure-body">
        <div class="transcript">${report.transcript ? esc(report.transcript) : '<span class="empty-note">No transcript returned.</span>'}</div>
      </div>
    </details>
    <details class="disclosure">
      <summary>Raw backend response</summary>
      <div class="disclosure-body">
        <pre class="json-dump">${esc(JSON.stringify(report, null, 2))}</pre>
      </div>
    </details>
  `;

  $('report-content').innerHTML = `
    ${head}
    ${hero}
    ${tiles}
    ${delivery}
    ${argumentation}
    ${structureCard}
    ${contentions.length ? `<h3 class="section-title">Contentions (${contentions.length})</h3>${contentionCards}` : ''}
    ${refutations.length ? `<h3 class="section-title">Refutations (${refutations.length})</h3>${refutationCards}` : ''}
    ${videoCard}
    <h3 class="section-title">Source</h3>
    ${disclosures}
  `;

  showView('report-view');
}

/* ---------------- boot ---------------- */
buildPositionPicker();
renderLibrary();
document.body.classList.add('on-landing');
