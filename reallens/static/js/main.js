/**
 * RealLens - Multi-Signal AI Provenance & Forensic Engine
 * Phase 2.5: Multi-Image Batch Queue, Image-Type Gating, Spread Inspection & Inconclusive Routing
 */

// Application State
const state = {
  queue: [],              // Array of { id, file, name, size, hash, status, result, thumbUrl, error }
  selectedIndex: -1,     // Index of currently inspected item
  currentFilter: 'all',  // 'all' | 'Likely AI' | 'Leaning AI' | 'Inconclusive' | 'Leaning real' | 'Likely real' | 'Failed'
  sortColumn: 'index',   // 'index' | 'filename' | 'type' | 'verdict' | 'probability' | 'spread' | 'time'
  sortDirection: 'asc',
  isProcessing: false,
  abortController: null,
  analysisCache: new Map(), // hash -> result
  elaMode: 'single',        // 'single' (toggle) | 'dual' (side-by-side)
  elaShowingOriginal: true,
  timesElapsed: [],
};

const ALLOWED_EXTS = ['.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp', '.tiff', '.tif'];
const MAX_FILE_SIZE = 25 * 1024 * 1024; // 25 MB
const MAX_QUEUE_SIZE = 50;

// DOM Elements
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');
const queueActionsBar = document.getElementById('queue-actions-bar');
const queueCountBadge = document.getElementById('queue-count-badge');
const analyzeAllBtn = document.getElementById('analyze-all-btn');
const cancelBtn = document.getElementById('cancel-btn');
const clearAllBtn = document.getElementById('clear-all-btn');
const progressContainer = document.getElementById('progress-container');
const progressFill = document.getElementById('progress-fill');
const progressText = document.getElementById('progress-text');
const progressEta = document.getElementById('progress-eta');
const summaryStrip = document.getElementById('summary-strip');
const queueTbody = document.getElementById('queue-tbody');
const exportBar = document.getElementById('export-bar');
const downloadCsvBtn = document.getElementById('download-csv-btn');
const downloadJsonBtn = document.getElementById('download-json-btn');
const copyReportBtn = document.getElementById('copy-report-btn');
const deviceStatus = document.getElementById('device-status');

// Detail Panel Elements
const detailEmpty = document.getElementById('detail-empty');
const detailSkeleton = document.getElementById('detail-skeleton');
const detailContent = document.getElementById('detail-content');
const detailFilename = document.getElementById('detail-filename');
const detailDims = document.getElementById('detail-dims');
const detailSize = document.getElementById('detail-size');
const detailTypeBadge = document.getElementById('detail-type-badge');
const detailTime = document.getElementById('detail-time');
const unreliableNotice = document.getElementById('unreliable-type-notice');
const unreliableNoticeText = document.getElementById('unreliable-notice-text');
const verdictBadge = document.getElementById('verdict-badge');
const verdictConfidenceBadge = document.getElementById('verdict-confidence-badge');
const verdictDisplayScore = document.getElementById('verdict-display-score');
const verdictOneLiner = document.getElementById('verdict-one-liner');
const whyVerdictText = document.getElementById('why-verdict-text');
const spreadIndicatorPill = document.getElementById('spread-indicator-pill');
const spreadBarFill = document.getElementById('spread-bar-fill');
const spreadRangeLabel = document.getElementById('spread-range-label');
const modelBarsContainer = document.getElementById('model-bars-container');
const elaActiveImg = document.getElementById('ela-active-img');
const elaViewCaption = document.getElementById('ela-view-caption');
const elaToggleBtn = document.getElementById('ela-toggle-btn');
const zoomImgBtn = document.getElementById('zoom-img-btn');
const elaSingleView = document.getElementById('ela-single-view');
const elaSideBySide = document.getElementById('ela-side-by-side');
const compOriginal = document.getElementById('comp-original');
const compEla = document.getElementById('comp-ela');
const elaInterpretationNote = document.getElementById('ela-interpretation-note');
const zoomModal = document.getElementById('zoom-modal');
const zoomModalImg = document.getElementById('zoom-modal-img');
const zoomTitle = document.getElementById('zoom-title');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
  initSystemHealth();
  setupEventListeners();
});

// Fetch Backend System Health
async function initSystemHealth() {
  try {
    const res = await fetch('/health');
    if (res.ok) {
      const data = await res.json();
      const modelsCount = (data.models_configured || []).length;
      deviceStatus.textContent = `Device: ${data.device.toUpperCase()} • ${modelsCount} Models`;
    }
  } catch (err) {
    deviceStatus.textContent = 'Device: Local Offline';
  }
}

// Event Listeners Setup
function setupEventListeners() {
  // Drag & drop on dropzone
  dropzone.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', (e) => handleFilesAdded(Array.from(e.target.files)));

  ['dragenter', 'dragover'].forEach(name => {
    dropzone.addEventListener(name, (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(name => {
    dropzone.addEventListener(name, (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
    });
  });

  dropzone.addEventListener('drop', (e) => {
    if (e.dataTransfer && e.dataTransfer.files) {
      handleFilesAdded(Array.from(e.dataTransfer.files));
    }
  });

  // Paste from clipboard anywhere
  document.addEventListener('paste', (e) => {
    if (e.clipboardData && e.clipboardData.files && e.clipboardData.files.length > 0) {
      handleFilesAdded(Array.from(e.clipboardData.files));
    }
  });

  // Action Buttons
  analyzeAllBtn.addEventListener('click', startBatchProcessing);
  cancelBtn.addEventListener('click', cancelBatchProcessing);
  clearAllBtn.addEventListener('click', clearQueue);

  // Filter Pills
  document.querySelectorAll('.summary-pill').forEach(pill => {
    pill.addEventListener('click', () => {
      document.querySelectorAll('.summary-pill').forEach(p => p.classList.remove('active'));
      pill.classList.add('active');
      state.currentFilter = pill.getAttribute('data-filter');
      renderTable();
    });
  });

  // Table Sorting
  document.querySelectorAll('th.sortable').forEach(th => {
    th.addEventListener('click', () => {
      const col = th.getAttribute('data-sort');
      if (state.sortColumn === col) {
        state.sortDirection = state.sortDirection === 'asc' ? 'desc' : 'asc';
      } else {
        state.sortColumn = col;
        state.sortDirection = 'asc';
      }
      renderTable();
    });
  });

  // Keyboard navigation: ArrowUp and ArrowDown
  document.addEventListener('keydown', (e) => {
    if (state.queue.length === 0) return;
    if (['ArrowDown', 'ArrowUp'].includes(e.key)) {
      // Don't hijack if user is typing in an input
      if (['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;
      e.preventDefault();
      navigateQueue(e.key === 'ArrowDown' ? 1 : -1);
    }
  });

  // ELA controls
  elaToggleBtn.addEventListener('click', toggleElaView);
  zoomImgBtn.addEventListener('click', openZoomModal);
  if (elaActiveImg) elaActiveImg.addEventListener('click', openZoomModal);

  // Export handlers
  downloadCsvBtn.addEventListener('click', exportToCsv);
  downloadJsonBtn.addEventListener('click', exportToJson);
  copyReportBtn.addEventListener('click', copyReportToClipboard);
}

// Compute SHA-256 for duplicate detection
async function computeFileHash(file) {
  try {
    const buffer = await file.slice(0, 65536).arrayBuffer(); // Hash first 64KB + length for speed
    const hashBuf = await crypto.subtle.digest('SHA-256', buffer);
    const hashArr = Array.from(new Uint8Array(hashBuf));
    return hashArr.map(b => b.toString(16).padStart(2, '0')).join('') + `_${file.size}`;
  } catch (e) {
    return `${file.name}_${file.size}_${file.lastModified}`;
  }
}

// Handle Files Added
async function handleFilesAdded(files) {
  if (!files || files.length === 0) return;

  const validFiles = files.filter(f => {
    const ext = '.' + f.name.split('.').pop().toLowerCase();
    return ALLOWED_EXTS.includes(ext) && f.size <= MAX_FILE_SIZE;
  });

  if (validFiles.length === 0) {
    alert('No supported image files found (allowed: JPEG, PNG, WEBP, GIF, BMP, max 25MB).');
    return;
  }

  const availableSlots = MAX_QUEUE_SIZE - state.queue.length;
  if (availableSlots <= 0) {
    alert(`Queue limit reached (maximum ${MAX_QUEUE_SIZE} images). Clear items to add more.`);
    return;
  }

  const toAdd = validFiles.slice(0, availableSlots);

  for (const file of toAdd) {
    const hash = await computeFileHash(file);
    const thumbUrl = URL.createObjectURL(file);
    const existing = state.queue.find(item => item.hash === hash);

    if (!existing) {
      state.queue.push({
        id: 'img_' + Math.random().toString(36).substr(2, 9),
        file,
        name: file.name,
        size: file.size,
        hash,
        status: 'queued',
        result: null,
        thumbUrl,
        error: null,
      });
    }
  }

  updateQueueVisibility();
  updateSummaryCounts();
  renderTable();

  // If nothing is selected, select first queued image
  if (state.selectedIndex === -1 && state.queue.length > 0) {
    selectImageByIndex(0);
  }
}

// Update UI Visibility
function updateQueueVisibility() {
  const hasItems = state.queue.length > 0;
  queueActionsBar.style.display = hasItems ? 'flex' : 'none';
  summaryStrip.style.display = hasItems ? 'flex' : 'none';
  exportBar.style.display = hasItems ? 'flex' : 'none';
  queueCountBadge.textContent = `${state.queue.length} image${state.queue.length === 1 ? '' : 's'}`;
}

// Update Summary Counts
function updateSummaryCounts() {
  const counts = {
    all: state.queue.length,
    'Likely AI': 0,
    'Leaning AI': 0,
    'Inconclusive': 0,
    'Leaning real': 0,
    'Likely real': 0,
    'Failed': 0,
  };

  state.queue.forEach(item => {
    if (item.status === 'failed') {
      counts['Failed']++;
    } else if (item.result && item.result.verdict) {
      if (counts[item.result.verdict] !== undefined) {
        counts[item.result.verdict]++;
      }
    }
  });

  document.getElementById('count-all').textContent = counts.all;
  document.getElementById('count-likely-ai').textContent = counts['Likely AI'];
  document.getElementById('count-leaning-ai').textContent = counts['Leaning AI'];
  document.getElementById('count-inconclusive').textContent = counts['Inconclusive'];
  document.getElementById('count-leaning-real').textContent = counts['Leaning real'];
  document.getElementById('count-likely-real').textContent = counts['Likely real'];
  document.getElementById('count-failed').textContent = counts['Failed'];
}

// Render Table
function renderTable() {
  queueTbody.innerHTML = '';

  let filtered = state.queue.map((item, originalIndex) => ({ ...item, originalIndex }));

  if (state.currentFilter !== 'all') {
    if (state.currentFilter === 'Failed') {
      filtered = filtered.filter(item => item.status === 'failed');
    } else {
      filtered = filtered.filter(item => item.result && item.result.verdict === state.currentFilter);
    }
  }

  // Sort
  filtered.sort((a, b) => {
    let valA = getSortValue(a, state.sortColumn);
    let valB = getSortValue(b, state.sortColumn);

    if (valA < valB) return state.sortDirection === 'asc' ? -1 : 1;
    if (valA > valB) return state.sortDirection === 'asc' ? 1 : -1;
    return 0;
  });

  if (filtered.length === 0) {
    const tr = document.createElement('tr');
    tr.className = 'empty-row';
    tr.innerHTML = `<td colspan="8">No images match the active filter '${state.currentFilter}'.</td>`;
    queueTbody.appendChild(tr);
    return;
  }

  filtered.forEach(item => {
    const tr = document.createElement('tr');
    if (item.originalIndex === state.selectedIndex) {
      tr.classList.add('selected-row');
    }

    const verdictBadgeHtml = getVerdictBadgeHtml(item);
    const scoreDisplay = getScoreDisplay(item);
    const spreadDisplay = item.result && item.result.spread !== undefined ? `${(item.result.spread * 100).toFixed(0)}%` : '—';
    const timeDisplay = item.result && item.result.time_taken_seconds !== undefined ? `${item.result.time_taken_seconds.toFixed(2)}s` : '—';
    const typeDisplay = item.result && item.result.image_type ? item.result.image_type : 'detecting...';

    tr.innerHTML = `
      <td><img src="${item.thumbUrl}" class="table-thumb" alt="Thumb"></td>
      <td class="table-filename" title="${escapeHtml(item.name)}">${escapeHtml(item.name)}</td>
      <td><span class="table-type">${escapeHtml(typeDisplay)}</span></td>
      <td>${verdictBadgeHtml}</td>
      <td class="table-score">${scoreDisplay}</td>
      <td class="table-spread">${spreadDisplay}</td>
      <td class="table-time">${timeDisplay}</td>
      <td>
        <button class="btn-remove-row" title="Remove" onclick="removeQueueItem('${item.id}', event)">✕</button>
      </td>
    `;

    tr.addEventListener('click', () => selectImageByIndex(item.originalIndex));
    queueTbody.appendChild(tr);
  });
}

function getSortValue(item, col) {
  switch (col) {
    case 'filename': return item.name.toLowerCase();
    case 'type': return (item.result && item.result.image_type) || '';
    case 'verdict': return (item.result && item.result.verdict) || item.status;
    case 'probability': return (item.result && item.result.ai_probability) || -1;
    case 'spread': return (item.result && item.result.spread) || -1;
    case 'time': return (item.result && item.result.time_taken_seconds) || -1;
    default: return item.originalIndex;
  }
}

function getVerdictBadgeHtml(item) {
  if (item.status === 'queued') return '<span class="status-badge badge-queued">Queued</span>';
  if (item.status === 'analyzing') return '<span class="status-badge badge-analyzing">Analyzing...</span>';
  if (item.status === 'failed') return '<span class="status-badge badge-failed">Failed</span>';

  const v = item.result.verdict;
  switch (v) {
    case 'Likely AI': return '<span class="status-badge badge-likely-ai">Likely AI</span>';
    case 'Leaning AI': return '<span class="status-badge badge-leaning-ai">Leaning AI</span>';
    case 'Inconclusive': return '<span class="status-badge badge-inconclusive">Inconclusive</span>';
    case 'Leaning real': return '<span class="status-badge badge-leaning-real">Leaning real</span>';
    case 'Likely real': return '<span class="status-badge badge-likely-real">Likely real</span>';
    default: return `<span class="status-badge badge-inconclusive">${escapeHtml(v)}</span>`;
  }
}

function getScoreDisplay(item) {
  if (item.status !== 'done' || !item.result) return '—';
  if (item.result.probability_display) {
    return escapeHtml(item.result.probability_display);
  }
  if (item.result.ai_percentage !== null && item.result.ai_percentage !== undefined) {
    return `${item.result.ai_percentage.toFixed(1)}%`;
  }
  return '—';
}

// Select Image for Detail View
function selectImageByIndex(index) {
  if (index < 0 || index >= state.queue.length) return;
  state.selectedIndex = index;
  renderTable();

  const item = state.queue[index];

  if (item.status === 'queued') {
    renderQueuedState(item);
  } else if (item.status === 'analyzing') {
    renderAnalyzingState(item);
  } else if (item.status === 'failed') {
    renderFailedState(item);
  } else if (item.status === 'done' && item.result) {
    renderDetailView(item);
  }
}

function renderQueuedState(item) {
  detailEmpty.style.display = 'none';
  detailSkeleton.style.display = 'none';
  detailContent.style.display = 'block';

  detailFilename.textContent = item.name;
  detailDims.textContent = 'Awaiting analysis';
  detailSize.textContent = formatBytes(item.size);
  detailTypeBadge.textContent = 'Queued';
  detailTime.textContent = '—';
  unreliableNotice.style.display = 'none';

  verdictBadge.className = 'verdict-badge verdict-inconclusive';
  verdictBadge.textContent = 'Queued';
  verdictConfidenceBadge.textContent = 'Pending';
  verdictConfidenceBadge.className = 'confidence-pill conf-low';
  verdictDisplayScore.textContent = 'Ready to Analyze';
  verdictOneLiner.textContent = "Click 'Analyze All' or inspect other images while queued.";
  whyVerdictText.textContent = "Analysis has not run yet. When started, RealLens evaluates provenance, metadata, ELA, Fourier spectra, and deep vision models.";

  spreadIndicatorPill.textContent = 'Spread: —';
  spreadBarFill.style.width = '0%';
  modelBarsContainer.innerHTML = '<div style="color: var(--muted); font-size: 12px;">Models will run upon analysis.</div>';
  elaActiveImg.src = item.thumbUrl;
  elaViewCaption.textContent = 'Input Preview';
  elaInterpretationNote.textContent = 'Error Level Analysis will be computed upon running.';
}

function renderAnalyzingState(item) {
  detailEmpty.style.display = 'none';
  detailSkeleton.style.display = 'flex';
  detailContent.style.display = 'none';
}

function renderFailedState(item) {
  detailEmpty.style.display = 'none';
  detailSkeleton.style.display = 'none';
  detailContent.style.display = 'block';

  detailFilename.textContent = item.name;
  detailDims.textContent = 'Error';
  detailSize.textContent = formatBytes(item.size);
  detailTypeBadge.textContent = 'Failed';
  detailTime.textContent = '—';
  unreliableNotice.style.display = 'none';

  verdictBadge.className = 'verdict-badge verdict-likely-ai';
  verdictBadge.textContent = 'Failed';
  verdictConfidenceBadge.textContent = 'Error';
  verdictConfidenceBadge.className = 'confidence-pill conf-low';
  verdictDisplayScore.textContent = 'Analysis Failed';
  verdictOneLiner.textContent = item.error || 'Failed to analyze image file.';
  whyVerdictText.textContent = `Error encountered: ${item.error || 'Server error'}. Please check file integrity.`;
}

// Render Complete Detail View (Phase 2.5 Architecture)
function renderDetailView(item) {
  const res = item.result;
  detailEmpty.style.display = 'none';
  detailSkeleton.style.display = 'none';
  detailContent.style.display = 'block';

  // 1. Header Metadata
  detailFilename.textContent = res.filename || item.name;
  detailDims.textContent = res.image_dimensions || '—';
  detailSize.textContent = formatBytes(item.size);
  detailTypeBadge.textContent = res.image_type || 'photo-like';
  detailTime.textContent = `${res.time_taken_seconds || 0}s`;

  // 2. Image Type Gating Notice
  if (res.is_photo_like === false) {
    unreliableNotice.style.display = 'flex';
    unreliableNoticeText.textContent = res.unreliable_notice ||
      "This looks like a screenshot/document/graphic. Photo-trained AI detectors are unreliable for this type.";
  } else {
    unreliableNotice.style.display = 'none';
  }

  // 3. Verdict Card & Probability Display
  const v = res.verdict || 'Inconclusive';
  verdictBadge.textContent = v;
  verdictBadge.className = 'verdict-badge ' + getVerdictClass(v);

  const conf = res.confidence || 'low';
  verdictConfidenceBadge.textContent = `${conf.toUpperCase()} CONFIDENCE`;
  verdictConfidenceBadge.className = `confidence-pill conf-${conf.toLowerCase()}`;

  // Big Probability or Range Display
  if (res.probability_display) {
    verdictDisplayScore.textContent = res.probability_display;
  } else if (res.ai_percentage !== null && res.ai_percentage !== undefined) {
    verdictDisplayScore.textContent = `${res.ai_percentage.toFixed(1)}%`;
  } else {
    verdictDisplayScore.textContent = '—';
  }

  verdictOneLiner.textContent = res.summary_reason || 'Multi-signal evaluation completed.';

  // 4. "Why this verdict?" Explanation Box
  whyVerdictText.textContent = res.why_this_verdict ||
    "Decision derived from consensus analysis across metadata, C2PA, physical ELA noise, and deep vision models.";

  // 5. Spread Indicator & Range Bar
  const spread = res.spread !== undefined ? res.spread : 0.0;
  spreadIndicatorPill.textContent = `Spread: ${(spread * 100).toFixed(1)}%`;

  const minScore = res.min_model_score !== undefined ? res.min_model_score : 0.5;
  const maxScore = res.max_model_score !== undefined ? res.max_model_score : 0.5;
  const leftPct = Math.max(0, Math.min(100, minScore * 100));
  const widthPct = Math.max(2, Math.min(100 - leftPct, (maxScore - minScore) * 100));

  spreadBarFill.style.left = `${leftPct}%`;
  spreadBarFill.style.width = `${widthPct}%`;
  spreadRangeLabel.textContent = `Min: ${(minScore * 100).toFixed(0)}% — Max: ${(maxScore * 100).toFixed(0)}%`;

  // 6. Individual Model Progress Bars
  renderModelBars(res.signals?.ml_ensemble);

  // 7. Forensic Visual Comparison (ELA)
  renderVisualEla(res.signals?.ela, item.thumbUrl);

  // 8. Collapsible Multi-Signal Accordion
  renderAccordionSignals(res.signals);
}

function getVerdictClass(verdict) {
  switch (verdict) {
    case 'Likely AI': return 'verdict-likely-ai';
    case 'Leaning AI': return 'verdict-leaning-ai';
    case 'Inconclusive': return 'verdict-inconclusive';
    case 'Leaning real': return 'verdict-leaning-real';
    case 'Likely real': return 'verdict-likely-real';
    default: return 'verdict-inconclusive';
  }
}

// Render Individual Model Progress Bars
function renderModelBars(mlSignal) {
  modelBarsContainer.innerHTML = '';
  if (!mlSignal || !mlSignal.per_model_scores) {
    modelBarsContainer.innerHTML = '<div style="color: var(--muted); font-size: 12px;">No vision models reported.</div>';
    return;
  }

  const scores = mlSignal.per_model_scores;
  for (const [rawName, score] of Object.entries(scores)) {
    const itemDiv = document.createElement('div');
    itemDiv.className = 'model-bar-item';

    const shortName = rawName.split('/').pop();
    const pct = score !== null && score !== undefined ? (score * 100).toFixed(1) : 'N/A';
    const numScore = score !== null && score !== undefined ? score : 0.5;

    // Confidence tag for individual model
    let confTag = 'LOW';
    let confClass = 'conf-low';
    if (numScore >= 0.85 || numScore <= 0.15) {
      confTag = 'HIGH';
      confClass = 'conf-high';
    } else if (numScore >= 0.70 || numScore <= 0.30) {
      confTag = 'MED';
      confClass = 'conf-medium';
    }

    // Color gradient for the bar
    let barColor = 'linear-gradient(90deg, #38BDF8, #6EA8FF)';
    if (numScore >= 0.65) barColor = 'linear-gradient(90deg, #F59E0B, #EF4444)';
    else if (numScore >= 0.45) barColor = 'linear-gradient(90deg, #A78BFA, #F59E0B)';

    itemDiv.innerHTML = `
      <div class="model-bar-header">
        <span class="model-name" title="${escapeHtml(rawName)}">${escapeHtml(shortName)}</span>
        <div class="model-score-tag">
          <span class="confidence-pill ${confClass}">${confTag}</span>
          <span>${pct}% AI</span>
        </div>
      </div>
      <div class="model-progress-track">
        <div class="model-progress-fill" style="width: ${numScore * 100}%; background: ${barColor};"></div>
      </div>
    `;
    modelBarsContainer.appendChild(itemDiv);
  }
}

// Render Visual ELA
function renderVisualEla(elaSignal, fallbackThumbUrl) {
  state.elaShowingOriginal = true;
  if (!elaSignal || !elaSignal.heatmap_base64) {
    elaActiveImg.src = fallbackThumbUrl;
    elaViewCaption.textContent = 'Original Input (No ELA Heatmap)';
    elaToggleBtn.style.display = 'none';
    return;
  }

  elaToggleBtn.style.display = 'inline-flex';
  const elaDataUri = `data:image/jpeg;base64,${elaSignal.heatmap_base64}`;

  elaActiveImg.dataset.original = fallbackThumbUrl;
  elaActiveImg.dataset.ela = elaDataUri;
  elaActiveImg.src = fallbackThumbUrl;
  elaViewCaption.textContent = 'Original Input (Click toggle to view ELA)';

  compOriginal.src = fallbackThumbUrl;
  compEla.src = elaDataUri;

  elaInterpretationNote.textContent = elaSignal.summary ||
    `ELA Mean Diff: ${elaSignal.mean_difference || 0}, Max: ${elaSignal.max_difference || 0}. Flat graphics or screenshots naturally display uniform difference patterns.`;
}

function toggleElaView() {
  if (!elaActiveImg.dataset.ela) return;
  state.elaShowingOriginal = !state.elaShowingOriginal;

  if (state.elaShowingOriginal) {
    elaActiveImg.src = elaActiveImg.dataset.original;
    elaViewCaption.textContent = 'Original Input';
  } else {
    elaActiveImg.src = elaActiveImg.dataset.ela;
    elaViewCaption.textContent = 'ELA Heatmap (Error Level Analysis)';
  }
}

// Render Accordion Signals
function renderAccordionSignals(signals) {
  if (!signals) return;

  // 1. C2PA
  const c2pa = signals.c2pa;
  const badgeC2pa = document.getElementById('badge-c2pa');
  const c2paDetails = document.getElementById('c2pa-details');
  if (c2pa && c2pa.manifest_present) {
    badgeC2pa.textContent = 'Manifest Present';
    badgeC2pa.style.color = 'var(--c-likely-real)';
    c2paDetails.innerHTML = `<strong>Authenticated Manifest:</strong> ${escapeHtml(c2pa.summary || 'Valid C2PA metadata found.')}`;
  } else {
    badgeC2pa.textContent = 'No Manifest';
    badgeC2pa.style.color = 'var(--muted)';
    c2paDetails.textContent = 'No cryptographic C2PA / CAI Content Credentials manifest found attached to this file.';
  }

  // 2. Metadata
  const meta = signals.metadata;
  const badgeMeta = document.getElementById('badge-meta');
  const metaDetails = document.getElementById('meta-details');
  if (meta && meta.ai_software_detected) {
    badgeMeta.textContent = 'AI Traces Found';
    badgeMeta.style.color = 'var(--c-likely-ai)';
    metaDetails.innerHTML = `<strong>Generator Artifacts:</strong> ${escapeHtml(meta.summary || 'AI metadata signature identified.')}`;
  } else if (meta && meta.camera_make) {
    badgeMeta.textContent = 'EXIF Camera Header';
    badgeMeta.style.color = 'var(--c-likely-real)';
    metaDetails.textContent = `Hardware EXIF tags: ${meta.camera_make} ${meta.camera_model || ''}. Note: EXIF is easily forged.`;
  } else {
    badgeMeta.textContent = 'Clean / Stripped';
    badgeMeta.style.color = 'var(--muted)';
    metaDetails.textContent = 'No generator prompts or hardware camera tags found in image chunks.';
  }

  // 3. ELA
  const ela = signals.ela;
  const badgeEla = document.getElementById('badge-ela');
  const elaDetails = document.getElementById('ela-details');
  if (ela) {
    badgeEla.textContent = ela.score !== null ? `${(ela.score * 100).toFixed(0)}% Score` : 'N/A';
    elaDetails.textContent = ela.summary || `Mean Diff: ${ela.mean_difference}, Max: ${ela.max_difference}.`;
  }

  // 4. FFT
  const fft = signals.frequency;
  const badgeFft = document.getElementById('badge-fft');
  const fftDetails = document.getElementById('fft-details');
  if (fft) {
    badgeFft.textContent = fft.score !== null ? `${(fft.score * 100).toFixed(0)}% Score` : 'N/A';
    fftDetails.textContent = fft.summary || `Upsampling artifact peak ratio: ${fft.periodic_peak_ratio || 0}. High frequency energy: ${fft.high_freq_energy || 0}.`;
  }

  // 5. ML Ensemble
  const ml = signals.ml_ensemble;
  const badgeMl = document.getElementById('badge-ml');
  const mlDetails = document.getElementById('ml-details');
  if (ml) {
    badgeMl.textContent = ml.score !== null ? `${(ml.score * 100).toFixed(0)}% Median` : 'N/A';
    mlDetails.innerHTML = `<strong>Ensemble Consensus:</strong> ${escapeHtml(ml.summary || 'Evaluated across models.')}`;
  }

  // 6. Gemini
  const gemini = signals.gemini;
  const badgeGemini = document.getElementById('badge-gemini');
  const geminiDetails = document.getElementById('gemini-details');
  if (gemini && gemini.score !== null) {
    badgeGemini.textContent = `${(gemini.score * 100).toFixed(0)}% Score`;
    geminiDetails.textContent = gemini.reasoning || (gemini.evidence || []).join('; ');
  } else {
    badgeGemini.textContent = 'Disabled';
    geminiDetails.textContent = 'Gemini visual reasoning inactive. Add GEMINI_API_KEY in .env to enable optional LLM verification.';
  }
}

// Toggle Accordion Row
window.toggleAccordion = function(id) {
  const el = document.getElementById(id);
  if (el) {
    el.classList.toggle('open');
  }
};

// Sequential Batch Processing (CPU-Friendly)
async function startBatchProcessing() {
  if (state.isProcessing) return;

  const queuedItems = state.queue.filter(item => item.status === 'queued' || item.status === 'failed');
  if (queuedItems.length === 0) {
    alert('No queued images to analyze.');
    return;
  }

  state.isProcessing = true;
  state.abortController = new AbortController();
  state.timesElapsed = [];

  analyzeAllBtn.style.display = 'none';
  cancelBtn.style.display = 'inline-flex';
  progressContainer.style.display = 'block';

  const total = queuedItems.length;
  let processed = 0;

  for (const item of queuedItems) {
    if (state.abortController.signal.aborted) break;

    // Check SHA-256 cache
    if (state.analysisCache.has(item.hash)) {
      item.result = state.analysisCache.get(item.hash);
      item.status = 'done';
      processed++;
      updateBatchProgress(processed, total);
      updateSummaryCounts();
      renderTable();
      if (state.selectedIndex === state.queue.indexOf(item)) {
        renderDetailView(item);
      }
      continue;
    }

    item.status = 'analyzing';
    renderTable();
    if (state.selectedIndex === state.queue.indexOf(item)) {
      renderAnalyzingState(item);
    }

    const t0 = performance.now();
    try {
      const formData = new FormData();
      formData.append('image', item.file, item.name);

      const res = await fetch('/analyze', {
        method: 'POST',
        body: formData,
        signal: state.abortController.signal,
      });

      const data = await res.json();
      if (res.ok && data.status === 'success' && data.result) {
        item.result = data.result;
        item.status = 'done';
        state.analysisCache.set(item.hash, data.result);
      } else {
        item.status = 'failed';
        item.error = data.message || 'Analysis error';
      }
    } catch (err) {
      if (err.name === 'AbortError') {
        item.status = 'queued';
        break;
      }
      item.status = 'failed';
      item.error = err.message || 'Network request failed';
    }

    const elapsedSec = (performance.now() - t0) / 1000;
    state.timesElapsed.push(elapsedSec);

    processed++;
    updateBatchProgress(processed, total);
    updateSummaryCounts();
    renderTable();

    if (state.selectedIndex === state.queue.indexOf(item)) {
      if (item.status === 'done') renderDetailView(item);
      else renderFailedState(item);
    }
  }

  state.isProcessing = false;
  state.abortController = null;
  analyzeAllBtn.style.display = 'inline-flex';
  cancelBtn.style.display = 'none';
  progressContainer.style.display = 'none';
}

function updateBatchProgress(processed, total) {
  const pct = Math.round((processed / total) * 100);
  progressFill.style.width = `${pct}%`;
  progressText.textContent = `Analyzing ${processed} of ${total} images (${pct}%)`;

  if (state.timesElapsed.length > 0) {
    const avgSec = state.timesElapsed.reduce((a, b) => a + b, 0) / state.timesElapsed.length;
    const remaining = Math.max(0, (total - processed) * avgSec);
    progressEta.textContent = `~${avgSec.toFixed(1)}s/img • Est. remaining: ${remaining.toFixed(0)}s`;
  } else {
    progressEta.textContent = 'Calculating ETA...';
  }
}

function cancelBatchProcessing() {
  if (state.abortController) {
    state.abortController.abort();
  }
}

function clearQueue() {
  if (state.isProcessing) {
    cancelBatchProcessing();
  }
  state.queue.forEach(item => {
    if (item.thumbUrl) URL.revokeObjectURL(item.thumbUrl);
  });
  state.queue = [];
  state.selectedIndex = -1;
  updateQueueVisibility();
  updateSummaryCounts();
  renderTable();

  detailEmpty.style.display = 'flex';
  detailSkeleton.style.display = 'none';
  detailContent.style.display = 'none';
}

window.removeQueueItem = function(id, event) {
  if (event) event.stopPropagation();
  const idx = state.queue.findIndex(item => item.id === id);
  if (idx !== -1) {
    const item = state.queue[idx];
    if (item.thumbUrl) URL.revokeObjectURL(item.thumbUrl);
    state.queue.splice(idx, 1);

    if (state.selectedIndex === idx) {
      state.selectedIndex = state.queue.length > 0 ? Math.min(idx, state.queue.length - 1) : -1;
    } else if (state.selectedIndex > idx) {
      state.selectedIndex--;
    }

    updateQueueVisibility();
    updateSummaryCounts();
    renderTable();

    if (state.selectedIndex >= 0) {
      selectImageByIndex(state.selectedIndex);
    } else {
      detailEmpty.style.display = 'flex';
      detailSkeleton.style.display = 'none';
      detailContent.style.display = 'none';
    }
  }
};

// Keyboard Arrow Navigation
function navigateQueue(delta) {
  if (state.queue.length === 0) return;
  let newIdx = state.selectedIndex + delta;
  if (newIdx < 0) newIdx = state.queue.length - 1;
  if (newIdx >= state.queue.length) newIdx = 0;
  selectImageByIndex(newIdx);

  // Scroll active table row into view
  const activeTr = queueTbody.querySelector('.selected-row');
  if (activeTr) {
    activeTr.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }
}

// Modal Zoom
function openZoomModal() {
  const imgSrc = (elaActiveImg && elaActiveImg.src) || (compOriginal && compOriginal.src);
  if (!imgSrc) return;
  zoomModalImg.src = imgSrc;
  zoomTitle.textContent = `${detailFilename.textContent} — ${elaViewCaption.textContent}`;
  zoomModal.classList.add('active');
}

window.closeZoomModal = function() {
  zoomModal.classList.remove('active');
};

// Export Handlers
function exportToCsv() {
  if (state.queue.length === 0) return;
  const rows = [
    ['Filename', 'Type', 'Verdict', 'Score_or_Range', 'AI_Probability', 'Spread', 'Time_Sec', 'Confidence', 'Summary', 'DINOv2_Probe', 'Dima806', 'SDXL_Detector', 'Ateeqq']
  ];

  state.queue.forEach(item => {
    const r = item.result;
    if (!r) {
      rows.push([item.name, 'unknown', item.status, '—', '—', '—', '—', '—', item.error || 'Pending', '', '', '', '']);
    } else {
      const perModel = r.signals?.ml_ensemble?.per_model_scores || {};
      rows.push([
        r.filename || item.name,
        r.image_type || 'photo-like',
        r.verdict || 'Inconclusive',
        r.probability_display || (r.ai_percentage ? `${r.ai_percentage}%` : '—'),
        r.ai_probability !== undefined ? r.ai_probability : '',
        r.spread !== undefined ? r.spread : '',
        r.time_taken_seconds || '',
        r.confidence || '',
        `"${(r.summary_reason || '').replace(/"/g, '""')}"`,
        perModel['generalizing_dinov2_probe'] !== undefined ? perModel['generalizing_dinov2_probe'] : '',
        perModel['dima806/ai_vs_real_image_detection'] !== undefined ? perModel['dima806/ai_vs_real_image_detection'] : '',
        perModel['Organika/sdxl-detector'] !== undefined ? perModel['Organika/sdxl-detector'] : '',
        perModel['Ateeqq/ai-vs-human-image-detector'] !== undefined ? perModel['Ateeqq/ai-vs-human-image-detector'] : '',
      ]);
    }
  });

  const csvContent = 'data:text/csv;charset=utf-8,' + rows.map(e => e.join(',')).join('\n');
  const encodedUri = encodeURI(csvContent);
  const link = document.createElement('a');
  link.setAttribute('href', encodedUri);
  link.setAttribute('download', `reallens_batch_report_${Date.now()}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

function exportToJson() {
  if (state.queue.length === 0) return;
  const payload = {
    exported_at: new Date().toISOString(),
    total_images: state.queue.length,
    results: state.queue.map(item => ({
      name: item.name,
      size: item.size,
      status: item.status,
      result: item.result,
      error: item.error,
    })),
  };

  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `reallens_forensics_${Date.now()}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function copyReportToClipboard() {
  if (state.queue.length === 0) return;

  const counts = {
    'Likely AI': 0, 'Leaning AI': 0, 'Inconclusive': 0, 'Leaning real': 0, 'Likely real': 0, 'Failed': 0,
  };

  state.queue.forEach(item => {
    if (item.status === 'failed') counts['Failed']++;
    else if (item.result && counts[item.result.verdict] !== undefined) counts[item.result.verdict]++;
  });

  let md = `# RealLens Forensics Summary Report\n`;
  md += `**Date:** ${new Date().toLocaleString()}\n`;
  md += `**Total Analyzed:** ${state.queue.length}\n\n`;
  md += `### Verdict Summary\n`;
  md += `- Likely AI: ${counts['Likely AI']}\n`;
  md += `- Leaning AI: ${counts['Leaning AI']}\n`;
  md += `- Inconclusive: ${counts['Inconclusive']}\n`;
  md += `- Leaning real: ${counts['Leaning real']}\n`;
  md += `- Likely real: ${counts['Likely real']}\n`;
  if (counts['Failed'] > 0) md += `- Failed: ${counts['Failed']}\n`;

  md += `\n### Detailed Breakdown\n`;
  md += `| File | Image Type | Verdict | Probability / Range | Spread | Time |\n`;
  md += `| :--- | :--- | :--- | :--- | :--- | :--- |\n`;

  state.queue.forEach(item => {
    const r = item.result;
    if (r) {
      md += `| ${r.filename || item.name} | ${r.image_type} | ${r.verdict} | ${r.probability_display || r.ai_percentage + '%'} | ${(r.spread * 100).toFixed(0)}% | ${r.time_taken_seconds}s |\n`;
    } else {
      md += `| ${item.name} | unknown | ${item.status} | — | — | — |\n`;
    }
  });

  navigator.clipboard.writeText(md).then(() => {
    alert('Forensics summary copied to clipboard in Markdown format!');
  }).catch(() => {
    alert('Failed to copy report to clipboard.');
  });
}

// Helpers
function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function formatBytes(bytes) {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}
