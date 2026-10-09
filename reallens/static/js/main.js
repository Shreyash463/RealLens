/**
 * RealLens - Phase 3 Frontend Controller
 * Complete client-side management for multi-image queue, batch forensics,
 * fixed-column results table, summary filters, and detailed visual inspection.
 */

(function () {
  'use strict';

  // State Management
  const state = {
    queue: [],            // Array of { id, file, name, size, status, result, thumbUrl }
    results: [],          // Completed analysis result objects
    selectedId: null,     // Currently inspected item ID
    analyzing: false,     // Batch execution state
    cancelRequested: false,
    activeFilter: 'all',  // 'all' | 'Likely AI' | 'Leaning AI' | 'Inconclusive' | 'Leaning Real' | 'Likely Real' | 'Failed'
    sortColumn: null,
    sortAsc: true,
    viewMode: 'ela',      // 'orig' | 'ela' | 'dual'
    avgDurationSec: 0.65,
  };

  // DOM Elements Cache
  const el = {
    dropzone: document.getElementById('dropzone'),
    fileInput: document.getElementById('file-input'),
    analyzeAllBtn: document.getElementById('analyze-all-btn'),
    cancelBtn: document.getElementById('cancel-btn'),
    clearAllBtn: document.getElementById('clear-all-btn'),
    queueCountBadge: document.getElementById('queue-count-badge'),
    queueSummaryLabel: document.getElementById('queue-summary-label'),
    queueList: document.getElementById('queue-list'),
    queueEmptyState: document.getElementById('queue-empty-state'),
    progressWrapper: document.getElementById('progress-wrapper'),
    progressStatusText: document.getElementById('progress-status-text'),
    progressEtaText: document.getElementById('progress-eta-text'),
    progressFill: document.getElementById('progress-fill'),
    downloadCsvBtn: document.getElementById('download-csv-btn'),
    downloadJsonBtn: document.getElementById('download-json-btn'),
    summaryStrip: document.getElementById('summary-strip'),
    resultsTbody: document.getElementById('results-tbody'),
    filterIndicator: document.getElementById('filter-indicator'),
    currentFilterName: document.getElementById('current-filter-name'),
    clearFilterBtn: document.getElementById('clear-filter-btn'),
    detailCard: document.getElementById('detail-card'),
    detailEmptyState: document.getElementById('detail-empty-state'),
    detailActiveContainer: document.getElementById('detail-active-container'),
    detailFilename: document.getElementById('detail-filename'),
    detailDimensions: document.getElementById('detail-dimensions'),
    detailSize: document.getElementById('detail-size'),
    detailImageType: document.getElementById('detail-image-type'),
    detailCondition: document.getElementById('detail-condition'),
    detailAutocrop: document.getElementById('detail-autocrop'),
    detailWatermark: document.getElementById('detail-watermark'),
    detailDuration: document.getElementById('detail-duration'),
    detailCached: document.getElementById('detail-cached'),
    autoAnalyzeToggle: document.getElementById('auto-analyze-toggle'),
    rerunBtn: document.getElementById('rerun-btn'),
    copyReportBtn: document.getElementById('copy-report-btn'),
    deepCheckGeminiBtn: document.getElementById('deep-check-gemini-btn'),
    unreliableNoticeBox: document.getElementById('unreliable-notice-box'),
    unreliableNoticeText: document.getElementById('unreliable-notice-text'),
    compressionNoticeBox: document.getElementById('compression-notice-box'),
    compressionNoticeText: document.getElementById('compression-notice-text'),
    bigVerdictBox: document.getElementById('big-verdict-box'),
    verdictBadgeMain: document.getElementById('verdict-badge-main'),
    confidenceBadgeMain: document.getElementById('confidence-badge-main'),
    verdictScoreDisplay: document.getElementById('verdict-score-display'),
    verdictSummaryLine: document.getElementById('verdict-summary-line'),
    verdictNoticesRow: document.getElementById('verdict-notices-row'),
    noticeChipType: document.getElementById('notice-chip-type'),
    noticeChipCondition: document.getElementById('notice-chip-condition'),
    noticeChipAutocrop: document.getElementById('notice-chip-autocrop'),
    noticeChipWatermark: document.getElementById('notice-chip-watermark'),
    noticeChipCompression: document.getElementById('notice-chip-compression'),
    noticeChipRange: document.getElementById('notice-chip-range'),
    whyVerdictContent: document.getElementById('why-verdict-content'),
    badgeAiCount: document.getElementById('badge-ai-count'),
    evidenceAiList: document.getElementById('evidence-ai-list'),
    badgeRealCount: document.getElementById('badge-real-count'),
    evidenceRealList: document.getElementById('evidence-real-list'),
    badgeUnreliableCount: document.getElementById('badge-unreliable-count'),
    evidenceUnreliableList: document.getElementById('evidence-unreliable-list'),
    whatNextChecklist: document.getElementById('what-next-checklist'),
    spreadValueBadge: document.getElementById('spread-value-badge'),
    spreadRangeBar: document.getElementById('spread-range-bar'),
    spreadLabelMinmax: document.getElementById('spread-label-minmax'),
    modelBarsList: document.getElementById('model-bars-list'),
    viewModeOrig: document.getElementById('view-mode-orig'),
    viewModeEla: document.getElementById('view-mode-ela'),
    viewModeDual: document.getElementById('view-mode-dual'),
    zoomForensicBtn: document.getElementById('zoom-forensic-btn'),
    elaFrameSingle: document.getElementById('ela-frame-single'),
    elaFrameDual: document.getElementById('ela-frame-dual'),
    elaImgSingle: document.getElementById('ela-img-single'),
    elaCaptionSingle: document.getElementById('ela-caption-single'),
    elaImgOriginal: document.getElementById('ela-img-original'),
    elaImgHeatmap: document.getElementById('ela-img-heatmap'),
    elaNoteText: document.getElementById('ela-note-text'),
    badgeMetaStatus: document.getElementById('badge-meta-status'),
    metaDetailsBody: document.getElementById('meta-details-body'),
    badgeC2paStatus: document.getElementById('badge-c2pa-status'),
    c2paDetailsBody: document.getElementById('c2pa-details-body'),
    badgeElaStatus: document.getElementById('badge-ela-status'),
    elaDetailsBody: document.getElementById('ela-details-body'),
    badgeFftStatus: document.getElementById('badge-fft-status'),
    fftDetailsBody: document.getElementById('fft-details-body'),
    badgeMlStatus: document.getElementById('badge-ml-status'),
    mlDetailsBody: document.getElementById('ml-details-body'),
    badgeGeminiStatus: document.getElementById('badge-gemini-status'),
    geminiDetailsBody: document.getElementById('gemini-details-body'),
    geminiResultCard: document.getElementById('gemini-result-card'),
    geminiCardBadge: document.getElementById('gemini-card-badge'),
    geminiFindingsList: document.getElementById('gemini-findings-list'),
    geminiCounterList: document.getElementById('gemini-counter-list'),
    geminiFindingsSection: document.getElementById('gemini-findings-section'),
    geminiCounterSection: document.getElementById('gemini-counter-section'),
    zoomModal: document.getElementById('zoom-modal'),
    zoomModalImage: document.getElementById('zoom-modal-image'),
    zoomModalTitle: document.getElementById('zoom-modal-title'),
    deviceText: document.getElementById('device-text'),
    cacheText: document.getElementById('cache-text'),
    versionText: document.getElementById('version-text'),
    modelsText: document.getElementById('models-text'),
  };

  // ===========================================================================
  // Initialization & System Health
  // ===========================================================================

  async function checkHealth() {
    try {
      const res = await fetch('/health');
      if (res.ok) {
        const data = await res.json();
        if (el.deviceText) {
          el.deviceText.textContent = `Device: ${data.device ? data.device.toUpperCase() : 'CPU'}`;
        }
        if (el.versionText && data.pipeline_version) {
          el.versionText.textContent = `Pipeline: ${data.pipeline_version}`;
        }
        if (el.modelsText) {
          const loaded = data.models ? Object.keys(data.models).length : 4;
          el.modelsText.textContent = `Models: ${loaded}/4 Loaded`;
        }
        if (el.cacheText && typeof data.cache_entries === 'number') {
          el.cacheText.textContent = `⚡ SHA-256 Cache (${data.cache_entries} cached)`;
        }
      }
    } catch (e) {
      console.warn('Health check unreachable:', e);
    }
  }

  // ===========================================================================
  // Queue & Upload Handling
  // ===========================================================================

  function formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  }

  function addFilesToQueue(fileList) {
    const maxFiles = 50;
    const maxBytes = 25 * 1024 * 1024; // 25 MB
    const remainingSlots = maxFiles - state.queue.length;

    if (remainingSlots <= 0) {
      alert(`Queue limit reached (maximum ${maxFiles} images).`);
      return;
    }

    const files = Array.from(fileList).slice(0, remainingSlots);
    let addedCount = 0;

    files.forEach(file => {
      if (!file.type.startsWith('image/') && !file.name.match(/\.(jpg|jpeg|png|webp|gif|bmp|tiff|tif)$/i)) {
        return;
      }
      if (file.size > maxBytes) {
        alert(`File "${file.name}" exceeds the 25MB limit and was skipped.`);
        return;
      }

      const id = 'img_' + Date.now() + '_' + Math.random().toString(36).substr(2, 6);
      const thumbUrl = URL.createObjectURL(file);

      const item = {
        id,
        file,
        name: file.name,
        size: file.size,
        status: 'queued', // 'queued' | 'analyzing' | 'done' | 'failed'
        verdict: null,
        result: null,
        thumbUrl,
      };

      state.queue.push(item);
      addedCount++;
    });

    if (addedCount > 0) {
      renderQueue();
      updateQueueStats();
      if (el.autoAnalyzeToggle && el.autoAnalyzeToggle.checked && !state.analyzing) {
        setTimeout(analyzeAll, 50);
      }
    }
  }

  function updateQueueStats() {
    const total = state.queue.length;
    const queuedCount = state.queue.filter(i => i.status === 'queued').length;
    const doneCount = state.queue.filter(i => i.status === 'done').length;

    el.queueCountBadge.textContent = `${total} items in queue`;
    el.queueSummaryLabel.textContent = total > 0 ? `${doneCount} analyzed, ${queuedCount} ready` : 'No images loaded';

    el.analyzeAllBtn.disabled = queuedCount === 0 || state.analyzing;
    el.clearAllBtn.disabled = total === 0 || state.analyzing;
    el.downloadCsvBtn.disabled = doneCount === 0;
    el.downloadJsonBtn.disabled = doneCount === 0;

    if (total === 0) {
      el.queueEmptyState.style.display = 'flex';
    } else {
      el.queueEmptyState.style.display = 'none';
    }
  }

  function renderQueue() {
    // Preserve empty state element
    const items = state.queue;
    el.queueList.innerHTML = '';
    el.queueList.appendChild(el.queueEmptyState);

    items.forEach(item => {
      const row = document.createElement('div');
      row.className = `queue-row ${item.id === state.selectedId ? 'active' : ''}`;
      row.id = `qrow-${item.id}`;
      row.onclick = () => selectItem(item.id);

      const thumb = document.createElement('img');
      thumb.className = 'queue-thumb';
      thumb.src = item.thumbUrl;
      thumb.alt = item.name;

      const col = document.createElement('div');
      col.className = 'queue-info-col';

      const fname = document.createElement('div');
      fname.className = 'queue-filename';
      fname.textContent = item.name;
      fname.title = item.name;

      const submeta = document.createElement('div');
      submeta.className = 'queue-submeta';

      const dot = document.createElement('span');
      dot.className = `queue-verdict-dot ${getVerdictDotClass(item.verdict)}`;
      if (!item.verdict) dot.style.display = 'none';

      const sizeSpan = document.createElement('span');
      sizeSpan.textContent = formatBytes(item.size);

      submeta.appendChild(dot);
      submeta.appendChild(sizeSpan);

      col.appendChild(fname);
      col.appendChild(submeta);

      const badge = document.createElement('span');
      badge.className = `queue-status-badge ${item.status}`;
      badge.textContent = item.status === 'analyzing' ? 'Analyzing...' :
                          item.status === 'queued' ? 'Queued' :
                          item.status === 'done' ? (item.verdict || 'Done') : 'Failed';

      const removeBtn = document.createElement('button');
      removeBtn.className = 'queue-remove-btn';
      removeBtn.innerHTML = '✕';
      removeBtn.title = 'Remove item';
      removeBtn.onclick = (e) => {
        e.stopPropagation();
        removeItem(item.id);
      };

      row.appendChild(thumb);
      row.appendChild(col);
      row.appendChild(badge);
      row.appendChild(removeBtn);

      el.queueList.appendChild(row);
    });
  }

  function removeItem(id) {
    const idx = state.queue.findIndex(i => i.id === id);
    if (idx !== -1) {
      URL.revokeObjectURL(state.queue[idx].thumbUrl);
      state.queue.splice(idx, 1);
    }
    const resIdx = state.results.findIndex(r => r._id === id);
    if (resIdx !== -1) {
      state.results.splice(resIdx, 1);
    }
    if (state.selectedId === id) {
      state.selectedId = state.results.length > 0 ? state.results[0]._id : null;
      renderDetailView();
    }
    renderQueue();
    renderResultsTable();
    updateSummaryStrip();
    updateQueueStats();
  }

  function clearAll() {
    if (state.analyzing) return;
    state.queue.forEach(i => URL.revokeObjectURL(i.thumbUrl));
    state.queue = [];
    state.results = [];
    state.selectedId = null;
    state.activeFilter = 'all';

    renderQueue();
    renderResultsTable();
    updateSummaryStrip();
    renderDetailView();
    updateQueueStats();

    el.progressWrapper.style.display = 'none';
  }

  // ===========================================================================
  // Batch Execution Controller
  // ===========================================================================

  async function analyzeAll() {
    if (state.analyzing) return;
    const toAnalyze = state.queue.filter(i => i.status === 'queued');
    if (toAnalyze.length === 0) return;

    state.analyzing = true;
    state.cancelRequested = false;

    el.analyzeAllBtn.style.display = 'none';
    el.cancelBtn.style.display = 'inline-flex';
    el.clearAllBtn.disabled = true;
    el.progressWrapper.style.display = 'block';

    const totalToRun = toAnalyze.length;
    let completedCount = 0;
    const startTime = performance.now();

    for (let i = 0; i < toAnalyze.length; i++) {
      if (state.cancelRequested) {
        break;
      }

      const item = toAnalyze[i];
      item.status = 'analyzing';
      updateQueueRow(item.id);

      // Progress calculation
      const pct = Math.round((completedCount / totalToRun) * 100);
      el.progressFill.style.width = `${pct}%`;
      el.progressStatusText.textContent = `Analyzing ${completedCount + 1} of ${totalToRun}: ${item.name}`;

      const elapsedSec = (performance.now() - startTime) / 1000;
      if (completedCount > 0) {
        const avgPerItem = elapsedSec / completedCount;
        const remainingItems = totalToRun - completedCount;
        const etaSec = Math.max(0, Math.round(avgPerItem * remainingItems));
        el.progressEtaText.textContent = `Est. remaining: ~${etaSec}s`;
      } else {
        el.progressEtaText.textContent = `Est. remaining: calculating...`;
      }

      try {
        const formData = new FormData();
        formData.append('image', item.file, item.name);

        const res = await fetch('/analyze', {
          method: 'POST',
          body: formData,
        });

        if (!res.ok) {
          throw new Error(`Server returned HTTP ${res.status}`);
        }

        const data = await res.json();
        if (data.status === 'success' && data.result) {
          const r = data.result;
          r._id = item.id;
          r._thumbUrl = item.thumbUrl;
          item.status = 'done';
          item.verdict = r.verdict;
          item.result = r;

          state.results.push(r);

          // If this is first result or current selection, activate detail view
          if (!state.selectedId || state.selectedId === item.id) {
            selectItem(item.id);
          }
        } else {
          throw new Error(data.message || 'Analysis failed');
        }
      } catch (err) {
        console.error(`Error analyzing ${item.name}:`, err);
        item.status = 'failed';
        item.verdict = 'Failed';
        const errResult = {
          _id: item.id,
          _thumbUrl: item.thumbUrl,
          filename: item.name,
          status: 'error',
          verdict: 'Failed',
          ai_percentage: null,
          probability_display: 'Error',
          confidence: 'none',
          image_type: 'unknown',
          time_taken_seconds: 0,
          summary_reason: err.message || 'Inference error',
          why_this_verdict: 'Failed to process this file through the forensic pipeline.',
        };
        item.result = errResult;
        state.results.push(errResult);
      }

      completedCount++;
      updateQueueRow(item.id);
      renderResultsTable();
      updateSummaryStrip();
      updateQueueStats();
    }

    // Finalize Batch Run
    state.analyzing = false;
    el.analyzeAllBtn.style.display = 'inline-flex';
    el.cancelBtn.style.display = 'none';
    el.clearAllBtn.disabled = false;
    el.progressFill.style.width = '100%';
    el.progressStatusText.textContent = state.cancelRequested ? 'Batch analysis cancelled.' : `Completed ${completedCount} of ${totalToRun} images.`;
    el.progressEtaText.textContent = 'Done';

    setTimeout(() => {
      if (!state.analyzing) {
        el.progressWrapper.style.display = 'none';
      }
    }, 3500);

    updateQueueStats();
  }

  function updateQueueRow(id) {
    const item = state.queue.find(i => i.id === id);
    if (!item) return;
    const row = document.getElementById(`qrow-${id}`);
    if (!row) return;

    const badge = row.querySelector('.queue-status-badge');
    if (badge) {
      badge.className = `queue-status-badge ${item.status}`;
      badge.textContent = item.status === 'analyzing' ? 'Analyzing...' :
                          item.status === 'queued' ? 'Queued' :
                          item.status === 'done' ? (item.verdict || 'Done') : 'Failed';
    }

    const dot = row.querySelector('.queue-verdict-dot');
    if (dot) {
      dot.className = `queue-verdict-dot ${getVerdictDotClass(item.verdict)}`;
      dot.style.display = item.verdict ? 'inline-block' : 'none';
    }
  }

  // ===========================================================================
  // Summary Strip & Filter Logic
  // ===========================================================================

  function normalizeVerdict(v) {
    if (!v) return 'Failed';
    const s = String(v).trim().toLowerCase();
    if (s.includes('strong') || s === 'likely ai' || s.includes('ai evidence found')) return 'AI evidence found (strong)';
    if (s === 'leaning ai') return 'Leaning AI';
    if (s.includes('inconclusive')) return 'Inconclusive';
    if (s.includes('no ai evidence') || s === 'leaning real') return 'No AI evidence found (not proof of real)';
    if (s === 'likely real') return 'Likely real';
    if (s === 'failed') return 'Failed';
    return 'Inconclusive';
  }

  function updateSummaryStrip() {
    const counts = {
      all: state.results.length,
      'AI evidence found (strong)': 0,
      'Leaning AI': 0,
      'Inconclusive': 0,
      'No AI evidence found (not proof of real)': 0,
      'Likely real': 0,
      'Failed': 0,
    };

    state.results.forEach(r => {
      const v = normalizeVerdict(r.verdict);
      if (counts[v] !== undefined) {
        counts[v]++;
      } else {
        counts['Inconclusive']++;
      }
    });

    if (document.getElementById('count-all')) document.getElementById('count-all').textContent = counts.all;
    if (document.getElementById('count-strong-ai')) document.getElementById('count-strong-ai').textContent = counts['AI evidence found (strong)'];
    if (document.getElementById('count-leaning-ai')) document.getElementById('count-leaning-ai').textContent = counts['Leaning AI'];
    if (document.getElementById('count-inconclusive')) document.getElementById('count-inconclusive').textContent = counts['Inconclusive'];
    if (document.getElementById('count-no-evidence')) document.getElementById('count-no-evidence').textContent = counts['No AI evidence found (not proof of real)'];
    if (document.getElementById('count-likely-real')) document.getElementById('count-likely-real').textContent = counts['Likely real'];
    if (document.getElementById('count-failed')) document.getElementById('count-failed').textContent = counts['Failed'];

    // Update filter pills active status
    const pills = el.summaryStrip.querySelectorAll('.summary-pill');
    pills.forEach(p => {
      const f = p.getAttribute('data-filter');
      p.classList.toggle('active', f === state.activeFilter);
    });

    if (state.activeFilter !== 'all') {
      el.filterIndicator.style.display = 'flex';
      el.currentFilterName.textContent = state.activeFilter;
    } else {
      el.filterIndicator.style.display = 'none';
    }
  }

  function setFilter(filterName) {
    state.activeFilter = filterName;
    updateSummaryStrip();
    renderResultsTable();
  }

  // ===========================================================================
  // Results Table (Fixed Widths, Sort, Selection, Arrow Navigation)
  // ===========================================================================

  function getFilteredResults() {
    if (state.activeFilter === 'all') {
      return [...state.results];
    }
    return state.results.filter(r => normalizeVerdict(r.verdict) === state.activeFilter);
  }

  function renderResultsTable() {
    let items = getFilteredResults();

    // Sort if column set
    if (state.sortColumn) {
      items.sort((a, b) => {
        let va = a[state.sortColumn];
        let vb = b[state.sortColumn];

        if (state.sortColumn === 'probability') {
          va = a.ai_probability !== undefined && a.ai_probability !== null ? a.ai_probability : 0.5;
          vb = b.ai_probability !== undefined && b.ai_probability !== null ? b.ai_probability : 0.5;
        } else if (state.sortColumn === 'time') {
          va = a.time_taken_seconds || 0;
          vb = b.time_taken_seconds || 0;
        }

        if (va < vb) return state.sortAsc ? -1 : 1;
        if (va > vb) return state.sortAsc ? 1 : -1;
        return 0;
      });
    }

    el.resultsTbody.innerHTML = '';

    if (items.length === 0) {
      const emptyRow = document.createElement('tr');
      emptyRow.className = 'empty-table-row';
      const msg = state.results.length === 0
        ? 'No images analyzed yet. Add images on the left and click "Analyze All".'
        : `No results matching filter "${state.activeFilter}".`;
      emptyRow.innerHTML = `<td colspan="7">${msg}</td>`;
      el.resultsTbody.appendChild(emptyRow);
      return;
    }

    items.forEach(r => {
      const tr = document.createElement('tr');
      tr.id = `row-${r._id}`;
      if (r._id === state.selectedId) {
        tr.classList.add('selected');
      }

      tr.onclick = () => selectItem(r._id);

      // 1. Thumb
      const tdThumb = document.createElement('td');
      const img = document.createElement('img');
      img.className = 'table-thumb';
      img.src = r._thumbUrl || '';
      img.alt = r.filename || '';
      tdThumb.appendChild(img);

      // 2. Filename (truncated + tooltip)
      const tdName = document.createElement('td');
      tdName.className = 'td-filename';
      tdName.textContent = r.filename || 'unknown.jpg';
      tdName.title = r.filename || '';

      // 3. Image Type
      const tdType = document.createElement('td');
      tdType.textContent = r.image_type || 'photo-like';

      // 4. Verdict Badge
      const tdVerdict = document.createElement('td');
      const badge = document.createElement('span');
      const normV = normalizeVerdict(r.verdict);
      badge.className = `verdict-badge ${getVerdictBadgeClass(normV)}`;
      badge.textContent = normV;
      tdVerdict.appendChild(badge);

      // 5. Probability / Range
      const tdProb = document.createElement('td');
      tdProb.textContent = r.probability_display || (r.ai_percentage !== null && r.ai_percentage !== undefined ? `${r.ai_percentage}%` : 'N/A');

      // 6. Confidence
      const tdConf = document.createElement('td');
      const confSpan = document.createElement('span');
      confSpan.className = 'conf-pill';
      confSpan.textContent = r.confidence || 'medium';
      tdConf.appendChild(confSpan);

      // 7. Time
      const tdTime = document.createElement('td');
      tdTime.style.fontFamily = 'var(--font-mono)';
      if (r.cached) {
        tdTime.textContent = '<2ms ⚡';
      } else {
        tdTime.textContent = r.time_taken_seconds !== undefined ? `${r.time_taken_seconds}s` : '--';
      }

      tr.appendChild(tdThumb);
      tr.appendChild(tdName);
      tr.appendChild(tdType);
      tr.appendChild(tdVerdict);
      tr.appendChild(tdProb);
      tr.appendChild(tdConf);
      tr.appendChild(tdTime);

      el.resultsTbody.appendChild(tr);
    });
  }

  function handleSort(col) {
    if (state.sortColumn === col) {
      state.sortAsc = !state.sortAsc;
    } else {
      state.sortColumn = col;
      state.sortAsc = true;
    }

    // Update sort arrows
    document.querySelectorAll('th.sortable').forEach(th => {
      const arrow = th.querySelector('.sort-arrow');
      if (th.getAttribute('data-sort') === col) {
        arrow.textContent = state.sortAsc ? '▲' : '▼';
        arrow.style.opacity = '1';
      } else {
        arrow.textContent = '⇅';
        arrow.style.opacity = '0.5';
      }
    });

    renderResultsTable();
  }

  // Keyboard navigation between rows (↑ and ↓)
  window.addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    if (e.key !== 'ArrowUp' && e.key !== 'ArrowDown') return;

    const items = getFilteredResults();
    if (items.length === 0) return;

    e.preventDefault();
    const currentIdx = items.findIndex(r => r._id === state.selectedId);
    let nextIdx = 0;

    if (e.key === 'ArrowDown') {
      nextIdx = currentIdx < items.length - 1 ? currentIdx + 1 : 0;
    } else if (e.key === 'ArrowUp') {
      nextIdx = currentIdx > 0 ? currentIdx - 1 : items.length - 1;
    }

    selectItem(items[nextIdx]._id);
  });

  // ===========================================================================
  // Detail View Controller
  // ===========================================================================

  function selectItem(id) {
    state.selectedId = id;

    // Highlight row in table
    document.querySelectorAll('.results-table tbody tr').forEach(tr => {
      tr.classList.toggle('selected', tr.id === `row-${id}`);
    });

    // Highlight row in left queue
    document.querySelectorAll('.queue-row').forEach(row => {
      row.classList.toggle('active', row.id === `qrow-${id}`);
    });

    renderDetailView();
  }

  function renderDetailView() {
    if (!state.selectedId) {
      el.detailEmptyState.style.display = 'flex';
      el.detailActiveContainer.style.display = 'none';
      return;
    }

    const item = state.queue.find(i => i.id === state.selectedId);
    const r = state.results.find(res => res._id === state.selectedId) || (item ? item.result : null);

    if (!r) {
      el.detailEmptyState.style.display = 'flex';
      el.detailActiveContainer.style.display = 'none';
      return;
    }

    el.detailEmptyState.style.display = 'none';
    el.detailActiveContainer.style.display = 'flex';

    // 1. File Metadata Header
    el.detailFilename.textContent = r.filename || item?.name || 'image.jpg';
    const dimText = r.cropped_dimensions ? `${r.image_dimensions} (cropped: ${r.cropped_dimensions})` : (r.image_dimensions || 'N/A');
    el.detailDimensions.textContent = dimText;
    el.detailSize.textContent = item ? formatBytes(item.size) : 'N/A';
    el.detailImageType.textContent = r.image_type || 'photo-like';

    if (el.detailCondition) {
      el.detailCondition.textContent = r.condition_label ? `Condition: ${r.condition_label}` : 'Condition: Standard';
    }

    if (r.auto_cropped) {
      if (el.detailAutocrop) {
        el.detailAutocrop.style.display = 'inline-block';
        el.detailAutocrop.textContent = `Auto-cropped (${(r.detected_container || 'UI').replace('_', ' ')})`;
      }
    } else {
      if (el.detailAutocrop) el.detailAutocrop.style.display = 'none';
    }

    if (r.watermark && r.watermark.watermark_detected) {
      if (el.detailWatermark) {
        el.detailWatermark.style.display = 'inline-block';
        el.detailWatermark.textContent = `Watermark: ${(r.watermark.watermark_type || 'AI').replace('_', ' ')}`;
      }
    } else {
      if (el.detailWatermark) el.detailWatermark.style.display = 'none';
    }

    if (r.cached) {
      const origTime = r.original_time_taken_seconds !== undefined ? r.original_time_taken_seconds : (r.time_taken_seconds || 0.45);
      el.detailDuration.textContent = `analyzed in ${origTime}s (cached)`;
      el.detailCached.style.display = 'inline-block';
    } else {
      el.detailDuration.textContent = r.time_taken_seconds !== undefined ? `${r.time_taken_seconds}s` : '--';
      el.detailCached.style.display = 'none';
    }

    // 2. Big Verdict Banner (Headline NEVER says range alone)
    const normV = normalizeVerdict(r.verdict);
    el.verdictBadgeMain.className = `verdict-badge ${getVerdictBadgeClass(normV)}`;
    el.verdictBadgeMain.textContent = normV;
    el.bigVerdictBox.className = `verdict-banner ${getVerdictBannerBorderClass(normV)}`;
    el.confidenceBadgeMain.textContent = `${r.confidence || 'medium'} confidence`;

    if (normV === 'Inconclusive') {
      el.verdictScoreDisplay.textContent = 'Inconclusive';
    } else if (normV === 'AI evidence found (strong)') {
      el.verdictScoreDisplay.textContent = `AI Evidence Found (${(r.probability * 100).toFixed(0)}%)`;
    } else if (normV === 'No AI evidence found (not proof of real)') {
      el.verdictScoreDisplay.textContent = `No AI Evidence Found (${(r.probability * 100).toFixed(0)}%)`;
    } else {
      el.verdictScoreDisplay.textContent = `${normV} (${(r.probability * 100).toFixed(0)}%)`;
    }
    el.verdictSummaryLine.textContent = r.summary_reason || r.summary || 'Multi-signal analysis complete.';

    // 2b. Compact Notice Chips Row
    if (el.verdictNoticesRow) {
      const typeProb = r.photo_probability !== undefined && r.photo_probability !== null
        ? ` (${(r.photo_probability * 100).toFixed(0)}% photo)`
        : '';
      el.noticeChipType.textContent = (r.image_type || 'photo-like') + typeProb;

      if (el.noticeChipCondition) {
        el.noticeChipCondition.textContent = r.condition_label || 'Standard';
      }

      if (r.auto_cropped) {
        if (el.noticeChipAutocrop) {
          el.noticeChipAutocrop.style.display = 'inline-flex';
          el.noticeChipAutocrop.textContent = `Auto-cropped (${(r.detected_container || 'UI').replace('_', ' ')})`;
        }
      } else {
        if (el.noticeChipAutocrop) el.noticeChipAutocrop.style.display = 'none';
      }

      if (r.watermark && r.watermark.watermark_detected) {
        if (el.noticeChipWatermark) {
          el.noticeChipWatermark.style.display = 'inline-flex';
          el.noticeChipWatermark.textContent = `Watermark: ${(r.watermark.watermark_type || 'AI').replace('_', ' ')}`;
        }
      } else {
        if (el.noticeChipWatermark) el.noticeChipWatermark.style.display = 'none';
      }

      const compNotice = r.compression_notice || (r.image_type_res && r.image_type_res.compression_notice) || (r.signals && r.signals.image_type && r.signals.image_type.compression_notice);
      if (r.is_heavily_recompressed || compNotice || r.condition === 'whatsapp' || r.condition === 'jpeg_compressed') {
        el.noticeChipCompression.style.display = 'inline-flex';
        el.noticeChipCompression.textContent = r.condition === 'whatsapp' ? 'WhatsApp Recompression' : 'Lossy Recompression';
      } else {
        el.noticeChipCompression.style.display = 'none';
      }

      const spreadVal = r.spread !== undefined ? (r.spread * 100).toFixed(1) : '0.0';
      const minScore = r.min_model_score !== undefined ? (r.min_model_score * 100).toFixed(0) : '0';
      const maxScore = r.max_model_score !== undefined ? (r.max_model_score * 100).toFixed(0) : '100';
      el.noticeChipRange.textContent = `Spread: ${spreadVal}% (${minScore}%–${maxScore}%)`;
    }

    // 2c. Unreliable Notice (for non-photos)
    if (r.unreliable_notice || (r.is_photo_like === false)) {
      el.unreliableNoticeBox.style.display = 'flex';
      el.unreliableNoticeText.textContent = r.unreliable_notice || 'This looks like a screenshot/document/graphic. Photo-trained AI detectors are unreliable for this type.';
    } else {
      el.unreliableNoticeBox.style.display = 'none';
    }

    // 2d. Compression Notice
    const compNotice = r.compression_notice || (r.image_type_res && r.image_type_res.compression_notice) || (r.signals && r.signals.image_type && r.signals.image_type.compression_notice);
    if (r.is_heavily_recompressed || compNotice || r.condition === 'whatsapp') {
      el.compressionNoticeBox.style.display = 'flex';
      el.compressionNoticeText.textContent = compNotice || (r.condition === 'whatsapp'
        ? 'WhatsApp recompression detected. High-frequency camera grain has been quantized. Precision guardrails prevent claiming definitive camera origin.'
        : 'Lossy recompression detected. Compression artifacts obscure fine camera noise.');
    } else {
      el.compressionNoticeBox.style.display = 'none';
    }

    // 3. "Why this verdict?" Box
    el.whyVerdictContent.textContent = r.why_this_verdict || r.summary_reason || r.summary || 'Multi-signal detector evaluation complete.';

    // 4. Three Evidence Lists
    renderEvidenceLists(r);

    // 5. "What to do next" Card
    renderWhatToDoNext(r);

    // 6. Model Spread & Individual Progress Bars
    const spreadVal = r.spread !== undefined ? (r.spread * 100).toFixed(1) : '0.0';
    el.spreadValueBadge.textContent = `Spread: ${spreadVal}%`;

    const minScore = r.min_model_score !== undefined ? (r.min_model_score * 100).toFixed(0) : '0';
    const maxScore = r.max_model_score !== undefined ? (r.max_model_score * 100).toFixed(0) : '100';
    el.spreadLabelMinmax.textContent = `Range: ${minScore}% – ${maxScore}%`;

    const leftPct = Math.max(0, Math.min(100, parseFloat(minScore)));
    const widthPct = Math.max(3, Math.min(100 - leftPct, parseFloat(maxScore) - leftPct));
    el.spreadRangeBar.style.left = `${leftPct}%`;
    el.spreadRangeBar.style.width = `${widthPct}%`;

    renderModelBars(r);

    // 7. Forensic Visuals (ELA Heatmap)
    renderForensicViews(r);

    // 8. Signal Accordion Rows
    renderSignalAccordion(r);

    // 9. Gemini Card Render
    renderGeminiCard(r);
  }

  function renderEvidenceLists(r) {
    // 1. Evidence pointing to AI
    const aiList = r.evidence_pointing_to_ai || [];
    if (el.badgeAiCount) el.badgeAiCount.textContent = aiList.length;
    if (el.evidenceAiList) {
      el.evidenceAiList.innerHTML = '';
      if (aiList.length === 0) {
        el.evidenceAiList.innerHTML = '<li class="evidence-empty-item">No strong AI indicators observed.</li>';
      } else {
        aiList.forEach(text => {
          const li = document.createElement('li');
          li.className = 'evidence-item ai';
          li.textContent = text;
          el.evidenceAiList.appendChild(li);
        });
      }
    }

    // 2. Evidence pointing to Real
    const realList = r.evidence_pointing_to_real || [];
    if (el.badgeRealCount) el.badgeRealCount.textContent = realList.length;
    if (el.evidenceRealList) {
      el.evidenceRealList.innerHTML = '';
      if (realList.length === 0) {
        el.evidenceRealList.innerHTML = '<li class="evidence-empty-item">No camera provenance confirmed.</li>';
      } else {
        realList.forEach(text => {
          const li = document.createElement('li');
          li.className = 'evidence-item real';
          li.textContent = text;
          el.evidenceRealList.appendChild(li);
        });
      }
    }

    // 3. Missing or Unreliable Evidence
    const unrepList = r.missing_or_unreliable_evidence || [];
    if (el.badgeUnreliableCount) el.badgeUnreliableCount.textContent = unrepList.length;
    if (el.evidenceUnreliableList) {
      el.evidenceUnreliableList.innerHTML = '';
      if (unrepList.length === 0) {
        el.evidenceUnreliableList.innerHTML = '<li class="evidence-empty-item">Standard signal clarity.</li>';
      } else {
        unrepList.forEach(text => {
          const li = document.createElement('li');
          li.className = 'evidence-item warn';
          li.textContent = text;
          el.evidenceUnreliableList.appendChild(li);
        });
      }
    }
  }

  function renderWhatToDoNext(r) {
    if (!el.whatNextChecklist) return;
    el.whatNextChecklist.innerHTML = '';
    const actions = r.what_to_do_next || [];
    if (actions.length === 0) {
      el.whatNextChecklist.innerHTML = '<div class="what-next-item"><span class="step-num">1</span><div><strong>Standard Verification Complete</strong><p>Review the multi-signal findings above.</p></div></div>';
      return;
    }

    actions.forEach((act, idx) => {
      const row = document.createElement('div');
      row.className = 'what-next-item';

      const num = document.createElement('span');
      num.className = 'step-num';
      num.textContent = (idx + 1);

      const content = document.createElement('div');
      content.className = 'step-content';

      const title = document.createElement('strong');
      title.className = 'step-label';
      title.textContent = act.label || act.title || `Step ${idx + 1}`;

      const desc = document.createElement('p');
      desc.className = 'step-desc';
      desc.textContent = act.description || '';

      content.appendChild(title);
      content.appendChild(desc);

      if (act.links && act.links.length > 0) {
        const linksDiv = document.createElement('div');
        linksDiv.className = 'step-links';
        act.links.forEach(l => {
          const a = document.createElement('a');
          a.className = 'btn btn-secondary btn-xs inline-link-btn';
          a.href = l.url;
          a.target = '_blank';
          a.rel = 'noopener noreferrer';
          a.textContent = l.title;
          linksDiv.appendChild(a);
        });
        content.appendChild(linksDiv);
      }

      row.appendChild(num);
      row.appendChild(content);
      el.whatNextChecklist.appendChild(row);
    });
  }

  function renderModelBars(r) {
    el.modelBarsList.innerHTML = '';

    // Robust extraction across multiple potential keys
    let modelsMap = r.model_scores;
    if (!modelsMap && r.signals && r.signals.ml_ensemble && r.signals.ml_ensemble.per_model_scores) {
      modelsMap = r.signals.ml_ensemble.per_model_scores;
    }
    if (!modelsMap && r.signals && r.signals.ml && r.signals.ml.models) {
      modelsMap = {};
      for (const k in r.signals.ml.models) {
        modelsMap[k] = r.signals.ml.models[k].score;
      }
    }

    const weightsMap = r.per_model_weights || (r.signals && r.signals.ml_ensemble && r.signals.ml_ensemble.per_model_weights) || {};
    const modelMetaMap = r.model_metadata || (r.signals && r.signals.ml_ensemble && r.signals.ml_ensemble.model_metadata) || {};

    if (!modelsMap || Object.keys(modelsMap).length === 0) {
      el.modelBarsList.innerHTML = '<div style="font-size: 0.74rem; color: var(--muted);">No individual model scores recorded.</div>';
      return;
    }

    const friendlyNames = {
      'dima806/ai_vs_real_image_detection': 'ViT-B/16 (CIFAKE)',
      'Organika/sdxl-detector': 'SDXL Specialist',
      'Ateeqq/ai-vs-human-image-detector': 'SigLIP Generalist',
      'generalizing_dinov2_probe': 'Multi-Backbone Probe (DINOv2+CLIP)',
      'gemini_semantic': 'Gemini Forensic Reasoning',
    };

    Object.keys(modelsMap).forEach(mKey => {
      const rawScore = modelsMap[mKey];
      const scoreNum = typeof rawScore === 'number' ? rawScore : (rawScore && typeof rawScore.score === 'number' ? rawScore.score : null);
      const pct = scoreNum !== null ? (scoreNum * 100).toFixed(1) : null;
      const weightNum = weightsMap[mKey];
      const weightStr = (weightNum !== undefined && weightNum !== null) ? ` (weight: ${(weightNum * 100).toFixed(0)}%)` : '';

      const modelMeta = modelMetaMap[mKey] || {};
      const displayName = modelMeta.name || friendlyNames[mKey] || mKey.split('/').pop();
      const reliabilityBadge = modelMeta.reliability_badge || '';
      const conditionStrengths = modelMeta.condition_strengths || '';

      const row = document.createElement('div');
      row.className = 'model-bar-row';

      const labelLine = document.createElement('div');
      labelLine.className = 'model-bar-label-line';

      const nameSpan = document.createElement('span');
      nameSpan.className = 'model-name-text';
      nameSpan.innerHTML = `<strong>${displayName}</strong><span class="weight-label">${weightStr}</span>`;

      const valSpan = document.createElement('span');
      valSpan.className = 'model-val-text';
      valSpan.textContent = pct !== null ? `${pct}%` : 'Unavailable';

      labelLine.appendChild(nameSpan);
      labelLine.appendChild(valSpan);

      const track = document.createElement('div');
      track.className = 'model-track';

      const fill = document.createElement('div');
      fill.className = 'model-fill';
      fill.style.width = pct !== null ? `${pct}%` : '0%';
      if (pct !== null) {
        const p = parseFloat(pct);
        fill.style.backgroundColor = p >= 65 ? 'var(--error)' : (p >= 40 ? 'var(--warn)' : 'var(--success)');
      }

      track.appendChild(fill);
      row.appendChild(labelLine);
      row.appendChild(track);

      if (reliabilityBadge || conditionStrengths) {
        const metaLine = document.createElement('div');
        metaLine.className = 'model-meta-line';
        if (reliabilityBadge) {
          const bSpan = document.createElement('span');
          bSpan.className = 'model-badge-pill';
          bSpan.textContent = reliabilityBadge;
          metaLine.appendChild(bSpan);
        }
        if (conditionStrengths) {
          const sSpan = document.createElement('span');
          sSpan.className = 'model-condition-text';
          sSpan.textContent = conditionStrengths;
          metaLine.appendChild(sSpan);
        }
        row.appendChild(metaLine);
      }

      el.modelBarsList.appendChild(row);
    });
  }

  function renderGeminiCard(r) {
    if (!el.geminiResultCard) return;
    const g = r.gemini;
    if (!g || ( (!g.findings || g.findings.length === 0) && (!g.counter_evidence || g.counter_evidence.length === 0) )) {
      el.geminiResultCard.style.display = 'none';
      return;
    }

    el.geminiResultCard.style.display = 'block';
    if (el.geminiCardBadge) {
      el.geminiCardBadge.textContent = g.score !== null && g.score !== undefined
        ? `AI Likelihood: ${(g.score * 100).toFixed(0)}% (${g.confidence || 'medium'} confidence)`
        : 'Forensics Completed';
    }

    if (el.geminiFindingsList) {
      el.geminiFindingsList.innerHTML = '';
      if (g.findings && g.findings.length > 0) {
        if (el.geminiFindingsSection) el.geminiFindingsSection.style.display = 'block';
        g.findings.forEach(f => {
          const card = document.createElement('div');
          const sev = (f.severity || 'medium').toLowerCase();
          card.className = `gemini-finding-card severity-${sev}`;

          const header = document.createElement('div');
          header.className = 'gemini-finding-header';

          const catBadge = document.createElement('span');
          catBadge.className = 'gemini-category-badge';
          catBadge.textContent = f.category || 'Visual';

          const sevBadge = document.createElement('span');
          sevBadge.className = `gemini-severity-badge ${sev}`;
          sevBadge.textContent = f.severity || 'med';

          header.appendChild(catBadge);
          header.appendChild(sevBadge);

          if (f.region_description) {
            const regSpan = document.createElement('span');
            regSpan.className = 'gemini-region-text';
            regSpan.textContent = `📍 ${f.region_description}`;
            header.appendChild(regSpan);
          }

          const desc = document.createElement('div');
          desc.className = 'gemini-finding-desc';
          desc.textContent = f.description || '';

          card.appendChild(header);
          card.appendChild(desc);
          el.geminiFindingsList.appendChild(card);
        });
      } else {
        if (el.geminiFindingsSection) el.geminiFindingsSection.style.display = 'none';
      }
    }

    if (el.geminiCounterList) {
      el.geminiCounterList.innerHTML = '';
      if (g.counter_evidence && g.counter_evidence.length > 0) {
        if (el.geminiCounterSection) el.geminiCounterSection.style.display = 'block';
        g.counter_evidence.forEach(itemText => {
          const li = document.createElement('li');
          li.className = 'gemini-counter-item';
          li.textContent = itemText;
          el.geminiCounterList.appendChild(li);
        });
      } else {
        if (el.geminiCounterSection) el.geminiCounterSection.style.display = 'none';
      }
    }
  }

  function renderForensicViews(r) {
    const thumb = r._thumbUrl || '';
    const elaBase64 = (r.signals && r.signals.ela && r.signals.ela.heatmap_base64) || null;
    const elaSrc = elaBase64 ? (elaBase64.startsWith('data:') ? elaBase64 : `data:image/jpeg;base64,${elaBase64}`) : thumb;

    el.elaImgOriginal.src = thumb;
    el.elaImgHeatmap.src = elaSrc;

    updateViewMode(state.viewMode, thumb, elaSrc);

    // ELA note
    if (r.signals && r.signals.ela && r.signals.ela.interpretation) {
      el.elaNoteText.textContent = r.signals.ela.interpretation;
    } else {
      el.elaNoteText.textContent = 'ELA highlights recompression error variance across 8x8 DCT grid macroblocks.';
    }
  }

  function updateViewMode(mode, origSrc, elaSrc) {
    state.viewMode = mode;
    [el.viewModeOrig, el.viewModeEla, el.viewModeDual].forEach(b => b.classList.remove('active'));

    const thumb = origSrc || el.elaImgOriginal.src;
    const heatmap = elaSrc || el.elaImgHeatmap.src;

    if (mode === 'orig') {
      el.viewModeOrig.classList.add('active');
      el.elaFrameSingle.style.display = 'flex';
      el.elaFrameDual.style.display = 'none';
      el.elaImgSingle.src = thumb;
      el.elaCaptionSingle.textContent = 'Original Input';
    } else if (mode === 'ela') {
      el.viewModeEla.classList.add('active');
      el.elaFrameSingle.style.display = 'flex';
      el.elaFrameDual.style.display = 'none';
      el.elaImgSingle.src = heatmap;
      el.elaCaptionSingle.textContent = 'ELA Heatmap';
    } else if (mode === 'dual') {
      el.viewModeDual.classList.add('active');
      el.elaFrameSingle.style.display = 'none';
      el.elaFrameDual.style.display = 'grid';
    }
  }

  function renderSignalAccordion(r) {
    const sig = r.signals || {};

    // 1. Metadata
    const meta = sig.metadata || {};
    const camMake = meta.camera_make || meta.make;
    const camModel = meta.camera_model || meta.model;
    const camStr = (camMake || camModel) ? `${camMake || ''} ${camModel || ''}`.trim() : 'None';
    const soft = meta.software ? `Software: ${meta.software}. ` : '';
    const aiTrace = meta.ai_tool_detected ? `🚨 AI generator tag found: ${meta.ai_tool_detected}. ` : '';

    el.badgeMetaStatus.textContent = camStr !== 'None' ? 'Camera EXIF' : (meta.ai_tool_detected ? 'AI Traces' : 'None');
    el.badgeMetaStatus.className = `acc-status-pill ${camStr !== 'None' ? 'found' : (meta.ai_tool_detected ? 'flagged' : 'neutral')}`;
    el.metaDetailsBody.innerHTML = `<strong>Camera Hardware:</strong> ${camStr}<br>${soft}${aiTrace}${meta.evidence ? meta.evidence.join('<br>') : ''}`;

    // 2. C2PA
    const c2pa = sig.c2pa || {};
    const c2paFound = c2pa.manifest_found;
    el.badgeC2paStatus.textContent = c2paFound ? 'Manifest Found' : 'Absent';
    el.badgeC2paStatus.className = `acc-status-pill ${c2paFound ? 'found' : 'neutral'}`;
    el.c2paDetailsBody.innerHTML = c2paFound
      ? `<strong>C2PA Manifest Verified:</strong> ${c2pa.claim_generator || 'Standard claim'}. Cryptographic chain intact.`
      : `No C2PA Content Credentials manifest present in image header. (Standard for raw downloads and social media).`;

    // 3. ELA
    const ela = sig.ela || {};
    if (ela.applicable === false || ela.score === null || ela.score === undefined) {
      el.badgeElaStatus.textContent = 'Not applicable';
      el.badgeElaStatus.className = 'acc-status-pill neutral';
      el.elaDetailsBody.innerHTML = `<strong>Status:</strong> ${ela.status_text || 'Not applicable'}.<br>${ela.interpretation || 'ELA is only calibrated for photographic JPEG compression.'}`;
    } else {
      const elaScore = (ela.score * 100).toFixed(0);
      el.badgeElaStatus.textContent = `Score: ${elaScore}%`;
      el.badgeElaStatus.className = `acc-status-pill ${ela.score >= 0.65 ? 'flagged' : (ela.score <= 0.35 ? 'found' : 'neutral')}`;
      el.elaDetailsBody.innerHTML = `<strong>Error Level Variance:</strong> ${ela.interpretation || 'Evaluated recompression differences.'}`;
    }

    // 4. FFT
    const fft = sig.frequency || {};
    if (fft.applicable === false || fft.score === null || fft.score === undefined) {
      el.badgeFftStatus.textContent = 'Not applicable';
      el.badgeFftStatus.className = 'acc-status-pill neutral';
      el.fftDetailsBody.innerHTML = `<strong>Status:</strong> ${fft.status_text || 'Not applicable'}.<br>${fft.note || 'Spectral frequency analysis is not calibrated for non-photographic graphics.'}`;
    } else {
      const fftScore = (fft.score * 100).toFixed(0);
      el.badgeFftStatus.textContent = `Score: ${fftScore}%`;
      el.badgeFftStatus.className = `acc-status-pill ${fft.score >= 0.65 ? 'flagged' : (fft.score <= 0.35 ? 'found' : 'neutral')}`;
      el.fftDetailsBody.innerHTML = `<strong>Spectral Decay Slope:</strong> ${fft.note || 'Examined radial frequency decay and periodic lattice peaks.'}`;
    }

    // 5. ML
    const ml = sig.ml || sig.ml_ensemble || {};
    const mlScore = ml.score !== undefined ? ml.score : (r.model_scores ? Object.values(r.model_scores).reduce((a, b) => a + b, 0) / Object.values(r.model_scores).length : null);
    el.badgeMlStatus.textContent = mlScore !== null ? `Ensemble: ${(mlScore * 100).toFixed(0)}%` : 'Active';
    el.mlDetailsBody.innerHTML = `<strong>Vision Ensemble:</strong> Evaluated across SigLIP Generalist, Multi-Backbone Probe (DINOv2+CLIP), SDXL Specialist, and ViT-B/16 with condition-aware dynamic weighting.`;

    // 6. Gemini
    const gem = sig.gemini || r.gemini;
    if (gem && gem.score !== null && gem.score !== undefined) {
      el.badgeGeminiStatus.textContent = `Score: ${(gem.score * 100).toFixed(0)}%`;
      el.badgeGeminiStatus.className = `acc-status-pill ${gem.score >= 0.65 ? 'flagged' : (gem.score <= 0.35 ? 'found' : 'neutral')}`;
      el.geminiDetailsBody.innerHTML = `<strong>Gemini Semantic Reasoning:</strong> Model ${gem.model_used || 'Gemini'} evaluated anatomy, physics, text, and texture. (${(gem.findings || []).length} artifact findings recorded).`;
    } else {
      el.badgeGeminiStatus.textContent = 'Available';
      el.badgeGeminiStatus.className = 'acc-status-pill neutral';
      el.geminiDetailsBody.textContent = 'Click "Deep Check (Gemini)" at top right to inspect semantic anomalies via Gemini multimodal reasoning.';
    }
  }

  // ===========================================================================
  // Gemini Deep Check (On-Demand)
  // ===========================================================================

  async function triggerGeminiDeepCheck() {
    if (!state.selectedId) return;
    const item = state.queue.find(i => i.id === state.selectedId);
    if (!item) return;

    el.deepCheckGeminiBtn.disabled = true;
    el.deepCheckGeminiBtn.textContent = '✨ Analyzing with Gemini...';

    try {
      const formData = new FormData();
      formData.append('image', item.file, item.name);

      const res = await fetch('/analyze_gemini', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) throw new Error(`HTTP error ${res.status}`);
      const data = await res.json();

      if (!data.api_key_configured) {
        alert('Gemini API key is not configured in .env. RealLens operates completely locally.');
        return;
      }

      if (data.status === 'success' && data.gemini) {
        const itemResult = state.results.find(res => res._id === state.selectedId) || item.result;
        if (itemResult) {
          itemResult.gemini = data.gemini;
          renderGeminiCard(itemResult);
          renderSignalAccordion(itemResult);
        }
      }
    } catch (err) {
      alert(`Gemini check failed: ${err.message}`);
    } finally {
      el.deepCheckGeminiBtn.disabled = false;
      el.deepCheckGeminiBtn.textContent = '✨ Deep Check (Gemini)';
    }
  }

  // ===========================================================================
  // Re-run Analysis (Bypassing Cache)
  // ===========================================================================

  async function rerunSelected() {
    if (!state.selectedId) return;
    const item = state.queue.find(i => i.id === state.selectedId);
    if (!item) return;

    if (el.rerunBtn) {
      el.rerunBtn.disabled = true;
      el.rerunBtn.textContent = '🔄 Re-running...';
    }

    try {
      const formData = new FormData();
      formData.append('image', item.file, item.name);
      formData.append('ignore_cache', '1');

      const res = await fetch('/analyze?ignore_cache=1', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      if (data.status === 'success' && data.result) {
        const r = data.result;
        r._id = item.id;
        r._thumbUrl = item.thumbUrl;
        item.status = 'done';
        item.verdict = r.verdict;
        item.result = r;

        // Update in state.results
        const idx = state.results.findIndex(res => res._id === item.id);
        if (idx !== -1) {
          state.results[idx] = r;
        } else {
          state.results.push(r);
        }

        updateQueueRow(item.id);
        renderResultsTable();
        updateSummaryStrip();
        renderDetailView();
      } else {
        throw new Error(data.message || 'Re-run failed');
      }
    } catch (err) {
      alert(`Re-run failed: ${err.message}`);
    } finally {
      if (el.rerunBtn) {
        el.rerunBtn.disabled = false;
        el.rerunBtn.textContent = '🔄 Re-run';
      }
    }
  }

  // ===========================================================================
  // Copy Report & Export (CSV, JSON)
  // ===========================================================================

  function copySelectedReport() {
    if (!state.selectedId) return;
    const item = state.queue.find(i => i.id === state.selectedId);
    const r = state.results.find(res => res._id === state.selectedId) || (item ? item.result : null);
    if (!r) return;

    const normV = normalizeVerdict(r.verdict);
    const prob = r.probability_display || `${r.ai_percentage}%`;
    const md = [
      `# RealLens Forensic Inspection Report`,
      `**File Name:** ${r.filename || item?.name}`,
      `**Dimensions:** ${r.image_dimensions || 'N/A'} | **Image Type:** ${r.image_type || 'photo-like'}`,
      `**Analysis Verdict:** ${normV} (${prob}, ${r.confidence || 'medium'} confidence)`,
      `**Model Spread:** ${r.spread !== undefined ? (r.spread * 100).toFixed(1) + '%' : 'N/A'}`,
      `\n## Why This Verdict?`,
      `${r.why_this_verdict || r.summary_reason}`,
      `\n## Multi-Signal Summary`,
      `- **Camera Metadata:** ${r.signals?.metadata?.camera_model || 'None detected'}`,
      `- **C2PA Credentials:** ${r.signals?.c2pa?.manifest_found ? 'Verified' : 'Absent'}`,
      `- **Error Level Analysis (ELA):** Score ${r.signals?.ela?.score !== undefined ? (r.signals.ela.score * 100).toFixed(0) + '%' : 'N/A'}`,
      `- **FFT Spectral Frequency:** Score ${r.signals?.frequency?.score !== undefined ? (r.signals.frequency.score * 100).toFixed(0) + '%' : 'N/A'}`,
      `\n*Generated by RealLens Forensic Engine • Privacy-Preserving Local Inference*`,
    ].join('\n');

    navigator.clipboard.writeText(md).then(() => {
      const origText = el.copyReportBtn.textContent;
      el.copyReportBtn.textContent = '✓ Copied!';
      setTimeout(() => {
        el.copyReportBtn.textContent = origText;
      }, 2000);
    }).catch(err => {
      alert('Failed to copy to clipboard.');
    });
  }

  function downloadCsv() {
    if (state.results.length === 0) return;
    const headers = ['Filename', 'Image Type', 'Verdict', 'Score / Range', 'Spread', 'Confidence', 'Time (s)'];
    const rows = state.results.map(r => [
      `"${(r.filename || '').replace(/"/g, '""')}"`,
      `"${r.image_type || 'photo-like'}"`,
      `"${normalizeVerdict(r.verdict)}"`,
      `"${r.probability_display || r.ai_percentage || ''}"`,
      `"${r.spread !== undefined ? (r.spread * 100).toFixed(1) + '%' : ''}"`,
      `"${r.confidence || ''}"`,
      `"${r.time_taken_seconds || ''}"`,
    ]);

    const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows.map(e => e.join(','))].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `reallens_results_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }

  function downloadJson() {
    if (state.results.length === 0) return;
    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(state.results, null, 2));
    const link = document.createElement('a');
    link.setAttribute('href', dataStr);
    link.setAttribute('download', `reallens_results_${Date.now()}.json`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }

  // ===========================================================================
  // Modals & Helpers
  // ===========================================================================

  window.toggleAccordion = function (rowId) {
    const row = document.getElementById(rowId);
    if (row) {
      row.classList.toggle('open');
    }
  };

  window.closeZoomModal = function () {
    el.zoomModal.classList.remove('open');
  };

  function openZoomModal() {
    if (!state.selectedId) return;
    const src = state.viewMode === 'orig' ? el.elaImgOriginal.src : el.elaImgHeatmap.src;
    el.zoomModalImage.src = src;
    el.zoomModalTitle.textContent = state.viewMode === 'orig' ? 'Original Image (Full Res)' : 'ELA Heatmap (Full Res)';
    el.zoomModal.classList.add('open');
  }

  function getVerdictBadgeClass(normV) {
    switch (normV) {
      case 'AI evidence found (strong)': return 'verdict-strong-ai';
      case 'Leaning AI': return 'verdict-leaning-ai';
      case 'Inconclusive': return 'verdict-inconclusive';
      case 'No AI evidence found (not proof of real)': return 'verdict-no-evidence';
      case 'Likely real': return 'verdict-likely-real';
      case 'Failed': return 'verdict-failed';
      default: return 'verdict-inconclusive';
    }
  }

  function getVerdictDotClass(v) {
    const normV = normalizeVerdict(v);
    switch (normV) {
      case 'AI evidence found (strong)': return 'red';
      case 'Leaning AI': return 'orange';
      case 'Inconclusive': return 'yellow';
      case 'No AI evidence found (not proof of real)': return 'teal';
      case 'Likely real': return 'green';
      case 'Failed': return 'dark-red';
      default: return 'gray';
    }
  }

  function getVerdictBannerBorderClass(normV) {
    switch (normV) {
      case 'AI evidence found (strong)': return 'verdict-border-strong-ai';
      case 'Leaning AI': return 'verdict-border-leaning-ai';
      case 'Inconclusive': return 'verdict-border-inconclusive';
      case 'No AI evidence found (not proof of real)': return 'verdict-border-no-evidence';
      case 'Likely real': return 'verdict-border-likely-real';
      default: return '';
    }
  }

  // ===========================================================================
  // Event Listeners Wiring
  // ===========================================================================

  // Dropzone click & drag
  el.dropzone.addEventListener('click', () => el.fileInput.click());
  el.fileInput.addEventListener('change', e => {
    if (e.target.files.length > 0) {
      addFilesToQueue(e.target.files);
      el.fileInput.value = '';
    }
  });

  ['dragenter', 'dragover'].forEach(name => {
    el.dropzone.addEventListener(name, e => {
      e.preventDefault();
      e.stopPropagation();
      el.dropzone.classList.add('drag-over');
    });
  });

  ['dragleave', 'drop'].forEach(name => {
    el.dropzone.addEventListener(name, e => {
      e.preventDefault();
      e.stopPropagation();
      el.dropzone.classList.remove('drag-over');
    });
  });

  el.dropzone.addEventListener('drop', e => {
    if (e.dataTransfer && e.dataTransfer.files) {
      addFilesToQueue(e.dataTransfer.files);
    }
  });

  // Clipboard Paste (Ctrl+V anywhere on window)
  window.addEventListener('paste', e => {
    if (e.clipboardData && e.clipboardData.files && e.clipboardData.files.length > 0) {
      addFilesToQueue(e.clipboardData.files);
    }
  });

  // Buttons
  el.analyzeAllBtn.addEventListener('click', analyzeAll);
  el.cancelBtn.addEventListener('click', () => { state.cancelRequested = true; });
  el.clearAllBtn.addEventListener('click', clearAll);
  if (el.rerunBtn) el.rerunBtn.addEventListener('click', rerunSelected);
  el.downloadCsvBtn.addEventListener('click', downloadCsv);
  el.downloadJsonBtn.addEventListener('click', downloadJson);
  el.copyReportBtn.addEventListener('click', copySelectedReport);
  el.deepCheckGeminiBtn.addEventListener('click', triggerGeminiDeepCheck);
  el.zoomForensicBtn.addEventListener('click', openZoomModal);
  el.clearFilterBtn.addEventListener('click', () => setFilter('all'));

  // ELA View Mode Toggles
  el.viewModeOrig.addEventListener('click', () => updateViewMode('orig'));
  el.viewModeEla.addEventListener('click', () => updateViewMode('ela'));
  el.viewModeDual.addEventListener('click', () => updateViewMode('dual'));

  // Summary Strip Filter Pills
  el.summaryStrip.querySelectorAll('.summary-pill').forEach(pill => {
    pill.addEventListener('click', () => {
      const f = pill.getAttribute('data-filter');
      setFilter(f);
    });
  });

  // Sortable table headers
  document.querySelectorAll('th.sortable').forEach(th => {
    th.addEventListener('click', () => {
      const col = th.getAttribute('data-sort');
      handleSort(col);
    });
  });

  // Start health check
  checkHealth();

  // Seed demo items if ?demo=1 is in query parameters
  if (window.location.search.includes('demo=1')) {
    const demoItems = [
      {
        id: 'demo_1',
        name: 'portrait_real_camera.jpg',
        size: 2450000,
        status: 'done',
        verdict: 'Likely real',
        thumbUrl: 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64"><rect width="64" height="64" fill="%231E293B"/><text x="32" y="36" fill="%233DDC97" font-size="10" text-anchor="middle">REAL</text></svg>',
        result: {
          _id: 'demo_1',
          _thumbUrl: 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64"><rect width="64" height="64" fill="%231E293B"/><text x="32" y="36" fill="%233DDC97" font-size="10" text-anchor="middle">REAL</text></svg>',
          filename: 'portrait_real_camera.jpg',
          image_dimensions: '1920x1080',
          image_type: 'photo-like',
          is_photo_like: true,
          verdict: 'Likely real',
          confidence: 'high',
          ai_probability: 0.12,
          ai_percentage: 12.0,
          probability_display: '12.0%',
          spread: 0.14,
          min_model_score: 0.05,
          max_model_score: 0.19,
          time_taken_seconds: 0.48,
          cached: false,
          summary_reason: 'Photographic sensor noise characteristics, natural spectral decay, and vision models align with authentic origin.',
          why_this_verdict: 'All active vision models agree on authentic photographic origin. Natural camera sensor noise floor, Fourier spectral slope, and compression characteristics align with real-world capture.',
          signals: {
            metadata: { camera_model: 'Canon EOS 5D Mark IV', evidence: ['Valid EXIF headers present.', 'Exposure 1/250s at f/2.8, ISO 200.'] },
            c2pa: { manifest_found: false },
            ela: { score: 0.14, interpretation: 'Natural, uniform error level across all photographic JPEG blocks.' },
            frequency: { score: 0.18, note: 'Consistent radial 1/f^alpha spectral power decay.' },
            ml: {
              score: 0.11,
              models: {
                'dima806/ai_vs_real_image_detection': { score: 0.05 },
                'Organika/sdxl-detector': { score: 0.12 },
                'Ateeqq/ai-vs-human-image-detector': { score: 0.08 },
                'generalizing_dinov2_probe': { score: 0.19 }
              }
            }
          }
        }
      },
      {
        id: 'demo_2',
        name: 'synthetic_cyberpunk_sdxl.png',
        size: 3200000,
        status: 'done',
        verdict: 'AI evidence found (strong)',
        thumbUrl: 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64"><rect width="64" height="64" fill="%231E293B"/><text x="32" y="36" fill="%23FF6B6B" font-size="10" text-anchor="middle">AI</text></svg>',
        result: {
          _id: 'demo_2',
          _thumbUrl: 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64"><rect width="64" height="64" fill="%231E293B"/><text x="32" y="36" fill="%23FF6B6B" font-size="10" text-anchor="middle">AI</text></svg>',
          filename: 'synthetic_cyberpunk_sdxl.png',
          image_dimensions: '1024x1024',
          image_type: 'photo-like',
          is_photo_like: true,
          verdict: 'AI evidence found (strong)',
          confidence: 'high',
          ai_probability: 0.96,
          ai_percentage: 96.0,
          probability_display: '96.0%',
          spread: 0.08,
          min_model_score: 0.91,
          max_model_score: 0.99,
          time_taken_seconds: 0.52,
          cached: false,
          summary_reason: 'Consistent AI visual characteristics and artifact patterns identified across detectors.',
          why_this_verdict: 'Vision models demonstrate strong consensus on synthetic diffusion textures, supported by spectral frequency patterns consistent with generative upsampling.',
          signals: {
            metadata: { camera_model: null, ai_tool_detected: 'Stable Diffusion XL', evidence: ['Software chunk: NovelAI / WebUI'] },
            c2pa: { manifest_found: false },
            ela: { score: 0.88, interpretation: 'Abnormal variance gradient across flat and detailed regions.' },
            frequency: { score: 0.92, note: 'Spectral peaks at high-frequency periodic lattice coordinates.' },
            ml: {
              score: 0.96,
              models: {
                'dima806/ai_vs_real_image_detection': { score: 0.98 },
                'Organika/sdxl-detector': { score: 0.99 },
                'Ateeqq/ai-vs-human-image-detector': { score: 0.91 },
                'generalizing_dinov2_probe': { score: 0.95 }
              }
            }
          }
        }
      },
      {
        id: 'demo_3',
        name: 'whatsapp_compressed_street.jpg',
        size: 180000,
        status: 'done',
        verdict: 'Inconclusive',
        thumbUrl: 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64"><rect width="64" height="64" fill="%231E293B"/><text x="32" y="36" fill="%23F5B454" font-size="10" text-anchor="middle">INCON</text></svg>',
        result: {
          _id: 'demo_3',
          _thumbUrl: 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64"><rect width="64" height="64" fill="%231E293B"/><text x="32" y="36" fill="%23F5B454" font-size="10" text-anchor="middle">INCON</text></svg>',
          filename: 'whatsapp_compressed_street.jpg',
          image_dimensions: '1280x720',
          image_type: 'photo-like',
          is_photo_like: true,
          is_heavily_recompressed: true,
          verdict: 'Inconclusive',
          confidence: 'low (heavy recompression)',
          ai_probability: 0.50,
          ai_percentage: null,
          probability_display: 'between 35% and 72% across models',
          spread: 0.37,
          min_model_score: 0.35,
          max_model_score: 0.72,
          time_taken_seconds: 0.54,
          cached: false,
          summary_reason: 'Heavy lossy recompression detected (DCT blockiness: 1.28, 1280x720). Compression artifacts obscure subtle sensor noise.',
          why_this_verdict: 'This image shows clear evidence of lossy compression (such as WhatsApp/social messaging export, downscaling, or low JPEG quality). Compression macroblocks destroy fine camera sensor noise and soften generative artifacts. Because model predictions span from 35% to 72%, RealLens refuses to force an unreliable guess and issues an honest Inconclusive verdict.',
          evidence_pointing_to_ai: [],
          evidence_pointing_to_real: ['Natural facial and scene geometry coherence.'],
          missing_or_unreliable_evidence: [
            'Camera metadata stripped by messaging platform.',
            'No C2PA Content Credentials manifest found.',
            'Fine sensor noise destroyed by lossy compression.',
            'High model spread (37%) indicates uncertainty.'
          ],
          what_to_do_next: [
            { label: 'Request Original File', description: 'Ask the sender to share the original uncompressed photo as a Document.', action_type: 'ask_source' },
            { label: 'Reverse Image Search', description: 'Check if this image was published elsewhere earlier.', action_type: 'reverse_search', links: [{ title: 'Google Lens', url: 'https://lens.google.com/' }, { title: 'TinEye', url: 'https://tineye.com/' }] },
            { label: 'Verify Sender Context', description: 'Verify caller or sender identity through a secondary trusted channel before acting.', action_type: 'caution' }
          ],
          signals: {
            metadata: { camera_model: null, evidence: [] },
            c2pa: { manifest_found: false },
            ela: { score: null, applicable: false, status_text: 'Not applicable', interpretation: 'Uniform macroblock variance caused by heavy secondary lossy recompression.' },
            frequency: { score: null, applicable: false, status_text: 'Not applicable', note: 'High frequencies severely truncated by JPEG quantization matrix.' },
            ml: {
              score: 0.52,
              models: {
                'dima806/ai_vs_real_image_detection': { score: 0.35 },
                'Organika/sdxl-detector': { score: 0.72 },
                'Ateeqq/ai-vs-human-image-detector': { score: 0.52 },
                'generalizing_dinov2_probe': { score: 0.50 }
              }
            }
          }
        }
      },
      {
        id: 'demo_4',
        name: 'web_screenshot_dashboard.png',
        size: 420000,
        status: 'done',
        verdict: 'Inconclusive',
        thumbUrl: 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64"><rect width="64" height="64" fill="%231E293B"/><text x="32" y="36" fill="%23A78BFA" font-size="10" text-anchor="middle">UI</text></svg>',
        result: {
          _id: 'demo_4',
          _thumbUrl: 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64"><rect width="64" height="64" fill="%231E293B"/><text x="32" y="36" fill="%23A78BFA" font-size="10" text-anchor="middle">UI</text></svg>',
          filename: 'web_screenshot_dashboard.png',
          image_dimensions: '1920x1080',
          image_type: 'screenshot/UI',
          is_photo_like: false,
          unreliable_notice: 'This looks like a screenshot/document/graphic. Photo-trained AI detectors are unreliable for this type.',
          verdict: 'Inconclusive',
          confidence: 'low',
          ai_probability: 0.50,
          ai_percentage: null,
          probability_display: 'between 2% and 99% across models',
          spread: 0.97,
          min_model_score: 0.02,
          max_model_score: 0.99,
          time_taken_seconds: 0.42,
          cached: false,
          summary_reason: 'Non-photographic image detected. Classified as screenshot/UI.',
          why_this_verdict: 'Identified as a screenshot/UI graphic with sharp font edges and flat fills. Vision models trained on camera photographs produce false alarms on digital graphics. Without cryptographic provenance, verdict is honestly Inconclusive.',
          evidence_pointing_to_ai: [],
          evidence_pointing_to_real: [],
          missing_or_unreliable_evidence: [
            'Non-photographic digital graphic: vision detectors suppressed.',
            'No physical camera sensor noise or optics expected.'
          ],
          what_to_do_next: [
            { label: 'Check Document Authenticity', description: 'Verify issuers digital signature, domain name, or cryptographic seals.', action_type: 'verify_doc' }
          ],
          signals: {
            metadata: { camera_model: null, evidence: [] },
            c2pa: { manifest_found: false },
            ela: { score: null, applicable: false, status_text: 'Not applicable', interpretation: 'ELA is not calibrated for non-photographic graphics.' },
            frequency: { score: null, applicable: false, status_text: 'Not applicable', note: 'Artificial step functions from raster text.' },
            ml: {
              score: 0.65,
              models: {
                'dima806/ai_vs_real_image_detection': { score: 0.02 },
                'Organika/sdxl-detector': { score: 0.99 },
                'Ateeqq/ai-vs-human-image-detector': { score: 0.95 },
                'generalizing_dinov2_probe': { score: 0.40 }
              }
            }
          }
        }
      },
      {
        id: 'demo_5',
        name: 'vacation_coastal_sunset.jpg',
        size: 1900000,
        status: 'done',
        verdict: 'Leaning Real',
        thumbUrl: 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64"><rect width="64" height="64" fill="%231E293B"/><text x="32" y="36" fill="%232DD4BF" font-size="10" text-anchor="middle">LEAN</text></svg>',
        result: {
          _id: 'demo_5',
          _thumbUrl: 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64"><rect width="64" height="64" fill="%231E293B"/><text x="32" y="36" fill="%232DD4BF" font-size="10" text-anchor="middle">LEAN</text></svg>',
          filename: 'vacation_coastal_sunset.jpg',
          image_dimensions: '2048x1536',
          image_type: 'photo-like',
          is_photo_like: true,
          verdict: 'Leaning Real',
          confidence: 'medium',
          ai_probability: 0.38,
          ai_percentage: 38.0,
          probability_display: '38.0%',
          spread: 0.16,
          min_model_score: 0.28,
          max_model_score: 0.44,
          time_taken_seconds: 0.46,
          cached: false,
          summary_reason: 'Primarily natural characteristics, though compression leaves slight ambiguity.',
          why_this_verdict: 'Image exhibits photographic characteristics and natural spectral decay, though compression re-encoding leaves minor residual uncertainty.',
          signals: {
            metadata: { camera_model: 'iPhone 13', evidence: ['Apple iPhone EXIF tags.'] },
            c2pa: { manifest_found: false },
            ela: { score: 0.32, interpretation: 'Expected camera JPEG compression profile.' },
            frequency: { score: 0.36, note: 'Normal optical decay slope.' },
            ml: {
              score: 0.38,
              models: {
                'dima806/ai_vs_real_image_detection': { score: 0.32 },
                'Organika/sdxl-detector': { score: 0.44 },
                'Ateeqq/ai-vs-human-image-detector': { score: 0.28 },
                'generalizing_dinov2_probe': { score: 0.39 }
              }
            }
          }
        }
      }
    ];

    demoItems.forEach(item => {
      state.queue.push(item);
      state.results.push(item.result);
    });

    renderQueue();
    renderResultsTable();
    updateSummaryStrip();
    updateQueueStats();
    selectItem('demo_3'); // Focus on the WhatsApp inconclusive image by default
  }

})();

