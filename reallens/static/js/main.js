// RealLens Frontend Controller
document.addEventListener("DOMContentLoaded", () => {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");
  const previewContainer = document.getElementById("preview-container");
  const previewImage = document.getElementById("preview-image");
  const analyzeBtn = document.getElementById("analyze-btn");
  const loadingBox = document.getElementById("loading-box");
  const errorBox = document.getElementById("error-box");
  const resultsCard = document.getElementById("results-card");
  const deviceStatus = document.getElementById("device-status");

  // Results elements
  const verdictBanner = document.getElementById("verdict-banner");
  const verdictTitle = document.getElementById("verdict-title");
  const verdictSummary = document.getElementById("verdict-summary");
  const verdictScore = document.getElementById("verdict-score");
  const compOriginal = document.getElementById("comp-original");
  const compEla = document.getElementById("comp-ela");
  const signalTbody = document.getElementById("signal-tbody");

  let selectedFile = null;

  // 1. Fetch system health on startup
  fetch("/health")
    .then((res) => res.json())
    .then((data) => {
      if (data && data.status === "healthy") {
        const dev = data.device ? data.device.toUpperCase() : "CPU";
        deviceStatus.textContent = `Engine: ${dev} (${data.models_loaded.length || 3} models ready)`;
      }
    })
    .catch(() => {
      deviceStatus.textContent = "Engine: Local Mode";
    });

  // 2. Drag & Drop Handlers
  dropzone.addEventListener("click", () => fileInput.click());

  ["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove("dragover");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileSelected(files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      handleFileSelected(e.target.files[0]);
    }
  });

  function handleFileSelected(file) {
    selectedFile = file;
    errorBox.style.display = "none";
    resultsCard.style.display = "none";

    const reader = new FileReader();
    reader.onload = (event) => {
      previewImage.src = event.target.result;
      previewContainer.style.display = "block";
    };
    reader.readAsDataURL(file);
  }

  // 3. Analyze Click Handler
  analyzeBtn.addEventListener("click", () => {
    if (!selectedFile) return;

    analyzeBtn.disabled = true;
    loadingBox.style.display = "block";
    errorBox.style.display = "none";
    resultsCard.style.display = "none";

    const formData = new FormData();
    formData.append("image", selectedFile);

    fetch("/analyze", {
      method: "POST",
      body: formData,
    })
      .then(async (response) => {
        const data = await response.json();
        if (!response.ok || data.status === "error") {
          throw new Error(data.message || `Server returned ${response.status}`);
        }
        return data.result;
      })
      .then((result) => {
        renderResults(result);
      })
      .catch((err) => {
        errorBox.textContent = `Analysis error: ${err.message}`;
        errorBox.style.display = "block";
      })
      .finally(() => {
        analyzeBtn.disabled = false;
        loadingBox.style.display = "none";
      });
  });

  // 4. Render Results
  function renderResults(result) {
    resultsCard.style.display = "block";

    // Verdict styling
    verdictTitle.textContent = result.verdict;
    verdictSummary.textContent = result.summary_reason;
    verdictScore.textContent = `${result.ai_percentage}%`;

    // Clear previous verdict classes
    verdictBanner.className = "verdict-banner";
    if (result.verdict === "Likely AI") {
      verdictBanner.classList.add("verdict-likely-ai");
    } else if (result.verdict === "Uncertain leaning AI") {
      verdictBanner.classList.add("verdict-uncertain-ai");
    } else if (result.verdict === "Uncertain leaning real") {
      verdictBanner.classList.add("verdict-uncertain-real");
    } else {
      verdictBanner.classList.add("verdict-likely-real");
    }

    // Time and dimensions badge
    const timeBadge = document.getElementById("time-badge");
    if (timeBadge && result.time_taken_seconds !== undefined) {
      const modeLabel = result.fusion_mode === "learned_stacker" ? "Learned Stacker" : "Hand Weighted";
      timeBadge.textContent = `⚡ Analyzed in ${result.time_taken_seconds}s (${modeLabel}) • ${result.image_dimensions || ''}`;
    }

    // Vision Models Bar List
    const modelBarsContainer = document.getElementById("model-bars-container");
    if (modelBarsContainer) {
      modelBarsContainer.innerHTML = "";
      const perModels = (result.signals.ml_ensemble && result.signals.ml_ensemble.per_model_scores) || {};
      for (const [modelKey, score] of Object.entries(perModels)) {
        const shortName = modelKey.includes("probe")
          ? "DINOv2 Generalizing Probe"
          : modelKey.includes("dima806")
          ? "dima806 (CIFAKE ViT)"
          : modelKey.includes("Organika")
          ? "Organika (SDXL ViT)"
          : modelKey.includes("Ateeqq")
          ? "Ateeqq (SigLIP Generalist)"
          : modelKey.split("/").pop();

        const pct = score !== null ? (score * 100).toFixed(1) : 0;
        const isAi = score !== null && score >= 0.50;
        const barColor = isAi ? "#EF4444" : "#10B981";

        const row = document.createElement("div");
        row.style.display = "flex";
        row.style.flexDirection = "column";
        row.style.gap = "4px";

        row.innerHTML = `
          <div style="display: flex; justify-content: space-between; font-size: 12px; font-weight: 500;">
            <span style="color: #CBD5E1;">${shortName}</span>
            <span style="font-family: monospace; color: ${score !== null ? (score >= 0.5 ? '#F87171' : '#34D399') : '#94A3B8'};">
              ${score !== null ? `${pct}% AI` : 'Unavailable'}
            </span>
          </div>
          <div style="height: 6px; background: #1E293B; border-radius: 3px; overflow: hidden;">
            <div style="height: 100%; width: ${score !== null ? pct : 0}%; background: ${barColor}; border-radius: 3px; transition: width 0.4s ease;"></div>
          </div>
        `;
        modelBarsContainer.appendChild(row);
      }
    }

    // Image comparison
    compOriginal.src = previewImage.src;
    if (result.signals.ela && result.signals.ela.heatmap_base64) {
      compEla.src = result.signals.ela.heatmap_base64;
    } else {
      compEla.src = previewImage.src;
    }

    // Populate table
    signalTbody.innerHTML = "";

    const signalsList = [
      {
        name: "Metadata & EXIF",
        key: "metadata",
        formatScore: (s) => (s.score !== null ? `${(s.score * 100).toFixed(0)}%` : "N/A"),
      },
      {
        name: "C2PA Provenance",
        key: "c2pa",
        formatScore: (s) => (s.score !== null ? `${(s.score * 100).toFixed(0)}%` : "N/A"),
      },
      {
        name: "Error Level Analysis",
        key: "ela",
        formatScore: (s) => (s.score !== null ? `${(s.score * 100).toFixed(0)}%` : "N/A"),
      },
      {
        name: "Fourier FFT Spectrum",
        key: "frequency",
        formatScore: (s) => (s.score !== null ? `${(s.score * 100).toFixed(0)}%` : "N/A"),
      },
      {
        name: "ML Ensemble Classifiers",
        key: "ml_ensemble",
        formatScore: (s) => (s.score !== null ? `${(s.score * 100).toFixed(0)}%` : "N/A"),
      },
    ];

    // Optional Gemini
    if (result.signals.gemini && result.signals.gemini.score !== null) {
      signalsList.push({
        name: "Gemini Reasoning",
        key: "gemini",
        formatScore: (s) => `${(s.score * 100).toFixed(0)}%`,
      });
    }

    signalsList.forEach((sig) => {
      const data = result.signals[sig.key] || { score: null, confidence: "low", evidence: [] };
      const tr = document.createElement("tr");

      const tdName = document.createElement("td");
      tdName.innerHTML = `<div class="detector-name">${sig.name}</div>
        <span class="confidence-tag conf-${data.confidence || 'low'}">${data.confidence || 'low'} conf</span>`;

      const tdScore = document.createElement("td");
      tdScore.className = "signal-score";
      tdScore.textContent = sig.formatScore(data);

      const tdEvidence = document.createElement("td");
      const ul = document.createElement("ul");
      ul.className = "evidence-list";

      (data.evidence || []).forEach((ev) => {
        const li = document.createElement("li");
        li.textContent = ev;
        ul.appendChild(li);
      });

      tdEvidence.appendChild(ul);
      tr.appendChild(tdName);
      tr.appendChild(tdScore);
      tr.appendChild(tdEvidence);
      signalTbody.appendChild(tr);
    });

    resultsCard.scrollIntoView({ behavior: "smooth" });
  }
});
