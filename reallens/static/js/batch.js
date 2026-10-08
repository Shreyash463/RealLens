/**
 * RealLens - Batch Testing & CSV Export Frontend Script
 */
document.addEventListener("DOMContentLoaded", () => {
  const batchDropzone = document.getElementById("batch-dropzone");
  const batchFileInput = document.getElementById("batch-file-input");
  const batchControls = document.getElementById("batch-controls");
  const batchCount = document.getElementById("batch-count");
  const batchRunBtn = document.getElementById("batch-run-btn");
  const batchDownloadBtn = document.getElementById("batch-download-btn");
  const batchLoading = document.getElementById("batch-loading");
  const batchProgressText = document.getElementById("batch-progress-text");
  const batchErrorBox = document.getElementById("batch-error-box");
  const batchResultsCard = document.getElementById("batch-results-card");
  const batchTbody = document.getElementById("batch-tbody");
  const batchStatsBadge = document.getElementById("batch-stats-badge");
  const deviceStatus = document.getElementById("device-status");

  let selectedFiles = [];
  let completedRecords = [];

  // 1. Health & Device Check
  fetch("/health")
    .then((r) => r.json())
    .then((data) => {
      if (deviceStatus) {
        deviceStatus.textContent = `Device: ${data.device.toUpperCase()}`;
        deviceStatus.classList.add("device-ready");
      }
    })
    .catch(() => {
      if (deviceStatus) deviceStatus.textContent = "Device: CPU";
    });

  // 2. Dropzone events
  batchDropzone.addEventListener("click", () => batchFileInput.click());

  ["dragenter", "dragover"].forEach((evt) => {
    batchDropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      batchDropzone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((evt) => {
    batchDropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      batchDropzone.classList.remove("dragover");
    });
  });

  batchDropzone.addEventListener("drop", (e) => {
    const files = Array.from(e.dataTransfer.files).filter((f) => f.type.startsWith("image/"));
    if (files.length > 0) {
      handleFiles(files);
    }
  });

  batchFileInput.addEventListener("change", (e) => {
    const files = Array.from(e.target.files);
    if (files.length > 0) {
      handleFiles(files);
    }
  });

  function handleFiles(files) {
    selectedFiles = files;
    batchCount.textContent = `${files.length} image file(s) selected`;
    batchControls.style.display = "flex";
    batchResultsCard.style.display = "none";
    batchDownloadBtn.style.display = "none";
    batchErrorBox.style.display = "none";
  }

  // 3. Run Batch Analysis
  batchRunBtn.addEventListener("click", async () => {
    if (selectedFiles.length === 0) return;

    batchRunBtn.disabled = true;
    batchLoading.style.display = "block";
    batchErrorBox.style.display = "none";
    batchResultsCard.style.display = "block";
    batchTbody.innerHTML = "";
    completedRecords = [];

    let total = selectedFiles.length;
    let done = 0;

    for (let i = 0; i < total; i++) {
      const file = selectedFiles[i];
      batchProgressText.textContent = `Analyzing batch... (${i + 1} / ${total}) - ${file.name}`;

      const formData = new FormData();
      formData.append("image", file);

      try {
        const resp = await fetch("/analyze", {
          method: "POST",
          body: formData,
        });

        const data = await resp.json();
        if (resp.ok && data.status === "success") {
          const res = data.result;
          completedRecords.push(res);
          appendTableRow(completedRecords.length, res);
        } else {
          appendErrorRow(completedRecords.length + 1, file.name, data.message || "Failed");
        }
      } catch (err) {
        appendErrorRow(completedRecords.length + 1, file.name, err.message);
      }
      done++;
    }

    batchLoading.style.display = "none";
    batchRunBtn.disabled = false;
    batchDownloadBtn.style.display = "inline-block";

    // Summary stats
    const likelyAi = completedRecords.filter((r) => r.verdict === "Likely AI").length;
    const likelyReal = completedRecords.filter((r) => r.verdict === "Likely real").length;
    const uncertain = completedRecords.filter((r) => r.verdict.includes("Uncertain")).length;
    batchStatsBadge.textContent = `Total: ${completedRecords.length} | AI: ${likelyAi} | Real: ${likelyReal} | Uncertain: ${uncertain}`;
  });

  function appendTableRow(index, res) {
    const tr = document.createElement("tr");

    let verdictBadgeClass = "badge-neutral";
    if (res.verdict === "Likely AI") verdictBadgeClass = "badge-danger";
    else if (res.verdict === "Likely real") verdictBadgeClass = "badge-success";
    else verdictBadgeClass = "badge-warning";

    tr.innerHTML = `
      <td style="font-family: monospace; color: #94A3B8;">${index}</td>
      <td style="font-weight: 500; color: #F1F5F9; max-width: 180px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${res.filename}</td>
      <td style="font-family: monospace; font-size: 12px; color: #94A3B8;">${res.image_dimensions || 'N/A'}</td>
      <td style="font-weight: 600; font-family: monospace; color: ${res.ai_percentage >= 50 ? '#F87171' : '#34D399'};">${res.ai_percentage}%</td>
      <td><span class="badge ${verdictBadgeClass}">${res.verdict}</span></td>
      <td style="text-transform: capitalize; font-size: 12px;">${res.confidence}</td>
      <td style="font-family: monospace; font-size: 12px; color: #94A3B8;">${res.time_taken_seconds || 0}s</td>
      <td style="font-size: 12px; color: #CBD5E1; max-width: 250px;">${res.summary_reason || 'N/A'}</td>
    `;
    batchTbody.appendChild(tr);
  }

  function appendErrorRow(index, filename, errorMsg) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td style="font-family: monospace; color: #94A3B8;">${index}</td>
      <td style="font-weight: 500; color: #F1F5F9;">${filename}</td>
      <td colspan="5" style="color: #EF4444; font-size: 12px;">Error: ${errorMsg}</td>
      <td>-</td>
    `;
    batchTbody.appendChild(tr);
  }

  // 4. Download CSV Handler
  batchDownloadBtn.addEventListener("click", () => {
    if (completedRecords.length === 0) return;

    const headers = [
      "Filename",
      "Dimensions",
      "AI_Probability",
      "Verdict",
      "Confidence",
      "Latency_Seconds",
      "Disagreement_Flag",
      "In_Uncertain_Zone",
      "Summary_Reason"
    ];

    const rows = completedRecords.map((r) => [
      `"${r.filename}"`,
      `"${r.image_dimensions || ''}"`,
      r.ai_probability,
      `"${r.verdict}"`,
      `"${r.confidence}"`,
      r.time_taken_seconds || 0,
      r.disagreement_flag ? "TRUE" : "FALSE",
      r.in_uncertain_zone ? "TRUE" : "FALSE",
      `"${(r.summary_reason || '').replace(/"/g, '""')}"`
    ]);

    const csvContent = [headers.join(","), ...rows.map((row) => row.join(","))].join("\n");
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute("download", `reallens_batch_results_${new Date().toISOString().slice(0, 10)}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  });
});
