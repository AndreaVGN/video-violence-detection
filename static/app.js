// ===============================
// CONFIG
// ===============================
const API_BASE = window.location.origin;

// ===============================
// API CALLS
// ===============================
async function fetchState() {
  const r = await fetch(`${API_BASE}/api/state`);
  return await r.json();
}

async function confirmEvent(clipId, label) {
  const r = await fetch(`${API_BASE}/api/confirm`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ clip_id: clipId, label: label })
  });
  return await r.json();
}

async function uploadVideo() {
  const fileInput = document.getElementById("videoFile");
  if (!fileInput.files.length) {
    alert("Select a video first");
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  document.getElementById("status").innerText = "Uploading and processing video...";

  const r = await fetch(`${API_BASE}/api/process`, {
    method: "POST",
    body: formData
  });

  const res = await r.json();
  document.getElementById("status").innerText =
    "Video processed: " + res.prediction;
}

// ===============================
// HELPERS
// ===============================
function fmtTs(ms) {
  if (!ms) return "-";
  return new Date(ms).toISOString();
}

function isViolence(pred) {
  return pred === "Violence" || pred === "Fight";
}

// ===============================
// MAIN LOOP
// ===============================
let CURRENT_CLIP_ID = null;

async function tick() {
  try {
    const st = await fetchState();
    const ev = st.last_event;

    document.getElementById("serverMode").innerText =
      st.server_mode || "-";

    const confirmBtn = document.getElementById("confirmBtn");
    const notViolentBtn = document.getElementById("notViolentBtn");
    const videoPlayer = document.getElementById("videoPlayer");

    if (!ev) {
      document.getElementById("status").innerText = "No events yet.";
      confirmBtn.style.display = "none";
      notViolentBtn.style.display = "none";
      videoPlayer.style.display = "none";
      return;
    }

    document.getElementById("status").innerText = "Last event received.";
    document.getElementById("mode").innerText = ev.mode || "-";
    document.getElementById("device").innerText = ev.device_id || "-";
    document.getElementById("clip").innerText = ev.clip_id || "-";
    document.getElementById("ts").innerText = fmtTs(ev.ts_utc_ms);

    document.getElementById("pred").innerText = ev.prediction || "-";
    

    const timings = ev.timings_ms || {};

document.getElementById("timings").innerText =
  Object.keys(timings).length
    ? Object.entries(timings)
        .map(([k, v]) => {
          const n = Math.abs(parseFloat(v));
          return `${k}:${n.toFixed(1)}`;
        })
        .join(" | ")
    : "-";



    // VIDEO PREVIEW
    if (ev.clip_id && ev.clip_id !== CURRENT_CLIP_ID) {
  CURRENT_CLIP_ID = ev.clip_id;

  videoPlayer.src = `${API_BASE}/api/view/${ev.clip_id}`;
  videoPlayer.load();      // carica una volta
  videoPlayer.style.display = "block";
}


    // CONFIRMATION BUTTONS
    if (isViolence(ev.prediction) && !ev.confirmed) {
      document.getElementById("confirmMsg").innerText =
        "SUSPECT EVENT DETECTED, Please select the type:";

      confirmBtn.style.display = "inline-block";
      notViolentBtn.style.display = "inline-block";

      confirmBtn.onclick = async () => {
        const res = await confirmEvent(ev.clip_id, "violent");
        document.getElementById("confirmMsg").innerText =
          res.ok ? "Last event: Violent" : ("Error: " + (res.error || "unknown"));
      };

      notViolentBtn.onclick = async () => {
        const res = await confirmEvent(ev.clip_id, "not_violent");
        document.getElementById("confirmMsg").innerText =
          res.ok ? "Last event: Not Violent" : ("Error: " + (res.error || "unknown"));
      };

    } else {
      confirmBtn.style.display = "none";
      notViolentBtn.style.display = "none";
    }

  } catch (e) {
    document.getElementById("status").innerText =
      "Error fetching state: " + e;
  }
}

setInterval(tick, 500);
tick();
