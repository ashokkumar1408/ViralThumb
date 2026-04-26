/* ── State ────────────────────────────────────────────────────────────── */
let selectedStyle = null;
let photoReady    = false;

/* ── Boot ─────────────────────────────────────────────────────────────── */
document.addEventListener("DOMContentLoaded", () => {
  loadStyles();
  bindUpload();
  bindForm();
});

/* ── Style grid ────────────────────────────────────────────────────────── */
async function loadStyles() {
  const styles = await fetch("/api/styles").then(r => r.json());
  const grid   = document.getElementById("style-grid");

  styles.forEach(s => {
    const card = document.createElement("div");
    card.className  = "style-card";
    card.dataset.id = s.id;

    // Gradient swatch from the 3 preview colours
    const gradient = `linear-gradient(135deg,${s.preview.join(",")})`;

    card.innerHTML = `
      <span class="swatch" style="background:${gradient}"></span>
      <div class="style-name">${s.name}</div>`;

    card.addEventListener("click", () => {
      document.querySelectorAll(".style-card").forEach(c => c.classList.remove("selected"));
      card.classList.add("selected");
      selectedStyle = s.id;
      updateBtn();
    });

    grid.appendChild(card);
  });
}

/* ── Upload ────────────────────────────────────────────────────────────── */
function bindUpload() {
  const input   = document.getElementById("photo-input");
  const zone    = document.getElementById("upload-zone");
  const preview = document.getElementById("photo-preview");
  const icon    = document.getElementById("upload-icon");
  const hint    = document.getElementById("upload-hint");

  input.addEventListener("change", () => {
    const file = input.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = ev => {
      preview.src = ev.target.result;
      preview.classList.remove("hidden");
      icon.style.display = "none";
      hint.style.display = "none";
      photoReady = true;
      updateBtn();
    };
    reader.readAsDataURL(file);
  });

  // Drag and drop
  zone.addEventListener("dragover",  e => { e.preventDefault(); zone.style.borderColor = "#f59e0b"; });
  zone.addEventListener("dragleave", ()  => { zone.style.borderColor = ""; });
  zone.addEventListener("drop", e => {
    e.preventDefault();
    zone.style.borderColor = "";
    const file = e.dataTransfer.files[0];
    if (!file) return;
    const dt   = new DataTransfer();
    dt.items.add(file);
    input.files = dt.files;
    input.dispatchEvent(new Event("change"));
  });
}

/* ── Enable generate button only when both style + photo are ready ───── */
function updateBtn() {
  document.getElementById("generate-btn").disabled = !(selectedStyle && photoReady);
}

/* ── Generate ──────────────────────────────────────────────────────────── */
function bindForm() {
  document.getElementById("generate-btn").addEventListener("click", generate);
  document.getElementById("redo-btn").addEventListener("click", reset);
}

async function generate() {
  const btn     = document.getElementById("generate-btn");
  const spinner = document.getElementById("spinner");
  const errBox  = document.getElementById("error-box");
  const result  = document.getElementById("result-wrap");

  btn.disabled = true;
  spinner.classList.remove("hidden");
  errBox.classList.add("hidden");
  result.classList.add("hidden");

  const fd = new FormData();
  fd.append("style",    selectedStyle);
  fd.append("headline", document.getElementById("headline").value.trim());
  fd.append("sub_text", document.getElementById("sub-text").value.trim());
  fd.append("photo",    document.getElementById("photo-input").files[0]);

  try {
    const res  = await fetch("/api/generate", { method: "POST", body: fd });
    const json = await res.json();
    if (!res.ok || json.error) throw new Error(json.error || "Server error");

    const img  = document.getElementById("result-img");
    const dl   = document.getElementById("download-btn");
    img.src    = json.result_url + "?t=" + Date.now();
    dl.href    = json.result_url;
    result.classList.remove("hidden");
    result.scrollIntoView({ behavior: "smooth", block: "start" });

  } catch (err) {
    errBox.textContent = "Error: " + err.message;
    errBox.classList.remove("hidden");
  } finally {
    spinner.classList.add("hidden");
    btn.disabled = !(selectedStyle && photoReady);
  }
}

function reset() {
  document.getElementById("result-wrap").classList.add("hidden");
  window.scrollTo({ top: 0, behavior: "smooth" });
}
