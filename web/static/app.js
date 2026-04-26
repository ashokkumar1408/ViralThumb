/* ── State ────────────────────────────────────────────────────────────── */
let selectedTemplate = null;
let sessionId        = null;
let selectedVariant  = null;

const VARIANT_ORDER = ["original", "shock", "excited", "serious"];
const VARIANT_LABEL = { original: "Original", shock: "Shock", excited: "Excited", serious: "Serious" };

/* ── Boot ─────────────────────────────────────────────────────────────── */
document.addEventListener("DOMContentLoaded", () => {
  loadTemplates();
  bindUpload();
  bindActions();
});

/* ── Step 1: Template grid ─────────────────────────────────────────────── */
async function loadTemplates() {
  let templates;
  try {
    templates = await fetch("/api/templates").then(r => r.json());
  } catch (e) {
    showError("Could not load templates: " + e.message);
    return;
  }

  const grid = document.getElementById("template-grid");
  templates.forEach(t => {
    const card = document.createElement("div");
    card.className  = "tmpl-card";
    card.dataset.id = t.id;
    card.innerHTML  = `
      <img class="tmpl-preview" src="/api/templates/preview/${t.id}" alt="${t.name}" loading="lazy"/>
      <div class="tmpl-name">${t.name}</div>`;

    card.addEventListener("click", () => {
      document.querySelectorAll(".tmpl-card").forEach(c => c.classList.remove("selected"));
      card.classList.add("selected");
      selectedTemplate = t.id;
      updateGenerateBtn();
    });

    grid.appendChild(card);
  });
}

/* ── Step 2: Upload + expression variants ──────────────────────────────── */
function bindUpload() {
  const input = document.getElementById("photo-input");
  const zone  = document.getElementById("upload-zone");

  input.addEventListener("change", () => {
    if (input.files[0]) handlePhoto(input.files[0]);
  });

  zone.addEventListener("dragover",  e => { e.preventDefault(); zone.style.borderColor = "#f59e0b"; });
  zone.addEventListener("dragleave", ()  => { zone.style.borderColor = ""; });
  zone.addEventListener("drop", e => {
    e.preventDefault();
    zone.style.borderColor = "";
    const file = e.dataTransfer.files[0];
    if (!file) return;
    const dt = new DataTransfer();
    dt.items.add(file);
    input.files = dt.files;
    handlePhoto(file);
  });
}

async function handlePhoto(file) {
  sessionId       = null;
  selectedVariant = null;

  const wrap    = document.getElementById("variants-wrap");
  const grid    = document.getElementById("variants-grid");
  const spinner = document.getElementById("variants-spinner");

  grid.innerHTML = "";
  wrap.classList.remove("hidden");
  spinner.classList.remove("hidden");
  updateGenerateBtn();

  document.getElementById("upload-hint").textContent = file.name;

  const fd = new FormData();
  fd.append("photo", file);

  try {
    const res  = await fetch("/api/process-photo", { method: "POST", body: fd });
    const json = await res.json();
    if (!res.ok || json.error) throw new Error(json.error || "Server error");

    sessionId = json.session_id;
    renderVariants(json.variants);
  } catch (err) {
    showError("Photo processing failed: " + err.message);
    wrap.classList.add("hidden");
  } finally {
    spinner.classList.add("hidden");
  }
}

function renderVariants(variants) {
  const grid = document.getElementById("variants-grid");
  grid.innerHTML = "";

  VARIANT_ORDER.forEach(name => {
    const b64 = variants[name];
    if (!b64) return;

    const card = document.createElement("div");
    card.className  = "variant-card";
    card.dataset.id = name;
    card.innerHTML  = `
      <img class="variant-img" src="data:image/png;base64,${b64}" alt="${name}"/>
      <div class="variant-name">${VARIANT_LABEL[name] || name}</div>`;

    card.addEventListener("click", () => {
      document.querySelectorAll(".variant-card").forEach(c => c.classList.remove("selected"));
      card.classList.add("selected");
      selectedVariant = name;
      updateGenerateBtn();
    });

    grid.appendChild(card);
  });
}

/* ── Enable generate button ─────────────────────────────────────────────── */
function updateGenerateBtn() {
  document.getElementById("generate-btn").disabled =
    !(selectedTemplate && sessionId && selectedVariant);
}

/* ── Generate + Reset ───────────────────────────────────────────────────── */
function bindActions() {
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

  const payload = {
    session_id:  sessionId,
    template_id: selectedTemplate,
    variant:     selectedVariant,
    headline:    document.getElementById("headline").value.trim(),
    sub_text:    document.getElementById("sub-text").value.trim(),
  };

  try {
    const res  = await fetch("/api/generate", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(payload),
    });
    const json = await res.json();
    if (!res.ok || json.error) throw new Error(json.error || "Server error");

    const img = document.getElementById("result-img");
    const dl  = document.getElementById("download-btn");
    img.src   = json.result_url + "?t=" + Date.now();
    dl.href   = json.result_url;
    result.classList.remove("hidden");
    result.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (err) {
    showError("Generation failed: " + err.message);
  } finally {
    spinner.classList.add("hidden");
    btn.disabled = !(selectedTemplate && sessionId && selectedVariant);
  }
}

function reset() {
  selectedTemplate = null;
  sessionId        = null;
  selectedVariant  = null;

  document.querySelectorAll(".tmpl-card,.variant-card").forEach(c => c.classList.remove("selected"));
  document.getElementById("variants-wrap").classList.add("hidden");
  document.getElementById("variants-grid").innerHTML = "";
  document.getElementById("result-wrap").classList.add("hidden");
  document.getElementById("error-box").classList.add("hidden");
  document.getElementById("upload-hint").textContent = "Click or drag a photo here";
  document.getElementById("headline").value = "";
  document.getElementById("sub-text").value = "";
  document.getElementById("photo-input").value = "";

  updateGenerateBtn();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function showError(msg) {
  const box = document.getElementById("error-box");
  box.textContent = msg;
  box.classList.remove("hidden");
}
