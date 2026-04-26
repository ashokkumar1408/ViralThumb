/* ── State ─────────────────────────────────────────────────────────────── */
let selectedTemplateId = null;

const EMOTION_ICONS = {
  shock:       "😱",
  excitement:  "🔥",
  fear:        "😨",
  curiosity:   "🤔",
  mystery:     "🕵️",
  celebration: "🎉",
};

/* ── Boot ──────────────────────────────────────────────────────────────── */
document.addEventListener("DOMContentLoaded", () => {
  loadNiches();
  loadTemplates("");
  bindForm();
});

/* ── Niche tabs ─────────────────────────────────────────────────────────── */
async function loadNiches() {
  const res  = await fetch("/api/templates");
  const data = await res.json();
  const niches = [...new Set(data.map(t => t.niche))].sort();
  const bar = document.getElementById("niche-tabs");

  niches.forEach(niche => {
    const btn = document.createElement("button");
    btn.className = "tab";
    btn.dataset.niche = niche;
    btn.textContent = niche;
    btn.addEventListener("click", () => switchNiche(niche, btn));
    bar.appendChild(btn);
  });
}

function switchNiche(niche, btn) {
  document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
  btn.classList.add("active");
  loadTemplates(niche);
}

/* ── Template grid ──────────────────────────────────────────────────────── */
async function loadTemplates(niche) {
  const grid = document.getElementById("template-grid");
  grid.innerHTML = '<div class="loading">Loading templates…</div>';

  const url = niche ? `/api/templates?niche=${niche}` : "/api/templates";
  const res  = await fetch(url);
  const data = await res.json();

  if (!data.length) {
    grid.innerHTML = '<div class="loading">No templates yet. Run <code>python curator.py --niche gaming</code> then <code>python processor.py</code>.</div>';
    return;
  }

  grid.innerHTML = "";
  data.forEach(t => {
    const card = document.createElement("div");
    card.className = "template-card" + (t.ready ? "" : " not-ready");
    card.dataset.id = t.template_id;

    const imgSrc = t.thumbnail_url || "";
    const views  = t.view_count >= 1e6
      ? (t.view_count / 1e6).toFixed(1) + "M"
      : t.view_count >= 1e3
        ? (t.view_count / 1e3).toFixed(0) + "K"
        : String(t.view_count);

    const emotionIcon = EMOTION_ICONS[t.emotion] || "📹";

    card.innerHTML = `
      <img src="${imgSrc}" alt="${t.video_title}" loading="lazy"
           onerror="this.src='data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 width=%22320%22 height=%22180%22%3E%3Crect fill=%22%231e2130%22 width=%22320%22 height=%22180%22/%3E%3C/svg%3E'" />
      <div class="card-info">
        <div class="card-title">${t.video_title || t.template_id}</div>
        <div class="card-meta">
          <span>${views} views</span>
          <span class="${t.ready ? "badge-ready" : "badge-pending"}">
            ${t.ready ? emotionIcon + " ready" : "⏳ analyzing"}
          </span>
        </div>
      </div>`;

    if (t.ready) {
      card.addEventListener("click", () => selectTemplate(t, card, imgSrc));
    }
    grid.appendChild(card);
  });
}

/* ── Select template ────────────────────────────────────────────────────── */
function selectTemplate(t, card, imgSrc) {
  document.querySelectorAll(".template-card").forEach(c => c.classList.remove("selected"));
  card.classList.add("selected");
  selectedTemplateId = t.template_id;

  document.getElementById("template_id").value = t.template_id;
  document.getElementById("selected-img").src   = imgSrc;
  document.getElementById("selected-info").textContent =
    `${t.niche.toUpperCase()} · ${t.video_title}`;

  // DNA badges
  const dnaBadges = document.getElementById("selected-dna");
  dnaBadges.innerHTML = "";
  if (t.emotion) {
    const emotionIcon = EMOTION_ICONS[t.emotion] || "📹";
    dnaBadges.innerHTML += `<span class="dna-badge emotion">${emotionIcon} ${t.emotion}</span>`;
  }
  if (t.layout) {
    const layoutLabel = t.layout.replace(/_/g, " ");
    dnaBadges.innerHTML += `<span class="dna-badge layout">📐 ${layoutLabel}</span>`;
  }

  document.getElementById("no-selection").classList.add("hidden");
  document.getElementById("selected-preview").classList.remove("hidden");
  document.getElementById("generate-form").classList.remove("hidden");
  document.getElementById("result-section").classList.add("hidden");
  document.getElementById("error-box").classList.add("hidden");
}

/* ── Photo preview ──────────────────────────────────────────────────────── */
document.addEventListener("change", e => {
  if (e.target.id !== "photo-input") return;
  const file = e.target.files[0];
  if (!file) return;
  document.getElementById("upload-text").textContent = `📷 ${file.name}`;
  const reader = new FileReader();
  reader.onload = ev => {
    document.getElementById("photo-preview").src = ev.target.result;
    document.getElementById("photo-preview-wrap").classList.remove("hidden");
  };
  reader.readAsDataURL(file);
});

/* ── Generate form ──────────────────────────────────────────────────────── */
function bindForm() {
  document.getElementById("generate-form").addEventListener("submit", async e => {
    e.preventDefault();
    if (!selectedTemplateId) return;

    const form    = e.target;
    const btn     = document.getElementById("generate-btn");
    const spinner = document.getElementById("spinner");
    const errBox  = document.getElementById("error-box");

    btn.disabled = true;
    spinner.classList.remove("hidden");
    errBox.classList.add("hidden");
    document.getElementById("result-section").classList.add("hidden");

    try {
      const fd  = new FormData(form);
      const res = await fetch("/api/generate", { method: "POST", body: fd });
      const json = await res.json();

      if (!res.ok || json.error) throw new Error(json.error || "Server error");

      const resultImg   = document.getElementById("result-img");
      const downloadBtn = document.getElementById("download-btn");
      resultImg.src     = json.result_url + "?t=" + Date.now();
      downloadBtn.href  = json.result_url;
      document.getElementById("result-section").classList.remove("hidden");

    } catch (err) {
      errBox.textContent = "Error: " + err.message;
      errBox.classList.remove("hidden");
    } finally {
      btn.disabled = false;
      spinner.classList.add("hidden");
    }
  });

  document.getElementById("reset-btn").addEventListener("click", () => {
    document.getElementById("result-section").classList.add("hidden");
    document.getElementById("photo-input").value = "";
    document.getElementById("photo-preview-wrap").classList.add("hidden");
    document.getElementById("upload-text").textContent = "📷 Upload your photo (jpg / png)";
  });
}
