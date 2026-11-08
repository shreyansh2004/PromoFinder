"use strict";
const $ = (s, el = document) => el.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const state = { runId: null, result: null, sel: 0, files: [] };

/* ---------------- views */
function show(view) {
  for (const v of ["start", "results", "insights"]) $(`#view-${v}`).hidden = v !== view;
  document.querySelectorAll(".tab").forEach((t) => t.setAttribute("aria-current", t.dataset.view === view ? "page" : "false"));
  if (view === "insights") loadInsights();
}
document.querySelectorAll("[data-view]").forEach((b) => b.addEventListener("click", (e) => { e.preventDefault(); if (!b.disabled) show(b.dataset.view); }));

/* ---------------- setup info */
const PROVIDER_LABEL = { claude: "Claude", gemini: "Gemini", openai_compat: "Custom API" };
fetch("/api/config").then((r) => r.json()).then((c) => {
  $("#engine").innerHTML = c.llm_available ? `AI <b>${esc(c.llm.split(" -> ")[0])}</b>${c.llm.includes("->") ? " with failover" : ""}` : `No AI key set: <b>rule-based mode</b>`;
  if (!c.allow_local_path) $("#localwrap").hidden = true;
  if ((c.ai_requires_code || c.access_code_optional) && c.llm_available) {
    $("#accesscode").hidden = false;
    $("#accesscode").placeholder = c.ai_requires_code ? "Access code for the AI model" : "Access code (optional, skips the daily limit)";
    const syncCode = () => { $("#accesscode").hidden = !$("#usellm").checked; };
    $("#usellm").addEventListener("change", syncCode);
  }
  state.aiLeft = c.ai_runs_left_today;
  const sel = $("#model");
  if (!c.llm_available) {
    $("#usellm").checked = false; $("#usellm").disabled = true; sel.disabled = true;
    sel.innerHTML = `<option>No AI key in .env</option>`;
    $("#llmnote").textContent = "Add ANTHROPIC_API_KEY or GEMINI_API_KEY to .env to enable the AI model.";
    return;
  }
  const groups = {};
  c.models.forEach((m) => (groups[m.provider] ||= []).push(m));
  sel.innerHTML = `<option value="">Automatic: ${esc(c.llm.replaceAll(":", " ").replaceAll("->", "then"))}</option>` +
    Object.entries(groups).map(([p, ms]) => `<optgroup label="${esc(PROVIDER_LABEL[p] || p)}">${ms.map((m) => `<option value="${esc(m.model)}">${esc(m.model)}</option>`).join("")}</optgroup>`).join("");
  $("#llmnote").textContent = "The chosen model goes first; if it is rate-limited or unavailable the next one takes over." +
    (c.ai_runs_left_today !== null && c.ai_runs_left_today !== undefined ? ` Free AI runs left today: ${c.ai_runs_left_today} of ${c.ai_daily_limit}; after that it runs rule-based.` : "");
  $("#usellm").addEventListener("change", () => { sel.disabled = !$("#usellm").checked; });
}).catch(() => { $("#engine").textContent = "Server not reachable"; });

/* ---------------- upload */
const drop = $("#drop"), input = $("#files");
drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
input.addEventListener("change", () => setFiles([...input.files]));
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => setFiles([...e.dataTransfer.files]));
function setFiles(fs) {
  state.files = fs.filter((f) => /\.(txt|zip)$/i.test(f.name));
  $("#picked").textContent = state.files.length ? `${state.files.length} file${state.files.length > 1 ? "s" : ""} ready` : "No .txt or .zip files in that selection";
}

$("#runform").addEventListener("submit", async (e) => {
  e.preventDefault();
  $("#formerror").textContent = "";
  const path = $("#localpath").value.trim();
  if (!state.files.length && !path) { $("#formerror").textContent = "Add episode scripts or a folder path first."; return; }
  $("#runbtn").disabled = true;
  try {
    let r;
    const useLlm = $("#usellm").checked, topN = +$("#topn").value || 15, name = $("#showname").value.trim();
    const model = useLlm ? $("#model").value : "", episodes = $("#episodes").value.trim();
    const access_code = $("#accesscode").value;
    if (state.files.length) {
      const fd = new FormData();
      state.files.forEach((f) => fd.append("files", f));
      fd.append("use_llm", useLlm); fd.append("top_n", topN); fd.append("show_name", name);
      fd.append("model", model); fd.append("episodes", episodes); fd.append("access_code", access_code);
      r = await fetch("/api/runs", { method: "POST", body: fd });
    } else {
      r = await fetch("/api/runs/local", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ path, use_llm: useLlm, top_n: topN, show_name: name, model, episodes, access_code }) });
    }
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || "Could not start the run");
    state.runId = j.id;
    $("#progress").hidden = false;
    poll();
  } catch (err) {
    $("#formerror").textContent = err.message;
    $("#runbtn").disabled = false;
  }
});

async function poll() {
  const r = await fetch(`/api/runs/${state.runId}`);
  const j = await r.json();
  $("#steps").innerHTML = j.log.map((l) => `<li><time>${l.t.toFixed(0)}s</time><span>${esc(l.msg)}</span></li>`).join("");
  if (j.status === "running") return setTimeout(poll, 1000);
  $("#runbtn").disabled = false;
  if (j.status === "error") { $("#formerror").textContent = `The run stopped: ${j.error}`; return; }
  state.result = j.result; state.sel = 0;
  $("#tab-results").disabled = false;
  renderResults();
  show("results");
}

/* ---------------- results */
function segLabel(s) { return `Ep ${s.episode} ${s.start}–${s.end}`; }

function renderResults() {
  const R = state.result;
  $("#showtitle").textContent = R.show_name || "Promos";
  $("#premise").textContent = R.bible?.premise || "";
  for (const [id, f] of [["#dl-csv", "promos.csv"], ["#dl-md", "report.md"], ["#dl-json", "promos.json"]])
    $(id).href = `/api/runs/${state.runId}/download/${f}`;
  renderRibbon();
  $("#ranklist").innerHTML = R.promos.map((p, i) => `
    <li class="${i === state.sel ? "on" : ""} ${p.decision ? "decision-" + p.decision : ""}" data-i="${i}">
      <button type="button" aria-label="Promo ${i + 1}: ${esc(p.title)}">
        <span class="rk">${i + 1}</span>
        <span><span class="rt">${esc(p.title || "Untitled clip")}</span>
          <span class="rm">${p.segments.map(segLabel).join(" + ")} · ${Math.round(p.duration_s)}s</span></span>
        <span class="rs">${(p.scores.final ?? 0).toFixed(2)}</span>
      </button></li>`).join("");
  document.querySelectorAll("#ranklist li").forEach((li) => li.querySelector("button").addEventListener("click", () => select(+li.dataset.i)));
  select(state.sel);
}

function renderRibbon() {
  const R = state.result;
  const eps = R.episodes;
  const durs = {};
  R.promos.forEach((p) => p.segments.forEach((s) => { durs[s.episode] = Math.max(durs[s.episode] || 0, s.end_s + 5); }));
  let html = "";
  for (let e = 1; e <= eps; e++) {
    const ticks = R.promos.map((p, i) => p.segments.filter((s) => s.episode === e).map((s) => {
      const left = Math.min(96, (s.start_s / (durs[e] || 150)) * 100);
      return `<button class="tick ${i === state.sel ? "on" : ""}" style="left:${left}%" data-i="${i}" title="#${i + 1} ${esc(p.title)}" aria-label="Promo ${i + 1} in episode ${e}"></button>`;
    }).join("")).join("");
    html += `<div class="ep">${ticks}${e === 1 || e % 5 === 0 ? `<span class="ep-num">${e}</span>` : ""}</div>`;
  }
  $("#ribbon").innerHTML = html;
  $("#ribbon").querySelectorAll(".tick").forEach((t) => t.addEventListener("click", () => select(+t.dataset.i)));
  const used = new Set(R.promos.flatMap((p) => p.segments.map((s) => s.episode)));
  $("#ribbon-caption").textContent = `${R.promos.length} promos from ${used.size} of ${eps} episodes · ${R.all_candidates} candidates checked · ${R.mode === "llm" ? "AI + rules" : "rules only"} · ${R.runtime_s}s`;
}

function select(i) {
  const R = state.result;
  if (!R || !R.promos.length) return;
  state.sel = Math.max(0, Math.min(R.promos.length - 1, i));
  document.querySelectorAll("#ranklist li").forEach((li) => li.classList.toggle("on", +li.dataset.i === state.sel));
  document.querySelectorAll(".tick").forEach((t) => t.classList.toggle("on", +t.dataset.i === state.sel));
  $(`#ranklist li[data-i="${state.sel}"]`)?.scrollIntoView({ block: "nearest" });
  const p = R.promos[state.sel];
  renderPhone(p);
  renderDetail(p);
}

function toSec(t) { const [m, s] = t.split(":"); return +m * 60 + +s; }

function renderPhone(p) {
  $("#ph-dur").textContent = `${Math.round(p.duration_s)} s`;
  $("#ph-src").textContent = p.segments.map((s) => `Ep ${s.episode}`).join(" + ");
  const lines = p.transcript.filter((t) => t.kind !== "CAMERA");
  const hook = lines.filter((t) => toSec(t.t) < 3), rest = lines.filter((t) => toSec(t.t) >= 3);
  const cutFrom = p.duration_s - 6;
  let prevEp = p.segments[0].episode;
  const row = (t) => {
    let jump = "";
    if (t.ep !== prevEp) { jump = `<div class="jump">cut to episode ${t.ep}</div>`; prevEp = t.ep; }
    const cls = t.kind === "DIALOGUE" ? "" : t.kind === "SFX" ? "sfx" : "act";
    const body = t.kind === "DIALOGUE"
      ? `<span class="who">${esc(t.speaker)}</span>${t.tone ? ` <span class="tone">(${esc(t.tone)})</span>` : ""}<br>${esc(t.text)}`
      : esc(t.text);
    const z = toSec(t.t) >= cutFrom ? " zone-cut" : "";
    return `${jump}<div class="ln ${cls}${z}"><time>${toSec(t.t).toFixed(1)}s</time><span>${body}</span></div>`;
  };
  $("#phone").innerHTML =
    `<div class="zone-hook"><div class="zone-label"><span>Hook</span><span>0–3 s</span></div>${hook.length ? hook.map(row).join("") : `<div class="ln act"><time></time><span>No line in the first 3 s. Opens on: ${esc(p.transcript[0]?.text || "")}</span></div>`}</div>` +
    rest.map(row).join("") +
    `<div class="cutline"><span>Cut here</span><span>${Math.round(p.duration_s)} s</span></div>`;
  $("#phone").scrollTop = 0;
  const hookPct = Math.min(100, (3 / p.duration_s) * 100);
  $("#scale").innerHTML = `<i style="top:0;height:${hookPct}%;background:var(--hook)"></i><i style="bottom:0;height:${(6 / p.duration_s) * 100}%;background:var(--cut)"></i>`;
}

function bar(label, v, cls = "") {
  if (v === null || v === undefined) return `<div class="bar ${cls}"><span>${label}</span><span class="track"></span><span class="val">n/a</span></div>`;
  return `<div class="bar ${cls}"><span>${label}</span><span class="track"><span class="fill" style="width:${Math.max(2, v * 100)}%"></span></span><span class="val">${v.toFixed(2)}</span></div>`;
}

function renderDetail(p) {
  const s = p.scores, j = p.judge || {}, t = p.tags || {};
  const flags = (p.flags || []).map((f) => ({ late_hook: "Hook starts after 3 s", trimmed_lead_in: "Lead-in trimmed to reach the hook", extended_back: "Extended backwards to reach 20 s", not_standalone: "A stranger found it confusing", weak_cliffhanger: "Ending feels resolved, not a cliffhanger", reopened_on_stronger_hook: "AI pick was over 60 s; re-opened on a stronger hook", tightened_to_hook_and_cliffhanger: "Tightened to the AI's hook and cliffhanger beats", montage_trimmed: "Montage trimmed to fit 60 s", review_failed: "AI review failed (all models busy); scored on script signals" }[f] || f));
  $("#detail").innerHTML = `
    <div><h3>${esc(p.title || "Untitled clip")}</h3>
      <p class="where">${p.segments.map(segLabel).join(" + ")} · ${Math.round(p.duration_s)} s</p></div>
    <div class="pair">
      <p class="h"><b>Hook</b>${esc(p.hook || p.first_3s_text || "")}</p>
      <p class="c"><b>Cliffhanger</b>${esc(p.cliffhanger || "")}</p>
    </div>
    <p class="why">${esc(p.why || "")}</p>
    <div class="scores">
      ${bar("Overall", s.final, "total")}
      ${bar(`Ad-success model <small>(weight ${(s.weights?.model ?? 0).toFixed(2)})</small>`, s.model_prob)}
      ${bar(`Stranger test <small>(weight ${(s.weights?.editor ?? 0).toFixed(2)})</small>`, s.editor)}
      ${bar(`Script signals <small>(weight ${(s.weights?.rules ?? 1).toFixed(2)})</small>`, s.rules_norm)}
    </div>
    ${j.what_is_happening ? `<div class="stranger"><b>What a stranger understood</b><p>${esc(j.what_is_happening)}</p>
      <p class="q">Wants to know next: ${esc(j.open_question)}${j.confusing && j.confusing !== "nothing" ? `<br>Confused by: ${esc(j.confusing)}` : ""}</p></div>` : ""}
    <dl class="facts">
      ${t.hook_type ? `<dt>Hook type</dt><dd>${esc(String(t.hook_type).replaceAll("_", " "))}</dd>` : ""}
      ${t.genre ? `<dt>Genre</dt><dd>${esc(String(t.genre).replaceAll("_", " "))}</dd>` : ""}
      ${t.tropes ? `<dt>Tropes</dt><dd>${esc([].concat(t.tropes).join(", ").replaceAll("_", " "))}</dd>` : ""}
      <dt>Send viewers to</dt><dd>Episode ${p.landing_episode}</dd>
      <dt>Found by</dt><dd>${p.source === "heuristic" ? "Script rules" : p.source === "llm-montage" ? "AI montage" : "AI editor"}</dd>
      ${flags.length ? `<dt>Check</dt><dd class="flag">${flags.map(esc).join("<br>")}</dd>` : ""}
    </dl>
    <div class="decide">
      <textarea id="note" placeholder="Note for the editor (optional)" aria-label="Note"></textarea>
      <div class="btns"><button class="approve" id="approve">Approve</button><button class="reject" id="reject">Reject</button></div>
    </div>
    <details><summary>Cut list</summary><pre class="cutlist">${p.cut_list.map(esc).join("\n")}</pre></details>`;
  $("#approve").addEventListener("click", () => decide("approved"));
  $("#reject").addEventListener("click", () => decide("rejected"));
}

async function decide(decision) {
  const p = state.result.promos[state.sel];
  const note = $("#note")?.value || "";
  const r = await fetch(`/api/runs/${state.runId}/feedback`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ promo_id: p.id, decision, note }) });
  if (r.ok) {
    p.decision = decision;
    const li = $(`#ranklist li[data-i="${state.sel}"]`);
    li.classList.remove("decision-approved", "decision-rejected");
    li.classList.add(`decision-${decision}`);
    select(state.sel + 1);
  }
}

document.addEventListener("keydown", (e) => {
  if ($("#view-results").hidden || ["INPUT", "TEXTAREA"].includes(document.activeElement.tagName)) return;
  if (e.key === "j") select(state.sel + 1);
  if (e.key === "k") select(state.sel - 1);
  if (e.key === "a") decide("approved");
  if (e.key === "r") decide("rejected");
});

/* ---------------- insights */
let insightsLoaded = false;
async function loadInsights() {
  if (insightsLoaded) return;
  const j = await (await fetch("/api/insights")).json();
  const el = $("#insights");
  if (!j.available) {
    el.innerHTML = `<h1>What wins in real ads</h1><p class="empty">The ad-success model has not been trained yet. Run <code>model/train.py</code> to learn from the real short-drama ad dataset.</p>`;
    return;
  }
  insightsLoaded = true;
  const block = (b) => `<div class="ins"><h2>${esc(b.title)}</h2><p class="sub">${esc(b.subtitle)}</p>
    ${b.rows.map((r) => `<div class="wr"><span>${esc(r.label)}</span><span class="track"><span class="fill" style="width:${r.rate * 100}%"></span><span class="base" style="left:${b.base * 100}%"></span></span><span class="val">${Math.round(r.rate * 100)}% <small>n=${r.n}</small></span></div>`).join("")}</div>`;
  const v = j.validation;
  const vrow = (label, a, w, note) => `<div class="wr"><span>${label}</span><span class="track"><span class="fill" style="width:${Math.max(0, Math.min(1, a)) * 100}%"></span><span class="base" style="left:50%"></span></span><span class="val">${a.toFixed(2)} <small>${note}</small></span></div>`;
  const totW = v ? Math.max(v.earned_weights.model, 0.02) + Math.max(v.earned_weights.editor, 0.02) + Math.max(v.earned_weights.rules, 0.02) : 1;
  const share = (x) => Math.round((Math.max(x, 0.02) / totW) * 100);
  const valBlock = v ? `<div class="ins trust"><h2>How each ranking signal was tested</h2>
      <p class="sub">Each signal scored ${v.n_ads} real ads; bars show how often it ranks a high-reach ad above a low-reach ad from the same app (0.50 = coin flip, pink line). Its share of the final score is earned from this.</p>
      ${vrow("Stranger test (AI)", v.stranger.within_app_auc, 0, `share ${share(v.earned_weights.editor)}%`)}
      ${vrow("Ad-success model", v.ml_model.within_app_auc, 0, `share ${share(v.earned_weights.model)}%`)}
      ${vrow("Script signals (rules)", v.rules.within_app_auc, 0, `share ${share(v.earned_weights.rules)}%`)}
      <p class="sub">The stranger test does best on American-style apps such as ReelShort (${v.stranger.per_source.named.toFixed(2)}) and worst on dubbed imports (${v.stranger.per_source.search.toFixed(2)}).</p></div>` : "";
  el.innerHTML = `<h1>${esc(j.headline)}</h1><p class="lede">${esc(j.lede)}</p>
    ${valBlock}
    <div class="ins-grid">${j.blocks.map(block).join("")}</div>
    <div class="modelcard"><h2>About the model</h2><ul>${j.model_card.map((x) => `<li>${esc(x)}</li>`).join("")}</ul></div>`;
}

show("start");
