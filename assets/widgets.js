/* =========================================================================
   Interactive widgets. Embed in any Markdown chapter with raw HTML:
     <div class="widget" data-widget="activation"></div>
     <div class="widget" data-widget="gradient-descent"></div>
     <div class="widget" data-widget="lr-schedule"></div>
     <div class="widget" data-widget="metrics-threshold"></div>
     <div class="widget" data-widget="attention"></div>
   Each is dependency-free (canvas/SVG) and reads CSS variables for theming.
   ========================================================================= */
(function () {
  "use strict";
  const cssv = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim() || "#888";
  const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };

  const registry = {};

  /* ---------- Activation functions ---------- */
  registry.activation = (host) => {
    const fns = {
      ReLU: (x) => Math.max(0, x),
      GELU: (x) => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3))),
      SiLU: (x) => x / (1 + Math.exp(-x)),
      Tanh: (x) => Math.tanh(x),
      Sigmoid: (x) => 1 / (1 + Math.exp(-x)),
      LeakyReLU: (x) => (x > 0 ? x : 0.1 * x),
    };
    let active = new Set(["ReLU", "GELU", "SiLU"]);
    const colors = { ReLU: cssv("--accent"), GELU: cssv("--accent-2"), SiLU: cssv("--accent-3"), Tanh: "#12a150", Sigmoid: "#d97706", LeakyReLU: "#dc2626" };
    host.appendChild(el("div", "w-title", "Activation functions"));
    const controls = el("div", "w-controls");
    Object.keys(fns).forEach((k) => {
      const b = el("button", "w-chip" + (active.has(k) ? " on" : ""), k);
      b.style.borderColor = colors[k];
      b.onclick = () => { active.has(k) ? active.delete(k) : active.add(k); b.classList.toggle("on"); draw(); };
      controls.appendChild(b);
    });
    host.appendChild(controls);
    const cv = el("canvas"); cv.width = 640; cv.height = 320; host.appendChild(cv);
    const ctx = cv.getContext("2d");
    function draw() {
      const W = cv.width, H = cv.height, xr = 6, yr = 4;
      ctx.clearRect(0, 0, W, H);
      const X = (x) => (x + xr) / (2 * xr) * W, Y = (y) => H - (y + yr) / (2 * yr) * H;
      ctx.strokeStyle = cssv("--border"); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(0, Y(0)); ctx.lineTo(W, Y(0)); ctx.moveTo(X(0), 0); ctx.lineTo(X(0), H); ctx.stroke();
      ctx.font = "12px Inter"; ctx.fillStyle = cssv("--text-faint");
      active.forEach((k) => {
        ctx.strokeStyle = colors[k]; ctx.lineWidth = 2.5; ctx.beginPath();
        for (let px = 0; px <= W; px++) { const x = px / W * 2 * xr - xr; const y = fns[k](x); px === 0 ? ctx.moveTo(X(x), Y(y)) : ctx.lineTo(X(x), Y(y)); }
        ctx.stroke();
      });
    }
    draw(); host._redraw = draw;
  };

  /* ---------- Gradient descent on a 1D loss ---------- */
  registry["gradient-descent"] = (host) => {
    host.appendChild(el("div", "w-title", "Gradient descent — watch the learning rate"));
    const controls = el("div", "w-controls");
    const lrLabel = el("span", "w-label", "lr = 0.10");
    const lr = el("input"); lr.type = "range"; lr.min = "0.01"; lr.max = "1.05"; lr.step = "0.01"; lr.value = "0.10"; lr.className = "w-range";
    const btn = el("button", "w-chip on", "▶ Restart");
    controls.append(lrLabel, lr, btn); host.appendChild(controls);
    const cv = el("canvas"); cv.width = 640; cv.height = 300; host.appendChild(cv);
    const ctx = cv.getContext("2d");
    const f = (x) => 0.15 * x ** 4 - 0.5 * x ** 2 + 0.2 * x + 1;   // double-well
    const g = (x) => 0.6 * x ** 3 - x + 0.2;
    let x = -2.4, t = 0, raf = null;
    lr.oninput = () => { lrLabel.textContent = "lr = " + (+lr.value).toFixed(2); };
    btn.onclick = start;
    function start() { cancelAnimationFrame(raf); x = -2.4 + Math.random() * 0.2; t = 0; step(); }
    function step() {
      x = x - (+lr.value) * g(x); t++;
      draw();
      if (t < 60) raf = requestAnimationFrame(() => setTimeout(step, 90));
    }
    function draw() {
      const W = cv.width, H = cv.height, xr = 3, ymin = 0, ymax = 3;
      ctx.clearRect(0, 0, W, H);
      const X = (xx) => (xx + xr) / (2 * xr) * W, Y = (yy) => H - (yy - ymin) / (ymax - ymin) * H;
      ctx.strokeStyle = cssv("--accent-2"); ctx.lineWidth = 2.5; ctx.beginPath();
      for (let px = 0; px <= W; px++) { const xx = px / W * 2 * xr - xr; px === 0 ? ctx.moveTo(px, Y(f(xx))) : ctx.lineTo(px, Y(f(xx))); }
      ctx.stroke();
      ctx.fillStyle = cssv("--accent");
      ctx.beginPath(); ctx.arc(X(x), Y(f(x)), 7, 0, 7); ctx.fill();
      ctx.fillStyle = cssv("--text-faint"); ctx.font = "12px Inter";
      ctx.fillText(`step ${t}   x=${x.toFixed(2)}   loss=${f(x).toFixed(3)}`, 12, 20);
      if (+lr.value > 0.9) { ctx.fillStyle = "#dc2626"; ctx.fillText("⚠ lr too high → divergence / oscillation", 12, 38); }
    }
    start(); host._redraw = draw;
  };

  /* ---------- LR schedules ---------- */
  registry["lr-schedule"] = (host) => {
    host.appendChild(el("div", "w-title", "Learning-rate schedules"));
    const cv = el("canvas"); cv.width = 640; cv.height = 280; host.appendChild(cv);
    const ctx = cv.getContext("2d");
    const T = 1000, warm = 100, base = 1;
    const sched = {
      "Cosine + warmup": (t) => t < warm ? base * t / warm : base * 0.5 * (1 + Math.cos(Math.PI * (t - warm) / (T - warm))),
      "Step decay": (t) => base * Math.pow(0.3, Math.floor(t / 300)),
      "Linear warmup+decay": (t) => t < warm ? base * t / warm : base * (1 - (t - warm) / (T - warm)),
      "Constant": () => base,
    };
    const colors = [cssv("--accent"), cssv("--accent-2"), cssv("--accent-3"), cssv("--text-faint")];
    const controls = el("div", "w-controls");
    const on = new Set(Object.keys(sched));
    Object.keys(sched).forEach((k, i) => { const b = el("button", "w-chip on", k); b.style.borderColor = colors[i]; b.onclick = () => { on.has(k) ? on.delete(k) : on.add(k); b.classList.toggle("on"); draw(); }; controls.appendChild(b); });
    host.insertBefore(controls, cv);
    function draw() {
      const W = cv.width, H = cv.height;
      ctx.clearRect(0, 0, W, H);
      ctx.strokeStyle = cssv("--border"); ctx.beginPath(); ctx.moveTo(0, H - 20); ctx.lineTo(W, H - 20); ctx.stroke();
      Object.keys(sched).forEach((k, i) => {
        if (!on.has(k)) return;
        ctx.strokeStyle = colors[i]; ctx.lineWidth = 2.5; ctx.beginPath();
        for (let t = 0; t <= T; t++) { const px = t / T * W, py = (H - 20) - sched[k](t) * (H - 40); t === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py); }
        ctx.stroke();
      });
    }
    draw(); host._redraw = draw;
  };

  /* ---------- Precision/Recall threshold ---------- */
  registry["metrics-threshold"] = (host) => {
    host.appendChild(el("div", "w-title", "Precision / Recall vs. threshold"));
    const controls = el("div", "w-controls");
    const lab = el("span", "w-label", "threshold = 0.50");
    const th = el("input"); th.type = "range"; th.min = "0"; th.max = "1"; th.step = "0.01"; th.value = "0.5"; th.className = "w-range";
    controls.append(lab, th); host.appendChild(controls);
    const cv = el("canvas"); cv.width = 640; cv.height = 300; host.appendChild(cv);
    const ctx = cv.getContext("2d");
    // two gaussians: negatives ~N(0.35,.15), positives ~N(0.65,.15)
    const N = 400, pos = [], neg = [];
    let seed = 42; const rnd = () => { seed = (seed * 9301 + 49297) % 233280; return seed / 233280; };
    const gauss = (m, s) => { let u = 0, v = 0; while (!u) u = rnd(); while (!v) v = rnd(); return m + s * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
    for (let i = 0; i < N; i++) { pos.push(Math.min(1, Math.max(0, gauss(0.65, 0.14)))); neg.push(Math.min(1, Math.max(0, gauss(0.35, 0.14)))); }
    const stat = el("div", "w-stat"); host.appendChild(stat);
    th.oninput = draw;
    function draw() {
      const t = +th.value; lab.textContent = "threshold = " + t.toFixed(2);
      const W = cv.width, H = cv.height, bins = 40; const hp = new Array(bins).fill(0), hn = new Array(bins).fill(0);
      pos.forEach((v) => hp[Math.min(bins - 1, v * bins | 0)]++); neg.forEach((v) => hn[Math.min(bins - 1, v * bins | 0)]++);
      const mx = Math.max(...hp, ...hn);
      ctx.clearRect(0, 0, W, H);
      for (let b = 0; b < bins; b++) {
        const x = b / bins * W, w = W / bins;
        ctx.fillStyle = "rgba(99,102,241,.45)"; ctx.fillRect(x, H - 30 - hn[b] / mx * (H - 60), w - 1, hn[b] / mx * (H - 60));
        ctx.fillStyle = "rgba(224,83,63,.5)"; ctx.fillRect(x, H - 30 - hp[b] / mx * (H - 60), w - 1, hp[b] / mx * (H - 60));
      }
      ctx.strokeStyle = cssv("--text-strong"); ctx.lineWidth = 2; ctx.setLineDash([5, 4]);
      ctx.beginPath(); ctx.moveTo(t * W, 10); ctx.lineTo(t * W, H - 30); ctx.stroke(); ctx.setLineDash([]);
      const TP = pos.filter((v) => v >= t).length, FN = pos.length - TP;
      const FP = neg.filter((v) => v >= t).length, TN = neg.length - FP;
      const prec = TP / (TP + FP || 1), rec = TP / (TP + FN || 1), f1 = 2 * prec * rec / (prec + rec || 1);
      ctx.fillStyle = cssv("--text-faint"); ctx.font = "12px Inter";
      ctx.fillText("negatives", 10, 20); ctx.fillText("positives", W - 70, 20);
      stat.innerHTML = `<span style="color:var(--ok)">Precision ${(prec * 100).toFixed(0)}%</span> · <span style="color:var(--accent-2)">Recall ${(rec * 100).toFixed(0)}%</span> · F1 ${(f1 * 100).toFixed(0)}% &nbsp;|&nbsp; TP ${TP} · FP ${FP} · FN ${FN} · TN ${TN}`;
    }
    draw(); host._redraw = draw;
  };

  /* ---------- Scaled dot-product attention ---------- */
  registry.attention = (host) => {
    host.appendChild(el("div", "w-title", "Scaled dot-product attention — hover a query row"));
    const tokens = ["The", "cat", "sat", "on", "the", "mat"];
    const n = tokens.length, d = 8;
    let seed = 7; const rnd = () => { seed = (seed * 9301 + 49297) % 233280; return seed / 233280 - 0.5; };
    const Q = [], K = [];
    for (let i = 0; i < n; i++) { Q.push(Array.from({ length: d }, rnd)); K.push(Array.from({ length: d }, rnd)); }
    // bias so nearby + repeated tokens attend
    const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
    const scores = Q.map((q, i) => K.map((k, j) => dot(q, k) / Math.sqrt(d) + (i === j ? 1.2 : 0) - Math.abs(i - j) * 0.25 + (tokens[i] === tokens[j] ? 0.8 : 0)));
    const soft = (r) => { const m = Math.max(...r); const e = r.map((x) => Math.exp(x - m)); const s = e.reduce((a, b) => a + b, 0); return e.map((x) => x / s); };
    const A = scores.map(soft);
    const grid = el("div", "w-attn");
    grid.style.gridTemplateColumns = `70px repeat(${n}, 1fr)`;
    grid.appendChild(el("div", "w-attn-corner", "q ＼ k"));
    tokens.forEach((t) => grid.appendChild(el("div", "w-attn-h", t)));
    for (let i = 0; i < n; i++) {
      grid.appendChild(el("div", "w-attn-h", tokens[i]));
      for (let j = 0; j < n; j++) {
        const c = el("div", "w-attn-cell", A[i][j].toFixed(2));
        c.style.background = `rgba(224,83,63,${A[i][j].toFixed(3)})`;
        c.dataset.row = i;
        grid.appendChild(c);
      }
    }
    grid.addEventListener("mouseover", (e) => { const r = e.target.dataset.row; if (r == null) return; grid.querySelectorAll(".w-attn-cell").forEach((c) => c.style.outline = c.dataset.row === r ? "2px solid var(--accent-2)" : "none"); });
    host.appendChild(grid);
    host.appendChild(el("div", "w-caption", "Each row is a softmax over keys → sums to 1. Darker = more attention. Note the diagonal (self) and the two “the” tokens attending to each other."));
  };

  /* ---------- Runnable code lab (Pyodide, lazy-loaded on first Run) ---------- */
  const PYODIDE_BASE = "https://cdn.jsdelivr.net/pyodide/v0.26.2/full/";
  let _pyodide = null;
  function ensurePyodide() {
    if (_pyodide) return _pyodide;
    _pyodide = new Promise((resolve, reject) => {
      const go = () => window.loadPyodide({ indexURL: PYODIDE_BASE }).then(resolve, reject);
      if (window.loadPyodide) return go();
      const s = document.createElement("script");
      s.src = PYODIDE_BASE + "pyodide.js";
      s.onload = go;
      s.onerror = () => { _pyodide = null; reject(new Error("Could not load the Python runtime (network blocked?).")); };
      document.head.appendChild(s);
    });
    return _pyodide;
  }
  function jsonEq(a, b, unordered) {
    if (unordered && Array.isArray(a) && Array.isArray(b)) {
      const sa = a.map((x) => JSON.stringify(x)).sort(), sb = b.map((x) => JSON.stringify(x)).sort();
      return JSON.stringify(sa) === JSON.stringify(sb);
    }
    return JSON.stringify(a) === JSON.stringify(b);
  }
  const fmt = (v) => JSON.stringify(v);

  registry.code = (host) => {
    let cfg;
    try { cfg = JSON.parse(host.querySelector("script.code-config").textContent); }
    catch (e) { host.innerHTML = "<em>code widget: invalid config</em>"; return; }
    host.innerHTML = "";
    if (cfg.title) host.appendChild(el("div", "w-title", cfg.title));
    const editor = el("textarea", "w-code-editor");
    editor.spellcheck = false;
    editor.value = cfg.starter || "";
    editor.rows = Math.max(6, (cfg.starter || "").split("\n").length + 1);
    editor.addEventListener("keydown", (e) => {
      if (e.key === "Tab") {
        e.preventDefault();
        const a = editor.selectionStart, b = editor.selectionEnd;
        editor.value = editor.value.slice(0, a) + "    " + editor.value.slice(b);
        editor.selectionStart = editor.selectionEnd = a + 4;
      }
    });
    host.appendChild(editor);
    const bar = el("div", "w-code-bar");
    const runBtn = el("button", "w-chip on", "▶ Run tests");
    const resetBtn = el("button", "w-chip", "Reset");
    const solBtn = el("button", "w-chip", "Solution");
    bar.append(runBtn, resetBtn);
    if (cfg.solution) bar.append(solBtn);
    host.appendChild(bar);
    const out = el("div", "w-code-out");
    host.appendChild(out);
    const solWrap = el("div");
    host.appendChild(solWrap);

    resetBtn.onclick = () => { editor.value = cfg.starter || ""; out.className = "w-code-out"; out.textContent = ""; };
    solBtn.onclick = () => {
      if (solWrap.firstChild) { solWrap.innerHTML = ""; solBtn.textContent = "Solution"; return; }
      const pre = el("pre"); const c = document.createElement("code");
      c.className = "language-python"; c.textContent = cfg.solution;
      pre.appendChild(c); solWrap.appendChild(pre);
      if (window.hljs) { try { hljs.highlightElement(c); } catch (e) {} }
      solBtn.textContent = "Hide solution";
    };
    runBtn.onclick = async () => {
      runBtn.disabled = true;
      out.className = "w-code-out"; out.textContent = "Loading Python… (first run downloads the runtime, ~10 MB — later runs are instant)";
      let py;
      try { py = await ensurePyodide(); }
      catch (e) { out.className = "w-code-out err"; out.textContent = "⚠ " + e.message; runBtn.disabled = false; return; }
      try { await py.runPythonAsync(editor.value); }
      catch (e) { out.className = "w-code-out err"; out.textContent = "❌ Your code raised an error:\n" + (e.message || e); runBtn.disabled = false; return; }
      const fn = py.globals.get(cfg.func);
      if (!fn) { out.className = "w-code-out err"; out.textContent = "❌ No function named `" + cfg.func + "` was defined."; runBtn.disabled = false; return; }
      let pass = 0; const fails = [];
      (cfg.tests || []).forEach((tc) => {
        const proxies = [];
        try {
          const args = (tc.args || []).map((a) => { const p = py.toPy(a); if (p && p.destroy) proxies.push(p); return p; });
          let r = fn(...args);
          if (r && r.toJs) r = r.toJs();
          const ok = jsonEq(r, tc.expect, tc.unordered);
          if (ok) pass++;
          else fails.push("  ✗ " + cfg.func + "(" + (tc.args || []).map(fmt).join(", ") + ") → got " + fmt(r) + ", want " + fmt(tc.expect));
        } catch (e) {
          fails.push("  ✗ " + cfg.func + "(" + (tc.args || []).map(fmt).join(", ") + ") → error: " + (e.message || e));
        }
        proxies.forEach((p) => { try { p.destroy(); } catch (e) {} });
      });
      try { if (fn.destroy) fn.destroy(); } catch (e) {}
      const total = (cfg.tests || []).length;
      if (pass === total) { out.className = "w-code-out ok"; out.textContent = "✓ All " + total + " tests passed. Well done."; }
      else { out.className = "w-code-out err"; out.textContent = pass + "/" + total + " passed\n" + fails.slice(0, 8).join("\n") + (fails.length > 8 ? "\n  … " + (fails.length - 8) + " more" : ""); }
      runBtn.disabled = false;
    };
  };

  function init(root) {
    (root || document).querySelectorAll(".widget:not([data-ready])").forEach((host) => {
      const name = host.dataset.widget;
      if (registry[name]) { host.setAttribute("data-ready", "1"); host.classList.add("w-host"); try { registry[name](host); } catch (e) { console.warn("widget", name, e); } }
    });
  }
  // Re-draw canvas widgets on theme change (colors are cached at draw time).
  window.addEventListener("mlcodex-theme", () => document.querySelectorAll(".w-host").forEach((h) => h._redraw && h._redraw()));
  window.Widgets = { init, registry };
})();
