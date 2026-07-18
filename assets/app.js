/* =========================================================================
   ML Interview Codex — application engine
   Client-side Markdown book: routing, rendering, math, diagrams, search, TOC.
   ========================================================================= */
(function () {
  "use strict";

  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const CONTENT_DIR = "content/";
  const contentEl = $("#content");
  const cache = new Map();       // file -> markdown text
  let searchIndex = null;        // built lazily

  /* ---------------- Theme ---------------- */
  const THEME_KEY = "mlcodex-theme";
  function currentTheme() { return document.documentElement.getAttribute("data-theme"); }
  function applyTheme(t) {
    document.documentElement.setAttribute("data-theme", t);
    localStorage.setItem(THEME_KEY, t);
    $("#hljs-theme").href = t === "light"
      ? "https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.9.0/build/styles/github.min.css"
      : "https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.9.0/build/styles/github-dark.min.css";
    if (window.mermaid) {
      mermaid.initialize({ startOnLoad: false, theme: t === "light" ? "neutral" : "dark", securityLevel: "loose", fontFamily: "Inter, sans-serif" });
    }
  }
  (function initTheme() {
    const saved = localStorage.getItem(THEME_KEY);
    const prefersLight = window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches;
    applyTheme(saved || (prefersLight ? "light" : "dark"));
  })();
  $("#theme-toggle").addEventListener("click", () => {
    const next = currentTheme() === "light" ? "dark" : "light";
    applyTheme(next);
    if (currentRoute) render(currentRoute.file, { rerender: true }); // recolor diagrams
  });

  /* ---------------- Language (EN default; KO keeps technical terms in English) ---------------- */
  const LANG_KEY = "mlcodex-lang";
  const I18N = {
    en: {
      onThisPage: "On this page",
      searchPlaceholder: "Search the whole book…",
      searchHint: 'Tip: search covers every chapter. Press <kbd>/</kbd> anywhere to open.',
      prev: "Previous", next: "Next",
      footer: "Built by Beomyoung Kim · content is a living document, revised continuously.",
      indexing: "Indexing the book…",
      notFound: (f) => `<h1>Not found</h1><p>The chapter <code>${f}</code> isn't written yet. This is a living book — check the <a href="#/resources/changelog">changelog</a> or jump back to the <a href="#/">introduction</a>.</p>`,
      noMatch: (q) => `No matches for “${q}”.`,
      fallbackNote: "🌏 This chapter isn't translated into Korean yet — showing the English version.",
    },
    ko: {
      onThisPage: "목차",
      searchPlaceholder: "책 전체 검색…",
      searchHint: '팁: 모든 챕터를 검색합니다. 어디서나 <kbd>/</kbd> 를 눌러 여세요.',
      prev: "이전", next: "다음",
      footer: "Beomyoung Kim 제작 · 지속적으로 갱신되는 살아있는 문서입니다.",
      indexing: "책을 색인하는 중…",
      notFound: (f) => `<h1>찾을 수 없음</h1><p><code>${f}</code> 챕터는 아직 작성되지 않았습니다. 살아있는 문서이니 <a href="#/resources/changelog">변경 이력</a>을 확인하거나 <a href="#/">소개</a>로 돌아가세요.</p>`,
      noMatch: (q) => `“${q}” 에 대한 검색 결과가 없습니다.`,
      fallbackNote: "🌏 이 챕터는 아직 한국어로 번역되지 않아 영어 원문을 표시합니다.",
    },
  };
  function currentLang() { return document.documentElement.getAttribute("data-lang") || "en"; }
  function t() { return I18N[currentLang()] || I18N.en; }
  function chTitle(ch) { return (currentLang() === "ko" && ch && ch.titleKo) ? ch.titleKo : (ch ? ch.title : ""); }
  function partTitle(p) { return (currentLang() === "ko" && p.titleKo) ? p.titleKo : p.title; }
  function applyI18n() {
    const s = t();
    const si = $("#search-input"); if (si) si.placeholder = s.searchPlaceholder;
    const sh = $(".search-hint"); if (sh) sh.innerHTML = s.searchHint;
    const pt = $(".page-toc-title"); if (pt) pt.textContent = s.onThisPage;
    const ff = $("#page-footer > div"); if (ff) ff.textContent = s.footer;
  }
  function applyLang(l) {
    document.documentElement.setAttribute("data-lang", l);
    document.documentElement.lang = l;
    localStorage.setItem(LANG_KEY, l);
    applyI18n();
  }
  (function initLang() { applyLang(localStorage.getItem(LANG_KEY) || "en"); })();
  $("#lang-toggle").addEventListener("click", () => {
    applyLang(currentLang() === "ko" ? "en" : "ko");
    searchIndex = null;             // rebuild search index in the new language
    buildSidebar();
    if (currentRoute) render(currentRoute.file);
  });

  /* ---------------- Sidebar ---------------- */
  function buildSidebar() {
    const toc = $("#toc");
    toc.innerHTML = "";
    window.BOOK.parts.forEach((part, pi) => {
      const wrap = document.createElement("div");
      wrap.className = "part";
      wrap.dataset.part = pi;
      const btn = document.createElement("button");
      btn.className = "part-title";
      btn.innerHTML = `<span class="part-emoji">${part.emoji || ""}</span><span>${partTitle(part)}</span><span class="chev">▾</span>`;
      btn.addEventListener("click", () => wrap.classList.toggle("collapsed"));
      const ul = document.createElement("ul");
      ul.className = "chapter-list";
      part.chapters.forEach((ch) => {
        const li = document.createElement("li");
        const a = document.createElement("a");
        a.href = "#/" + ch.file;
        a.dataset.file = ch.file;
        const badge = ch.badge ? ` <span class="badge badge-${ch.badge === "2026" ? "2026" : "new"}" style="font-size:9px;padding:0 6px">${ch.badge === "2026" ? "2026" : "NEW"}</span>` : "";
        a.innerHTML = `${chTitle(ch)}${badge}`;
        li.appendChild(a);
        ul.appendChild(li);
      });
      wrap.appendChild(btn);
      wrap.appendChild(ul);
      toc.appendChild(wrap);
    });
    $("#build-meta").textContent = `v${window.BOOK.updated} · ${window.BOOK_FLAT.length} chapters`;
  }
  function setActiveNav(file) {
    $$("#toc a").forEach((a) => a.classList.toggle("active", a.dataset.file === file));
    // expand the part containing the active chapter, collapse hint retained
    const active = $(`#toc a[data-file="${CSS.escape(file)}"]`);
    if (active) {
      const part = active.closest(".part");
      if (part) part.classList.remove("collapsed");
    }
  }

  /* ---------------- Math protection (keep LaTeX away from Markdown) ---------------- */
  function protectMath(md) {
    const store = [];
    const stash = (tex, display) => {
      store.push({ tex, display });
      return `KTXMATH${store.length - 1}ENDKTX`;
    };
    // Order matters: block first.
    md = md.replace(/\$\$([\s\S]+?)\$\$/g, (_, t) => stash(t, true));
    md = md.replace(/\\\[([\s\S]+?)\\\]/g, (_, t) => stash(t, true));
    md = md.replace(/\\\(([\s\S]+?)\\\)/g, (_, t) => stash(t, false));
    // inline $...$ — avoid $ next to space/digit boundaries to skip currency
    md = md.replace(/(?<![\\$])\$(?!\s)([^\n$]+?)(?<!\s)\$(?!\d)/g, (_, t) => stash(t, false));
    return { md, store };
  }
  function restoreMath(html, store) {
    return html.replace(/KTXMATH(\d+)ENDKTX/g, (_, i) => {
      const { tex, display } = store[+i];
      try {
        return katex.renderToString(tex.trim(), { displayMode: display, throwOnError: false, output: "html" });
      } catch (e) {
        return `<code>${tex}</code>`;
      }
    });
  }

  /* ---------------- Markdown rendering ---------------- */
  marked.setOptions({ gfm: true, breaks: false, headerIds: false, mangle: false });

  function slugify(s) {
    return s.toLowerCase().replace(/<[^>]+>/g, "").replace(/[^\w\s-]/g, "").trim().replace(/\s+/g, "-").slice(0, 60);
  }

  function renderMarkdown(md) {
    const { md: protectedMd, store } = protectMath(md);
    let html = marked.parse(protectedMd);
    html = restoreMath(html, store);
    return html;
  }

  /* Convert GitHub-style callouts (> [!NOTE] ...) into styled boxes. */
  function processCallouts(root) {
    $$("blockquote", root).forEach((bq) => {
      const first = bq.querySelector("p");
      if (!first) return;
      const m = first.innerHTML.match(/^\[!(\w+)\]\s*(.*)$/s);
      if (!m) return;
      const type = m[1].toLowerCase();
      const known = { note: 1, tip: 1, warning: 1, danger: 1, question: 1, example: 1, quote: 1, important: 1, caution: 1 };
      const cls = { important: "note", caution: "warning" }[type] || (known[type] ? type : "note");
      const div = document.createElement("div");
      div.className = "callout callout-" + cls;
      const titleText = m[2].trim();
      first.innerHTML = m[2];
      const title = document.createElement("div");
      title.className = "callout-title";
      title.textContent = titleText && titleText.length < 60 && !/[.!?]$/.test(titleText) ? titleText : (cls[0].toUpperCase() + cls.slice(1));
      if (titleText && title.textContent === titleText) first.remove();
      div.appendChild(title);
      while (bq.firstChild) div.appendChild(bq.firstChild);
      bq.replaceWith(div);
    });
  }

  /* Convert ```mermaid code fences into live diagrams. */
  function processMermaid(root) {
    $$("pre code.language-mermaid", root).forEach((code) => {
      const div = document.createElement("div");
      div.className = "mermaid";
      div.textContent = code.textContent;
      code.closest("pre").replaceWith(div);
    });
    const nodes = $$(".mermaid", root);
    if (nodes.length && window.mermaid) {
      try { mermaid.run({ nodes }); } catch (e) { console.warn("mermaid", e); }
    }
  }

  function processCode(root) {
    $$("pre code", root).forEach((code) => {
      if (code.classList.contains("language-mermaid")) return;
      try { hljs.highlightElement(code); } catch (e) {}
      const pre = code.closest("pre");
      const btn = document.createElement("button");
      btn.className = "code-copy";
      btn.textContent = "Copy";
      btn.addEventListener("click", () => {
        navigator.clipboard.writeText(code.textContent).then(() => {
          btn.textContent = "Copied!"; btn.classList.add("copied");
          setTimeout(() => { btn.textContent = "Copy"; btn.classList.remove("copied"); }, 1500);
        });
      });
      pre.appendChild(btn);
    });
  }

  function processTables(root) {
    $$("table", root).forEach((t) => {
      if (t.parentElement.classList.contains("table-wrap")) return;
      const wrap = document.createElement("div");
      wrap.className = "table-wrap";
      t.replaceWith(wrap); wrap.appendChild(t);
    });
  }

  function processHeadings(root) {
    const toc = [];
    $$("h2, h3", root).forEach((h) => {
      const id = h.id || slugify(h.textContent);
      h.id = id;
      const a = document.createElement("a");
      a.className = "heading-anchor"; a.href = "#/" + currentRoute.file + "::" + id; a.textContent = "#";
      h.appendChild(a);
      toc.push({ id, text: h.textContent.replace(/#$/, "").trim(), level: h.tagName === "H2" ? 2 : 3 });
    });
    return toc;
  }

  /* Rewrite internal links so relative content links route correctly. */
  function processLinks(root) {
    $$("a", root).forEach((a) => {
      const href = a.getAttribute("href") || "";
      if (/^https?:|^mailto:|^#\//.test(href)) {
        if (/^https?:/.test(href)) { a.target = "_blank"; a.rel = "noopener"; }
        return;
      }
      // support markdown links like (content/foo.md) or (foo)
      if (href.endsWith(".md")) a.setAttribute("href", "#/" + href.replace(/^content\//, "").replace(/\.md$/, ""));
    });
  }

  /* ---------------- On-page TOC + scrollspy ---------------- */
  let spyObserver = null;
  function buildPageToc(toc) {
    const nav = $("#page-toc-nav");
    nav.innerHTML = "";
    if (toc.length < 2) { $("#page-toc").style.visibility = "hidden"; return; }
    $("#page-toc").style.visibility = "visible";
    toc.forEach((h) => {
      const a = document.createElement("a");
      a.href = "#" + h.id; a.textContent = h.text; a.dataset.target = h.id;
      if (h.level === 3) a.classList.add("h3");
      a.addEventListener("click", (e) => {
        e.preventDefault();
        const el = document.getElementById(h.id);
        if (el) el.scrollIntoView({ behavior: "smooth" });
      });
      nav.appendChild(a);
    });
    if (spyObserver) spyObserver.disconnect();
    const links = $$("#page-toc-nav a");
    spyObserver = new IntersectionObserver((entries) => {
      entries.forEach((en) => {
        if (en.isIntersecting) {
          links.forEach((l) => l.classList.toggle("active", l.dataset.target === en.target.id));
        }
      });
    }, { rootMargin: "-80px 0px -70% 0px" });
    toc.forEach((h) => { const el = document.getElementById(h.id); if (el) spyObserver.observe(el); });
  }

  /* ---------------- Pager ---------------- */
  function buildPager(file) {
    const idx = window.BOOK_FLAT.findIndex((c) => c.file === file);
    const pager = $("#pager");
    pager.innerHTML = "";
    if (idx < 0) return;
    const prev = window.BOOK_FLAT[idx - 1], next = window.BOOK_FLAT[idx + 1];
    if (prev) pager.innerHTML += `<a class="prev" href="#/${prev.file}"><div class="dir">‹ ${t().prev}</div><div class="ttl">${chTitle(prev)}</div></a>`;
    else pager.innerHTML += `<span style="flex:1"></span>`;
    if (next) pager.innerHTML += `<a class="next" href="#/${next.file}"><div class="dir">${t().next} ›</div><div class="ttl">${chTitle(next)}</div></a>`;
  }

  /* ---------------- Fetch + render a page ---------------- */
  async function fetchMd(file) {
    const lang = currentLang();
    const key = lang + ":" + file;
    if (cache.has(key)) return cache.get(key);
    if (lang === "ko") {
      // Prefer a Korean translation; fall back to English if it doesn't exist yet.
      const rk = await fetch(CONTENT_DIR + file + ".ko.md", { cache: "no-cache" });
      if (rk.ok) { const tk = await rk.text(); cache.set(key, { text: tk, translated: true }); return { text: tk, translated: true }; }
    }
    const res = await fetch(CONTENT_DIR + file + ".md", { cache: "no-cache" });
    if (!res.ok) throw new Error("404");
    const text = await res.text();
    const val = { text, translated: lang === "en" };
    cache.set(key, val);
    return val;
  }

  let currentRoute = { file: "index", anchor: null };

  async function render(file, opts = {}) {
    currentRoute.file = file;
    if (!opts.rerender) { contentEl.innerHTML = `<div class="loading"><div class="spinner"></div></div>`; }
    let doc;
    try {
      doc = await fetchMd(file);
    } catch (e) {
      contentEl.innerHTML = t().notFound(file);
      $("#pager").innerHTML = ""; buildPageToc([]);
      return;
    }
    // strip optional YAML frontmatter
    const md = doc.text.replace(/^---\n[\s\S]*?\n---\n/, "");
    let html = renderMarkdown(md);
    if (currentLang() === "ko" && !doc.translated) {
      html = `<div class="callout callout-note lang-fallback">${t().fallbackNote}</div>` + html;
    }
    contentEl.innerHTML = html;
    processLinks(contentEl);
    processCallouts(contentEl);
    processMermaid(contentEl);
    processCode(contentEl);
    processTables(contentEl);
    const toc = processHeadings(contentEl);
    if (window.Widgets) window.Widgets.init(contentEl);
    buildPageToc(toc);
    buildPager(file);
    setActiveNav(file);
    const meta = window.BOOK_BY_FILE[file];
    document.title = (meta ? chTitle(meta) + " · " : "") + "ML Interview Codex";

    if (opts.rerender) return;
    // scroll to anchor or top
    if (currentRoute.anchor) {
      requestAnimationFrame(() => { const el = document.getElementById(currentRoute.anchor); if (el) el.scrollIntoView(); });
    } else {
      window.scrollTo(0, 0);
    }
  }

  /* ---------------- Router ---------------- */
  function parseHash() {
    let h = location.hash.replace(/^#\/?/, "");
    let anchor = null;
    if (h.includes("::")) { const [f, a] = h.split("::"); h = f; anchor = a; }
    if (!h) h = "index";
    return { file: h, anchor };
  }
  function route() {
    const { file, anchor } = parseHash();
    currentRoute.anchor = anchor;
    if (!window.BOOK_BY_FILE[file] && file !== "index") {
      // still try to fetch (allows non-manifest files)
    }
    closeNav();
    render(file);
  }
  window.addEventListener("hashchange", route);

  /* ---------------- Progress bar ---------------- */
  window.addEventListener("scroll", () => {
    const h = document.documentElement;
    const scrolled = (h.scrollTop) / (h.scrollHeight - h.clientHeight || 1);
    $("#progress-bar").style.width = Math.min(100, Math.max(0, scrolled * 100)) + "%";
  }, { passive: true });

  /* ---------------- Mobile nav ---------------- */
  function openNav() { document.body.classList.add("nav-open"); }
  function closeNav() { document.body.classList.remove("nav-open"); }
  $("#menu-toggle").addEventListener("click", () => document.body.classList.toggle("nav-open"));
  $("#scrim").addEventListener("click", closeNav);

  /* ---------------- Search ---------------- */
  const searchModal = $("#search-modal");
  const searchInput = $("#search-input");
  const searchResults = $("#search-results");

  function stripMd(md) {
    return md
      .replace(/^---\n[\s\S]*?\n---\n/, "")
      .replace(/```[\s\S]*?```/g, " ")
      .replace(/`[^`]*`/g, " ")
      .replace(/!\[[^\]]*\]\([^)]*\)/g, " ")
      .replace(/\[([^\]]*)\]\([^)]*\)/g, "$1")
      .replace(/[#>*_~|]/g, " ")
      .replace(/\$[^$]*\$/g, " ")
      .replace(/\s+/g, " ")
      .trim();
  }
  async function buildSearchIndex() {
    if (searchIndex) return searchIndex;
    searchResults.innerHTML = `<div class="search-empty">${t().indexing}</div>`;
    const entries = await Promise.all(window.BOOK_FLAT.map(async (ch) => {
      try {
        const doc = await fetchMd(ch.file);
        return { ...ch, text: stripMd(doc.text) };
      } catch (e) { return null; }
    }));
    searchIndex = entries.filter(Boolean);
    return searchIndex;
  }
  function escapeHtml(s) { return s.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c])); }
  function snippet(text, terms) {
    const low = text.toLowerCase();
    let pos = -1;
    for (const t of terms) { const p = low.indexOf(t); if (p >= 0 && (pos < 0 || p < pos)) pos = p; }
    if (pos < 0) pos = 0;
    const start = Math.max(0, pos - 40);
    let snip = text.slice(start, start + 160);
    snip = escapeHtml(snip);
    terms.forEach((t) => { snip = snip.replace(new RegExp("(" + t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + ")", "ig"), "<mark>$1</mark>"); });
    return (start > 0 ? "…" : "") + snip + "…";
  }
  async function runSearch(q) {
    q = q.trim().toLowerCase();
    if (!q) { searchResults.innerHTML = ""; return; }
    const idx = await buildSearchIndex();
    const terms = q.split(/\s+/).filter(Boolean);
    const scored = [];
    idx.forEach((e) => {
      const hayTitle = (e.title + " " + (e.titleKo || "")).toLowerCase(), hayText = e.text.toLowerCase();
      let score = 0, allInText = true;
      terms.forEach((t) => {
        if (hayTitle.includes(t)) score += 12;
        const c = hayText.split(t).length - 1;
        if (c === 0 && !hayTitle.includes(t)) allInText = false;
        score += Math.min(c, 8);
      });
      if (allInText && score > 0) scored.push({ e, score });
    });
    scored.sort((a, b) => b.score - a.score);
    if (!scored.length) { searchResults.innerHTML = `<div class="search-empty">${t().noMatch(escapeHtml(q))}</div>`; return; }
    searchResults.innerHTML = scored.slice(0, 20).map(({ e }) =>
      `<a class="search-result" href="#/${e.file}"><div class="sr-part">${e.partEmoji} ${currentLang() === "ko" && e.partKo ? e.partKo : e.part}</div><div class="sr-title">${chTitle(e)}</div><div class="sr-snip">${snippet(e.text, terms)}</div></a>`
    ).join("");
    $$(".search-result").forEach((a) => a.addEventListener("click", closeSearch));
  }
  let searchTimer = null;
  searchInput.addEventListener("input", () => { clearTimeout(searchTimer); searchTimer = setTimeout(() => runSearch(searchInput.value), 120); });
  function openSearch() { searchModal.hidden = false; searchInput.focus(); searchInput.select(); buildSearchIndex(); }
  function closeSearch() { searchModal.hidden = true; }
  $("#search-open").addEventListener("click", openSearch);
  searchModal.addEventListener("click", (e) => { if (e.target === searchModal) closeSearch(); });
  document.addEventListener("keydown", (e) => {
    if (e.key === "/" && !/input|textarea/i.test(document.activeElement.tagName)) { e.preventDefault(); openSearch(); }
    else if (e.key === "Escape") closeSearch();
    else if (e.key === "Enter" && !searchModal.hidden) { const first = $(".search-result"); if (first) { location.hash = first.getAttribute("href"); closeSearch(); } }
  });

  /* ---------------- Boot ---------------- */
  buildSidebar();
  route();
})();
