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
  const cache = new Map();       // language:file -> markdown response
  let searchIndex = null;        // built lazily
  let searchIndexLang = null;
  let searchIndexPromise = null;
  let searchIndexGeneration = 0;

  /* ---------------- Theme ---------------- */
  const THEME_KEY = "mlcodex-theme";
  function currentTheme() { return document.documentElement.getAttribute("data-theme"); }
  function applyTheme(t) {
    document.documentElement.setAttribute("data-theme", t);
    document.documentElement.style.colorScheme = t;
    localStorage.setItem(THEME_KEY, t);
    const toggle = $("#theme-toggle");
    if (toggle) toggle.setAttribute("aria-pressed", String(t === "dark"));
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
    applyI18n();
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
      searchStart: "Type a topic, model, interview stage, or chapter title.",
      resultCount: (shown, total) => shown < total ? `Top ${shown} of ${total} results` : `${total} result${total === 1 ? "" : "s"}`,
      searchDialog: "Search the book",
      searchButton: "Search",
      searchResults: "Search results",
      searchClose: "Close search",
      bookPosition: "Book position",
      skipLink: "Skip to main content",
      sidebarLabel: "Table of contents",
      chaptersLabel: "Book chapters",
      menuOpen: "Open navigation",
      menuClose: "Close navigation",
      themeToLight: "Switch to light theme",
      themeToDark: "Switch to dark theme",
      langToggle: "Switch to Korean",
      sampleAnswer: "Sample answer",
      conceptCode: "Concept code",
      copy: "Copy code",
      copied: "Copied",
      codeRegion: "Scrollable code example",
      tableRegion: "Scrollable table",
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
      searchStart: "주제, 모델, 면접 단계 또는 챕터 제목을 입력하세요.",
      resultCount: (shown, total) => shown < total ? `${total}개 중 상위 ${shown}개` : `검색 결과 ${total}개`,
      searchDialog: "책 전체 검색",
      searchButton: "검색",
      searchResults: "검색 결과",
      searchClose: "검색 닫기",
      bookPosition: "책에서의 위치",
      skipLink: "본문으로 건너뛰기",
      sidebarLabel: "목차",
      chaptersLabel: "책 챕터",
      menuOpen: "목차 열기",
      menuClose: "목차 닫기",
      themeToLight: "라이트 테마로 전환",
      themeToDark: "다크 테마로 전환",
      langToggle: "영어로 전환",
      sampleAnswer: "예시 답변",
      conceptCode: "개념 코드",
      copy: "코드 복사",
      copied: "복사됨",
      codeRegion: "가로로 스크롤 가능한 코드 예시",
      tableRegion: "가로로 스크롤 가능한 표",
      notFound: (f) => `<h1>찾을 수 없음</h1><p><code>${f}</code> 챕터는 아직 작성되지 않았습니다. 살아있는 문서이니 <a href="#/resources/changelog">변경 이력</a>을 확인하거나 <a href="#/">소개</a>로 돌아가세요.</p>`,
      noMatch: (q) => `“${q}” 에 대한 검색 결과가 없습니다.`,
      fallbackNote: "🌏 이 챕터는 아직 한국어로 번역되지 않아 영어 원문을 표시합니다.",
    },
  };
  function currentLang() { return document.documentElement.getAttribute("data-lang") || "en"; }
  function t() { return I18N[currentLang()] || I18N.en; }
  function chTitle(ch) { return (currentLang() === "ko" && ch && ch.titleKo) ? ch.titleKo : (ch ? ch.title : ""); }
  function partTitle(p) { return (currentLang() === "ko" && p.titleKo) ? p.titleKo : p.title; }
  function updateMenuButton() {
    const button = $("#menu-toggle");
    if (!button) return;
    const open = document.body.classList.contains("nav-open");
    button.setAttribute("aria-expanded", String(open));
    const label = open ? t().menuClose : t().menuOpen;
    button.setAttribute("aria-label", label); button.title = label;
  }
  function applyI18n() {
    const s = t();
    const si = $("#search-input"); if (si) { si.placeholder = s.searchPlaceholder; si.setAttribute("aria-label", s.searchPlaceholder); }
    const sh = $(".search-hint"); if (sh) sh.innerHTML = s.searchHint;
    const pt = $(".page-toc-title"); if (pt) pt.textContent = s.onThisPage;
    const pa = $("#page-toc"); if (pa) pa.setAttribute("aria-label", s.onThisPage);
    const ff = $("#page-footer > div"); if (ff) ff.textContent = s.footer;
    const skip = $(".skip-link"); if (skip) skip.textContent = s.skipLink;
    const side = $("#sidebar"); if (side) side.setAttribute("aria-label", s.sidebarLabel);
    const toc = $("#toc"); if (toc) toc.setAttribute("aria-label", s.chaptersLabel);
    const sp = $("#sidebar-progress-label"); if (sp) sp.textContent = s.bookPosition;
    const ss = $(".sidebar-status"); if (ss) ss.setAttribute("aria-label", s.bookPosition);
    const sd = $("#search-dialog-title"); if (sd) sd.textContent = s.searchDialog;
    const sm = $("#search-modal"); if (sm) sm.setAttribute("aria-label", s.searchDialog);
    const sr = $("#search-results"); if (sr) sr.setAttribute("aria-label", s.searchResults);
    const sc = $("#search-close"); if (sc) { sc.setAttribute("aria-label", s.searchClose); sc.title = s.searchClose; }
    const so = $("#search-open");
    if (so) {
      so.setAttribute("aria-label", s.searchDialog);
      const label = $("span", so); if (label) label.textContent = s.searchButton;
    }
    const lt = $("#lang-toggle"); if (lt) { lt.setAttribute("aria-label", s.langToggle); lt.title = s.langToggle; }
    const tt = $("#theme-toggle");
    if (tt) {
      const label = currentTheme() === "dark" ? s.themeToLight : s.themeToDark;
      tt.setAttribute("aria-label", label); tt.title = label;
    }
    updateMenuButton();
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
    resetSearchIndex();            // rebuild search index in the new language
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
      btn.type = "button";
      btn.id = `part-title-${pi}`;
      btn.setAttribute("aria-controls", `part-chapters-${pi}`);
      btn.setAttribute("aria-expanded", "true");
      btn.innerHTML = `<span class="part-emoji" aria-hidden="true">${escapeHtml(part.emoji || "")}</span><span class="part-name">${escapeHtml(partTitle(part))}</span><span class="part-count" aria-hidden="true">${part.chapters.length}</span><span class="chev" aria-hidden="true">▾</span>`;
      btn.addEventListener("click", () => {
        const collapsed = wrap.classList.toggle("collapsed");
        btn.setAttribute("aria-expanded", String(!collapsed));
      });
      const ul = document.createElement("ul");
      ul.className = "chapter-list";
      ul.id = `part-chapters-${pi}`;
      ul.setAttribute("aria-labelledby", btn.id);
      part.chapters.forEach((ch) => {
        const li = document.createElement("li");
        const a = document.createElement("a");
        a.href = "#/" + ch.file;
        a.dataset.file = ch.file;
        const badgeKey = String(ch.badge || "");
        const badge = badgeKey
          ? ` <span class="badge sidebar-badge badge-${badgeKey === "2026" ? "2026" : "new"}">${escapeHtml(badgeKey.toUpperCase())}</span>`
          : "";
        a.innerHTML = `<span>${escapeHtml(chTitle(ch))}</span>${badge}`;
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
    $$("#toc a").forEach((a) => {
      const active = a.dataset.file === file;
      a.classList.toggle("active", active);
      if (active) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
    });
    $$("#toc .part").forEach((part) => part.classList.remove("current"));
    // expand the part containing the active chapter, collapse hint retained
    const active = $(`#toc a[data-file="${CSS.escape(file)}"]`);
    if (active) {
      const part = active.closest(".part");
      if (part) {
        part.classList.remove("collapsed"); part.classList.add("current");
        const button = $(".part-title", part); if (button) button.setAttribute("aria-expanded", "true");
      }
      requestAnimationFrame(() => active.scrollIntoView({ block: "nearest" }));
    }
    const idx = window.BOOK_FLAT.findIndex((ch) => ch.file === file);
    const count = $("#sidebar-progress-count");
    const title = $("#sidebar-progress-title");
    const track = $("#sidebar-progress-track");
    const fill = $("#sidebar-progress-fill");
    if (idx >= 0) {
      const value = idx + 1, total = window.BOOK_FLAT.length;
      if (count) count.textContent = `${value} / ${total}`;
      if (title) title.textContent = `${window.BOOK_FLAT[idx].partEmoji || ""} ${chTitle(window.BOOK_FLAT[idx])}`.trim();
      if (track) { track.setAttribute("aria-valuemax", String(total)); track.setAttribute("aria-valuenow", String(value)); track.setAttribute("aria-valuetext", `${value} / ${total} · ${chTitle(window.BOOK_FLAT[idx])}`); }
      if (fill) fill.style.width = `${(value / total) * 100}%`;
    } else {
      if (count) count.textContent = `— / ${window.BOOK_FLAT.length}`;
      if (title) title.textContent = file;
      if (fill) fill.style.width = "0%";
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
    const slug = String(s)
      .normalize("NFKC")
      .toLowerCase()
      .replace(/<[^>]+>/g, "")
      .replace(/[^\p{L}\p{N}\s_-]/gu, "")
      .trim()
      .replace(/[\s_]+/g, "-")
      .replace(/-+/g, "-")
      .slice(0, 60);
    return slug || "section";
  }

  function renderMarkdown(md) {
    const { md: protectedMd, store } = protectMath(md);
    let html = marked.parse(protectedMd);
    html = restoreMath(html, store);
    return html;
  }

  /* CommonMark leaves Markdown literal inside same-line raw HTML such as
     <dd>**term** and [link](#/route)</dd>. Re-run only those inline containers
     so authoring helpers render consistently without touching block layout. */
  function processInlineMarkdownInHtml(root) {
    $$(`dt, dd, figcaption`, root).forEach((el) => {
      const html = el.innerHTML;
      const hasRawInline = /\*\*|__|`|\[[^\]]+\]\((?:#\/|https?:\/\/)[^)]+\)|(^|[\s(])\*[^*\n]+\*/m.test(html);
      if (!hasRawInline || el.querySelector(".katex")) return;
      try { el.innerHTML = marked.parseInline(html); } catch (e) {}
    });
  }

  /* Convert GitHub-style callouts (> [!NOTE] ...) into styled boxes. */
  function processCallouts(root) {
    $$("blockquote", root).forEach((bq) => {
      const first = bq.querySelector("p");
      if (!first) return;
      const m = first.innerHTML.match(/^\s*\[!(\w+)\]\s*([^\n]*)(?:\n([\s\S]*))?$/i);
      if (!m) return;
      const type = m[1].toLowerCase();
      const known = { note: 1, tip: 1, warning: 1, danger: 1, question: 1, example: 1, quote: 1, important: 1, caution: 1 };
      const cls = { important: "note", caution: "warning" }[type] || (known[type] ? type : "note");
      const div = document.createElement("div");
      div.className = "callout callout-" + cls;
      const inlineTitleHtml = m[2].trim();
      const remainderHtml = (m[3] || "").trim();
      const titleProbe = document.createElement("span");
      titleProbe.innerHTML = inlineTitleHtml;
      const titleText = titleProbe.textContent.trim();
      const title = document.createElement("div");
      title.className = "callout-title";
      const useInlineTitle = titleText && titleText.length < 60 && !/[.!?]$/.test(titleText);
      if (useInlineTitle) {
        title.innerHTML = inlineTitleHtml;
        if (remainderHtml) first.innerHTML = remainderHtml; else first.remove();
      } else {
        title.textContent = cls[0].toUpperCase() + cls.slice(1);
        const bodyHtml = [inlineTitleHtml, remainderHtml].filter(Boolean).join("\n");
        if (bodyHtml) first.innerHTML = bodyHtml; else first.remove();
      }
      div.appendChild(title);
      while (bq.firstChild) div.appendChild(bq.firstChild);
      bq.replaceWith(div);
    });
  }

  /* Distinguish expandable sample answers from collapsible reference code. */
  function processDetails(root) {
    $$("details", root).forEach((details) => {
      const summary = $(":scope > summary", details);
      if (!summary) return;
      const cleanLabel = summary.textContent.trim();
      if (!details.classList.contains("qa") && /개념\s*코드|concept(?:ual)?\s+code/i.test(summary.textContent)) {
        details.classList.add("concept-code");
      }
      const kind = details.classList.contains("answer-card")
        ? t().sampleAnswer
        : details.classList.contains("concept-code") ? t().conceptCode : "";
      if (kind && !$(".detail-meta", summary)) {
        summary.setAttribute("aria-label", cleanLabel);
        const meta = document.createElement("span");
        meta.className = "detail-meta";
        meta.textContent = kind;
        summary.appendChild(meta);
      }
    });
    $$(".answer-card .qa-body p", root).forEach((p) => {
      const strong = p.firstElementChild;
      if (strong && strong.tagName === "STRONG" && /^(예시|의도|주의|확인|예상|대안|개인화|강한\s+follow-up|example|intent|caution|check|expected|alternative|follow-up)/i.test(strong.textContent.trim())) {
        p.classList.add("answer-meta");
      }
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
      // The pinned highlight.js common build ships JSON but not the JSONC
      // alias. JSON-with-comments still benefits from JSON highlighting and
      // should not emit a console warning on every render.
      if (code.classList.contains("language-jsonc")) {
        code.classList.remove("language-jsonc");
        code.classList.add("language-json");
      }
      try { hljs.highlightElement(code); } catch (e) {}
      const pre = code.closest("pre");
      const btn = document.createElement("button");
      btn.className = "code-copy";
      btn.type = "button";
      btn.textContent = currentLang() === "ko" ? "복사" : "Copy";
      btn.setAttribute("aria-label", t().copy);
      btn.addEventListener("click", () => {
        if (!navigator.clipboard || !navigator.clipboard.writeText) return;
        navigator.clipboard.writeText(code.textContent).then(() => {
          btn.textContent = t().copied; btn.classList.add("copied");
          setTimeout(() => {
            btn.textContent = currentLang() === "ko" ? "복사" : "Copy";
            btn.classList.remove("copied");
          }, 1500);
        }).catch(() => {});
      });
      pre.appendChild(btn);
      requestAnimationFrame(() => {
        if (pre.scrollWidth > pre.clientWidth) {
          pre.tabIndex = 0;
          pre.setAttribute("aria-label", t().codeRegion);
        }
      });
    });
  }

  function processTables(root) {
    $$("table", root).forEach((table) => {
      if (table.parentElement.classList.contains("table-wrap")) return;
      const wrap = document.createElement("div");
      wrap.className = "table-wrap";
      table.replaceWith(wrap); wrap.appendChild(table);
      requestAnimationFrame(() => {
        if (wrap.scrollWidth > wrap.clientWidth) {
          wrap.tabIndex = 0;
          wrap.setAttribute("role", "region");
          wrap.setAttribute("aria-label", t().tableRegion);
        }
      });
    });
  }

  function processHeadings(root) {
    const toc = [];
    const usedIds = new Set();
    $$("h2, h3", root).forEach((h) => {
      const baseId = h.id || slugify(h.textContent);
      let id = baseId;
      let suffix = 2;
      while (usedIds.has(id)) id = `${baseId}-${suffix++}`;
      usedIds.add(id);
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
        if (el) el.scrollIntoView({ behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
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
  async function fetchMd(file, lang = currentLang()) {
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
  let renderGeneration = 0;
  let hasRenderedRoute = false;
  let focusContentOnNextRoute = false;

  function focusMainContent(scroll = false) {
    if (scroll) contentEl.scrollIntoView({ block: "start" });
    contentEl.focus({ preventScroll: true });
  }

  function finishNavigation(generation, anchor, shouldFocus) {
    requestAnimationFrame(() => {
      if (generation !== renderGeneration) return;
      if (anchor) {
        const target = document.getElementById(anchor);
        if (target) target.scrollIntoView();
      } else {
        window.scrollTo(0, 0);
      }
      if (shouldFocus) focusMainContent();
    });
  }

  async function render(file, opts = {}) {
    const generation = ++renderGeneration;
    const renderLang = currentLang();
    const anchor = Object.prototype.hasOwnProperty.call(opts, "anchor") ? opts.anchor : currentRoute.anchor;
    currentRoute = { file, anchor };
    contentEl.classList.toggle("is-home", file === "index");
    document.body.dataset.route = file;
    contentEl.setAttribute("aria-busy", "true");
    if (!opts.rerender) { contentEl.innerHTML = `<div class="loading"><div class="spinner"></div></div>`; }
    let doc;
    try {
      doc = await fetchMd(file, renderLang);
    } catch (e) {
      if (generation !== renderGeneration) return;
      contentEl.innerHTML = t().notFound(escapeHtml(file));
      $("#pager").innerHTML = ""; buildPageToc([]);
      contentEl.setAttribute("aria-busy", "false");
      $("#route-announcer").textContent = $("h1", contentEl)?.textContent || file;
      if (!opts.rerender) finishNavigation(generation, null, opts.focusContent);
      return;
    }
    if (generation !== renderGeneration) return;
    // strip optional YAML frontmatter
    const md = doc.text.replace(/^---\n[\s\S]*?\n---\n/, "");
    let html = renderMarkdown(md);
    if (renderLang === "ko" && !doc.translated) {
      html = `<div class="callout callout-note lang-fallback">${t().fallbackNote}</div>` + html;
    }
    contentEl.innerHTML = html;
    processInlineMarkdownInHtml(contentEl);
    processLinks(contentEl);
    processCallouts(contentEl);
    processDetails(contentEl);
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
    contentEl.setAttribute("aria-busy", "false");
    const pageHeading = $("h1", contentEl)?.textContent.trim() || (meta ? chTitle(meta) : file);
    $("#route-announcer").textContent = pageHeading;

    if (opts.rerender) return;
    finishNavigation(generation, anchor, opts.focusContent);
  }

  /* ---------------- Router ---------------- */
  function parseHash() {
    let h = location.hash.replace(/^#\/?/, "");
    try { h = decodeURIComponent(h); } catch (_) { /* keep malformed hashes readable */ }
    let anchor = null;
    if (h.includes("::")) { const [f, a] = h.split("::"); h = f; anchor = a; }
    if (!h) h = "index";
    return { file: h, anchor };
  }
  function route() {
    if (location.hash && !location.hash.startsWith("#/")) {
      if (!hasRenderedRoute) {
        hasRenderedRoute = true;
        render("index", { focusContent: location.hash === "#content" });
      } else if (location.hash === "#content") {
        requestAnimationFrame(() => focusMainContent(true));
      }
      return;
    }
    const { file, anchor } = parseHash();
    const previousFile = currentRoute.file;
    const searchWasOpen = !searchModal.hidden;
    const navWasOpen = document.body.classList.contains("nav-open");
    const shouldFocus = focusContentOnNextRoute || searchWasOpen || navWasOpen || (hasRenderedRoute && file !== previousFile);
    focusContentOnNextRoute = false;
    if (!window.BOOK_BY_FILE[file] && file !== "index") {
      // still try to fetch (allows non-manifest files)
    }
    closeSearch(false);
    closeNav();
    hasRenderedRoute = true;
    render(file, { anchor, focusContent: shouldFocus });
  }
  window.addEventListener("hashchange", route);
  function activateSkipLink(event) {
    event.preventDefault();
    // Defer until the activation event finishes so keyboard/browser focus does
    // not snap back to the skip link after we move it to the article.
    setTimeout(() => focusMainContent(true), 0);
  }
  const skipLink = $(".skip-link");
  skipLink.addEventListener("click", activateSkipLink);
  skipLink.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") activateSkipLink(event);
  });

  /* ---------------- Progress bar ---------------- */
  window.addEventListener("scroll", () => {
    const h = document.documentElement;
    const scrolled = (h.scrollTop) / (h.scrollHeight - h.clientHeight || 1);
    $("#progress-bar").style.width = Math.min(100, Math.max(0, scrolled * 100)) + "%";
  }, { passive: true });

  /* ---------------- Mobile nav ---------------- */
  const mobileNavMedia = window.matchMedia("(max-width: 860px)");
  const sidebar = $("#sidebar");
  const menuButton = $("#menu-toggle");
  const main = $("#main");
  let navReturnFocus = null;
  function syncMobileNavA11y() {
    const mobile = mobileNavMedia.matches;
    const open = mobile && document.body.classList.contains("nav-open");
    if (!mobile) document.body.classList.remove("nav-open");
    if ("inert" in sidebar) sidebar.inert = mobile && !open;
    if (mobile && !open) sidebar.setAttribute("aria-hidden", "true"); else sidebar.removeAttribute("aria-hidden");
    if ("inert" in main) main.inert = open;
    if (open) main.setAttribute("aria-hidden", "true"); else main.removeAttribute("aria-hidden");
    updateMenuButton();
  }
  function openNav() {
    if (!mobileNavMedia.matches) return;
    navReturnFocus = document.activeElement;
    document.body.classList.add("nav-open");
    syncMobileNavA11y();
    requestAnimationFrame(() => ($("#toc a.active") || $("#toc .part-title"))?.focus());
  }
  function closeNav(restoreFocus = false) {
    const wasOpen = document.body.classList.contains("nav-open");
    document.body.classList.remove("nav-open");
    syncMobileNavA11y();
    if (restoreFocus && wasOpen && navReturnFocus instanceof HTMLElement) navReturnFocus.focus();
  }
  menuButton.addEventListener("click", () => document.body.classList.contains("nav-open") ? closeNav(true) : openNav());
  $("#scrim").addEventListener("click", () => closeNav(true));
  if (mobileNavMedia.addEventListener) mobileNavMedia.addEventListener("change", syncMobileNavA11y);
  else mobileNavMedia.addListener(syncMobileNavA11y);
  syncMobileNavA11y();
  $("#toc").addEventListener("click", (event) => {
    const link = event.target.closest("a[data-file]");
    if (!link || !mobileNavMedia.matches || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    focusContentOnNextRoute = true;
    if (link.getAttribute("href") === location.hash) {
      event.preventDefault();
      focusContentOnNextRoute = false;
      closeNav();
      requestAnimationFrame(() => focusMainContent(true));
    }
  });

  /* ---------------- Search ---------------- */
  const searchModal = $("#search-modal");
  const searchInput = $("#search-input");
  const searchResults = $("#search-results");
  const searchOpenButton = $("#search-open");
  const searchCloseButton = $("#search-close");
  let searchReturnFocus = null;
  let selectedSearchIndex = -1;
  let searchRun = 0;

  function resetSearchIndex() {
    searchIndex = null;
    searchIndexLang = null;
    searchIndexPromise = null;
    searchIndexGeneration++;
    searchRun++;
  }

  function stripMd(md) {
    return md
      .replace(/^---\n[\s\S]*?\n---\n/, "")
      .replace(/```[\s\S]*?```/g, " ")
      .replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, " ")
      .replace(/<style\b[^>]*>[\s\S]*?<\/style>/gi, " ")
      .replace(/<[^>]+>/g, " ")
      .replace(/`[^`]*`/g, " ")
      .replace(/!\[[^\]]*\]\([^)]*\)/g, " ")
      .replace(/\[([^\]]*)\]\([^)]*\)/g, "$1")
      .replace(/\[!(?:NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]/gi, " ")
      .replace(/[#>*_~|]/g, " ")
      .replace(/\$[^$]*\$/g, " ")
      .replace(/\s+/g, " ")
      .trim();
  }
  async function buildSearchIndex() {
    const lang = currentLang();
    if (searchIndex && searchIndexLang === lang) return searchIndex;
    if (searchIndexPromise && searchIndexPromise.lang === lang) return searchIndexPromise.promise;

    const generation = ++searchIndexGeneration;
    const promise = Promise.all(window.BOOK_FLAT.map(async (ch) => {
      try {
        const doc = await fetchMd(ch.file, lang);
        return { ...ch, text: stripMd(doc.text) };
      } catch (e) { return null; }
    })).then((entries) => entries.filter(Boolean));
    const request = { lang, generation, promise };
    searchIndexPromise = request;
    try {
      const built = await promise;
      if (generation === searchIndexGeneration && currentLang() === lang) {
        searchIndex = built;
        searchIndexLang = lang;
      }
      return built;
    } finally {
      if (searchIndexPromise === request) searchIndexPromise = null;
    }
  }
  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }
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
    const run = ++searchRun;
    const lang = currentLang();
    q = q.trim().toLowerCase();
    selectedSearchIndex = -1;
    searchInput.removeAttribute("aria-activedescendant");
    if (!q) { searchResults.innerHTML = `<div class="search-empty">${t().searchStart}</div>`; return; }
    searchResults.innerHTML = `<div class="search-empty">${t().indexing}</div>`;
    const idx = await buildSearchIndex();
    if (run !== searchRun || lang !== currentLang() || q !== searchInput.value.trim().toLowerCase()) return;
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
    const visible = scored.slice(0, 20);
    searchResults.innerHTML = `<div class="search-results-meta" role="presentation">${t().resultCount(visible.length, scored.length)}</div>` + visible.map(({ e }, i) =>
      `<a id="search-result-${i}" class="search-result" role="option" aria-selected="false" tabindex="-1" href="#/${e.file}"><div class="sr-part">${escapeHtml(e.partEmoji || "")} ${escapeHtml(currentLang() === "ko" && e.partKo ? e.partKo : e.part)}</div><div class="sr-title">${escapeHtml(chTitle(e))}</div><div class="sr-snip">${snippet(e.text, terms)}</div></a>`
    ).join("");
    $$(".search-result", searchResults).forEach((a, i) => {
      a.addEventListener("mouseenter", () => selectSearchResult(i, false));
      a.addEventListener("click", (event) => {
        if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
        event.preventDefault();
        activateSearchResult(a);
      });
    });
  }
  let searchTimer = null;
  searchInput.addEventListener("input", () => { clearTimeout(searchTimer); searchTimer = setTimeout(() => runSearch(searchInput.value), 120); });
  function selectSearchResult(index, scroll = true) {
    const results = $$(".search-result", searchResults);
    if (!results.length) { selectedSearchIndex = -1; searchInput.removeAttribute("aria-activedescendant"); return; }
    selectedSearchIndex = (index + results.length) % results.length;
    results.forEach((result, i) => {
      const selected = i === selectedSearchIndex;
      result.classList.toggle("sel", selected);
      result.setAttribute("aria-selected", String(selected));
    });
    const selected = results[selectedSearchIndex];
    searchInput.setAttribute("aria-activedescendant", selected.id);
    if (scroll) selected.scrollIntoView({ block: "nearest" });
  }
  function openSearch() {
    if (!searchModal.hidden) return;
    searchReturnFocus = document.activeElement;
    searchModal.hidden = false;
    document.body.classList.add("search-open");
    searchOpenButton.setAttribute("aria-expanded", "true");
    searchInput.setAttribute("aria-expanded", "true");
    const existingQuery = searchInput.value.trim();
    searchResults.innerHTML = `<div class="search-empty">${t().searchStart}</div>`;
    selectedSearchIndex = -1;
    requestAnimationFrame(() => { searchInput.focus(); searchInput.select(); });
    if (existingQuery) runSearch(existingQuery);
    else buildSearchIndex().catch(() => {});
  }
  function closeSearch(restoreFocus = true) {
    if (searchModal.hidden) return;
    searchRun++;
    searchModal.hidden = true;
    document.body.classList.remove("search-open");
    searchOpenButton.setAttribute("aria-expanded", "false");
    searchInput.setAttribute("aria-expanded", "false");
    searchInput.removeAttribute("aria-activedescendant");
    selectedSearchIndex = -1;
    if (restoreFocus && searchReturnFocus instanceof HTMLElement) searchReturnFocus.focus();
  }
  function activateSearchResult(link) {
    const targetHash = link.getAttribute("href");
    focusContentOnNextRoute = true;
    closeSearch(false);
    if (location.hash === targetHash) {
      focusContentOnNextRoute = false;
      requestAnimationFrame(() => focusMainContent(true));
    } else {
      location.hash = targetHash;
    }
  }
  function focusableWithin(container) {
    return $$("a[href], button:not([disabled]), input:not([disabled]), textarea:not([disabled]), select:not([disabled]), summary, [tabindex]:not([tabindex='-1'])", container)
      .filter((el) => !el.hidden && el.getClientRects().length && el.getAttribute("aria-hidden") !== "true");
  }
  function trapFocus(container, event) {
    const focusable = focusableWithin(container);
    if (!focusable.length) return;
    const first = focusable[0], last = focusable[focusable.length - 1];
    if (event.shiftKey && (document.activeElement === first || !container.contains(document.activeElement))) { event.preventDefault(); last.focus(); }
    else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
  }
  searchOpenButton.addEventListener("click", openSearch);
  searchCloseButton.addEventListener("click", () => closeSearch(true));
  searchModal.addEventListener("click", (e) => { if (e.target === searchModal) closeSearch(true); });
  document.addEventListener("keydown", (e) => {
    const editing = /input|textarea|select/i.test(document.activeElement.tagName) || document.activeElement.isContentEditable;
    if (e.key === "/" && searchModal.hidden && !editing) { e.preventDefault(); openSearch(); return; }
    if (e.key === "Escape") {
      if (!searchModal.hidden) closeSearch(true);
      else if (document.body.classList.contains("nav-open")) closeNav(true);
      return;
    }
    if (!searchModal.hidden) {
      if (e.key === "ArrowDown") { e.preventDefault(); selectSearchResult(selectedSearchIndex + 1); }
      else if (e.key === "ArrowUp") { e.preventDefault(); selectSearchResult(selectedSearchIndex < 0 ? -1 : selectedSearchIndex - 1); }
      else if (e.key === "Enter") {
        const results = $$(".search-result", searchResults);
        const selected = results[selectedSearchIndex >= 0 ? selectedSearchIndex : 0];
        if (selected) { e.preventDefault(); activateSearchResult(selected); }
      } else if (e.key === "Tab") trapFocus(searchModal, e);
      return;
    }
    if (e.key === "Tab" && mobileNavMedia.matches && document.body.classList.contains("nav-open")) trapFocus(sidebar, e);
  });

  /* ---------------- Boot ---------------- */
  buildSidebar();
  route();
})();
