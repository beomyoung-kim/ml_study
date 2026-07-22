#!/usr/bin/env node

import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { spawnSync } from 'node:child_process';

const root = process.cwd();
const contentRoot = path.join(root, 'content');
const manifestPath = path.join(root, 'assets', 'book.js');
const FENCE = String.fromCharCode(96).repeat(3);
const failures = [];
const warnings = [];
const pythonBlocks = [];

function fail(message) {
  failures.push(message);
}

function read(file) {
  return fs.readFileSync(file, 'utf8');
}

function count(text, needle) {
  return text.split(needle).length - 1;
}

function headingCounts(text) {
  const out = [0, 0, 0, 0, 0, 0, 0];
  for (const line of text.split(/\r?\n/)) {
    const match = /^(#{1,6})\s+/.exec(line);
    if (match) out[match[1].length] += 1;
  }
  return out.slice(1);
}

function structuralSignature(text) {
  return {
    headings: headingCounts(text),
    fences: count(text, FENCE),
    mermaid: count(text, FENCE + 'mermaid'),
    details: count(text, '<details'),
    answerCards: count(text, 'class="qa answer-card"'),
    conceptCode: count(text, 'class="concept-code"'),
    figures: count(text, '<figure'),
    svg: count(text, '<svg'),
    scripts: count(text, '<script'),
    codeWidgets: count(text, 'class="code-config"'),
  };
}

function slugify(value) {
  const slug = String(value)
    .normalize('NFKC')
    .toLowerCase()
    .replace(/<[^>]+>/g, '')
    .replace(/[^\p{L}\p{N}\s_-]/gu, '')
    .trim()
    .replace(/[\s_]+/g, '-')
    .replace(/-+/g, '-')
    .slice(0, 60);
  return slug || 'section';
}

function headingIds(text) {
  const ids = new Set();
  const seen = new Map();
  for (const line of text.split(/\r?\n/)) {
    const match = /^#{2,6}\s+(.+?)\s*$/.exec(line);
    if (!match) continue;
    const base = slugify(match[1]);
    const next = (seen.get(base) || 0) + 1;
    seen.set(base, next);
    ids.add(next === 1 ? base : base + '-' + next);
  }
  return ids;
}

function parseCodeWidgets(text, label) {
  const pattern = /<script\b[^>]*class="code-config"[^>]*>([\s\S]*?)<\/script>/g;
  let match;
  let total = 0;
  let tests = 0;
  const summaries = [];
  while ((match = pattern.exec(text))) {
    total += 1;
    try {
      const value = JSON.parse(match[1]);
      for (const key of ['func', 'starter', 'tests', 'solution']) {
        if (!(key in value)) fail(label + ': code widget missing key ' + key);
      }
      if (!Array.isArray(value.tests) || value.tests.length === 0) {
        fail(label + ': code widget has no tests');
      }
      tests += Array.isArray(value.tests) ? value.tests.length : 0;
      summaries.push({
        func: value.func,
        packages: value.packages || [],
        approx: Boolean(value.approx),
        tests: Array.isArray(value.tests) ? value.tests.length : 0,
      });
    } catch (error) {
      fail(label + ': invalid code-widget JSON: ' + error.message);
    }
  }
  return { total, tests, summaries };
}

function collectPythonFences(text, label) {
  const pattern = new RegExp('^' + FENCE + 'python[^\\n]*\\n([\\s\\S]*?)^' + FENCE, 'gm');
  let match;
  let index = 0;
  while ((match = pattern.exec(text))) {
    index += 1;
    pythonBlocks.push({ label: label + ' Python fence ' + index, code: match[1] });
  }
}

const sandbox = { window: {} };
vm.runInNewContext(read(manifestPath), sandbox, { filename: manifestPath });
const book = sandbox.window.BOOK;
const chapters = sandbox.window.BOOK_FLAT;

if (!book || !Array.isArray(book.parts)) fail('Manifest did not define BOOK.parts');
if (!Array.isArray(chapters)) fail('Manifest did not define BOOK_FLAT');

const ids = new Set();
const routes = new Set();
for (const chapter of chapters || []) {
  if (ids.has(chapter.id)) fail('Duplicate chapter id: ' + chapter.id);
  if (routes.has(chapter.file)) fail('Duplicate chapter route: ' + chapter.file);
  ids.add(chapter.id);
  routes.add(chapter.file);
}

const texts = { ko: new Map(), en: new Map() };
let widgetTotalKo = 0;
let widgetTotalEn = 0;
let widgetTestTotalKo = 0;
let widgetTestTotalEn = 0;

for (const route of routes) {
  const koPath = path.join(contentRoot, route + '.ko.md');
  const enPath = path.join(contentRoot, route + '.md');
  if (!fs.existsSync(koPath)) fail('Missing Korean chapter: ' + route);
  if (!fs.existsSync(enPath)) fail('Missing English chapter: ' + route);
  if (!fs.existsSync(koPath) || !fs.existsSync(enPath)) continue;

  const ko = read(koPath);
  const en = read(enPath);
  texts.ko.set(route, ko);
  texts.en.set(route, en);

  for (const [lang, text] of [['KO', ko], ['EN', en]]) {
    const label = lang + ' ' + route;
    if (count(text, FENCE) % 2 !== 0) fail(label + ': unbalanced code fences');
    if (count(text, '<details') !== count(text, '</details>')) fail(label + ': unbalanced details');
    if (count(text, '<figure') !== count(text, '</figure>')) fail(label + ': unbalanced figures');
    if (count(text, '<script') !== count(text, '</script>')) fail(label + ': unbalanced scripts');
    collectPythonFences(text, label);
  }

  const koSignature = structuralSignature(ko);
  const enSignature = structuralSignature(en);
  if (JSON.stringify(koSignature) !== JSON.stringify(enSignature)) {
    fail('KO/EN structural mismatch in ' + route + '\n  KO ' +
      JSON.stringify(koSignature) + '\n  EN ' + JSON.stringify(enSignature));
  }

  const koWidgets = parseCodeWidgets(ko, 'KO ' + route);
  const enWidgets = parseCodeWidgets(en, 'EN ' + route);
  widgetTotalKo += koWidgets.total;
  widgetTotalEn += enWidgets.total;
  widgetTestTotalKo += koWidgets.tests;
  widgetTestTotalEn += enWidgets.tests;
  if (JSON.stringify(koWidgets.summaries) !== JSON.stringify(enWidgets.summaries)) {
    fail('KO/EN code-widget contract mismatch in ' + route);
  }
}

for (const [lang, byRoute] of Object.entries(texts)) {
  const anchors = new Map();
  for (const [route, text] of byRoute) anchors.set(route, headingIds(text));

  for (const [sourceRoute, text] of byRoute) {
    const routePattern = /#\/([\p{L}\p{N}._/-]+)(?:::([\p{L}\p{N}_-]+))?/gu;
    let match;
    while ((match = routePattern.exec(text))) {
      const targetRoute = match[1].replace(/\/$/, '');
      const targetAnchor = match[2];
      if (!routes.has(targetRoute)) {
        fail(lang.toUpperCase() + ' ' + sourceRoute + ': broken route #/' + targetRoute);
      } else if (targetAnchor && !anchors.get(targetRoute)?.has(targetAnchor)) {
        fail(lang.toUpperCase() + ' ' + sourceRoute + ': broken anchor #/' +
          targetRoute + '::' + targetAnchor);
      }
    }
  }
}

const allowedHangulRoutes = new Set([
  'llm/prompting',
  'llm/next-token',
  'resources/changelog',
]);
for (const [route, text] of texts.en) {
  if (/[가-힣]/.test(text) && !allowedHangulRoutes.has(route)) {
    fail('Unexpected Hangul in English chapter: ' + route);
  }
}

if (widgetTotalKo !== widgetTotalEn) {
  fail('Code-widget count differs: KO ' + widgetTotalKo + ', EN ' + widgetTotalEn);
}
if (widgetTestTotalKo !== widgetTestTotalEn) {
  fail('Code-widget test count differs: KO ' + widgetTestTotalKo + ', EN ' + widgetTestTotalEn);
}

const pythonChecker = [
  'import ast, json, sys',
  'blocks = json.load(sys.stdin)',
  'errors = []',
  'for block in blocks:',
  '    try:',
  "        ast.parse(block['code'])",
  '    except SyntaxError as exc:',
  "        errors.append({'label': block['label'], 'line': exc.lineno, 'message': exc.msg})",
  'print(json.dumps(errors))',
].join('\n');
const pythonResult = spawnSync('python3', ['-c', pythonChecker], {
  cwd: root,
  input: JSON.stringify(pythonBlocks),
  encoding: 'utf8',
});
if (pythonResult.error || pythonResult.status !== 0) {
  fail('Could not parse Python fences: ' + (pythonResult.error?.message || pythonResult.stderr.trim()));
} else {
  const pythonErrors = JSON.parse(pythonResult.stdout || '[]');
  for (const error of pythonErrors) {
    fail(error.label + ': Python syntax error at line ' + error.line + ': ' + error.message);
  }
}

if ((book?.parts?.length || 0) !== 15) {
  warnings.push('Expected 15 parts; found ' + (book?.parts?.length || 0));
}

if (failures.length) {
  console.error('Content validation failed (' + failures.length + '):');
  for (const message of failures) console.error('- ' + message);
  process.exit(1);
}

console.log('Content validation passed.');
console.log('- ' + routes.size + ' unique routes');
console.log('- ' + texts.ko.size + ' Korean + ' + texts.en.size + ' English chapters');
console.log('- ' + widgetTotalKo + ' code widgets per language');
console.log('- ' + widgetTestTotalKo + ' widget tests per language');
console.log('- ' + pythonBlocks.length + ' Python fences parsed');
for (const message of warnings) console.warn('- warning: ' + message);
