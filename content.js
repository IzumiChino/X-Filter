"use strict";

// ============================================================
//  chrome.storage 封装
// ============================================================
function storageGet(key, defaultVal) {
  return chrome.storage.local.get(key).then(r => r[key] !== undefined ? r[key] : defaultVal).catch(() => defaultVal);
}
function storageSet(key, val) {
  chrome.storage.local.set({ [key]: val }).catch(() => {});
}

// ============================================================
//  默认配置
// ============================================================
const DEFAULT = {
  simThresholdReply: 0.32,
  simThresholdTimeline: 0.40,
  ngramN: 3,
  maxBadSamples: 350,
  maxGoodSamples: 150,
  markingMode: true,
  blockMode: "fold",
  previewLen: 80,
  followedByHover: true,
  structuralChannel: true,
  emojiOnlyMinCount: 3,
  digitsOnlyMinCount: 2,
  mixedMinEmoji: 2,
  mixedMinDigits: 1,
  structuralReplyOnly: false,
  heuristicChannel: true,
  heuristicThresholdReply: 0.40,
  heuristicThresholdTimeline: 0.70,
  mlEnabled: true,
  mlMinSamples: 8,
  mlThresholdReply: 0.55,
  mlThresholdTimeline: 0.65,
  mlLearningRate: 0.05,
  mlL2: 0.001,
  mlMaxTrainData: 500,
  mlMinPositiveSamples: 3,
  mlMinNegativeSamples: 3,
  debug: false,
};

// ============================================================
//  存储 key
// ============================================================
const KEY_BAD = "xls_samplesBad_v4";
const KEY_GOOD = "xls_samplesGood_v4";
const KEY_FOLLOWED = "xls_followedHandles_v1";
const KEY_MODEL = "xls_modelState_v1";
const KEY_TRAIN = "xls_trainData_v1";
const KEY_ACCT = "xls_accountCache_v1";
const KEY_HREP = "xls_handleRep_v1";

// ============================================================
//  运行态变量（start() 中异步填充）
// ============================================================
const cfg = {};
for (const k of Object.keys(DEFAULT)) cfg[k] = DEFAULT[k];

let log = (...a) => {};

let memBad = [];
let memGood = [];
let memFollowed = new Set();
let memTrain = [];
let memHRep = {};
let memAcct = {};

// ============================================================
//  存储工具
// ============================================================
function saveArr(key, arr) { storageSet(key, arr); }
function saveObj(key, obj) { storageSet(key, obj); }

function addFollowed(handle) {
  if (!handle || !handle.startsWith("@")) return;
  if (memFollowed.has(handle)) return;
  memFollowed.add(handle);
  saveArr(KEY_FOLLOWED, [...memFollowed]);
  log("learned followed:", handle);
}

function updateHandleRep(handle, label) {
  if (!handle) return;
  if (!memHRep[handle]) memHRep[handle] = { labels: [], avg: 0.5 };
  const entry = memHRep[handle];
  entry.labels.push(label);
  if (entry.labels.length > 20) entry.labels.shift();
  entry.avg = entry.labels.reduce((a, b) => a + b, 0) / entry.labels.length;
  saveObj(KEY_HREP, memHRep);
}

function getHandleRep(handle) {
  const e = handle && memHRep[handle];
  return e ? e.avg : 0.5;
}

function saveTrainExample(features, label, rawData) {
  memTrain.unshift({
    f: features.map(v => Math.round(v * 1e4) / 1e4),
    y: label,
    h: rawData.handle || "",
    txt: (rawData.text || "").slice(0, 500),
    nm: (rawData.name || "").slice(0, 60),
    rp: rawData.isReply ? 1 : 0,
    md: rawData.hasMedia ? 1 : 0,
    ts: Date.now(),
  });
  if (memTrain.length > cfg.mlMaxTrainData) memTrain.length = cfg.mlMaxTrainData;
  saveArr(KEY_TRAIN, memTrain);
}

// ============================================================
//  Toast
// ============================================================
function toast(msg) {
  const el = document.createElement("div");
  el.textContent = msg;
  Object.assign(el.style, {
    position: "fixed", right: "16px", bottom: "16px", zIndex: "999999",
    padding: "10px 12px", borderRadius: "12px",
    background: "rgba(0,0,0,.75)", color: "#fff", fontSize: "13px",
    backdropFilter: "blur(6px)", maxWidth: "60vw", pointerEvents: "none",
  });
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 900);
}

// ============================================================
//  文本预处理 + n-gram + Jaccard
// ============================================================
function normalizeText(s) {
  return (s || "")
    .toLowerCase()
    .replace(/\s+/g, "")
    .replace(/[，。！？、,.!?:;'"""''（）()\[\]【】<>]/g, "");
}

function makeNgrams(str, n) {
  const s = normalizeText(str);
  if (!s) return [];
  if (s.length <= n) return [s];
  const grams = [];
  for (let i = 0; i <= s.length - n; i++) grams.push(s.slice(i, i + n));
  return grams;
}

function jaccard(aArr, bArr) {
  const a = new Set(aArr);
  const b = new Set(bArr);
  if (a.size === 0 && b.size === 0) return 0;
  let inter = 0;
  const [small, big] = a.size <= b.size ? [a, b] : [b, a];
  for (const x of small) if (big.has(x)) inter++;
  const union = a.size + b.size - inter;
  return union === 0 ? 0 : inter / union;
}

function bestSimilarity(grams, samples) {
  let best = 0;
  const recentCount = Math.min(samples.length, 120);
  for (let i = 0; i < recentCount; i++) {
    const sim = jaccard(grams, samples[i].grams);
    if (sim > best) best = sim;
    if (best >= 0.95) return best;
  }
  if (samples.length > recentCount) {
    const rest = samples.length - recentCount;
    const sampleCount = Math.min(rest, 60);
    for (let j = 0; j < sampleCount; j++) {
      const idx = recentCount + Math.floor(Math.random() * rest);
      const sim = jaccard(grams, samples[idx].grams);
      if (sim > best) best = sim;
      if (best >= 0.95) return best;
    }
  }
  return best;
}

function addSample(key, fingerprint, rawPreview) {
  const grams = makeNgrams(fingerprint, cfg.ngramN);
  if (!grams.length) return false;
  const sig = normalizeText(fingerprint).slice(0, 120);
  const arr = key === KEY_BAD ? memBad : memGood;
  if (arr.some(x => x.sig === sig)) return false;
  arr.unshift({ sig, t: (rawPreview || fingerprint).slice(0, 360), grams, ts: Date.now() });
  const cap = key === KEY_BAD ? cfg.maxBadSamples : cfg.maxGoodSamples;
  if (arr.length > cap) arr.length = cap;
  saveArr(key, arr);
  return true;
}

// ============================================================
//  NLP 工具：字符熵、文本重复度
// ============================================================
function charEntropy(str) {
  if (!str || str.length < 2) return 0;
  const freq = {};
  for (const c of str) freq[c] = (freq[c] || 0) + 1;
  let ent = 0;
  const len = str.length;
  for (const c in freq) {
    const p = freq[c] / len;
    ent -= p * Math.log2(p);
  }
  return ent;
}

function textRepetition(str) {
  const s = normalizeText(str);
  if (s.length < 4) return 0;
  const seen = {};
  let total = 0, repeated = 0;
  for (let i = 0; i < s.length - 1; i++) {
    const bg = s.slice(i, i + 2);
    if (seen[bg]) repeated++;
    seen[bg] = true;
    total++;
  }
  return total > 0 ? repeated / total : 0;
}

function countMatches(str, re) {
  const m = (str || "").match(re);
  return m ? m.length : 0;
}

function compactRiskText(str) {
  return (str || "")
    .toLowerCase()
    .replace(/[\s\u3000]+/g, "")
    .replace(/[，。！？、,.!?:;'""'""''（）()\[\]【】<>《》~`|\\/]/g, "");
}

function countEmoji(str) {
  const m = (str || "").match(/[\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}\u{2300}-\u{23FF}\u{2B05}-\u{2B55}]/gu);
  return m ? m.length : 0;
}

function countDigits(str) {
  return (str || "").replace(/\D/g, "").length;
}

function textRiskProfile(text) {
  const raw = text || "";
  const compact = compactRiskText(raw);
  const len = raw.length;
  const cjkCount = countMatches(raw, /[\u4E00-\u9FFF]/gu);
  const latinCount = countMatches(raw, /[A-Za-z]/g);
  const digitCount = countDigits(raw);
  const cjkRatio = len > 0 ? cjkCount / len : 0;
  const latinRatio = len > 0 ? latinCount / len : 0;
  const mixedScript = len > 0 && cjkCount > 0 && latinCount > 0
    ? Math.min(1, 2 * Math.min(cjkRatio, latinRatio))
    : 0;
  const cjkDominant = cjkRatio >= 0.45 && latinRatio <= 0.15;

  const hasURL = /https?:\/\/|www\.|t\.me\/|telegram\.me|t\.co|wa\.me|line\.me|bit\.ly|tinyurl\.com/i.test(raw) ? 1 : 0;
  const contact = /(?:vx|wx|wechat|weixin|tg|telegram|qq|whatsapp|line|[\u79C1\u804A][\u4FE1\u5929]|[\u8054\u7CFB][\u6211]|[\u52A0][vV\u5FAE\u5FAE\u4FE1\u8587]|v[\u4FE1]|\u5FAE\s*[\u4FE1]|\u7535\u62A5|[\u7AD9\u5185][\u4FE1]|dm)/i.test(compact) ? 1 : 0;
  const sex = /(?:[\u7EA6][\u70AE\u556A\u0070]|[\u5305][\u591C]|[\u4E0A\u95E8][\u670D\u52A1]|[\u5916][\u7EA6]|[\u88F8][\u804A]|[\u5168\u5957]|[\u5916\u56F4]|[\u966A][\u7761]|[\u5B66\u751F][\u59B9]|[\u6210\u4EBA][\u89C6\u9891]|[\u65E0\u7801][\u89C6\u9891]|[\u5F00][\u623F]|[\u4E00\u591C\u60C5]|[\u540C\u57CE][\u7EA6]|[\u9644\u8FD1][\u59B9])/i.test(compact) ? 1 : 0;
  const fraud = /(?:[\u5237\u5355][\u8FD4][\u5229]|[\u8FD4\u5229][\u7FA4]|[\u63D0\u73B0][\u5F02\u5E38]|[\u7A33][\u8D5A]|[\u65E5\u7ED3][\u517C\u804C]|[\u65E0\u62B5\u62BC][\u8D37\u6B3E]|[\u5237][\u6D41\u6C34]|[\u535A\u5F69]|[\u6740\u732A\u76D8]|[\u9001][\u5F69\u91D1]|[\u4EE3][\u6295]|[\u5F00\u6237][\u94FE\u63A5]|[\u9AD8][\u6536\u76CA]|[\u5E26][\u5355]|[\u8D44\u91D1\u76D8]|[\u865A\u62DF\u5E01][\u642C\u7816]|usdt[\u642C\u7816])/i.test(compact) ? 1 : 0;
  const money = /(?:¥|￥|\$|\d+(?:\.\d+)?(?:w|W|万|k|K|元|块|刀|美元|usdt|USDT))/i.test(raw) ? 1 : 0;
  const rep = textRepetition(raw);

  const riskScore = Math.min(
    1,
    hasURL * 0.20 +
    contact * 0.30 +
    sex * 0.35 +
    fraud * 0.35 +
    money * 0.10 +
    (digitCount >= 8 ? 0.10 : 0) +
    (rep > 0.45 ? 0.10 : 0)
  );

  return {
    len,
    cjkCount,
    latinCount,
    digitCount,
    cjkRatio,
    latinRatio,
    mixedScript,
    cjkDominant,
    hasURL,
    contact,
    sex,
    fraud,
    money,
    rep,
    riskScore,
    hasStrongAnchor: riskScore >= 0.35,
  };
}

function isCjkLowRiskText(risk) {
  return risk.cjkDominant && !risk.hasStrongAnchor && risk.riskScore < 0.35;
}

function getCjkProtectionLevel(risk, replyMode) {
  if (!isCjkLowRiskText(risk)) return 0;
  return replyMode ? 0.18 : 0.12;
}

// ============================================================
//  n-gram 哈希特征 (Hashing Trick)
// ============================================================
function fnv32a(str) {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

const BASE_FEATURE_COUNT = 28;
const HASH_DIM = 256;

function hashNgramFeatures(text, handle, name, risk) {
  const features = new Float64Array(HASH_DIM);
  const r = risk || textRiskProfile(text);
  const parts = [
    { value: text, weight: r.cjkDominant && !r.hasStrongAnchor ? 0.35 : 1 },
    { value: handle, weight: 0.9 },
    { value: name, weight: r.cjkDominant && !r.hasStrongAnchor ? 0.6 : 0.85 },
  ].filter(part => part.value);
  for (const part of parts) {
    const s = normalizeText(part.value);
    if (s.length < 3) continue;
    for (let i = 0; i <= s.length - 3; i++) {
      const gram = s.slice(i, i + 3);
      const idx = fnv32a(gram) % HASH_DIM;
      features[idx] = Math.max(features[idx], part.weight);
    }
  }
  let normSq = 0;
  for (let i = 0; i < HASH_DIM; i++) normSq += features[i] * features[i];
  if (normSq > 0) {
    const norm = 1 / Math.sqrt(normSq);
    for (let i = 0; i < HASH_DIM; i++) if (features[i]) features[i] *= norm;
  }
  return features;
}

// ============================================================
//  上下文分析（基于 Jaccard n-gram 集合）
// ============================================================
let threadCtx = { url: null, opGrams: null, blocked: 0, total: 0 };
const recentBuf = [];
const RECENT_CAP = 30;

function refreshThreadCtx() {
  const url = location.pathname;
  if (url !== threadCtx.url) {
    threadCtx = { url, opGrams: null, blocked: 0, total: 0 };
    recentBuf.length = 0;
  }
}

function trackThreadResult(wasBlocked) {
  threadCtx.total++;
  if (wasBlocked) threadCtx.blocked++;
}

function computeContextBoost(text, handle, name) {
  const fp = [name, handle, text].filter(Boolean).join(" | ");
  const grams = makeNgrams(fp, cfg.ngramN);
  if (grams.length < 5) return 0;

  let boost = 0;

  if (inReplyContext()) {
    if (!threadCtx.opGrams) {
      threadCtx.opGrams = grams;
    } else if (threadCtx.opGrams.length >= 5) {
      const coherence = jaccard(grams, threadCtx.opGrams);
      if (coherence < 0.03) boost += 0.10;
      else if (coherence < 0.06) boost += 0.05;
    }

    if (threadCtx.total >= 5) {
      const ratio = threadCtx.blocked / threadCtx.total;
      if (ratio > 0.6) boost += 0.05;
    }
  }

  let maxCross = 0;
  for (const r of recentBuf) {
    if (r.handle === handle) continue;
    const sim = jaccard(grams, r.grams);
    if (sim > maxCross) maxCross = sim;
  }
  recentBuf.push({ grams, handle });
  if (recentBuf.length > RECENT_CAP) recentBuf.shift();

  if (maxCross > 0.70) boost += 0.12;
  else if (maxCross > 0.50) boost += 0.06;

  return Math.min(boost, 0.20);
}

// ============================================================
//  在线逻辑回归 (Online Logistic Regression)
// ============================================================
const FEATURE_COUNT = BASE_FEATURE_COUNT + HASH_DIM;

class OnlineLR {
  constructor(nf, lr, l2) {
    this.w = new Float64Array(nf);
    this.lr = lr;
    this.l2 = l2;
    this.n = 0;
  }

  sigmoid(z) {
    if (z > 500) return 1;
    if (z < -500) return 0;
    return 1 / (1 + Math.exp(-z));
  }

  predict(x) {
    let z = 0;
    for (let i = 0; i < this.w.length; i++) z += this.w[i] * (x[i] || 0);
    if (!isFinite(z)) return 0.5;
    return this.sigmoid(z);
  }

  update(x, y, weight) {
    const p = this.predict(x);
    const err = y - p;
    const w = weight || 1;
    const effectiveLr = this.lr / (1 + this.n * 0.002);
    for (let i = 0; i < this.w.length; i++) {
      const l2 = i === 0 ? 0 : (i >= BASE_FEATURE_COUNT ? this.l2 * 10 : this.l2);
      this.w[i] += effectiveLr * (w * err * (x[i] || 0) - l2 * this.w[i]);
    }
    this.n++;
  }

  train(x, y, passes, weight) {
    for (let i = 0; i < (passes || 3); i++) this.update(x, y, weight);
  }

  getState() {
    return { w: Array.from(this.w), n: this.n };
  }

  setState(st) {
    if (!st || !st.w) return;
    for (let i = 0; i < Math.min(st.w.length, this.w.length); i++) {
      this.w[i] = st.w[i] || 0;
    }
    this.n = st.n || 0;
  }
}

const FEATURE_VERSION = 3;

let model = new OnlineLR(FEATURE_COUNT, cfg.mlLearningRate, cfg.mlL2);

function saveModelState() {
  const st = model.getState();
  st.fv = FEATURE_VERSION;
  st.fc = FEATURE_COUNT;
  storageSet(KEY_MODEL, st);
}

function autoRetrain() {
  if (!memTrain.length) return;
  log("特征版本变化，自动重训模型");
  model = new OnlineLR(FEATURE_COUNT, cfg.mlLearningRate, cfg.mlL2);
  let posCount = 0;
  for (const ex of memTrain) if (ex.y === 1) posCount++;
  const negCount = memTrain.length - posCount;
  const posW = posCount > 0 ? Math.min(memTrain.length / (2 * posCount), 10) : 1;
  const negW = negCount > 0 ? Math.min(memTrain.length / (2 * negCount), 10) : 1;
  for (let epoch = 0; epoch < 5; epoch++) {
    const shuffled = [...memTrain].sort(() => Math.random() - 0.5);
    for (const ex of shuffled) {
      let f = null;
      const txt = ex.txt !== undefined ? ex.txt : ex.t;
      if (txt !== undefined) {
        f = computeFeatures({
          text: txt, handle: ex.h, name: ex.nm,
          isReply: !!ex.rp, hasMedia: !!ex.md,
        });
      }
      if (f) model.update(f, ex.y, ex.y === 1 ? posW : negW);
    }
  }
  saveModelState();
}

function getTrainCounts() {
  let pos = 0;
  for (const ex of memTrain) if (ex.y === 1) pos++;
  return { pos, neg: memTrain.length - pos, total: memTrain.length };
}

function hasEnoughMlTraining() {
  const c = getTrainCounts();
  return (
    c.total >= cfg.mlMinSamples &&
    c.pos >= cfg.mlMinPositiveSamples &&
    c.neg >= cfg.mlMinNegativeSamples
  );
}

function getClassWeight(label) {
  if (memTrain.length < 4) return 1;
  let posCount = 0;
  for (const ex of memTrain) if (ex.y === 1) posCount++;
  const negCount = memTrain.length - posCount;
  if (posCount === 0 || negCount === 0) return 1;
  const raw = label === 1
    ? memTrain.length / (2 * posCount)
    : memTrain.length / (2 * negCount);
  return Math.min(raw, 10);
}

function getTrainPasses() {
  if (memTrain.length < 20) return 3;
  if (memTrain.length < 100) return 2;
  return 1;
}

// ============================================================
//  特征提取  (28 维 + hashing trick, 全部归一化到 ~0-1)
// ============================================================
function computeFeatures({ text, handle, name, isReply, hasMedia }) {
  const hClean = (handle || "").replace(/^@/, "");
  const risk = textRiskProfile(text);

  const tLen = (text || "").length;
  const eN = countEmoji(text);
  const dN = countDigits(text);

  const fp = [name, handle, text].filter(Boolean).join(" | ");
  const grams = makeNgrams(fp, cfg.ngramN);
  const simBad = grams.length && memBad.length ? bestSimilarity(grams, memBad) : 0;
  const simGood = grams.length && memGood.length ? bestSimilarity(grams, memGood) : 0;

  const acct = (handle && memAcct[handle]) || {};
  const hasA = acct.followers !== undefined ? 1 : 0;

  const base = [
    1,
    Math.log1p(tLen) / 7,
    tLen > 0 ? eN / tLen : 0,
    tLen > 0 ? dN / tLen : 0,
    risk.riskScore,
    risk.hasURL,
    tLen > 0 ? new Set(text).size / tLen : 0,
    risk.rep,
    hClean.length > 0 ? hClean.replace(/\D/g, "").length / hClean.length : 0,
    /\d{4,}$/.test(hClean) ? 1 : 0,
    charEntropy(hClean) / 4.5,
    name ? countEmoji(name) / Math.max(name.length, 1) : 0,
    isReply ? 1 : 0,
    hasMedia ? 1 : 0,
    simBad,
    simGood,
    getHandleRep(handle),
    hasA,
    hasA ? Math.log1p(acct.followers || 0) / 14 : 0,
    hasA ? Math.log1p(acct.following || 0) / 14 : 0,
    hasA && (acct.following || 0) > 0
      ? Math.min((acct.followers || 0) / acct.following / 10, 1) : 0,
    hasA && acct.bio ? 1 : 0,
    hasA && acct.verified ? 1 : 0,
    risk.contact,
    risk.sex,
    risk.fraud,
    risk.mixedScript,
    risk.hasStrongAnchor ? 1 : 0,
  ];

  const hashed = hashNgramFeatures(text, handle, name, risk);
  for (let i = 0; i < HASH_DIM; i++) base.push(hashed[i]);
  return base;
}

function extractFeatures(article) {
  return computeFeatures({
    text: getTweetText(article),
    handle: getHandle(article),
    name: getDisplayName(article),
    isReply: inReplyContext(),
    hasMedia: hasMediaInArticle(article),
  });
}

// ============================================================
//  重新训练（从存储的训练数据）
// ============================================================
function retrainModel(silent = false) {
  if (!memTrain.length) {
    if (!silent) alert("没有训练数据。");
    return;
  }
  model = new OnlineLR(FEATURE_COUNT, cfg.mlLearningRate, cfg.mlL2);

  let posCount = 0;
  for (const ex of memTrain) if (ex.y === 1) posCount++;
  const negCount = memTrain.length - posCount;
  const posW = posCount > 0 ? Math.min(memTrain.length / (2 * posCount), 10) : 1;
  const negW = negCount > 0 ? Math.min(memTrain.length / (2 * negCount), 10) : 1;

  for (let epoch = 0; epoch < 5; epoch++) {
    const shuffled = [...memTrain].sort(() => Math.random() - 0.5);
    for (const ex of shuffled) {
      const txt = ex.txt !== undefined ? ex.txt : ex.t;
      if (txt === undefined) continue;
      const f = computeFeatures({
        text: txt, handle: ex.h, name: ex.nm,
        isReply: !!ex.rp, hasMedia: !!ex.md,
      });
      model.update(f, ex.y, ex.y === 1 ? posW : negW);
    }
  }
  saveModelState();
  if (!silent) alert(`重新训练完成。样本 ${memTrain.length}（垃圾 ${posCount} / 正常 ${negCount}），轮次 5。`);
}

function learnFromLabel(features, label, rawData) {
  if (!features || !cfg.mlEnabled) return;
  saveTrainExample(features, label, rawData);
  updateHandleRep(rawData.handle, label);
  if (hasEnoughMlTraining()) retrainModel(true);
  else saveModelState();
}

// ============================================================
//  导出 / 导入 / 清空
// ============================================================
function exportData() {
  const payload = {
    schema: "x-learnshield-export",
    version: 5,
    featureCount: FEATURE_COUNT,
    exportedAt: new Date().toISOString(),
    config: {},
    samplesBad: memBad,
    samplesGood: memGood,
    followedHandles: [...memFollowed],
    trainData: memTrain,
    modelState: model.getState(),
    accountCache: memAcct,
    handleReputation: memHRep,
  };
  for (const k of Object.keys(DEFAULT)) payload.config[k] = cfg[k];

  const blob = new Blob([JSON.stringify(payload)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `x-filter-${new Date().toISOString().replace(/[:.]/g, "-")}.json`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function mergeSamples(existing, incoming, cap) {
  const map = new Map();
  for (const x of existing) map.set(x.sig, x);
  for (const x of incoming) {
    if (x && typeof x === "object" && x.sig && Array.isArray(x.grams)) map.set(x.sig, x);
  }
  const merged = Array.from(map.values()).sort((a, b) => (b.ts || 0) - (a.ts || 0));
  if (merged.length > cap) merged.length = cap;
  return merged;
}

function mergeStrings(existing, incoming, cap) {
  const set = new Set(existing || []);
  for (const x of (incoming || [])) {
    if (typeof x === "string" && x.startsWith("@") && x.length <= 30) set.add(x);
  }
  const merged = Array.from(set);
  if (merged.length > cap) merged.length = cap;
  return merged;
}

function mergeTrainData(existing, incoming, cap) {
  const map = new Map();
  for (const x of existing) map.set(x.ts, x);
  for (const x of (incoming || [])) {
    if (x && Array.isArray(x.f) && (x.y === 0 || x.y === 1)) map.set(x.ts, x);
  }
  const merged = Array.from(map.values()).sort((a, b) => (b.ts || 0) - (a.ts || 0));
  if (merged.length > cap) merged.length = cap;
  return merged;
}

function mergeHandleRep(existing, incoming) {
  const out = { ...existing };
  for (const [k, v] of Object.entries(incoming || {})) {
    if (!v || !Array.isArray(v.labels)) continue;
    if (!out[k]) { out[k] = v; continue; }
    const combined = [...out[k].labels, ...v.labels];
    if (combined.length > 20) combined.splice(0, combined.length - 20);
    out[k] = { labels: combined, avg: combined.reduce((a, b) => a + b, 0) / combined.length };
  }
  return out;
}

function mergeAccountCache(existing, incoming) {
  const out = { ...existing };
  for (const [k, v] of Object.entries(incoming || {})) {
    if (!v || typeof v !== "object") continue;
    if (!out[k] || (v.ts || 0) > (out[k].ts || 0)) out[k] = v;
  }
  return out;
}

function importData(pastedJson) {
  let data;
  if (pastedJson !== undefined) {
    try { data = JSON.parse(pastedJson); } catch { alert("解析失败：不是合法 JSON"); return; }
  } else {
    const json = prompt("把导出的 JSON 粘贴到这里（会合并去重）");
    if (json === null) return;
    try { data = JSON.parse(json); } catch { alert("解析失败：不是合法 JSON"); return; }
  }
  if (!data || data.schema !== "x-learnshield-export") {
    alert("格式不对：不是本脚本导出的数据"); return;
  }

  memBad = mergeSamples(memBad, data.samplesBad || [], cfg.maxBadSamples);
  memGood = mergeSamples(memGood, data.samplesGood || [], cfg.maxGoodSamples);
  memFollowed = new Set(mergeStrings([...memFollowed], data.followedHandles, 8000));
  memTrain = mergeTrainData(memTrain, data.trainData, cfg.mlMaxTrainData);
  memHRep = mergeHandleRep(memHRep, data.handleReputation);
  memAcct = mergeAccountCache(memAcct, data.accountCache);

  saveArr(KEY_BAD, memBad);
  saveArr(KEY_GOOD, memGood);
  saveArr(KEY_FOLLOWED, [...memFollowed]);
  saveArr(KEY_TRAIN, memTrain);
  saveObj(KEY_HREP, memHRep);
  storageSet(KEY_ACCT, memAcct);

  if (data.trainData && data.trainData.length) {
    retrainModel();
  }

  alert(
    `导入完成（已合并去重）：\n` +
    `垃圾样本：${memBad.length}\n正常样本：${memGood.length}\n` +
    `ML 训练数据：${memTrain.length}\n已关注缓存：${memFollowed.size}\n` +
    `刷新页面后生效。`
  );
}

function clearSamples() {
  if (!confirm("确认清空学习样本（垃圾/正常 + ML 训练数据 + 模型权重）？")) return;
  memBad = []; memGood = []; memTrain = [];
  saveArr(KEY_BAD, []); saveArr(KEY_GOOD, []); saveArr(KEY_TRAIN, []);
  model = new OnlineLR(FEATURE_COUNT, cfg.mlLearningRate, cfg.mlL2);
  saveModelState();
  memHRep = {};
  saveObj(KEY_HREP, {});
  alert("已清空所有学习数据和 ML 模型。");
}

function clearFollowed() {
  if (!confirm("确认清空已关注缓存？")) return;
  memFollowed = new Set();
  saveArr(KEY_FOLLOWED, []);
  alert("已清空已关注缓存。");
}

function clearAccountCache() {
  if (!confirm("确认清空账号特征缓存？")) return;
  memAcct = {};
  storageSet(KEY_ACCT, {});
  alert("已清空账号缓存。");
}

// ============================================================
//  X DOM 辅助
// ============================================================
function getTweetTextNode(article) {
  return article.querySelector('div[data-testid="tweetText"]');
}

function getTweetText(article) {
  const n = getTweetTextNode(article);
  if (!n) return "";
  const t1 = (n.innerText || "").trim();
  if (t1) return t1;
  const t2 = (n.textContent || "").trim();
  const emojiParts = [];
  n.querySelectorAll("img").forEach(img => {
    const a = (img.getAttribute("alt") || img.getAttribute("aria-label") || "").trim();
    if (a) emojiParts.push(a);
  });
  return (t2 ? [t2] : []).concat(emojiParts).join("").trim();
}

function getHandle(article) {
  const spans = article.querySelectorAll("span");
  for (const s of spans) {
    const t = (s.innerText || "").trim();
    if (t.startsWith("@") && t.length >= 2 && t.length <= 30) return t;
  }
  return "";
}

function getDisplayName(article) {
  const nameBox = article.querySelector('[data-testid="User-Name"]');
  const spans = (nameBox || article).querySelectorAll("span");
  for (const s of spans) {
    const t = (s.innerText || "").trim();
    if (!t || t.startsWith("@") || /^\d/.test(t) || t === "·" || t.length > 40) continue;
    return t;
  }
  return "";
}

function inReplyContext() {
  return /\/status\/\d+/.test(location.pathname);
}

function hasMediaInArticle(article) {
  return !!(
    article.querySelector('[data-testid="tweetPhoto"]') ||
    article.querySelector("video") ||
    article.querySelector('[data-testid="card.wrapper"]')
  );
}

// ============================================================
//  结构通道：emoji-only / digits-only / mixed
// ============================================================
function hasAlphaOrCJK(str) {
  return /[a-zA-Z一-鿿぀-ゟ゠-ヿ가-힯]/.test(str || "");
}

function isEmojiOnly(str) {
  const t = (str || "").trim();
  if (!t || hasAlphaOrCJK(t) || /\d/.test(t)) return false;
  if (countEmoji(t) <= 0) return false;
  const stripped = t.replace(/[\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}\u{2300}-\u{23FF}\u{2B05}-\u{2B55}\u{FE00}-\u{FE0F}\u{200D}\u{20E3}]/gu, "");
  return /^[\s~`!@#$%^&*()_+\-=\[\]{};:'",.<>/?\\|。！？、，·…（）【】]*$/u.test(stripped);
}

function isDigitsOnly(str) {
  const t = (str || "").trim();
  if (!t || hasAlphaOrCJK(t)) return false;
  const stripped = t.replace(/[\s~`!@#$%^&*()_+\-=\[\]{};:'",.<>/?\\|。！？、，·…（）【】]/gu, "");
  return stripped.length > 0 && /^\d+$/.test(stripped);
}

function isDigitsEmojiOnly(str) {
  const t = (str || "").trim();
  if (!t || hasAlphaOrCJK(t)) return false;
  const stripped = t.replace(/[\s~`!@#$%^&*()_+\-=\[\]{};:'",.<>/?\\|。！？、，·…（）【】]/gu, "");
  return /^[\d\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}\u{2300}-\u{23FF}\u{2B05}-\u{2B55}\u{FE00}-\u{FE0F}\u{200D}\u{20E3}]+$/u.test(stripped);
}

function structuralDecision(text) {
  if (!cfg.structuralChannel) return { hit: false };
  if (cfg.structuralReplyOnly && !inReplyContext()) return { hit: false };

  const emojiN = countEmoji(text);
  const digitN = countDigits(text);

  if (isEmojiOnly(text) && emojiN >= cfg.emojiOnlyMinCount)
    return { hit: true, kind: "emoji-only", score: 1.0, emojiN, digitN };
  if (isDigitsOnly(text) && digitN >= cfg.digitsOnlyMinCount)
    return { hit: true, kind: "digits-only", score: 1.0, emojiN, digitN };
  if (isDigitsEmojiOnly(text) && emojiN >= cfg.mixedMinEmoji && digitN >= cfg.mixedMinDigits)
    return { hit: true, kind: "digits+emoji-only", score: 1.0, emojiN, digitN };

  return { hit: false };
}

// ============================================================
//  启发式通道：零训练即可工作，基于账号 + 内容特征评分
// ============================================================
function heuristicDecision(article) {
  if (!cfg.heuristicChannel) return { hit: false };

  const text = getTweetText(article);
  const handle = getHandle(article);
  const name = getDisplayName(article);
  const hClean = (handle || "").replace(/^@/, "");
  if (!hClean) return { hit: false };

  const replyMode = inReplyContext();
  let score = 0;
  const parts = [];

  const hLen = hClean.length;
  const hDigits = hClean.replace(/\D/g, "").length;
  const hDigitR = hLen > 0 ? hDigits / hLen : 0;

  if (hDigitR >= 0.5 && hDigits >= 4) {
    score += 0.30; parts.push("handle 高数字比");
  } else if (hDigitR >= 0.35 && hDigits >= 3) {
    score += 0.15; parts.push("handle 数字比偏高");
  }

  if (/\d{6,}$/.test(hClean)) {
    score += 0.20; parts.push("handle 尾部长数字串");
  } else if (/\d{4,5}$/.test(hClean)) {
    score += 0.10; parts.push("handle 尾部数字");
  }

  if (/^[a-z]{1,5}\d{6,}$/i.test(hClean) || /^[a-z]{1,4}_[a-z]{1,4}\d{5,}$/i.test(hClean)) {
    score += 0.20; parts.push("handle 疑似自动生成");
  }

  if (hLen > 14 && charEntropy(hClean) > 3.2) {
    score += 0.10; parts.push("handle 高随机度");
  }

  const tLen = (text || "").length;
  const risk = textRiskProfile(text);

  if (risk.contact && (risk.sex || risk.fraud || risk.money || risk.hasURL)) {
    score += 0.30; parts.push("联系方式+风险内容");
  } else if (risk.sex && (risk.money || risk.hasURL)) {
    score += 0.25; parts.push("色情服务锚点");
  } else if (risk.fraud && (risk.contact || risk.money || risk.hasURL)) {
    score += 0.25; parts.push("诈骗/返利锚点");
  } else if (risk.riskScore >= 0.45) {
    score += 0.15; parts.push("内容风险词");
  }

  if (tLen > 0 && tLen < 5) {
    score += 0.15; parts.push("超短回复");
  } else if (tLen >= 5 && tLen < 15) {
    const eR = countEmoji(text) / tLen;
    if (eR > 0.4) { score += 0.10; parts.push("短 emoji 回复"); }
  }

  if (tLen > 0 && !hasAlphaOrCJK(text) && !isEmojiOnly(text) && !isDigitsOnly(text)) {
    score += 0.10; parts.push("无文字内容");
  }

  if (tLen > 4 && textRepetition(text) > 0.6) {
    score += 0.10; parts.push("高重复度");
  }

  if (name) {
    const nEmoji = countEmoji(name);
    if (nEmoji >= 3 && nEmoji / Math.max(name.length, 1) > 0.3) {
      score += 0.10; parts.push("名字多 emoji");
    }
  }

  const acct = handle && memAcct[handle];
  if (acct) {
    const fol = acct.followers || 0;
    const fing = acct.following || 0;
    if (fol < 5 && fing > 50) {
      score += 0.15; parts.push("粉丝极少但关注多");
    } else if (fol < 20 && fing > 200) {
      score += 0.10; parts.push("粉/关比异常");
    }
    if (!acct.bio) {
      score += 0.05; parts.push("无 bio");
    }
  }

  const th = replyMode ? cfg.heuristicThresholdReply : cfg.heuristicThresholdTimeline;

  if (score >= th) {
    return { hit: true, score, reason: parts.join(" + ") };
  }
  return { hit: false, score };
}

// ============================================================
//  悬浮卡片：关注识别 + 账号特征缓存
// ============================================================
function parseCount(text) {
  if (!text) return null;
  const m = text.match(/([\d,]+\.?\d*)\s*([KkMm万]?)/);
  if (!m) return null;
  let n = parseFloat(m[1].replace(/,/g, ""));
  if (isNaN(n)) return null;
  const s = m[2];
  if (s === "K" || s === "k") n *= 1000;
  else if (s === "M" || s === "m") n *= 1000000;
  else if (s === "万") n *= 10000;
  return Math.round(n);
}

function extractAccountFromCard(card) {
  const info = {};
  const spans = card.querySelectorAll("span");
  for (const s of spans) {
    const t = (s.innerText || "").trim();
    if (t.startsWith("@") && t.length >= 2 && t.length <= 30) { info.handle = t; break; }
  }
  if (!info.handle) return null;

  const bioEl = card.querySelector('[data-testid="UserDescription"]');
  if (bioEl) info.bio = (bioEl.textContent || "").trim();

  const links = card.querySelectorAll("a[href]");
  for (const link of links) {
    const href = link.getAttribute("href") || "";
    const text = (link.textContent || "").trim();
    if (href.endsWith("/following")) {
      const n = parseCount(text);
      if (n !== null) info.following = n;
    }
    if (href.endsWith("/followers") || href.endsWith("/verified_followers")) {
      const n = parseCount(text);
      if (n !== null) info.followers = n;
    }
  }

  info.verified = !!(
    card.querySelector('[data-testid="icon-verified"]') ||
    card.querySelector('[aria-label="Verified account"]') ||
    card.querySelector('[aria-label="已验证帐号"]')
  );

  info.ts = Date.now();
  return info;
}

function detectFollowedFromHoverCard(root) {
  const cards = (root || document).querySelectorAll(
    '[data-testid="HoverCard"], [data-testid="UserHoverCard"], [role="dialog"]'
  );
  for (const card of cards) {
    const info = extractAccountFromCard(card);
    if (!info || !info.handle) continue;

    memAcct[info.handle] = info;
    storageSet(KEY_ACCT, memAcct);

    if (!cfg.followedByHover) continue;
    const btns = card.querySelectorAll("button");
    for (const b of btns) {
      const tx = (b.innerText || "").trim();
      if (
        tx === "Following" || tx === "已关注" || tx === "正在关注" ||
        tx === "关注中" || tx === "Unfollow" || tx === "取消关注"
      ) {
        addFollowed(info.handle);
        break;
      }
    }
  }
}

// ============================================================
//  指纹
// ============================================================
function getFingerprintPack(article) {
  const name = getDisplayName(article);
  const handle = getHandle(article);
  const text = getTweetText(article);
  const fingerprint = [name, handle, text].filter(Boolean).join(" | ");
  const raw = `${name ? name + " " : ""}${handle ? handle + " " : ""}${text || ""}`.trim();
  const preview = raw.length > cfg.previewLen ? raw.slice(0, cfg.previewLen) + "…" : raw;
  return { name, handle, text, fingerprint: fingerprint.trim(), preview };
}

// ============================================================
//  屏蔽：隐藏 / 折叠
// ============================================================
function blockHide(article) {
  article.classList.add("xls-hidden");
}

function blockFold(article, sim, where, reasonText) {
  if (article.dataset.xlsFolded === "1") return;

  const textNode = getTweetTextNode(article);
  if (!textNode) return;

  article.dataset.xlsFolded = "1";

  const pack = getFingerprintPack(article);
  const features = cfg.mlEnabled ? extractFeatures(article) : null;
  const preview = pack.preview || "";

  textNode.classList.add("xls-hidden");

  const card = document.createElement("div");
  card.className = "xls-card";

  const channelMap = { struct: "结构", heuristic: "启发式", ml: "ML", reply: "Jaccard", timeline: "Jaccard", manual: "手动" };
  const channelLabel = channelMap[where] || where;
  const scoreLabel = sim != null ? sim.toFixed(2) : "—";

  card.innerHTML = `
    <div class="xls-row">
      <div class="xls-title">已屏蔽：疑似垃圾</div>
      <div class="xls-meta">${channelLabel} ${scoreLabel}</div>
    </div>
    <div class="xls-preview"></div>
    <div class="xls-actions">
      <button class="xls-btn xls-show" type="button">展开本条</button>
      <button class="xls-btn xls-mark-good" type="button">标注：正常</button>
    </div>
    <div class="xls-reason"></div>
  `;
  card.querySelector(".xls-preview").textContent = preview;
  card.querySelector(".xls-reason").textContent =
    reasonText ? `${reasonText}` : `提示：误伤时点「标注：正常」`;

  card.querySelector(".xls-show").addEventListener("click", (e) => {
    e.preventDefault(); e.stopPropagation();
    textNode.classList.remove("xls-hidden");
    card.remove();
    article.dataset.xlsFolded = "0";
  }, true);

  card.querySelector(".xls-mark-good").addEventListener("click", (e) => {
    e.preventDefault(); e.stopPropagation();
    const ok = addSample(KEY_GOOD, pack.fingerprint, pack.preview);

    if (features && cfg.mlEnabled) {
      learnFromLabel(features, 0, {
        text: pack.text, handle: pack.handle, name: pack.name,
        isReply: inReplyContext(), hasMedia: hasMediaInArticle(article),
      });
    }

    toast(ok ? "已标注：正常 + ML 学习" : "已存在：正常样本");
  }, true);

  textNode.parentElement.insertBefore(card, textNode);
}

function block(article, sim, where, reasonText) {
  if (cfg.blockMode === "hide") blockHide(article);
  else blockFold(article, sim, where, reasonText);
}

function forceBlockCurrent(article, sim, where, reasonText) {
  if (!article) return;
  if (article.classList.contains("xls-hidden") || article.dataset.xlsFolded === "1") return;
  block(article, sim, where, reasonText);
}

// ============================================================
//  标记模式：注入按钮
// ============================================================
function injectMarkButtons(article) {
  if (!cfg.markingMode) return;
  if (article.querySelector(".xls-inline-actions")) return;

  const textNode = getTweetTextNode(article);
  if (!textNode) return;

  const bar = document.createElement("div");
  bar.className = "xls-inline-actions";
  bar.innerHTML = `
    <button class="xls-mini xls-mini-bad" type="button">标注垃圾</button>
    <button class="xls-mini xls-mini-good" type="button">标注正常</button>
  `;

  bar.querySelector(".xls-mini-bad").addEventListener("click", (e) => {
    e.preventDefault(); e.stopPropagation();
    const pack = getFingerprintPack(article);
    if (!pack.fingerprint) return toast("没取到内容");

    addSample(KEY_BAD, pack.fingerprint, pack.preview);
    const features = cfg.mlEnabled ? extractFeatures(article) : null;

    if (features && cfg.mlEnabled) {
      learnFromLabel(features, 1, {
        text: pack.text, handle: pack.handle, name: pack.name,
        isReply: inReplyContext(), hasMedia: hasMediaInArticle(article),
      });
    }

    forceBlockCurrent(article, 1.0, "manual", "手动标注垃圾");
    const c = getTrainCounts();
    toast(hasEnoughMlTraining()
      ? "已标注垃圾 + ML 学习"
      : `已标注垃圾；还需正常样本 ${Math.max(0, cfg.mlMinNegativeSamples - c.neg)} 条`);
  }, true);

  bar.querySelector(".xls-mini-good").addEventListener("click", (e) => {
    e.preventDefault(); e.stopPropagation();
    const pack = getFingerprintPack(article);
    if (!pack.fingerprint) return toast("没取到内容");

    const features = cfg.mlEnabled ? extractFeatures(article) : null;
    const ok = addSample(KEY_GOOD, pack.fingerprint, pack.preview);

    if (features && cfg.mlEnabled) {
      learnFromLabel(features, 0, {
        text: pack.text, handle: pack.handle, name: pack.name,
        isReply: inReplyContext(), hasMedia: hasMediaInArticle(article),
      });
    }

    const c = getTrainCounts();
    const msg = hasEnoughMlTraining()
      ? (ok ? "已标注正常 + ML 学习" : "已存在：正常样本")
      : `已标注正常；还需垃圾样本 ${Math.max(0, cfg.mlMinPositiveSamples - c.pos)} 条`;
    toast(msg);
  }, true);

  textNode.parentElement.appendChild(bar);
}

// ============================================================
//  判定：关注免屏蔽 → 结构通道 → 启发式 → ML → Jaccard
// ============================================================
function shouldBlock(article) {
  const handle = getHandle(article);
  const pack = getFingerprintPack(article);
  const textPreview = (pack.text || "").slice(0, 60).replace(/\n/g, " ");
  const risk = textRiskProfile(pack.text || "");

  if (handle && memFollowed.has(handle)) {
    log(`[PASS] ${handle} | 关注白名单 | "${textPreview}"`);
    return { block: false, reason: "followed-user" };
  }

  if (!pack.text && !pack.name && !pack.handle) return { block: false };

  const s = structuralDecision(pack.text || "");
  if (s.hit) {
    log(`[BLOCK] ${handle} | 结构通道: ${s.kind} | emoji=${s.emojiN} digits=${s.digitN} | "${textPreview}"`);
    return {
      block: true, sim: 1.0, where: "struct",
      reasonText: `${s.kind} (emoji=${s.emojiN}, digits=${s.digitN})`,
    };
  }

  const ctxBoost = computeContextBoost(pack.text, handle, pack.name);

  const replyMode = inReplyContext();
  const cjkProtection = getCjkProtectionLevel(risk, replyMode);
  const h = heuristicDecision(article);
  const mlActive = cfg.mlEnabled && hasEnoughMlTraining();
  let mlRaw = null, mlTh = null, features = null;

  if (mlActive) {
    features = extractFeatures(article);
    mlRaw = model.predict(features);
    mlTh = replyMode ? cfg.mlThresholdReply : cfg.mlThresholdTimeline;
    if (cjkProtection > 0) mlTh = Math.min(0.95, mlTh + cjkProtection);
  }

  const logParts = [];
  logParts.push(`handle=${handle}`);
  logParts.push(`reply=${replyMode}`);
  if (risk.riskScore > 0) logParts.push(`risk=${risk.riskScore.toFixed(2)}`);
  if (cjkProtection > 0) logParts.push(`cjk_guard=+${cjkProtection.toFixed(2)}`);
  if (h.score > 0) logParts.push(`heur=${h.score.toFixed(2)}(${h.hit ? "HIT" : "miss"}${h.reason ? ": " + h.reason : ""})`);
  if (mlActive) logParts.push(`ml_raw=${mlRaw.toFixed(3)} th=${mlTh}`);
  if (ctxBoost > 0) logParts.push(`ctx_boost=${ctxBoost.toFixed(2)}`);

  if (h.hit) {
    if (cjkProtection > 0 && h.score < (replyMode ? 0.75 : 0.85)) {
      log(`[OVERRIDE] ${logParts.join(" | ")} | 启发式命中但中文低风险保护 | "${textPreview}"`);
    } else if (!mlActive || mlRaw >= mlTh * 0.5) {
      log(`[BLOCK] ${logParts.join(" | ")} | 决策=启发式 | "${textPreview}"`);
      return {
        block: true, sim: h.score, where: "heuristic",
        reasonText: `启发式 (${h.reason})`,
      };
    } else if (mlRaw !== null) {
      log(`[OVERRIDE] ${logParts.join(" | ")} | 启发式命中但 ML 低分覆盖(${mlRaw.toFixed(3)}<${(mlTh * 0.5).toFixed(3)}) | "${textPreview}"`);
    }
  }

  if (mlActive) {
    const guardedCtxBoost = cjkProtection > 0 ? Math.min(ctxBoost, 0.06) : ctxBoost;
    const mlBoosted = Math.min(mlRaw + guardedCtxBoost, 1);
    logParts.push(`ml_final=${mlBoosted.toFixed(3)}`);
    if (mlBoosted >= mlTh) {
      log(`[BLOCK] ${logParts.join(" | ")} | 决策=ML | "${textPreview}"`);
      return {
        block: true, sim: mlBoosted, where: "ml",
        reasonText: `ML (p=${mlRaw.toFixed(3)}${guardedCtxBoost > 0 ? `+ctx${guardedCtxBoost.toFixed(2)}` : ""}, 训练${memTrain.length}例)`,
      };
    }
    log(`[PASS] ${logParts.join(" | ")} | 决策=ML放行 | "${textPreview}"`);
    return { block: false, prob: mlBoosted };
  }

  if (!pack.fingerprint) return { block: false };
  const grams = makeNgrams(pack.fingerprint, cfg.ngramN);
  if (!grams.length) return { block: false };
  if (!memBad.length) return { block: false };

  const simBad = bestSimilarity(grams, memBad);
  const simGood = memGood.length ? bestSimilarity(grams, memGood) : 0;
  const th = replyMode ? cfg.simThresholdReply : cfg.simThresholdTimeline;

  logParts.push(`jaccard_bad=${simBad.toFixed(3)} good=${simGood.toFixed(3)} th=${th}`);

  if (simGood >= simBad - 0.02) {
    log(`[PASS] ${logParts.join(" | ")} | 决策=Jaccard白样本保护 | "${textPreview}"`);
    return { block: false, simBad, simGood };
  }
  const jaccardGuard = cjkProtection > 0 ? Math.min(0.18, cjkProtection) : 0;
  const adjustedTh = Math.min(0.95, th + jaccardGuard);
  const adjusted = Math.min(simBad + (cjkProtection > 0 ? Math.min(ctxBoost, 0.04) : ctxBoost), 1);
  if (adjusted >= adjustedTh) {
    log(`[BLOCK] ${logParts.join(" | ")} | 决策=Jaccard(adj=${adjusted.toFixed(3)}) | "${textPreview}"`);
    return {
      block: true, sim: adjusted, where: replyMode ? "reply" : "timeline",
      reasonText: `Jaccard (bad=${simBad.toFixed(2)}, good=${simGood.toFixed(2)}${ctxBoost > 0 ? `, ctx+${ctxBoost.toFixed(2)}` : ""})`,
    };
  }
  log(`[PASS] ${logParts.join(" | ")} | 决策=Jaccard未达阈值(adj=${adjusted.toFixed(3)}) | "${textPreview}"`);
  return { block: false, simBad, simGood };
}

// ============================================================
//  主流程
// ============================================================
function processArticle(article) {
  if (!article || article.dataset.xlsChecked === "1") return;
  article.dataset.xlsChecked = "1";

  refreshThreadCtx();
  injectMarkButtons(article);

  const r = shouldBlock(article);
  trackThreadResult(r.block);
  if (r.block) {
    block(article, r.sim, r.where, r.reasonText);
  }
}

function scan(root) {
  const arts = (root || document).querySelectorAll("article");
  for (const a of arts) processArticle(a);
}

const mo = new MutationObserver((mutations) => {
  for (const m of mutations) {
    for (const node of m.addedNodes) {
      if (!(node instanceof HTMLElement)) continue;
      detectFollowedFromHoverCard(node);
      if (node.tagName === "ARTICLE") processArticle(node);
      else scan(node);
    }
  }
});

// ============================================================
//  控制面板（页面内浮动 UI）
// ============================================================
function createControlPanel() {
  const T = (key, label) =>
    `<div class="xls-opt"><span>${label}</span><div class="xls-sw${cfg[key] ? " on" : ""}" data-key="${key}"></div></div>`;
  const N = (key, label, min, max, step) =>
    `<div class="xls-opt"><span>${label}</span><input type="number" class="xls-ni" data-key="${key}" value="${cfg[key]}" min="${min}" max="${max}" step="${step}"></div>`;

  const fab = document.createElement("div");
  fab.className = "xls-fab";
  fab.textContent = "XLS";
  fab.title = "X-Filter 控制面板";

  const bd = document.createElement("div");
  bd.className = "xls-bd";

  const cp = document.createElement("div");
  cp.className = "xls-cp";

  const show = () => { bd.classList.add("open"); cp.classList.add("open"); refreshStats(); };
  const hide = () => { bd.classList.remove("open"); cp.classList.remove("open"); };
  fab.addEventListener("click", show);
  bd.addEventListener("click", hide);

  cp.innerHTML = `
<div class="xls-cp-hd"><span>X-Filter v5.1.0</span><button class="xls-cp-x">&times;</button></div>
<div class="xls-cp-bd">

  <div class="xls-sec">
    <div class="xls-sec-t">模式</div>
    ${T("markingMode", "标记模式（推文旁显示标注按钮）")}
    <div class="xls-opt">
      <span>屏蔽方式</span>
      <button class="xls-mdb" id="xls-bm">${cfg.blockMode === "fold" ? "折叠卡片" : "直接隐藏"}</button>
    </div>
    ${T("mlEnabled", "ML 分类器")}
    ${T("heuristicChannel", "启发式通道（零训练自动拦截）")}
    ${T("followedByHover", "关注免屏蔽")}
    ${T("structuralChannel", "结构通道（emoji/digits-only 直杀）")}
    ${T("debug", "调试日志")}
  </div>

  <div class="xls-sec">
    <div class="xls-sec-t">ML 参数</div>
    ${N("mlThresholdReply", "回复区阈值", 0.1, 0.95, 0.05)}
    ${N("mlThresholdTimeline", "时间线阈值", 0.1, 0.95, 0.05)}
    ${N("mlMinSamples", "最小训练样本数", 1, 100, 1)}
  </div>

  <div class="xls-sec">
    <div class="xls-sec-t">启发式参数</div>
    ${N("heuristicThresholdReply", "回复区阈值（越低越激进）", 0.15, 0.90, 0.05)}
    ${N("heuristicThresholdTimeline", "时间线阈值", 0.30, 0.95, 0.05)}
  </div>

  <div class="xls-sec">
    <div class="xls-sec-t">Jaccard 回退</div>
    ${N("simThresholdReply", "回复区阈值", 0.05, 0.95, 0.05)}
    ${N("simThresholdTimeline", "时间线阈值", 0.05, 0.95, 0.05)}
    ${N("ngramN", "n-gram N", 2, 6, 1)}
  </div>

  <div class="xls-sec">
    <div class="xls-sec-t">结构通道</div>
    ${N("emojiOnlyMinCount", "emoji-only 最小数", 1, 10, 1)}
    ${N("digitsOnlyMinCount", "digits-only 最小数", 1, 10, 1)}
    ${N("mixedMinEmoji", "混合 emoji 最小", 1, 10, 1)}
    ${N("mixedMinDigits", "混合 digits 最小", 1, 10, 1)}
  </div>

  <div class="xls-sec">
    <div class="xls-sec-t">统计</div>
    <div class="xls-sg" id="xls-stats"></div>
  </div>

  <div class="xls-sec">
    <div class="xls-sec-t">操作</div>
    <div class="xls-cp-acts">
      <button class="xls-cp-ab" id="xls-a-export">导出 JSON</button>
      <button class="xls-cp-ab" id="xls-a-import">导入 JSON</button>
      <button class="xls-cp-ab" id="xls-a-retrain">重新训练 ML</button>
      <button class="xls-cp-ab red" id="xls-a-clear">清空学习数据</button>
      <button class="xls-cp-ab red" id="xls-a-cfol">清空关注缓存</button>
      <button class="xls-cp-ab red" id="xls-a-cacct">清空账号缓存</button>
    </div>
  </div>

</div>`;

  cp.querySelector(".xls-cp-x").addEventListener("click", hide);

  cp.querySelectorAll(".xls-sw").forEach(sw => {
    sw.addEventListener("click", () => {
      const k = sw.dataset.key;
      cfg[k] = !cfg[k];
      storageSet(k, cfg[k]);
      sw.classList.toggle("on");
      toast(`${k}: ${cfg[k] ? "开" : "关"}`);
    });
  });

  cp.querySelector("#xls-bm").addEventListener("click", () => {
    cfg.blockMode = cfg.blockMode === "fold" ? "hide" : "fold";
    storageSet("blockMode", cfg.blockMode);
    cp.querySelector("#xls-bm").textContent = cfg.blockMode === "fold" ? "折叠卡片" : "直接隐藏";
    toast(`屏蔽方式: ${cfg.blockMode}`);
  });

  cp.querySelectorAll(".xls-ni").forEach(inp => {
    inp.addEventListener("change", () => {
      const k = inp.dataset.key;
      let v = parseFloat(inp.value);
      v = Math.max(parseFloat(inp.min), Math.min(parseFloat(inp.max), v));
      if (inp.step === "1") v = Math.round(v);
      inp.value = v;
      cfg[k] = v;
      storageSet(k, v);
      toast(`${k}: ${v}`);
    });
  });

  cp.querySelector("#xls-a-export").addEventListener("click", () => { hide(); exportData(); });
  cp.querySelector("#xls-a-import").addEventListener("click", () => { hide(); importData(); });
  cp.querySelector("#xls-a-retrain").addEventListener("click", () => { retrainModel(); refreshStats(); });
  cp.querySelector("#xls-a-clear").addEventListener("click", () => { clearSamples(); refreshStats(); });
  cp.querySelector("#xls-a-cfol").addEventListener("click", () => { clearFollowed(); refreshStats(); });
  cp.querySelector("#xls-a-cacct").addEventListener("click", () => { clearAccountCache(); refreshStats(); });

  function refreshStats() {
    const el = cp.querySelector("#xls-stats");
    if (!el) return;
    const ms = model.getState();
    const c = getTrainCounts();
    const active = cfg.mlEnabled && hasEnoughMlTraining();
    el.innerHTML = `
      <span>垃圾样本</span><span class="xls-sv">${memBad.length}</span>
      <span>正常样本</span><span class="xls-sv">${memGood.length}</span>
      <span>ML 训练数据</span><span class="xls-sv">${memTrain.length}</span>
      <span>ML 垃圾/正常</span><span class="xls-sv">${c.pos}/${c.neg}</span>
      <span>ML 训练步数</span><span class="xls-sv">${ms.n}</span>
      <span>ML 状态</span><span class="xls-sv">${active ? "已激活" : "等正/负样本"}</span>
      <span>已关注缓存</span><span class="xls-sv">${memFollowed.size}</span>
      <span>账号缓存</span><span class="xls-sv">${Object.keys(memAcct).length}</span>
      <span>Handle 声誉</span><span class="xls-sv">${Object.keys(memHRep).length}</span>
    `;
  }

  document.body.append(fab, bd, cp);
}

// ============================================================
//  启动（异步加载配置和数据）
// ============================================================
async function start() {
  const configKeys = Object.keys(DEFAULT);
  const configResult = await chrome.storage.local.get(configKeys);
  for (const k of configKeys) {
    cfg[k] = configResult[k] !== undefined ? configResult[k] : DEFAULT[k];
  }
  log = (...a) => cfg.debug && console.log("[X-Filter v5.1]", ...a);

  const dataKeys = [KEY_BAD, KEY_GOOD, KEY_FOLLOWED, KEY_TRAIN, KEY_HREP, KEY_ACCT, KEY_MODEL];
  const data = await chrome.storage.local.get(dataKeys);

  memBad = data[KEY_BAD] || [];
  memGood = data[KEY_GOOD] || [];
  memFollowed = new Set(data[KEY_FOLLOWED] || []);
  memTrain = data[KEY_TRAIN] || [];
  memHRep = data[KEY_HREP] || {};

  const raw = data[KEY_ACCT] || {};
  const now = Date.now();
  const TTL = 7 * 24 * 3600e3;
  memAcct = {};
  for (const [k, v] of Object.entries(raw)) {
    if (now - (v.ts || 0) < TTL) memAcct[k] = v;
  }

  model = new OnlineLR(FEATURE_COUNT, cfg.mlLearningRate, cfg.mlL2);
  const savedModel = data[KEY_MODEL];
  const savedModelCompatible = !!(
    savedModel &&
    savedModel.fv === FEATURE_VERSION &&
    savedModel.fc === FEATURE_COUNT &&
    Array.isArray(savedModel.w) &&
    savedModel.w.length === FEATURE_COUNT
  );
  if (savedModelCompatible) model.setState(savedModel);
  const needAutoRetrain = !savedModelCompatible;
  if (needAutoRetrain && memTrain.length) autoRetrain();

  createControlPanel();
  detectFollowedFromHoverCard(document);
  scan();
  mo.observe(document.body, { childList: true, subtree: true });
  log(
    "Started. bad=", memBad.length,
    "good=", memGood.length,
    "ml.n=", model.n,
    "train=", memTrain.length,
    "followed=", memFollowed.size,
    "acctCache=", Object.keys(memAcct).length,
  );
}

// ============================================================
//  消息监听（popup 通信）
// ============================================================
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "getStats") {
    const s = model.getState();
    const c = getTrainCounts();
    sendResponse({
      bad: memBad.length,
      good: memGood.length,
      train: memTrain.length,
      trainPos: c.pos,
      trainNeg: c.neg,
      trainSteps: s.n,
      mlActive: cfg.mlEnabled && hasEnoughMlTraining(),
      followed: memFollowed.size,
      accounts: Object.keys(memAcct).length,
      handleRep: Object.keys(memHRep).length,
    });
    return true;
  }
  if (message.type === "configChanged") {
    if (message.key && message.value !== undefined) {
      cfg[message.key] = message.value;
      if (message.key === "debug") {
        log = (...a) => cfg.debug && console.log("[X-Filter v5.1]", ...a);
      }
    } else {
      const configKeys = Object.keys(DEFAULT);
      chrome.storage.local.get(configKeys).then(result => {
        for (const k of configKeys) {
          if (result[k] !== undefined) cfg[k] = result[k];
        }
        log = (...a) => cfg.debug && console.log("[X-Filter v5.1]", ...a);
      });
    }
    return false;
  }
  if (message.type === "action") {
    switch (message.action) {
      case "export": exportData(); break;
      case "import": if (message.data) importData(message.data); break;
      case "retrain": retrainModel(); break;
      case "clearSamples": clearSamples(); break;
      case "clearFollowed": clearFollowed(); break;
      case "clearAccounts": clearAccountCache(); break;
    }
    return false;
  }
  return false;
});

// ============================================================
//  入口
// ============================================================
if (document.readyState === "complete" || document.readyState === "interactive") {
  setTimeout(start, 600);
} else {
  window.addEventListener("DOMContentLoaded", () => setTimeout(start, 600));
}
