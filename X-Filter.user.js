// ==UserScript==
// @name         X-filter
// @namespace    https://tampermonkey.net/
// @version      5.2.0
// @description  内置轻量在线逻辑回归分类器，从用户标注学习"文本特征+账号特征"联合判定垃圾；Jaccard 相似度作为特征之一；结构通道处理 emoji/digits-only；加入中文误伤保护、风险词锚点与软诱导组合判定；支持标记模式、折叠/隐藏、导入导出；关注免屏蔽
// @author       好奇猫a & Izumi Chino
// @match        https://x.com/*
// @match        https://twitter.com/*
// @grant        GM_getValue
// @grant        GM_setValue
// @grant        GM_registerMenuCommand
// ==/UserScript==

(() => {
  "use strict";

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

    softBaitChannel: true,
    softBaitThresholdReply: 0.62,
    softBaitThresholdTimeline: 0.88,
    lowInfoChannel: true,
    lowInfoThresholdReply: 0.70,
    lowInfoThresholdTimeline: 0.92,

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
  //  读取配置
  // ============================================================
  const cfg = {};
  for (const k of Object.keys(DEFAULT)) {
    cfg[k] = GM_getValue(k, DEFAULT[k]);
  }

  const log = (...a) => cfg.debug && console.log("[X ML-Shield v5.2]", ...a);

  // ============================================================
  //  存储 + 内存缓存
  // ============================================================
  function loadArr(key) { return GM_getValue(key, []); }
  function saveArr(key, a) { GM_setValue(key, a); }
  function loadObj(key) { return GM_getValue(key, {}); }
  function saveObj(key, o) { GM_setValue(key, o); }

  let memBad = loadArr(KEY_BAD);
  let memGood = loadArr(KEY_GOOD);
  let memFollowed = new Set(loadArr(KEY_FOLLOWED));
  let memTrain = loadArr(KEY_TRAIN);
  let memHRep = loadObj(KEY_HREP);

  // 账号缓存：加载时顺便清理 >7 天的
  function loadAccountCache() {
    const raw = GM_getValue(KEY_ACCT, {});
    const now = Date.now();
    const TTL = 7 * 24 * 3600e3;
    const out = {};
    for (const [k, v] of Object.entries(raw)) {
      if (now - (v.ts || 0) < TTL) out[k] = v;
    }
    return out;
  }
  let memAcct = loadAccountCache();

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
      .replace(/[，。！？、,.!?:;'""'“”‘’（）()\[\]【】<>《》~`|\\/]/g, "");
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
    const contact = /(?:vx|wx|wechat|weixin|tg|telegram|qq|whatsapp|line|私信|私聊|联系我|加v|加vx|加微|加微信|加薇|v信|微\s*信|电报|站内信|dm)/i.test(compact) ? 1 : 0;
    const sex = /(?:约炮|约啪|约p|包夜|上门服务|外约|裸聊|全套|外围|陪睡|学生妹|成人视频|无码视频|开房|一夜情|同城约|附近妹)/i.test(compact) ? 1 : 0;
    const fraud = /(?:刷单返利|返利群|提现异常|稳赚|日结兼职|无抵押贷款|刷流水|博彩|杀猪盘|送彩金|代投|开户链接|高收益|带单|资金盘|虚拟币搬砖|usdt搬砖)/i.test(compact) ? 1 : 0;
    const money = /(?:¥|￥|\$|\d+(?:\.\d+)?(?:w|W|万|k|K|元|块|刀|美元|usdt|USDT))/i.test(raw) ? 1 : 0;
    const softBait = computeSoftBaitSignal(raw, compact);
    const lowInfo = computeLowInfoSignal(raw, compact);
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
      softBait: softBait.score,
      softBaitHits: softBait.hits,
      lowInfo: lowInfo.score,
      lowInfoHits: lowInfo.hits,
      rep,
      riskScore,
      hasStrongAnchor: riskScore >= 0.35,
    };
  }

  function computeSoftBaitSignal(raw, compact) {
    const hits = [];
    const add = (name, re) => {
      if (re.test(compact)) hits.push(name);
    };

    add("affection", /(?:想被爱|缺爱|没人爱|求抱抱|想要抱抱|需要抱抱|抱抱我|哄哄我)/i);
    add("lonely", /(?:好孤独|好寂寞|没人陪|陪陪我|找人陪|一个人好难过|睡不着|失眠了)/i);
    add("breakup", /(?:刚分手|分手了|失恋了|被甩了|前任|单身了)/i);
    add("chat", /(?:想聊天|找人聊天|有人聊天吗|谁来聊天|来个人聊天|私聊我|私信我|dm我)/i);
    add("flirt", /(?:哥哥在吗|姐姐在吗|宝贝在吗|有人要我吗|想谈恋爱|想找对象|想被宠|想被疼)/i);

    const len = (raw || "").trim().length;
    const emojiN = countEmoji(raw);
    const emojiRatio = len > 0 ? emojiN / len : 0;
    const laugh = /(?:哈){3,}|(?:hhh){2,}|(?:lol){2,}/i.test(compact) ? 1 : 0;
    const questionish = /[?？]/.test(raw) || /(?:吗|嘛|呢|呀|啦)$/.test(compact) ? 1 : 0;

    if (!hits.length) return { score: 0, hits };

    let score = Math.min(hits.length, 3) * 0.18;
    if (len > 0 && len <= 36) score += 0.10;
    if (emojiN >= 2) score += 0.08;
    if (emojiRatio >= 0.12) score += 0.06;
    if (laugh) score += 0.04;
    if (questionish) score += 0.04;

    return { score: Math.min(score, 0.58), hits };
  }

  function computeLowInfoSignal(raw, compact) {
    const text = (raw || "").trim();
    const len = text.length;
    const hits = [];
    if (!len) return { score: 0, hits };

    const cjkCount = countMatches(text, /[\u4E00-\u9FFF]/gu);
    const latinCount = countMatches(text, /[A-Za-z]/g);
    const emojiN = countEmoji(text);
    const digitN = countDigits(text);
    const meaningfulChars = cjkCount + latinCount + digitN;
    const uniqueRatio = len > 0 ? new Set(text).size / len : 0;
    const rep = textRepetition(text);

    if (len <= 8) hits.push("极短");
    else if (len <= 18) hits.push("短文本");

    if (emojiN >= 2) hits.push("多 emoji");
    if (meaningfulChars > 0 && meaningfulChars <= 4 && len <= 18) hits.push("低文字量");
    if (rep > 0.45 || uniqueRatio < 0.45) hits.push("重复");
    if (/^(?:哈哈+|hhh+|笑死|顶|支持|厉害|不错|可以|确实|是的|对的|看看|来了|围观|mark|666|6+|牛+|赞+)$/i.test(compact)) hits.push("模板附和");
    if (/^(?:[!?？！，。,.~·…\-\s]|\p{Emoji_Presentation})+$/u.test(text)) hits.push("符号表情");

    if (!hits.length) return { score: 0, hits };

    let score = Math.min(hits.length, 4) * 0.12;
    if (len <= 8) score += 0.08;
    if (emojiN >= 2 && meaningfulChars <= 8) score += 0.08;
    if (rep > 0.45) score += 0.06;

    return { score: Math.min(score, 0.56), hits };
  }

  function accountSuspicionProfile(handle, name) {
    const hClean = (handle || "").replace(/^@/, "");
    const hLen = hClean.length;
    const hDigits = hClean.replace(/\D/g, "").length;
    const hDigitRatio = hLen > 0 ? hDigits / hLen : 0;
    const trailingSeq = (hClean.match(/\d+$/) || [""])[0];
    const trailingDigits = trailingSeq.length;
    const hasYearSuffix = /^(?:19|20)\d{2}$/.test(trailingSeq);
    const generated =
      (!hasYearSuffix && /^[a-z]{1,7}\d{4,}$/i.test(hClean)) ||
      /^[a-z]{1,4}_[a-z]{1,6}\d{4,}$/i.test(hClean) ||
      /^[a-z]+[a-z]\d{5,}$/i.test(hClean);
    const nameEmojiRatio = name ? countEmoji(name) / Math.max(name.length, 1) : 0;

    let score = 0;
    const parts = [];

    if (hDigitRatio >= 0.50 && hDigits >= 4) { score += 0.30; parts.push("handle 高数字比"); }
    else if (hDigitRatio >= 0.35 && hDigits >= 3) { score += 0.15; parts.push("handle 数字比偏高"); }

    if (trailingDigits >= 6) { score += 0.20; parts.push("handle 尾部长数字串"); }
    else if (trailingDigits >= 4 && !hasYearSuffix) { score += 0.10; parts.push("handle 尾部数字"); }
    else if (trailingDigits >= 4) { score += 0.03; parts.push("handle 年份尾号"); }

    if (generated) { score += 0.20; parts.push("handle 疑似自动生成"); }
    if (hLen > 14 && charEntropy(hClean) > 3.2) { score += 0.10; parts.push("handle 高随机度"); }
    if (nameEmojiRatio > 0.30 && countEmoji(name) >= 2) { score += 0.08; parts.push("名字 emoji 偏多"); }

    return {
      score: Math.min(score, 0.65),
      parts,
      hLen,
      hDigits,
      hDigitRatio,
      trailingDigits,
      hasYearSuffix,
      generated,
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
  const BASE_FEATURE_COUNT = 33;
  const HASH_DIM = 256;
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

  const FEATURE_VERSION = 5;

  let model = new OnlineLR(FEATURE_COUNT, cfg.mlLearningRate, cfg.mlL2);
  const savedModel = GM_getValue(KEY_MODEL, null);
  const savedModelCompatible = !!(
    savedModel &&
    savedModel.fv === FEATURE_VERSION &&
    savedModel.fc === FEATURE_COUNT &&
    Array.isArray(savedModel.w) &&
    savedModel.w.length === FEATURE_COUNT
  );
  if (savedModelCompatible) model.setState(savedModel);

  function saveModelState() {
    const st = model.getState();
    st.fv = FEATURE_VERSION;
    st.fc = FEATURE_COUNT;
    GM_setValue(KEY_MODEL, st);
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

  const needAutoRetrain = !savedModelCompatible;

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
  //  特征提取  (33 维 + hashing trick, 全部归一化到 ~0-1)
  // ============================================================
  //  0  bias               1
  //  1  textLen             log(1+len)/7
  //  2  emojiRatio          emoji/len
  //  3  digitRatio          digits/len
  //  4  contentRisk         约炮/诈骗/联系方式/金额/URL 等综合风险
  //  5  hasURL              0/1
  //  6  charDiversity       unique/len
  //  7  repetition          bigram repeat ratio
  //  8  handleDigitRatio    digits in handle / handle len
  //  9  handleTrailingDig   has 4+ trailing digits
  // 10  handleEntropy       char entropy / 4.5
  // 11  nameEmojiRatio      emoji in name / name len
  // 12  isReply             0/1
  // 13  hasMedia            0/1
  // 14  simBad              Jaccard vs bad samples
  // 15  simGood             Jaccard vs good samples
  // 16  handleRep           handle reputation (0.5=unknown)
  // 17  hasAcctData         0/1
  // 18  followers           log(1+n)/14
  // 19  following           log(1+n)/14
  // 20  ffRatio             min(followers/following/10, 1)
  // 21  hasBio              0/1
  // 22  isVerified          0/1
  // 23  contactSignal       联系方式/私聊/加微信/TG 等
  // 24  sexualSpamSignal    约炮/色情服务锚点
  // 25  fraudSignal         诈骗/返利/博彩/贷款等锚点
  // 26  mixedScript         中英混排比例
  // 27  strongRiskAnchor    明确风险锚点
  // 28  softBaitSignal      情感/暧昧软诱导弱信号
  // 29  accountSuspicion    handle/name 可疑度
  // 30  softBaitCombo       软诱导 + 账号/场景组合信号
  // 31  lowInfoSignal       低信息量/模板化短回复弱信号
  // 32  lowInfoCombo        低信息量 + 账号/场景组合信号

  function computeFeatures({ text, handle, name, isReply, hasMedia }) {
    const hClean = (handle || "").replace(/^@/, "");
    const risk = textRiskProfile(text);
    const acctSus = accountSuspicionProfile(handle, name);

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
      risk.softBait,
      acctSus.score,
      Math.min(1, risk.softBait + acctSus.score + (isReply ? 0.10 : 0)),
      risk.lowInfo,
      Math.min(1, risk.lowInfo + acctSus.score + (isReply ? 0.10 : 0)),
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

  if (needAutoRetrain && memTrain.length) autoRetrain();

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
    a.download = `x-mlshield-${new Date().toISOString().replace(/[:.]/g, "-")}.json`;
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

  function importData() {
    const json = prompt("把导出的 JSON 粘贴到这里（会合并去重）");
    if (json === null) return;
    let data;
    try { data = JSON.parse(json); } catch { alert("解析失败：不是合法 JSON"); return; }
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
    GM_setValue(KEY_ACCT, memAcct);

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
    GM_setValue(KEY_ACCT, {});
    alert("已清空账号缓存。");
  }

  // ============================================================
  //  样式
  // ============================================================
  function ensureStyle() {
    if (document.getElementById("x-learnshield-style")) return;
    const st = document.createElement("style");
    st.id = "x-learnshield-style";
    st.textContent = `
      .xls-hidden{ display:none !important; }

      .xls-card{
        margin: 8px 12px;
        border: 1px solid rgba(255,255,255,.12);
        background: rgba(255,255,255,.06);
        border-radius: 14px;
        padding: 10px 12px;
        backdrop-filter: blur(6px);
      }
      .xls-row{display:flex;justify-content:space-between;gap:10px;align-items:baseline;}
      .xls-title{font-weight:700;font-size:14px;}
      .xls-meta{font-size:12px;opacity:.7;white-space:nowrap;}
      .xls-preview{
        margin-top:6px;font-size:13px;opacity:.85;line-height:1.35;
        display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;
      }
      .xls-actions{margin-top:8px;display:flex;flex-wrap:wrap;gap:8px;}
      .xls-btn{
        border:1px solid rgba(255,255,255,.16);
        background:rgba(255,255,255,.08);
        color:inherit;border-radius:999px;padding:6px 10px;font-size:12px;cursor:pointer;
      }
      .xls-btn:hover{background:rgba(255,255,255,.12);}
      .xls-reason{margin-top:6px;font-size:12px;opacity:.6;}

      .xls-inline-actions{
        margin-top:6px;
        display:flex; gap:8px; flex-wrap:wrap;
        opacity:.90;
        position: relative;
        z-index: 9999;
        pointer-events: auto !important;
      }
      .xls-mini{
        border:1px solid rgba(255,255,255,.14);
        background:rgba(255,255,255,.06);
        border-radius:999px;
        padding:4px 8px;
        font-size:12px;
        cursor:pointer;
        position: relative;
        z-index: 10000;
        pointer-events: auto !important;
      }
      .xls-mini:hover{background:rgba(255,255,255,.10);}
    `;
    document.head.appendChild(st);
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
  function countEmoji(str) {
    const m = (str || "").match(/[\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}\u{2300}-\u{23FF}\u{2B05}-\u{2B55}]/gu);
    return m ? m.length : 0;
  }

  function countDigits(str) {
    return (str || "").replace(/\D/g, "").length;
  }

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

    // --- 账号维度 ---
    const acctSus = accountSuspicionProfile(handle, name);
    score += acctSus.score;
    parts.push(...acctSus.parts);

    // --- 内容维度 ---
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

    // --- 名字维度 ---
    if (name) {
      const nEmoji = countEmoji(name);
      if (nEmoji >= 3 && nEmoji / Math.max(name.length, 1) > 0.3) {
        score += 0.10; parts.push("名字多 emoji");
      }
    }

    // --- 账号缓存加成 ---
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

  function softBaitDecision({ text, handle, name, risk, ctxBoost, replyMode }) {
    if (!cfg.softBaitChannel) return { hit: false, score: 0 };
    if (!risk || risk.hasStrongAnchor || risk.softBait <= 0 || !risk.softBaitHits.length) return { hit: false, score: 0 };

    const acctSus = accountSuspicionProfile(handle, name);
    const len = (text || "").trim().length;
    const emojiN = countEmoji(text);
    const shortText = len > 0 && len <= 36;

    let score = 0;
    const parts = [];

    score += risk.softBait;
    parts.push(`软诱导${risk.softBait.toFixed(2)}`);

    if (replyMode) { score += 0.12; parts.push("回复区"); }
    if (shortText) { score += 0.08; parts.push("短文本"); }
    if (emojiN >= 2) { score += 0.06; parts.push("emoji"); }

    if (acctSus.score > 0) {
      score += Math.min(acctSus.score, 0.35);
      parts.push(...acctSus.parts);
    }

    if (ctxBoost >= 0.10) { score += 0.12; parts.push("跨账号相似"); }
    else if (ctxBoost >= 0.06) { score += 0.06; parts.push("上下文相似"); }

    const acctGate = acctSus.score >= 0.20;
    const contextGate = replyMode || ctxBoost >= 0.06;
    const formatGate = shortText || emojiN >= 2;
    const th = replyMode ? cfg.softBaitThresholdReply : cfg.softBaitThresholdTimeline;

    if (acctGate && contextGate && formatGate && score >= th) {
      return { hit: true, score, reason: parts.join(" + ") };
    }
    return { hit: false, score, reason: parts.join(" + ") };
  }

  function lowInfoDecision({ text, handle, name, risk, ctxBoost, replyMode }) {
    if (!cfg.lowInfoChannel) return { hit: false, score: 0 };
    if (!risk || risk.hasStrongAnchor || risk.lowInfo <= 0 || !risk.lowInfoHits.length) return { hit: false, score: 0 };

    const acctSus = accountSuspicionProfile(handle, name);
    const len = (text || "").trim().length;
    const emojiN = countEmoji(text);

    let score = 0;
    const parts = [];

    score += risk.lowInfo;
    parts.push(`低信息${risk.lowInfo.toFixed(2)}`);

    if (replyMode) { score += 0.12; parts.push("回复区"); }
    if (len > 0 && len <= 18) { score += 0.08; parts.push("短文本"); }
    if (emojiN >= 2) { score += 0.05; parts.push("emoji"); }

    if (acctSus.score > 0) {
      score += Math.min(acctSus.score, 0.35);
      parts.push(...acctSus.parts);
    }

    if (ctxBoost >= 0.10) { score += 0.14; parts.push("跨账号相似"); }
    else if (ctxBoost >= 0.06) { score += 0.07; parts.push("上下文相似"); }

    const acctGate = acctSus.score >= 0.25;
    const contextGate = replyMode || ctxBoost >= 0.06;
    const repeatGate = ctxBoost >= 0.06 || risk.lowInfoHits.includes("模板附和") || risk.lowInfoHits.includes("符号表情") || risk.lowInfoHits.includes("重复");
    const th = replyMode ? cfg.lowInfoThresholdReply : cfg.lowInfoThresholdTimeline;

    if (acctGate && contextGate && repeatGate && score >= th) {
      return { hit: true, score, reason: parts.join(" + ") };
    }
    return { hit: false, score, reason: parts.join(" + ") };
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
      GM_setValue(KEY_ACCT, memAcct);

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

    ensureStyle();
    article.dataset.xlsFolded = "1";

    const pack = getFingerprintPack(article);
    const features = cfg.mlEnabled ? extractFeatures(article) : null;
    const preview = pack.preview || "";

    textNode.classList.add("xls-hidden");

    const card = document.createElement("div");
    card.className = "xls-card";

    const channelMap = { struct: "结构", heuristic: "启发式", softbait: "软诱导", lowinfo: "低信息", ml: "ML", reply: "Jaccard", timeline: "Jaccard", manual: "手动" };
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

    ensureStyle();

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
  //  判定：关注免屏蔽 → 结构通道 → ML/Jaccard
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
    const sb = softBaitDecision({
      text: pack.text,
      handle,
      name: pack.name,
      risk,
      ctxBoost,
      replyMode,
    });
    const li = lowInfoDecision({
      text: pack.text,
      handle,
      name: pack.name,
      risk,
      ctxBoost,
      replyMode,
    });
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
    if (risk.softBait > 0) logParts.push(`soft=${risk.softBait.toFixed(2)}(${sb.hit ? "HIT" : "miss"}${sb.reason ? ": " + sb.reason : ""})`);
    if (risk.lowInfo > 0) logParts.push(`lowinfo=${risk.lowInfo.toFixed(2)}(${li.hit ? "HIT" : "miss"}${li.reason ? ": " + li.reason : ""})`);
    if (cjkProtection > 0) logParts.push(`cjk_guard=+${cjkProtection.toFixed(2)}`);
    if (h.score > 0) logParts.push(`heur=${h.score.toFixed(2)}(${h.hit ? "HIT" : "miss"}${h.reason ? ": " + h.reason : ""})`);
    if (mlActive) logParts.push(`ml_raw=${mlRaw.toFixed(3)} th=${mlTh}`);
    if (ctxBoost > 0) logParts.push(`ctx_boost=${ctxBoost.toFixed(2)}`);

    if (sb.hit) {
      log(`[BLOCK] ${logParts.join(" | ")} | 决策=软诱导组合 | "${textPreview}"`);
      return {
        block: true, sim: sb.score, where: "softbait",
        reasonText: `软诱导组合 (${sb.reason})`,
      };
    }

    if (li.hit) {
      log(`[BLOCK] ${logParts.join(" | ")} | 决策=低信息组合 | "${textPreview}"`);
      return {
        block: true, sim: li.score, where: "lowinfo",
        reasonText: `低信息组合 (${li.reason})`,
      };
    }

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
  //  菜单
  // ============================================================
  GM_registerMenuCommand(`标记模式：${cfg.markingMode ? "开" : "关"}（点击切换）`, () => {
    GM_setValue("markingMode", !cfg.markingMode);
    alert("已切换，刷新生效。");
  });

  GM_registerMenuCommand(`屏蔽模式：${cfg.blockMode === "hide" ? "隐藏" : "折叠"}（点击切换）`, () => {
    GM_setValue("blockMode", cfg.blockMode === "hide" ? "fold" : "hide");
    alert("已切换，刷新生效。");
  });

  GM_registerMenuCommand(`ML 模型：${cfg.mlEnabled ? "开" : "关"}（点击切换）`, () => {
    GM_setValue("mlEnabled", !cfg.mlEnabled);
    alert("已切换 ML 模型，刷新生效。");
  });

  GM_registerMenuCommand(`关注免屏蔽：${cfg.followedByHover ? "开" : "关"}`, () => {
    GM_setValue("followedByHover", !cfg.followedByHover);
    alert("已切换，刷新生效。");
  });

  GM_registerMenuCommand(`启发式通道：${cfg.heuristicChannel ? "开" : "关"}（零训练自动拦截）`, () => {
    GM_setValue("heuristicChannel", !cfg.heuristicChannel);
    alert("已切换，刷新生效。");
  });

  GM_registerMenuCommand(`结构通道：${cfg.structuralChannel ? "开" : "关"}`, () => {
    GM_setValue("structuralChannel", !cfg.structuralChannel);
    alert("已切换，刷新生效。");
  });

  GM_registerMenuCommand("学习库：导出 JSON", exportData);
  GM_registerMenuCommand("学习库：导入 JSON（粘贴）", importData);
  GM_registerMenuCommand("学习库：清空全部（样本+ML+声誉）", clearSamples);
  GM_registerMenuCommand("关注缓存：清空", clearFollowed);
  GM_registerMenuCommand("账号缓存：清空", clearAccountCache);
  GM_registerMenuCommand("ML：重新训练模型", retrainModel);

  GM_registerMenuCommand("统计：查看当前状态", () => {
    const s = model.getState();
    const c = getTrainCounts();
    const active = cfg.mlEnabled && hasEnoughMlTraining();
    const topW = Array.from(model.w)
      .map((v, i) => ({ i, v: Math.abs(v) }))
      .sort((a, b) => b.v - a.v)
      .slice(0, 5)
      .map(x => `  #${x.i} = ${model.w[x.i].toFixed(3)}`)
      .join("\n");

    alert(
      `=== X ML-Shield v5.2 ===\n` +
      `垃圾样本：${memBad.length}  正常样本：${memGood.length}\n` +
      `ML 训练数据：${memTrain.length}（垃圾 ${c.pos} / 正常 ${c.neg}）  训练次数：${s.n}\n` +
      `已关注缓存：${memFollowed.size}\n` +
      `账号特征缓存：${Object.keys(memAcct).length}\n` +
      `Handle 声誉库：${Object.keys(memHRep).length}\n\n` +
      `--- ML 设置 ---\n` +
      `启用：${cfg.mlEnabled ? "是" : "否"}  状态：${active ? "已激活" : "等待正/负样本"}\n` +
      `最小样本：总数 ${cfg.mlMinSamples} / 垃圾 ${cfg.mlMinPositiveSamples} / 正常 ${cfg.mlMinNegativeSamples}  学习率：${cfg.mlLearningRate}\n` +
      `阈值：reply=${cfg.mlThresholdReply}  timeline=${cfg.mlThresholdTimeline}\n\n` +
      `--- Jaccard 回退 ---\n` +
      `阈值：reply=${cfg.simThresholdReply}  timeline=${cfg.simThresholdTimeline}\n` +
      `ngramN=${cfg.ngramN}\n\n` +
      `--- Top 5 权重 ---\n${topW}\n` +
      `模式：${cfg.blockMode}  标记：${cfg.markingMode ? "开" : "关"}`
    );
  });

  GM_registerMenuCommand("设置：ML 回复区阈值", () => {
    const v = prompt("mlThresholdReply（0~1，越低越激进；建议 0.45~0.70）", String(cfg.mlThresholdReply));
    if (v === null) return;
    GM_setValue("mlThresholdReply", Math.min(0.95, Math.max(0.1, parseFloat(v))));
    alert("已保存，刷新生效。");
  });

  GM_registerMenuCommand("设置：ML 时间线阈值", () => {
    const v = prompt("mlThresholdTimeline（0~1，建议 0.50~0.80）", String(cfg.mlThresholdTimeline));
    if (v === null) return;
    GM_setValue("mlThresholdTimeline", Math.min(0.95, Math.max(0.1, parseFloat(v))));
    alert("已保存，刷新生效。");
  });

  GM_registerMenuCommand("设置：ML 最小训练样本数", () => {
    const v = prompt("mlMinSamples（少于此数则回退 Jaccard；建议 5~20）", String(cfg.mlMinSamples));
    if (v === null) return;
    GM_setValue("mlMinSamples", Math.max(1, Math.min(100, parseInt(v, 10) || DEFAULT.mlMinSamples)));
    alert("已保存，刷新生效。");
  });

  GM_registerMenuCommand("设置：Jaccard 回复区阈值", () => {
    const v = prompt("simThresholdReply（0~1；建议 0.30~0.55）", String(cfg.simThresholdReply));
    if (v === null) return;
    GM_setValue("simThresholdReply", Math.min(0.95, Math.max(0.05, parseFloat(v))));
    alert("已保存，刷新生效。");
  });

  GM_registerMenuCommand("设置：Jaccard 时间线阈值", () => {
    const v = prompt("simThresholdTimeline（0~1；建议 0.35~0.70）", String(cfg.simThresholdTimeline));
    if (v === null) return;
    GM_setValue("simThresholdTimeline", Math.min(0.95, Math.max(0.05, parseFloat(v))));
    alert("已保存，刷新生效。");
  });

  GM_registerMenuCommand("设置：ngram N", () => {
    const v = prompt("ngramN（整数，推荐 3 或 4）", String(cfg.ngramN));
    if (v === null) return;
    GM_setValue("ngramN", Math.max(2, Math.min(6, parseInt(v, 10) || DEFAULT.ngramN)));
    alert("已保存，刷新生效。");
  });

  GM_registerMenuCommand("设置：emoji-only 最小数量", () => {
    const v = prompt("emojiOnlyMinCount（建议 2~4）", String(cfg.emojiOnlyMinCount));
    if (v === null) return;
    GM_setValue("emojiOnlyMinCount", Math.max(1, Math.min(10, parseInt(v, 10) || DEFAULT.emojiOnlyMinCount)));
    alert("已保存，刷新生效。");
  });

  GM_registerMenuCommand("设置：digits-only 最小数量", () => {
    const v = prompt("digitsOnlyMinCount（建议 2~4）", String(cfg.digitsOnlyMinCount));
    if (v === null) return;
    GM_setValue("digitsOnlyMinCount", Math.max(1, Math.min(10, parseInt(v, 10) || DEFAULT.digitsOnlyMinCount)));
    alert("已保存，刷新生效。");
  });

  GM_registerMenuCommand(`调试日志：${cfg.debug ? "开" : "关"}`, () => {
    GM_setValue("debug", !cfg.debug);
    alert("已切换，刷新生效。");
  });

  // ============================================================
  //  控制面板（页面内浮动 UI）
  // ============================================================
  function createControlPanel() {
    const ps = document.createElement("style");
    ps.textContent = `
.xls-fab{position:fixed;left:16px;bottom:80px;z-index:99999;width:44px;height:44px;border-radius:50%;background:rgba(29,155,240,.85);color:#fff;font-size:11px;font-weight:700;display:flex;align-items:center;justify-content:center;cursor:pointer;box-shadow:0 2px 8px rgba(0,0,0,.35);transition:transform .15s;user-select:none;letter-spacing:.5px;}
.xls-fab:hover{transform:scale(1.12);background:rgba(29,155,240,1);}
.xls-bd{display:none;position:fixed;inset:0;background:rgba(0,0,0,.55);z-index:100000;backdrop-filter:blur(2px);}
.xls-bd.open{display:block;}
.xls-cp{display:none;position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);width:420px;max-height:82vh;z-index:100001;background:#16181c;border:1px solid rgba(255,255,255,.12);border-radius:16px;overflow:hidden;box-shadow:0 8px 32px rgba(0,0,0,.6);color:#e7e9ea;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;flex-direction:column;}
.xls-cp.open{display:flex;}
.xls-cp-hd{display:flex;justify-content:space-between;align-items:center;padding:14px 16px;border-bottom:1px solid rgba(255,255,255,.08);font-weight:700;font-size:15px;}
.xls-cp-x{background:none;border:none;color:#e7e9ea;font-size:22px;cursor:pointer;padding:2px 8px;border-radius:50%;line-height:1;}
.xls-cp-x:hover{background:rgba(255,255,255,.1);}
.xls-cp-bd{overflow-y:auto;padding:0 0 14px;flex:1;}
.xls-sec{padding:14px 16px 2px;}
.xls-sec-t{font-size:11px;font-weight:700;color:#71767b;text-transform:uppercase;letter-spacing:.6px;margin-bottom:8px;}
.xls-opt{display:flex;justify-content:space-between;align-items:center;padding:7px 0;font-size:13px;border-bottom:1px solid rgba(255,255,255,.04);}
.xls-opt:last-child{border-bottom:none;}
.xls-sw{position:relative;width:38px;height:20px;background:rgba(255,255,255,.18);border-radius:10px;cursor:pointer;transition:background .2s;flex-shrink:0;}
.xls-sw.on{background:#1d9bf0;}
.xls-sw::after{content:'';position:absolute;top:2px;left:2px;width:16px;height:16px;background:#fff;border-radius:50%;transition:transform .2s;}
.xls-sw.on::after{transform:translateX(18px);}
.xls-ni{width:62px;background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.12);border-radius:6px;color:#e7e9ea;padding:4px 6px;font-size:12px;text-align:center;outline:none;}
.xls-ni:focus{border-color:#1d9bf0;}
.xls-sg{display:grid;grid-template-columns:1fr auto;gap:4px 16px;font-size:13px;padding:4px 0;}
.xls-sv{font-weight:700;color:#1d9bf0;text-align:right;}
.xls-cp-acts{display:flex;flex-wrap:wrap;gap:6px;padding:8px 0 0;}
.xls-cp-ab{flex:1 1 calc(50% - 3px);text-align:center;border:1px solid rgba(255,255,255,.12);background:rgba(255,255,255,.06);color:#e7e9ea;border-radius:8px;padding:8px 4px;font-size:12px;cursor:pointer;transition:background .15s;}
.xls-cp-ab:hover{background:rgba(255,255,255,.1);}
.xls-cp-ab.red{border-color:rgba(244,33,46,.25);color:#f4212e;}
.xls-cp-ab.red:hover{background:rgba(244,33,46,.1);}
.xls-mdb{border:1px solid rgba(255,255,255,.12);background:rgba(255,255,255,.06);color:#e7e9ea;border-radius:6px;padding:4px 10px;font-size:12px;cursor:pointer;}
.xls-mdb:hover{background:rgba(255,255,255,.1);}
`;
    document.head.appendChild(ps);

    const T = (key, label) =>
      `<div class="xls-opt"><span>${label}</span><div class="xls-sw${cfg[key] ? " on" : ""}" data-key="${key}"></div></div>`;
    const N = (key, label, min, max, step) =>
      `<div class="xls-opt"><span>${label}</span><input type="number" class="xls-ni" data-key="${key}" value="${cfg[key]}" min="${min}" max="${max}" step="${step}"></div>`;

    const fab = document.createElement("div");
    fab.className = "xls-fab";
    fab.textContent = "XLS";
    fab.title = "X ML-Shield 控制面板";

    const bd = document.createElement("div");
    bd.className = "xls-bd";

    const cp = document.createElement("div");
    cp.className = "xls-cp";

    const show = () => { bd.classList.add("open"); cp.classList.add("open"); refreshStats(); };
    const hide = () => { bd.classList.remove("open"); cp.classList.remove("open"); };
    fab.addEventListener("click", show);
    bd.addEventListener("click", hide);

    cp.innerHTML = `
<div class="xls-cp-hd"><span>X ML-Shield v5.2</span><button class="xls-cp-x">&times;</button></div>
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
    ${T("softBaitChannel", "软诱导组合通道")}
    ${T("lowInfoChannel", "低信息短回复组合通道")}
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
    ${N("softBaitThresholdReply", "软诱导回复区阈值", 0.35, 0.95, 0.05)}
    ${N("softBaitThresholdTimeline", "软诱导时间线阈值", 0.50, 0.98, 0.05)}
    ${N("lowInfoThresholdReply", "低信息回复区阈值", 0.45, 0.98, 0.05)}
    ${N("lowInfoThresholdTimeline", "低信息时间线阈值", 0.60, 0.99, 0.05)}
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

    // --- 关闭按钮 ---
    cp.querySelector(".xls-cp-x").addEventListener("click", hide);

    // --- 开关 ---
    cp.querySelectorAll(".xls-sw").forEach(sw => {
      sw.addEventListener("click", () => {
        const k = sw.dataset.key;
        cfg[k] = !cfg[k];
        GM_setValue(k, cfg[k]);
        sw.classList.toggle("on");
        toast(`${k}: ${cfg[k] ? "开" : "关"}`);
      });
    });

    // --- 屏蔽方式切换 ---
    cp.querySelector("#xls-bm").addEventListener("click", () => {
      cfg.blockMode = cfg.blockMode === "fold" ? "hide" : "fold";
      GM_setValue("blockMode", cfg.blockMode);
      cp.querySelector("#xls-bm").textContent = cfg.blockMode === "fold" ? "折叠卡片" : "直接隐藏";
      toast(`屏蔽方式: ${cfg.blockMode}`);
    });

    // --- 数值输入 ---
    cp.querySelectorAll(".xls-ni").forEach(inp => {
      inp.addEventListener("change", () => {
        const k = inp.dataset.key;
        let v = parseFloat(inp.value);
        v = Math.max(parseFloat(inp.min), Math.min(parseFloat(inp.max), v));
        if (inp.step === "1") v = Math.round(v);
        inp.value = v;
        cfg[k] = v;
        GM_setValue(k, v);
        toast(`${k}: ${v}`);
      });
    });

    // --- 操作按钮 ---
    cp.querySelector("#xls-a-export").addEventListener("click", () => { hide(); exportData(); });
    cp.querySelector("#xls-a-import").addEventListener("click", () => { hide(); importData(); });
    cp.querySelector("#xls-a-retrain").addEventListener("click", () => { retrainModel(); refreshStats(); });
    cp.querySelector("#xls-a-clear").addEventListener("click", () => { clearSamples(); refreshStats(); });
    cp.querySelector("#xls-a-cfol").addEventListener("click", () => { clearFollowed(); refreshStats(); });
    cp.querySelector("#xls-a-cacct").addEventListener("click", () => { clearAccountCache(); refreshStats(); });

    // --- 统计刷新 ---
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
  //  启动
  // ============================================================
  function start() {
    ensureStyle();
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

  if (document.readyState === "complete" || document.readyState === "interactive") {
    setTimeout(start, 600);
  } else {
    window.addEventListener("DOMContentLoaded", () => setTimeout(start, 600));
  }
})();
