// regen-rules.js — 用新版 normalizeText 重新生成 rules.json 中所有样本的 grams + sig，
// 并补充至少 15 条多样化白样本，最后更新 _meta.summary。
//
// 用法：node regen-rules.js

'use strict';

const fs = require('fs');
const path = require('path');

const RULES_PATH = path.join(__dirname, 'rules.json');

// ============================================================
//  与 content.js 保持一致的 normalizeText / makeNgrams
// ============================================================
function normalizeText(s) {
  return (s || '')
    // 1. 去除零宽字符、不可见格式字符
    .replace(/[​-‏‪-‮⁠-⁯﻿­͏]/g, '')
    // 2. 剥离 Unicode 组合附加符号（如 ͓̽ ͏ 等）
    .replace(/[̀-ͯ᷀-᷿⃐-⃿]/g, '')
    // 3. 全角转半角
    .replace(/[！-～]/g, c => String.fromCharCode(c.charCodeAt(0) - 0xfee0))
    // 4. 去除末尾随机短尾缀（1-5 位字母数字，前有空白）
    .replace(/\s+[a-z0-9]{1,5}$/i, '')
    // 以下保留原有逻辑
    .toLowerCase()
    .replace(/\s+/g, '')
    .replace(/[，。！？、,.!?:;'"“”‘’（）()\[\]【】<>]/g, '');
}

function makeNgrams(str, n) {
  if (n === undefined || n === null) n = 3;
  const s = normalizeText(str);
  if (!s) return [];
  if (s.length <= n) return [s];
  const grams = [];
  for (let i = 0; i <= s.length - n; i++) grams.push(s.slice(i, i + n));
  return grams;
}

function recompute(sample, ngramN) {
  const t = sample.t || '';
  const sig = normalizeText(t).slice(0, 120);
  const grams = makeNgrams(t, ngramN);
  return { ...sample, sig, grams };
}

// ============================================================
//  追加的多样化白样本（仅文本，sig/grams 由脚本计算）
// ============================================================
const NEW_GOOD_TEXTS = [
  // 普通中文闲聊
  '小明 @xiaoming_daily 今天天气真好，出门散步了一圈，看到很多猫',
  '阿真 @azhen2024 早上吃了肠粉超级好吃，午饭都不想吃了',
  '林深 @linshen_zh 周末在家整理书架，发现好多以前买的书都没看',
  // 日语
  'タナカ @tanaka_jp 今日のラーメンは本当に最高でした、また行きたい',
  '桜 @sakura_dev 新しいアニメのオープニングが好きすぎる',
  // 韩语
  '민지 @minji_kr 오늘 카페에서 공부 좀 했어요 커피가 진짜 맛있어요',
  '준호 @junho_seoul 주말에 친구들이랑 영화 보러 갔는데 재밌었어요',
  // 英文讨论
  'Alex @alexcodes Just finished reading a great book on system design, highly recommend',
  'Sam @samdev_io Anyone else having issues with the latest npm update on macOS?',
  'Maya @maya_writes Working on a new short story this weekend, feels good to write again',
  // emoji 短回复
  '玲 @ling_chan 太可爱了 😊',
  'Zoe @zoe_pix 哈哈哈哈这个梗笑死我了 😂',
  // 游戏
  'GamerCat @gamercat_zh 塞尔达传说王国之泪终于通关了，神作没什么好说的',
  'ProPlayer @apex_pro_jp Apex 上周末打到了大师段位，队友很给力',
  // 动漫
  '阿宅 @otaku_fan_zh 鬼灭之刃柱训练编动画作画真的太燃了',
  'AniLover @anifan2025 推荐大家看葬送的芙莉莲，剧情很治愈节奏也舒服',
  // 技术讨论
  'dev @devhub_cn 今天用 Rust 重写了一个小工具，性能比原来的 Python 版本快了十几倍',
  'codeNinja @code_ninja TypeScript 5.4 的新特性挺实用的，类型推断更聪明了',
  'ml_kid @ml_kid_dev 在复现一篇 NLP 论文，训练时显存差点炸了，得换更小的 batch',
];

// ============================================================
//  主流程
// ============================================================
function main() {
  const raw = fs.readFileSync(RULES_PATH, 'utf-8');
  const data = JSON.parse(raw);

  const ngramN =
    (data.config && data.config.ngramN && typeof data.config.ngramN.value === 'number')
      ? data.config.ngramN.value
      : 3;

  if (!data.data) data.data = {};
  if (!Array.isArray(data.data.badSamples)) data.data.badSamples = [];
  if (!Array.isArray(data.data.goodSamples)) data.data.goodSamples = [];

  const badBefore = data.data.badSamples.length;
  const goodBefore = data.data.goodSamples.length;

  // 1) 重算所有 badSamples
  data.data.badSamples = data.data.badSamples.map(s => recompute(s, ngramN));

  // 2) 重算所有 goodSamples
  data.data.goodSamples = data.data.goodSamples.map(s => recompute(s, ngramN));

  // 3) 追加新白样本（去重：sig 不在已有 goodSamples 中）
  const existingSigs = new Set(data.data.goodSamples.map(s => s.sig));
  let added = 0;
  const baseTs = Date.now();
  NEW_GOOD_TEXTS.forEach((text, i) => {
    const sig = normalizeText(text).slice(0, 120);
    if (!sig || existingSigs.has(sig)) return;
    existingSigs.add(sig);
    data.data.goodSamples.push({
      sig,
      t: text,
      grams: makeNgrams(text, ngramN),
      ts: baseTs + i,
    });
    added++;
  });

  // 4) 更新 _meta.summary
  if (!data._meta) data._meta = {};
  if (!data._meta.summary) data._meta.summary = {};
  data._meta.summary.badSamples = data.data.badSamples.length;
  data._meta.summary.goodSamples = data.data.goodSamples.length;
  data._meta.exportedAt = new Date().toISOString();

  fs.writeFileSync(RULES_PATH, JSON.stringify(data, null, 2) + '\n', 'utf-8');

  console.log('[regen-rules] done');
  console.log(`  badSamples:  ${badBefore} -> ${data.data.badSamples.length} (sig+grams 重算)`);
  console.log(`  goodSamples: ${goodBefore} -> ${data.data.goodSamples.length} (sig+grams 重算, 新增 ${added})`);
  console.log(`  ngramN=${ngramN}`);
  console.log(`  _meta.exportedAt = ${data._meta.exportedAt}`);
}

main();
