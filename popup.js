const DEFAULTS = {
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

const STORAGE_KEYS = [
  "xls_samplesBad_v4",
  "xls_samplesGood_v4",
  "xls_followedHandles_v1",
  "xls_modelState_v1",
  "xls_trainData_v1",
  "xls_accountCache_v1",
  "xls_handleRep_v1",
];

let config = {};
let tabId = null;

async function getActiveTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (tab && tab.url && (tab.url.includes("x.com") || tab.url.includes("twitter.com"))) {
    tabId = tab.id;
    return tab;
  }
  return null;
}

function sendToContent(msg) {
  if (!tabId) return Promise.reject(new Error("no tab"));
  return chrome.tabs.sendMessage(tabId, msg).catch(() => {});
}

async function loadConfig() {
  const keys = Object.keys(DEFAULTS);
  const result = await chrome.storage.local.get(keys);
  for (const k of keys) {
    config[k] = result[k] !== undefined ? result[k] : DEFAULTS[k];
  }
}

function updateUI() {
  document.querySelectorAll(".toggle").forEach(el => {
    const key = el.dataset.key;
    if (key) el.classList.toggle("on", !!config[key]);
  });

  document.querySelectorAll(".num-input").forEach(el => {
    const key = el.dataset.key;
    if (key && config[key] !== undefined) el.value = config[key];
  });

  const bmBtn = document.getElementById("btn-blockMode");
  if (bmBtn) {
    bmBtn.textContent = config.blockMode === "fold" ? "折叠卡片" : "直接隐藏";
  }
}

async function refreshStats() {
  if (!tabId) return;
  try {
    const stats = await sendToContent({ type: "getStats" });
    if (!stats) return;
    document.getElementById("s-bad").textContent = stats.bad || 0;
    document.getElementById("s-good").textContent = stats.good || 0;
    document.getElementById("s-train").textContent = stats.train || 0;
    document.getElementById("s-train-ratio").textContent = `${stats.trainPos || 0}/${stats.trainNeg || 0}`;
    document.getElementById("s-steps").textContent = stats.trainSteps || 0;
    document.getElementById("s-followed").textContent = stats.followed || 0;
    document.getElementById("s-accounts").textContent = stats.accounts || 0;
    document.getElementById("s-hrep").textContent = stats.handleRep || 0;
    const statusEl = document.getElementById("stats-status");
    if (statusEl) {
      const dot = stats.mlActive ? "active" : "inactive";
      statusEl.innerHTML = `<span class="status-dot ${dot}"></span>${stats.mlActive ? "ML 已激活" : "等正/负样本"}`;
    }
  } catch (e) {
    // tab might not have content script loaded
  }
}

async function saveAndNotify(key, value) {
  config[key] = value;
  await chrome.storage.local.set({ [key]: value });
  if (tabId) {
    chrome.tabs.sendMessage(tabId, { type: "configChanged", key, value }).catch(() => {});
  }
}

document.addEventListener("DOMContentLoaded", async () => {
  await loadConfig();
  await getActiveTab();
  updateUI();
  if (tabId) refreshStats();

  document.querySelectorAll(".toggle").forEach(el => {
    el.addEventListener("click", async () => {
      const key = el.dataset.key;
      if (!key) return;
      const val = !config[key];
      await saveAndNotify(key, val);
      el.classList.toggle("on", val);
    });
  });

  document.querySelectorAll(".num-input").forEach(el => {
    el.addEventListener("change", async () => {
      const key = el.dataset.key;
      if (!key) return;
      let v = parseFloat(el.value);
      v = Math.max(parseFloat(el.min), Math.min(parseFloat(el.max), v));
      if (el.step === "1") v = Math.round(v);
      el.value = v;
      await saveAndNotify(key, v);
    });
  });

  const bmBtn = document.getElementById("btn-blockMode");
  if (bmBtn) {
    bmBtn.addEventListener("click", async () => {
      const val = config.blockMode === "fold" ? "hide" : "fold";
      await saveAndNotify("blockMode", val);
      bmBtn.textContent = val === "fold" ? "折叠卡片" : "直接隐藏";
    });
  }

  document.getElementById("action-export").addEventListener("click", async () => {
    if (tabId) {
      sendToContent({ type: "action", action: "export" });
    } else {
      const data = {};
      const keys = [...Object.keys(DEFAULTS), ...STORAGE_KEYS];
      const all = await chrome.storage.local.get(keys);
      const payload = {
        schema: "x-learnshield-export",
        version: 5,
        exportedAt: new Date().toISOString(),
        config: {},
        samplesBad: all["xls_samplesBad_v4"] || [],
        samplesGood: all["xls_samplesGood_v4"] || [],
        followedHandles: all["xls_followedHandles_v1"] || [],
        trainData: all["xls_trainData_v1"] || [],
        modelState: all["xls_modelState_v1"] || {},
        accountCache: all["xls_accountCache_v1"] || {},
        handleReputation: all["xls_handleRep_v1"] || {},
      };
      for (const k of Object.keys(DEFAULTS)) payload.config[k] = all[k] !== undefined ? all[k] : DEFAULTS[k];
      const blob = new Blob([JSON.stringify(payload)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `x-filter-${new Date().toISOString().replace(/[:.]/g, "-")}.json`;
      a.click();
      URL.revokeObjectURL(url);
    }
  });

  document.getElementById("action-import").addEventListener("click", async () => {
    const json = prompt("把导出的 JSON 粘贴到这里（会合并去重）");
    if (!json) return;
    if (tabId) {
      sendToContent({ type: "action", action: "import", data: json });
    }
  });

  document.getElementById("action-retrain").addEventListener("click", () => {
    sendToContent({ type: "action", action: "retrain" });
    setTimeout(refreshStats, 500);
  });

  document.getElementById("action-clear").addEventListener("click", () => {
    if (!confirm("确认清空学习样本（垃圾/正常 + ML 训练数据 + 模型权重）？")) return;
    sendToContent({ type: "action", action: "clearSamples" });
    setTimeout(refreshStats, 500);
  });

  document.getElementById("action-cfol").addEventListener("click", () => {
    if (!confirm("确认清空已关注缓存？")) return;
    sendToContent({ type: "action", action: "clearFollowed" });
    setTimeout(refreshStats, 500);
  });

  document.getElementById("action-cacct").addEventListener("click", () => {
    if (!confirm("确认清空账号特征缓存？")) return;
    sendToContent({ type: "action", action: "clearAccounts" });
    setTimeout(refreshStats, 500);
  });
});
