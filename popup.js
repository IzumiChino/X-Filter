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

// ============================================================
//  云同步
// ============================================================
function syncStatus(msg, type) {
  const el = document.getElementById("sync-status");
  if (!el) return;
  el.textContent = msg;
  el.className = "sync-status";
  if (type) el.classList.add(type);
}

async function loadSyncSettings() {
  const res = await chrome.storage.local.get(["xls_sync_token", "xls_sync_gist_id", "xls_default_rules_url"]);
  const tokenInput = document.getElementById("sync-token");
  const gistInput = document.getElementById("sync-gist-id");
  const defaultUrlInput = document.getElementById("sync-default-url");
  if (tokenInput && res.xls_sync_token) tokenInput.value = res.xls_sync_token;
  if (gistInput && res.xls_sync_gist_id) gistInput.value = res.xls_sync_gist_id;
  if (defaultUrlInput) {
    defaultUrlInput.value = res.xls_default_rules_url || DEFAULT_RULES_URL;
  }
}

async function saveSyncSettings() {
  const token = document.getElementById("sync-token").value.trim();
  const gistId = document.getElementById("sync-gist-id").value.trim();
  const defaultUrl = document.getElementById("sync-default-url").value.trim();
  if (token) await chrome.storage.local.set({ xls_sync_token: token });
  if (gistId) await chrome.storage.local.set({ xls_sync_gist_id: gistId });
  if (defaultUrl) await chrome.storage.local.set({ xls_default_rules_url: defaultUrl });
}

const DEFAULT_RULES_URL = "https://raw.githubusercontent.com/acnekot/X-Filter/main/rules.json";

async function doLoadDefaultRules() {
  let url = document.getElementById("sync-default-url").value.trim();
  if (!url) url = DEFAULT_RULES_URL;
  if (!tabId) { syncStatus("请在 x.com 页面使用此功能", "error"); return; }

  await saveSyncSettings();
  syncStatus("正在下载默认规则…", "loading");

  try {
    const res = await fetch(url);
    if (!res.ok) {
      syncStatus(`下载失败: HTTP ${res.status} (请确认 rules.json 已推送)`, "error"); return;
    }
    const text = await res.text();
    let payload;
    try { payload = JSON.parse(text); } catch {
      syncStatus("规则内容不是有效 JSON", "error"); return;
    }

    const resp = await sendToContent({ type: "syncImport", payload: payload });
    if (resp && resp.success) {
      const now = new Date().toLocaleString();
      chrome.storage.local.set({ xls_last_default_fetch: Date.now() });
      syncStatus(`默认规则已加载 (${now})`, "success");
      setTimeout(refreshStats, 300);
    } else {
      syncStatus("合并失败", "error");
    }
  } catch (e) {
    syncStatus(`网络错误: ${e.message}`, "error");
  }
}

async function doSyncPush() {
  const token = document.getElementById("sync-token").value.trim();
  const gistId = document.getElementById("sync-gist-id").value.trim();
  if (!token) { syncStatus("请填写 GitHub Token", "error"); return; }
  if (!gistId) { syncStatus("请填写 Gist ID", "error"); return; }
  if (!tabId) { syncStatus("请在 x.com 页面使用此功能", "error"); return; }

  await saveSyncSettings();
  syncStatus("正在导出数据…", "loading");

  try {
    const resp = await sendToContent({ type: "syncExport" });
    if (!resp || !resp.success) {
      syncStatus("导出失败", "error"); return;
    }
    const payload = resp.payload;

    const filename = `x-filter-data.json`;
    const res = await fetch(`https://api.github.com/gists/${gistId}`, {
      method: "PATCH",
      headers: {
        "Authorization": `token ${token}`,
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        files: {
          [filename]: {
            content: JSON.stringify(payload, null, 2),
          },
        },
      }),
    });

    if (res.ok || res.status === 200) {
      const now = new Date().toLocaleString();
      syncStatus(`已推送到 Gist (${now})`, "success");
      chrome.storage.local.set({ xls_last_sync: Date.now() });
    } else {
      const err = await res.json().catch(() => ({}));
      syncStatus(`推送失败: ${err.message || res.status}`, "error");
    }
  } catch (e) {
    syncStatus(`网络错误: ${e.message}`, "error");
  }
}

async function doSyncPull() {
  const token = document.getElementById("sync-token").value.trim();
  const gistId = document.getElementById("sync-gist-id").value.trim();
  if (!token) { syncStatus("请填写 GitHub Token", "error"); return; }
  if (!gistId) { syncStatus("请填写 Gist ID", "error"); return; }
  if (!tabId) { syncStatus("请在 x.com 页面使用此功能", "error"); return; }

  await saveSyncSettings();
  syncStatus("正在从 Gist 拉取…", "loading");

  try {
    const res = await fetch(`https://api.github.com/gists/${gistId}`, {
      headers: {
        "Authorization": `token ${token}`,
        "Accept": "application/vnd.github.v3+json",
      },
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      syncStatus(`拉取失败: ${err.message || res.status}`, "error"); return;
    }

    const gist = await res.json();
    const files = gist.files || {};
    const content = Object.values(files)[0];
    if (!content || !content.content) {
      syncStatus("Gist 中没有找到数据文件", "error"); return;
    }

    let payload;
    try { payload = JSON.parse(content.content); } catch {
      syncStatus("Gist 内容不是有效 JSON", "error"); return;
    }

    const resp = await sendToContent({ type: "syncImport", payload: payload });
    if (resp && resp.success) {
      const now = new Date().toLocaleString();
      syncStatus(`已从 Gist 拉取并合并 (${now})`, "success");
      chrome.storage.local.set({ xls_last_sync: Date.now() });
      setTimeout(refreshStats, 300);
    } else {
      syncStatus("合并失败", "error");
    }
  } catch (e) {
    syncStatus(`网络错误: ${e.message}`, "error");
  }
}

async function doImportFromUrl() {
  const url = document.getElementById("sync-url").value.trim();
  if (!url) { syncStatus("请输入 JSON 文件的 URL", "error"); return; }
  if (!tabId) { syncStatus("请在 x.com 页面使用此功能", "error"); return; }

  syncStatus("正在下载…", "loading");

  try {
    const res = await fetch(url);
    if (!res.ok) {
      syncStatus(`下载失败: HTTP ${res.status}`, "error"); return;
    }
    const text = await res.text();
    let payload;
    try { payload = JSON.parse(text); } catch {
      syncStatus("下载内容不是有效 JSON", "error"); return;
    }

    const resp = await sendToContent({ type: "syncImport", payload: payload });
    if (resp && resp.success) {
      syncStatus("已从 URL 导入并合并", "success");
      setTimeout(refreshStats, 300);
    } else {
      syncStatus("合并失败", "error");
    }
  } catch (e) {
    syncStatus(`网络错误: ${e.message}`, "error");
  }
}

// ============================================================
//  首次安装检测
// ============================================================
async function checkAndOfferDefaults() {
  const res = await chrome.storage.local.get(["xls_samplesBad_v4", "xls_last_default_fetch"]);
  const bad = res["xls_samplesBad_v4"];
  if (!bad || bad.length === 0) {
    if (!res["xls_last_default_fetch"]) {
      syncStatus("首次使用：点击「加载默认规则」获取最新规则列表", "");
    }
  }
}

// ============================================================
//  入口
// ============================================================
document.addEventListener("DOMContentLoaded", async () => {
  await loadConfig();
  await getActiveTab();
  updateUI();
  loadSyncSettings();
  if (tabId) {
    refreshStats();
    document.getElementById("sync-info").textContent = "纯本地自学习过滤器";
    checkAndOfferDefaults();
  } else {
    document.getElementById("sync-info").textContent = "请打开 x.com 使用";
  }

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
      const all = await chrome.storage.local.get([...Object.keys(DEFAULTS), ...STORAGE_KEYS]);
      const payload = {
        _meta: {
          schema: "x-filter-export-v2",
          version: 5,
          exportedAt: new Date().toISOString(),
          exportedFrom: "Browser Extension (standalone)",
          summary: {},
        },
        config: {},
        data: {
          badSamples: all["xls_samplesBad_v4"] || [],
          goodSamples: all["xls_samplesGood_v4"] || [],
          trainingData: all["xls_trainData_v1"] || [],
          followedHandles: all["xls_followedHandles_v1"] || [],
          modelState: all["xls_modelState_v1"] || {},
          accountCache: all["xls_accountCache_v1"] || {},
          handleReputation: all["xls_handleRep_v1"] || {},
        },
      };
      for (const k of Object.keys(DEFAULTS)) {
        payload.config[k] = { value: all[k] !== undefined ? all[k] : DEFAULTS[k], default: DEFAULTS[k] };
      }
      const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
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
      setTimeout(refreshStats, 500);
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

  // Sync buttons
  document.getElementById("sync-load-default").addEventListener("click", doLoadDefaultRules);
  document.getElementById("sync-load-url").addEventListener("click", doImportFromUrl);
  document.getElementById("sync-push").addEventListener("click", doSyncPush);
  document.getElementById("sync-pull").addEventListener("click", doSyncPull);

  // Save sync settings on input change
  document.getElementById("sync-token").addEventListener("change", saveSyncSettings);
  document.getElementById("sync-gist-id").addEventListener("change", saveSyncSettings);
  document.getElementById("sync-default-url").addEventListener("change", saveSyncSettings);
});
