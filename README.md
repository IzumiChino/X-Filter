# X-filter

一个运行在 [Tampermonkey](https://www.tampermonkey.net/) / Greasemonkey / Violentmonkey 上的 **X (Twitter) 客户端垃圾推文过滤器**。

纯前端、纯本地、自学习。所有学习样本与模型权重都保存在浏览器本地，不发起任何外部网络请求。

- 当前版本：**v5.0.0** (base)
- 作者：好奇猫a & Izumi Chino
- 适配域名：`x.com` / `twitter.com`

---

## 安装

1. 安装浏览器扩展 [Tampermonkey](https://www.tampermonkey.net/)（推荐）或 Violentmonkey / Greasemonkey。
2. 打开 [X-Filter.user.js](./X-Filter.user.js) 原始文件链接，扩展会自动弹出安装确认。
   - 直链：`https://raw.githubusercontent.com/acnekot/X-Filter/main/X-Filter.user.js`
3. 刷新 `x.com`，左下角出现蓝色 **XLS** 浮动按钮即代表安装成功。

## 工作原理：四级判定流水线

命中即停，按优先级从上到下：

| 顺序 | 通道 | 说明 |
| --- | --- | --- |
| 1 | **关注免屏蔽** | 鼠标悬浮到用户卡片时，从 "Following / 已关注" 按钮反向识别并写入白名单 |
| 2 | **结构通道** | 纯 emoji / 纯数字 / emoji+数字 直接拦截 |
| 3 | **启发式通道** | 零训练即可工作，基于 handle 数字比、尾部数字串、粉丝/关注比、bio 缺失等加权打分 |
| 4 | **ML 通道** | 23 维特征的在线逻辑回归（sigmoid + L2 正则 + SGD），训练样本不足时自动回退 |
| 5 | **Jaccard 回退** | n-gram + Jaccard 相似度对比黑/白样本库 |

### ML 特征（23 维）

包含文本长度、emoji 比、数字比、CJK、URL、字符多样性、重复度、handle 数字/熵/尾随数字、名字 emoji 比、是否回复、是否含媒体、与黑/白样本的 Jaccard、handle 历史声誉、粉丝数、关注数、粉关比、有无 bio、是否认证 等。

## 使用

### 标记模式

每条推文旁会注入 **"标注垃圾 / 标注正常"** 两个小按钮，点击即增量训练 ML 模型 + 更新样本库 + 更新 handle 声誉。

### 折叠 vs 隐藏

两种屏蔽呈现方式：

- **折叠卡片**（默认）：把推文替换成一张提示卡片，可点击"展开本条"撤销，或点击"标注：正常"反向训练。
- **直接隐藏**：`display:none`，从信息流里完全消失。

### 控制面板

左下角蓝色 **XLS** 浮动按钮 → 弹出模态面板：

- 模式开关：标记模式 / ML / 启发式 / 关注免屏蔽 / 结构通道 / 调试日志
- ML 参数：回复区阈值、时间线阈值、最小训练样本数
- 启发式参数：回复区/时间线阈值
- Jaccard 回退：阈值、n-gram N
- 结构通道：emoji/digits-only 最小数、混合最小数
- 实时统计：黑/白样本、训练步数、关注缓存、账号缓存、handle 声誉数
- 操作：导出/导入 JSON、重新训练、清空学习数据/关注/账号缓存

同样的功能也通过 **Tampermonkey 菜单** 暴露。

## 数据存储

全部走 `GM_setValue` 存在浏览器本地：

| Key | 内容 |
| --- | --- |
| `xls_samplesBad_v4` | 黑样本（带 n-gram 指纹） |
| `xls_samplesGood_v4` | 白样本 |
| `xls_trainData_v1` | ML 训练集（最多 500 条） |
| `xls_modelState_v1` | ML 模型权重 |
| `xls_followedHandles_v1` | 已关注白名单 |
| `xls_accountCache_v1` | 账号特征缓存（TTL 7 天） |
| `xls_handleRep_v1` | handle 声誉滚动均值（最近 20 次） |

### 导入 / 导出

控制面板 → "导出 JSON" 会把上面所有数据 + 当前配置打包成一个 JSON 文件下载。"导入 JSON" 接受同结构的 JSON 并按 `sig` / `ts` / `handle` 去重合并，导入完成会自动重训模型。

## 隐私

- **不发起任何外部网络请求**，不上报数据。
- 所有样本、模型、缓存只存在你自己浏览器的扩展存储里。
- 想换电脑？用导出/导入即可迁移。

## 调优建议

- 刚装好时 ML 还没训练，主要靠**结构通道 + 启发式通道**工作；标注 8 条以上垃圾后 ML 自动接管。
- 误伤多 → 调高 `mlThresholdReply` / `heuristicThresholdReply`；漏判多 → 调低。
- 不希望短 emoji 被拦 → 关闭结构通道，或调高 `emojiOnlyMinCount`。
- 主页（时间线）默认比回复区严格，因为评论区垃圾密度更高。

## License

MIT
