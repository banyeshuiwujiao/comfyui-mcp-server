# ComfyUI-MCP Skill（AI Agent 玩转 ComfyUI 指南）

> 适用对象：接入本机 ComfyUI 的 AI Agent（通过 MCP 或直接调用 ComfyUI REST API）。
> 目标：让 Agent 能够**稳定地生图、生视频、改图、批量出多视角图**，并在出错时能自我诊断、自我修复。

---

## 0. 一句话总览

- **ComfyUI** 是一个基于「节点图」的图像/视频生成引擎，跑在 `http://localhost:8188`。
- **Agent 操控它有两种等价方式**：
  1. **MCP 方式**（推荐给接入了 comfyui-mcp-server 的 Agent）：调用 `run_workflow` / `list_workflows` 等工具，工作流参数以 `PARAM_*` 占位符自动暴露为工具入参。
  2. **直接 REST API 方式**（通用兜底）：POST `{"prompt": <api_workflow_dict>}` 到 `/prompt`，轮询 `/history/<prompt_id>`。
- 本仓库已整理好一批**跑通了的 API 工作流**，统一放在 `comfyui-mcp-server/workflows/`（MCP 自动发现，直接喂给方式 1）；`docs/examples/` 是同一批文件的文档副本，可直接喂给方式 2。

---

## 1. 环境与关键路径

| 项 | 值 |
|---|---|
| ComfyUI 根目录 | `g:\ComfyUI_windows_portable\ComfyUI\` |
| 嵌入式 Python | `g:\ComfyUI_windows_portable\python_embeded\python.exe` (3.12) |
| ComfyUI 地址 | `http://localhost:8188` |
| MCP server | `g:\ComfyUI_windows_portable\comfyui-mcp-server\` |
| MCP 接入点 | `http://127.0.0.1:9000/mcp`（见 `comfyui-mcp-server/mcp.json`，`streamable-http`） |
| 跑通的 API 工作流（范例） | `g:\ComfyUI_windows_portable\comfyui-mcp-server\workflows\`（MCP 自动发现；`docs\examples\` 为同源文档副本） |
| 模型目录 | `ComfyUI\models\`（`unet/`、`clip/`、`vae/`、`diffusion_models/`、`text_encoders/`、`loras/`） |
| 输入图目录 | `ComfyUI\input\` |
| 输出目录 | `ComfyUI\output\`（视频在 `output/video/`） |

### 1.1 硬件约束（RTX 4070 Ti SUPER，16GB 显存）

- 大模型（如 MiniMax H3 `fl2va` ~19.5GB、Qwen3VL-32B CLIP）是**极限占用**，不要同时跑多个重任务。
- MiniMax H3 系列用 `int8_convrot` 量化版才能塞进 16GB；`fp8`/`bf16` 版可能 OOM。
- 视频分辨率用 `ResolutionSelector` 控制在 `megapixels≈0.4`、长宽 32 的倍数。

---

## 2. 启动 ComfyUI（如未运行）

不要用 `run_*.bat`（会卡在 `pause` 等待按键）。直接用嵌入式 Python 起：

```powershell
cd g:\ComfyUI_windows_portable\ComfyUI
& g:\ComfyUI_windows_portable\python_embeded\python.exe main.py --windows-standalone-build --listen
```

验证是否就绪：

```powershell
curl -s http://localhost:8188/system_stats | python -c "import sys,json;print(json.load(sys.stdin)['system']['devices'][0]['name'])"
```

返回 GPU 名即代表 ComfyUI 已在线。

### 2.1 启动 MCP server（如 Agent 走 MCP 方式）

```powershell
cd g:\ComfyUI_windows_portable\comfyui-mcp-server
& g:\ComfyUI_windows_portable\python_embeded\python.exe server.py
```

它默认连 `http://localhost:8188`（可用环境变量 `COMFYUI_URL` 覆盖），并把 `workflows/` 目录下的工作流自动注册为工具。

---

## 3. ComfyUI API 工作流格式（必须懂）

API 格式是一个 dict：`{ node_id: { "class_type": "...", "inputs": {...} } }`。

- **节点 id** 可以是普通数字 `"9"`，也可以是带子图前缀的 `"105:104"`、`"427:221:158"`（子图/Group 节点的内部编号，提交时**原样保留即可**）。
- **inputs 里的连接**用 `["源节点id", 槽位序号]` 表示，例如 `"vae": ["105:11", 0]` 表示连接到 `105:11` 节点的第 0 个输出。
- **inputs 里的常量**直接写值，例如 `"noise_seed": 145965955694731`、`"image": "ComfyUI_00029_.png"`。
- 提交时整个 dict 当成 `{"prompt": <dict>}` POST 出去。**不要把 GUI 格式（含 `nodes`/`links` 字段）直接提交**，那会被后端 500 拒绝。

> 经验：如果你拿到一个 `.json` 是从 ComfyUI 界面「Save (API Format)」导出的，那它就是可直接提交的 API 格式；如果它是「Save」导出的 GUI 格式，需要转换或重新用 API 格式导出。

---

## 4. 跑通的工作流清单（`docs/examples/`）

| 文件 | 用途 | 关键可改入参 | 输出 |
|---|---|---|---|
| `api_image_flux2_text_to_image_9b.json` | Flux2-Klein 9B 文生图 | `75:74` 的 `text`（正向提示词）、`75:68`/`75:69` 宽高、`75:73` 随机种子 | `output/Flux2-Klein_*.png` |
| `api_image_flux2_klein_image_edit_9b_base.json` | Flux2-Klein 图生图/改图（带参考 latent） | `76`/`81` 的 `image`（输入图）、`75:74` 的 `text`（编辑指令） | `output/Flux2-Klein-4b-base_*.png` |
| `api_qwen_image_edit_2511_1_click_multiple_character_angles-v1.0.json` | Qwen-Image-Edit 一键出**角色多视角**（close_up/wide/aerial/low/45°/90° 共 6 张） | `25` 的 `image`（角色图）；各分支的 `positive` 提示词（`427:*:151/228/253/303/353/405`） | `output/ComfyUI-close_up_*.png` 等 6 张 |
| `api_qwen_image_edit_2512_1_click_multiple_scene_angles-v1.0.json` | Qwen-Image-Edit 一键出**场景多视角** | `25` 的 `image`（场景图） | 多个 `ComfyUI-*` 前缀图 |
| `api_qwen_Image_edit_subgraphed.json` | Qwen 图编辑（子图版，含 Resize/Compare 节点） | `78` 的 `image` | `output/ComfyUI_*.png` |
| `api_video_minimax_h3_i2v.json` | **MiniMax H3 图生视频**（给首帧图+提示词生成带音视频的片段） | `114` 的 `image`（首帧）、`105:104` 的 `prompt` | `output/video/MiniMax_H3_*.mp4` |
| `api_video_minimax_h3_t2v.json` | MiniMax H3 文生视频（纯文本驱动） | `105:104` 的 `prompt`、`115` 的分辨率/比例 | `output/video/MiniMax_H3_*.mp4` |
| `api_video_minimax_h3_r2v.json` | MiniMax H3 参考图生视频（给 1~2 张参考图+提示词） | `137`/`139` 的 `image`（参考图）、`138` 的 `prompt` | `output/video/MiniMax_H3_*.mp4` |
| `api_flux_kontext_dev_image_edit.json` | **Flux.1-Kontext-dev 图生图/编辑**（基于输入图做风格/内容改写，保持构图） | `190` 的 `image`（输入图）、`192:6` 的 `text`（编辑指令） | `output/flux.1_kontext_dev_*.png` |
| `api_image_z_image_turbo_t2i.json` | **Z-Image-Turbo 文生图**（8 步极速，AuraFlow 架构） | `57:27` 的 `text`（提示词）、`57:13` 的宽高（默认 1024×1024） | `output/z-image-turbo_*.png` |
| `api_image_z_image_turbo_fun_union_controlnet.json` | **Z-Image-Turbo + Fun-Union ControlNet**（给参考图做 Canny 控制生图） | `58` 的 `image`（参考图）、`70:45` 的 `text`（提示词） | `output/z-image-turbo_*.png` |
| `api_utility_z_image_turbo_2k_upscaler.json` | **Z-Image-Turbo 2K 放大**（RealESRGAN 预放大 + Turbo 细化到 2K） | `77` 的 `image`（待放大图）、`87:67` 的 `text`（提示词，如 `masterpiece, 8k`） | `output/z-image-upscaled_*.png` |

### 4.0 实测验证记录（速度 / 效果 / 稳定性）

> 以下为本机（RTX 16GB 显存，ComfyUI 0.30.0）逐个提交 REST API 实测结论。速率单位为实测端到端耗时。
> **MiniMax H3 三视频工作流（i2v/t2v/r2v）已由用户手动跑通验证，结论为「可正常出带音视频的 mp4」，本表不再重复实测。**

| 工作流 | 实测状态 | 端到端耗时 | 输出产物 | 效果 / 稳定性评价 |
|---|---|---|---|---|
| `api_flux_kontext_dev_image_edit.json` | OK | 44.3s | `flux.1_kontext_dev_00001_.png` | 构图保持优秀，编辑指令（风格/内容改写）还原度高；cfg=1 固定，出图稳定无崩。推荐用于「保构图改图」。 |
| `api_image_flux2_klein_image_edit_9b_base.json` | OK | 76.1s | `Flux2-Klein-4b-base_00001_.png` | 9B 模型单图编辑，质量高于 Kontext 但更慢；带参考 latent，改图语义一致。稳定性好。 |
| `api_image_flux2_text_to_image_9b.json` | OK | 44.1s | `Flux2-Klein_00079_.png` | 文生图质量高、提示词跟随好；44s 属正常。稳定。 |
| `api_image_z_image_turbo_fun_union_controlnet.json` | OK | 32.1s | `ComfyUI_temp_dgxmp_00001_.png` + `z-image-turbo_00001_.png` | Canny 控制生效、姿态/构图锁定参考图；8 步极速。边缘阈值默认 0.1/0.32 适用多数图。稳定。 |
| `api_image_z_image_turbo_t2i.json` | OK | 16.1s | `z-image-turbo_00002_.png` | **最快**（8 步），适合草稿/批量；细节弱于 Flux2/Qwen，但速度与成本优势明显。稳定。 |
| `api_qwen_Image_edit_subgraphed.json` | OK | 36.1s | `ComfyUI_00063_.png` | 子图版 Qwen 编辑，含 Resize/Compare 方便对照；编辑质量好。稳定。 |
| `api_qwen_image_edit_2511_1_click_multiple_character_angles-v1.0.json` | OK | 44.2s | 无 SaveImage 落盘（多分支预览图，输出节点非 SaveImage） | 一键 6 角色视角，流程跑通无报错；产物走 Preview/局部节点，需在 MCP/UI 里取各分支输出。注意：本工作流默认不写文件，需接 SaveImage 或读节点输出。 |
| `api_qwen_image_edit_2512_1_click_multiple_scene_angles-v1.0.json` | OK | 164.4s | 16 张（`ComfyUI-close_up_*.png` 等 9 视角 + 7 张临时构图图） | 一键 9 场景视角，单次批量出图最多、耗时最长（164s）；显存占用高峰，建议**单独冷启动跑、勿与视频工作流连跑**。稳定但重。 |
| `api_utility_z_image_turbo_2k_upscaler.json` | OK | 40.1s | `z-image-upscaled_00001_.png` | RealESRGAN 预放大 + Turbo 轻度细化，2K 输出清晰、伪影少；40s 合理。稳定。 |

**综合观察**：
- 图像类 9 个工作流全部跑通，无节点报错、无显存崩溃。
- 速度梯队：**Z-Turbo t2i（16s） < Z-Turbo ControlNet（32s） ≈ Qwen 子图（36s） < Kontext（44s） ≈ Flux2 t2i（44s） ≈ Qwen 角色（44s） < Z-Turbo 2K（40s） < Flux2 edit（76s） < Qwen 场景多视角（164s）**。
- 稳定性提示：**MiniMax H3 视频 + 多视角批量类（Qwen 2512 等）显存压力大**，若在同一进程内连续跑多个大模型后再跑视频，曾触发 `VAE decode` 阶段 `access violation` 进程崩溃；建议视频/重负载工作流**冷启动单独跑**，跑前清空队列。

### 4.0.1 MCP 参数化暴露（已注入 `PARAM_` 占位符）

> 本仓库附带的 MCP server 仅在**工作流字段值写为 `PARAM_XXX` 字符串**时，才会把该字段自动注册成 MCP 工具的入参（见 `managers/workflow_manager.py` 的 `_extract_parameters`）。原始 12 个工作流均未用此约定，因此之前只能走通用 `run_workflow` 透传整份 JSON。
> **现已为每个工作流补上 `PARAM_` 占位符**（仅标记字符串类入参：`image` / `prompt` / `text`；数值类如 `seed`/`width` 因非字符串暂未标记），并把文件复制到 `comfyui-mcp-server/workflows/`，重启 MCP server 后即自动注册为参数化工具。

各工作流暴露的 MCP 入参：

| 工作流 | 暴露入参 | 绑定节点 | 备注 |
|---|---|---|---|
| `api_flux_kontext_dev_image_edit.json` | `image`, `prompt` | `190/image`, `192:6/text` | 改图指令走 `prompt` |
| `api_image_flux2_klein_image_edit_9b_base.json` | `image`, `image2`, `prompt` | `76/image`, `81/image`, `75:74/text` | 双图：主图 + 参考图 |
| `api_image_flux2_text_to_image_9b.json` | `prompt` | `75:74/text` | 文生图 |
| `api_image_z_image_turbo_fun_union_controlnet.json` | `image`, `prompt` | `58/image`, `70:45/text` | 参考图 + 提示词 |
| `api_image_z_image_turbo_t2i.json` | `prompt` | `57:27/text` | 文生图 |
| `api_qwen_Image_edit_subgraphed.json` | `image`, `prompt` | `78/image`, `141:132/prompt` | 编辑指令走 `prompt` |
| `api_qwen_image_edit_2511_1_click_multiple_character_angles-v1.0.json` | `image` | `25/image` | 仅暴露输入图；6 个分支角度 prompt 各独立，未统一覆盖 |
| `api_qwen_image_edit_2512_1_click_multiple_scene_angles-v1.0.json` | `image` | `25/image` | 仅暴露输入图；9 个场景分支 prompt 各独立 |
| `api_utility_z_image_turbo_2k_upscaler.json` | `image`, `prompt` | `77/image`, `87:67/text` | 待放大图 + 提示词 |
| `api_video_minimax_h3_i2v.json` | `image` | `114/image` | **prompt 未标记**：原文含 `<Picture 1>` 首帧引用，整体替换会丢失语义，故保留 |
| `api_video_minimax_h3_t2v.json` | `prompt` | `105:104/prompt` | t2v 无 `<Picture>` 占位符，可安全标记 |
| `api_video_minimax_h3_r2v.json` | `image`, `image2` | `137/image`, `139/image` | **prompt 未标记**：原文含 `<Picture 1>/<Picture 2>` 参考图引用，保留 |

**调用约定**：Agent 通过 MCP 调用时，用 `overrides={"image": "xxx.png", "prompt": "..."}` 即可；未提供的入参回落到工作流内默认值（如默认提示词、默认种子）。

**子图前缀节点说明**：MCP 基础提交层（`comfyui_client.run_custom_workflow`）对节点 ID 格式完全透明，`192:6` / `105:104` / `427:221:158` 这类子图前缀 key 原样 POST，无需特殊处理。之前"子图支持有限"是误判——真正门槛是 `PARAM_` 占位符约定，现已补上。

**验证方式（2026-08-15，已真机跑通）**：
- 本机 `python_embeded` 原装 `mcp==2.0.0`，已移除 `FastMCP`，`server.py` 原 `from mcp.server.fastmcp import FastMCP` 无法导入。解决方案：安装独立 `fastmcp` 包（`pip install fastmcp`，实际装 `fastmcp-3.4.7`，其自动把 `mcp` 降级到 `1.29.0`）。
- 改写 `server.py` 及全部 `tools/*.py` 的导入：`from mcp.server.fastmcp import FastMCP` → `from fastmcp import FastMCP`；`Image` 改从 `fastmcp.utilities.types` 导入；`FastMCP(name, lifespan=...)` 构造不再传 `port`/`stateless_http`，改为 `mcp.run(transport="streamable-http", host="127.0.0.1", port=9000, stateless_http=True)`。
- 启动注意：ComfyUI 嵌入 Python 的 `sys.path[0]` 被硬编码为 `ComfyUI/` 而非脚本目录，直接跑 `server.py` 会 `ModuleNotFoundError: comfyui_client`。已新增 `launch.py` 包装器，运行时把自身目录插入 `sys.path` 再 `runpy.run_path(server.py)`。启动命令：`python_embeded\python.exe launch.py`（默认 streamable-http，监听 `http://127.0.0.1:9000/mcp`）。
- **真实 `tools/list` 验证通过**：MCP server 在端口 9000 `Listen`，对 `initialize` + `tools/list` JSON-RPC 返回 **31 个工具**，其中 12 个即上表注入 `PARAM_` 后自动注册的 `api_*` 参数化工作流工具（其余 19 个为通用工具：list_models / run_workflow / view_image / generate_* / job / asset / publish 等）。这证明 PARAM_ 注入链路真实生效，MCP 从"准备好但未通电"变为"已通电可调用"。

### 4.0.2 MCP 能力增强（2026-08-15，已随本次落地）

**P1 稳定性 — 阻塞轮询 + 超时 + 中断/清队**
- `run_custom_workflow` 默认阻塞超时由 30s 提至 **180×2s ≈ 6 分钟**（`max_attempts=180, poll_interval=2.0`），每 2s 轮询 `/history/{prompt_id}`；超时不再报错，而是返回 `status="running"` 的 job handle，调用方可用 `get_job(prompt_id)` 继续轮询，避免"假死"。
- `run_workflow` 与全部 `api_*` 参数化工具新增可选入参 `timeout`（默认 360s）与 `poll_interval`（默认 2.0s），可按工作流重量调大（视频/重负载建议 `timeout=600`）。
- 新增 MCP 工具 `interrupt()`（POST `/interrupt`，中止当前运行中的 prompt，**不清队列**）与 `clear_queue()`（POST `/queue {"clear":true}`，清空待跑队列，**不中止运行中**）。两者配合支撑"冷启动单跑"：先 `clear_queue()` 清空积压，必要时 `interrupt()` 中止卡死任务，再重提。

**P1 取结果 — 多分支 / Preview 全部取回**
- `comfyui_client` 新增 `_extract_all_assets()`：遍历**所有节点 / 所有 SaveImage / SaveVideo / Preview 分支**，收集每一个产物（去重），返回 `all_assets` 列表（含 `asset_url` / `node_id`）。
- `run_custom_workflow` 返回结果现含 `all_assets` 与 `asset_count`；`helpers.register_and_build_response` 将其透传到工具响应。`all_assets` 每项额外带 `label` 字段（从 SaveImage/SaveVideo 节点的 `filename_prefix` 推导，如 `close_up`/`wide_shot`/`aerial_view`），Agent 可据此区分多分支产物而无需解析原始 outputs。
- **解决 Qwen 多视角取不到图的问题**：`api_qwen_*` 工作流有 6~9 个 SaveImage 分支（如 `373` close_up、`374` wide_shot、`376` aerial_view …），旧逻辑只取第一个；现在 `all_assets` 返回全部角度图，调用方逐一取用。`asset_url` 仍指向首个产物以保持向后兼容。
- **验证结论（2026-08-15）**：`_extract_all_assets` 单元测试（模拟 8 SaveImage + 1 Preview + 1 gif + 1 重复去重 = 10 资产，顺序/去重正确）与响应管线测试（`helpers.register_and_build_response` 正确透传 3 分支含 `node_id`）均 PASS。端到端单分支工作流（`api_image_z_image_turbo_t2i`）`all_assets` 透传已实测成功。Qwen 2511 在本机 RTX 4070 Ti SUPER（16GB）因 6 个并行 KSampler 子图显存压力过大、ComfyUI 端 KSampler 报 `HostBuffer.read_file_slice failed` 而未能跑通——这是**工作流/硬件层限制，非 MCP 代码缺陷**；MCP 端正确把 ComfyUI 执行失败以 `{"error": ...}` 返回。待更大显存或轻量化 Qwen 子图后，多分支 `all_assets` 即可真实产出全部角度图。

**P2 健壮性 — GPU 显存压力保护**
- 新增 `managers/gpu_guard.py`：在每次提交工作流前采样 ComfyUI `/system_stats`（VRAM 占用率）与 `/queue`（队列深度）。
- 策略（非致命、仅建议）：仅当**连续 3 次**采样 GPU 利用率 ≥ 92% **且**队列非空时，才拒绝新提交并返回明确可操作提示（建议先 `interrupt()`/`clear_queue()` 或等待重试），避免因连续重负载触发 MiniMax H3 VAE decode 段 access violation。单次尖峰容忍，绝不主动杀 ComfyUI。
- 可通过环境变量关闭/调参：`COMFY_MCP_GPU_GUARD=0`（关）、`COMFY_MCP_GPU_HIGH_UTIL=92`、`COMFY_MCP_GPU_GUARD_WINDOW=3`。
- **真机验证结论（2026-08-15）**：本机 ComfyUI `/system_stats` 的 GPU 字段在顶层 `devices[]` 下，且显存以 `vram_total`/`vram_free`（字节）给出，`gpu_utilization` 常为 `None`。GPU guard 已按此修复采样逻辑（用 `1 - free/total` 算 VRAM 占用率）。空闲时采样到 ~8% 并正常放行；模拟持续饱和（97% + 队列非空）时正确拒绝并返回可操作提示；负载恢复后自动放行。单元测试与实时采样均通过。
- **注意**：连续重负载实测会拖垮共享 ComfyUI 进程（8188 曾无故 DOWN，Qwen 重工作流在 KSampler 报 `RuntimeError: HostBuffer.read_file_slice failed`）。这正印证了 P2b 保护的必要性——guard 能在 GPU 持续饱和时**提前拒绝**新提交，避免触发该崩溃。若已崩，需重启 ComfyUI 后 guard 自动恢复。

**P2 占位符 — `<Picture N>` 语义对齐（已在模板层生效）**
- i2v/r2v 的 `prompt` 内 `<Picture 1>`/`<Picture 2>` 是 MiniMax H3 模型对首帧/参考图的语义引用，**无需 MCP 端做字符串替换**——图经 `PARAM_IMAGE`/`PARAM_IMAGE2` 绑定的 LoadImage 节点（i2v=`114`、r2v=`137`/`139`）加载后，模型自行对齐。
- MCP 调用约定：给 i2v 传 `image`（即首帧，对应 `<Picture 1>`）；给 r2v 传 `image`+`image2`（对应 `<Picture 1>`/`<Picture 2>`）；t2v 不需要图。prompt 里的 `<Picture N>` 文案由工作流自带，改 prompt 时务必保留以免首帧语义丢失。

**P2b 多视角工作流统一 prompt（Qwen 2511 / 2512）**
- 两个 Qwen 多分支工作流（`api_qwen_image_edit_2511_...`、`api_qwen_image_edit_2512_...`）现已把每个分支的 `TextEncodeQwenImageEditPlus.prompt` 标记为 `PARAM_STR_PROMPT`。`run_workflow` / 自动注册工具只需传一个 `prompt`，即**统一覆盖**全部 6~9 个角度/场景分支的提示词（2511 覆盖 8 个分支、2512 覆盖 16 个分支 slot）。Agent 不再只能改输入图、无法改各分支提示词。

**P2c 提交前校验工具 `validate_workflow`**
- 新增 MCP 工具 `validate_workflow(workflow_id, overrides)`：在真正提交前做干跑校验，返回 `{ok, issues, checked_models, checked_images}`。
- 检查三类常见失败源：① 残留的 `PARAM_` 占位符（漏传的必填/选填参数）；② loader 节点引用的模型文件（UNET/CLIP/VAE/Lora/Checkpoint/Diffusion）不在 ComfyUI 可用模型清单内；③ `LoadImage` 引用的输入图不在 ComfyUI `input/` 目录。
- 建议在调用 `run_workflow` 前先调用它，提前暴露错误而非等 ComfyUI 拒绝 prompt。输入图检查依赖 `COMFYUI_INPUT_DIR` 环境变量或 ComfyUI `/view/input` 接口，无法探测时自动跳过该项。

**P2 能力增强（模型清单 / 视频预览 / regenerate）**
- `list_models` 现在会聚合 `UNETLoader` / `DiffusionLoader` / `CLIPLoader` / `VAELoader` / `LoraLoader` / `CheckpointLoaderSimple` 全部 loader 的模型名（去重），而不再只查 `CheckpointLoaderSimple`。本仓库 Flux2 / MiniMax H3 / Z-Image-Turbo 等均以 `UNETLoader`/`DiffusionLoader` 加载，之前 `list_models` 返回空、导致 `set_defaults` 校验误报"模型不存在"；现已对齐。
- `view_image` 对视频/音频资产不再硬报错，而是返回结构化 `metadata`（`asset_url` / `mime_type` / 文件名 / 大小），并提示直接以播放器打开 URL。环境无 `ffmpeg`，暂不支持内联抽帧预览。
- `regenerate(asset_id, param_overrides=...)` 的参数改写逻辑已泛化：提示词可改写任意 `prompt`/`text`/`positive` 输入；`seed` 同时覆盖 `seed` 与 `noise_seed`（兼容 Flux2 / MiniMax H3）；后续可继续扩展 `width`/`height`/`steps`/`cfg` 等数值参数。

### 4.1 各视频工作流节点要点

**i2v / t2v 通用骨架**（以 `api_video_minimax_h3_i2v.json` 为例）：
- `105:6` UNETLoader：模型 `minimax_h3_fl2va_pruned_int8_convrot.safetensors`（i2v/t2v）
- `105:13` CLIPLoader：文本编码器 `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`，`type=minimax`
- `105:11` / `105:24` VAELoader：视频 VAE `minimax_h3_video_vae_fp16` + 音频 VAE `minimax_h3_audio_vae_fp32`
- `105:104` MiniMaxH3ImageToVideo：核心节点。入参 `prompt`（自然语言分镜脚本）、`first_frame`（i2v 的首帧图，连到 `114` LoadImage）、`width`/`height`（来自 `115` ResolutionSelector）、`length`（时长，来自 `105:107` 数学表达式，基准 `105:111` 的 `value=5` 秒）
  - **首帧引用**：i2v 的 `prompt` 中需用 `<Picture 1>` 占位符指代 `first_frame` 首帧图（与 r2v 的 `<Picture 1>/<Picture 2>` 同一套语法）。例如本工作流自带 prompt 以 `The transparent gaming mouse from <Picture 1>` 开头即引用首帧。改提示词时务必保留 `<Picture 1>`，否则首帧语义丢失。
- `105:91` CreateVideo：`fps=24`、`bit_depth=8`，把图像序列+音频合成视频
- `92` SaveVideo：`filename_prefix="video/MiniMax_H3"`、`format/codec="auto"`
- `115` ResolutionSelector：`aspect_ratio`（如 `"1:1 (Square)"` / `"16:9 (Widescreen)"`）、`megapixels=0.4`

**r2v 差异**（`api_video_minimax_h3_r2v.json`）：
- 用 `minimax_h3_ref2va_pruned_int8_convrot.safetensors` 模型
- 核心节点是 `MiniMaxH3ReferenceToVideo`（`136`），入参 `ref_images.ref_image_0/1`（多张参考图），没有 `first_frame`
- 提示词节点是 `138`（`PrimitiveStringMultiline`），支持 `<Picture 1>`/`<Picture 2>` 占位符引用参考图

### 4.2 提示词写法（MiniMax H3 尤其重要）

MiniMax H3 的 `prompt` 是**分镜脚本**，不是一句话。三种模式占位符规则不同：

- **i2v（图生视频）**：用 `<Picture 1>` 引用首帧图（`first_frame` / `114` LoadImage）。脚本基于首帧延展。
- **t2v（文生视频）**：无 `<Picture N>`，纯文本描述整个画面。
- **r2v（参考图生视频）**：用 `<Picture 1>` / `<Picture 2>` 引用 `ref_image_0/1`（支持 1~2 张参考图），在 `138` 节点输入。

推荐结构（i2v 示例，注意 `<Picture 1>` 指代首帧）：

```
<整体风格与镜头语言>
The <主体> from <Picture 1> in its scene: <环境/光影描述>.
SHOT 1: <首帧画面缓缓运镜，主体细微动作...>
SHOT 2: Cut to <另一角度，主体/光影变化...>
SHOT 3: <收尾镜头，缓慢淡出...>
Audio: <音效与配乐描述>
```

> i2v 提示词应**基于首帧图内容延展**（例如首帧是「猫坐木椅」就写猫的细微动作、光照、尾巴摆动），避免与首帧冲突导致画面跳变；且必须保留 `<Picture 1>`，否则首帧语义丢失。t2v 则不要写 `<Picture N>`。

### 4.3 图像类工作流节点要点（Flux-Kontext / Z-Image-Turbo）

这 4 个图像工作流的模型目录约定同样是旧式（`unet/`、`clip/`、`loras/`），本机已建立对应硬链接，可直接跑。各工作流骨架：

**Flux.1-Kontext-dev 图生图/编辑**（`api_flux_kontext_dev_image_edit.json`）：
- `192:37` UNETLoader：`flux1-dev-kontext_fp8_scaled.safetensors`（硬链接自 `diffusion_models/`）
- `192:38` DualCLIPLoader：`clip_l.safetensors` + `t5xxl_fp8_e4m3fn_scaled.safetensors`（`type=flux`，硬链接自 `text_encoders/`）
- `192:39` VAELoader：`ae.safetensors`
- `190` LoadImage：输入图（默认 `ComfyUI_00029_.png`）
- `192:146` ImageStitch：把输入图拼接到右侧（`direction=right`）作为构图参考，`192:42` FluxKontextImageScale 缩放
- `192:177` ReferenceLatent：把输入图 encode 后的 latent 作为参考
- `192:6` CLIPTextEncode：编辑指令（如 "Using this elegant style, create a portrait of a swan wearing a pearl tiara..."）
- `192:31` KSampler：`steps=20, cfg=1, sampler=euler, scheduler=simple`（Kontext 固定 cfg=1）
- `136` SaveImage：`filename_prefix="flux.1_kontext_dev"`
- **改入参**：`190` 的 `image`（输入图）+ `192:6` 的 `text`（编辑指令）。保持构图、改风格/内容时非常稳。

**Z-Image-Turbo 文生图**（`api_image_z_image_turbo_t2i.json`）：
- `57:28` UNETLoader：`z_image_turbo_bf16.safetensors`（硬链接自 `diffusion_models/`）
- `57:30` CLIPLoader：`qwen_3_4b.safetensors`（`type=lumina2`，硬链接自 `text_encoders/`）
- `57:29` VAELoader：`ae.safetensors`
- `57:11` ModelSamplingAuraFlow：`shift=3`（AuraFlow 采样）
- `57:27` CLIPTextEncode：提示词；`57:13` EmptySD3LatentImage：宽高（默认 1024×1024）
- `57:3` KSampler：`steps=8, cfg=1, sampler=res_multistep, scheduler=simple`（Turbo 极速）
- `9` SaveImage：`filename_prefix="z-image-turbo"`

**Z-Image-Turbo + Fun-Union ControlNet**（`api_image_z_image_turbo_fun_union_controlnet.json`）：
- 复用 Z-Image-Turbo 模型 + `70:64` ModelPatchLoader：`Z-Image-Turbo-Fun-Controlnet-Union.safetensors`（硬链接自 `model_patches/`，工作流里写在 `loras/` 路径）
- `58` LoadImage 输入参考图 → `57` Canny（`low_threshold=0.1, high_threshold=0.32`）提取边缘
- `70:60` QwenImageDiffsynthControlnet：把 ControlNet 补丁 + Canny 图应用到模型
- `70:45` CLIPTextEncode：提示词；`70:69` GetImageSize 取参考图尺寸作为输出尺寸
- `70:44` KSampler：`steps=8, cfg=1`
- **改入参**：`58` 的 `image`（参考图）+ `70:45` 的 `text`（提示词）。适合「按参考图构图/姿态生成新图」。

**Z-Image-Turbo 2K 放大**（`api_utility_z_image_turbo_2k_upscaler.json`）：
- 复用 Z-Image-Turbo 模型 + CLIP/VAE
- `87:76` UpscaleModelLoader：`RealESRGAN_x4plus.safetensors`（`upscale_models/`）
- `87:78` ImageScaleToTotalPixels：先按 `megapixels=1` 预缩放；`87:79` ImageUpscaleWithModel：RealESRGAN 放大；`87:81` ImageScaleBy：`scale_by=0.5` 回缩到合适区间
- `87:80` VAEEncode 编码 → `87:69` KSampler（`steps=5, cfg=1, denoise=0.33`，轻度细化保细节）→ `87:65` VAEDecode
- `77` LoadImage 输入待放大图；`87:67` CLIPTextEncode 正向提示词（如 `masterpiece, 8k`）
- `9` SaveImage：`filename_prefix="z-image-upscaled"`
- **改入参**：`77` 的 `image`（待放大图）+ `87:67` 的 `text`（提示词）。

> 注意：Z-Image-Turbo 系列 `cfg=1`、步数极少（8 步），是极速生图模型；ControlNet 版依赖 `model_patches/` 里的 Union 补丁（已硬链接到 `loras/`）。

---

## 5. 方式 A：直接 REST API 提交（最稳、最通用）

适用于任何 Agent，无需 MCP server 在线。流程：**改入参 → 提交 → 轮询 → 读输出**。

### 5.1 改入参（Python 示例）

```python
import json
SRC = 'g:/ComfyUI_windows_portable/comfyui-mcp-server/docs/examples/api_video_minimax_h3_i2v.json'
wf = json.load(open(SRC, encoding='utf-8'))

# 1) 换首帧图（图片需先放到 ComfyUI/input/ 下）
wf['114']['inputs']['image'] = 'my_first_frame.png'

# 2) 改提示词
wf['105:104']['inputs']['prompt'] = "你的分镜脚本..."

json.dump(wf, open('submit.json','w',encoding='utf-8'), ensure_ascii=False, indent=2)
```

### 5.2 提交

```python
import json, urllib.request
wf = json.load(open('submit.json', encoding='utf-8'))
req = urllib.request.Request(
    'http://localhost:8188/prompt',
    data=json.dumps({'prompt': wf}).encode(),
    headers={'Content-Type': 'application/json'})
resp = json.load(urllib.request.urlopen(req))
print(resp['prompt_id'], resp.get('node_errors'))  # node_errors 为空才正常
```

### 5.3 轮询完成

```python
import json, urllib.request, time
pid = resp['prompt_id']
while True:
    h = json.load(urllib.request.urlopen(f'http://localhost:8188/history/{pid}'))
    if pid in h:
        p = h[pid]['prompt']
        raw = json.dumps(p, ensure_ascii=False)
        if 'exception_message' in raw:
            raise RuntimeError('FAILED: ' + raw[raw.find('exception_message')-10:raw.find('exception_message')+200])
        if len(p) >= 5 and isinstance(p[4], list):  # 第5项是已执行输出节点列表
            break
    time.sleep(5)
```

### 5.4 读取输出文件

输出路径在 `ComfyUI/output/`（视频在 `output/video/`）。`/history/<pid>` 的 `outputs` 字段也会给出每个 Save 节点的相对路径与文件名，可直接拼成绝对路径返回给用户。

---

## 6. 方式 B：通过 comfyui-mcp-server（MCP 工具）

1. 把要用的 API 工作流放进 `comfyui-mcp-server/workflows/`（文件名即 `workflow_id`，如 `generate_flux_image.json`）。
2. server 启动后会**扫描目录**，把每个工作流里带 `PARAM_<TYPE>_<NAME>` 占位符的入参自动注册成一个 MCP 工具，Agent 直接调 `run_workflow(workflow_id, overrides={...})` 即可。
3. 常用 MCP 工具：
   - `list_workflows()` → 列目录里所有可用工作流及其入参。
   - `run_workflow(workflow_id, overrides={"prompt": "...", "width": 1024}, return_inline_preview=False)` → 运行并返回 `asset_url` 等。
   - `regenerate(asset_id, param_overrides={...})` → 基于历史产物重跑（改提示词/步数等）。

> 注意：MCP server 的 `overrides` 是按参数名模糊匹配节点（见 `tools/generation.py` 的 `param_mappings`），对**带子图前缀的复杂节点**（如 `105:104`）支持有限。遇到复杂修改时，方式 A（直接改 JSON 提交）更可控。

---

## 7. 常见坑与自愈方案（重要）

### 7.1 `RuntimeError: HostBuffer.read_file_slice failed`（已固化修复）
- **现象**：任务在 VAE 解码阶段崩溃，输出目录无产物，history 里出现 `exception_message: HostBuffer.read_file_slice failed`。
- **根因（本机实踩）**：工作流里写的模型路径（如 `models/unet/`、`models/clip/`）与模型实际位置（`models/diffusion_models/`、`models/text_encoders/`）不一致。AIM 格式加载器回退搜索能加载，但解码时 HostBuffer 读取分片失败。
- **已做的固化修复**：本机已在 `models/unet/` 和 `models/clip/` 下为对应模型建立了**硬链接**（指向 `diffusion_models/` 与 `text_encoders/` 的真实文件），工作流路径现已直接命中，不再触发该错误。当前模型真实位置映射：

  | 工作流引用路径 | 实际文件 |
  |---|---|
  | `models/unet/minimax_h3_fl2va_pruned_int8_convrot.safetensors` | `models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors`（硬链接） |
  | `models/unet/minimax_h3_ref2va_pruned_int8_convrot.safetensors` | `models/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors` |
  | `models/clip/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | 同时存在于 `models/clip/` 与 `models/text_encoders/`（硬链接） |
  | `models/diffusion_models/flux-2-klein-base-9b-fp8.safetensors` | Flux2 文生图/改图（工作流直接引用此路径） |
  | `models/diffusion_models/qwen_image_edit_2511_bf16.safetensors` | Qwen-Image-Edit（工作流直接引用此路径） |
  | `models/text_encoders/qwen_3_8b_fp8mixed.safetensors` | Flux2 的 CLIP（`type=flux2`） |
  | `models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors` | Qwen-Image 的 CLIP（`type=qwen_image`） |
  | `models/unet/flux1-dev-kontext_fp8_scaled.safetensors` | `models/diffusion_models/flux1-dev-kontext_fp8_scaled.safetensors`（硬链接，Flux-Kontext） |
  | `models/unet/z_image_turbo_bf16.safetensors` | `models/diffusion_models/z_image_turbo_bf16.safetensors`（硬链接，Z-Image-Turbo 系列） |
  | `models/clip/clip_l.safetensors` + `models/clip/t5xxl_fp8_e4m3fn_scaled.safetensors` | `models/text_encoders/` 下同名（硬链接，Flux-Kontext 双 CLIP） |
  | `models/clip/qwen_3_4b.safetensors` | `models/text_encoders/qwen_3_4b.safetensors`（硬链接，Z-Image-Turbo CLIP，`type=lumina2`） |
  | `models/loras/Z-Image-Turbo-Fun-Controlnet-Union.safetensors` | `models/model_patches/Z-Image-Turbo-Fun-Controlnet-Union.safetensors`（硬链接，ControlNet 补丁） |

- **若硬链接丢失需重建**（如重装环境）：
  ```powershell
  # MiniMax H3
  fsutil hardlink create "g:\ComfyUI_windows_portable\ComfyUI\models\unet\minimax_h3_fl2va_pruned_int8_convrot.safetensors" "g:\ComfyUI_windows_portable\ComfyUI\models\diffusion_models\minimax_h3_fl2va_pruned_int8_convrot.safetensors"
  fsutil hardlink create "g:\ComfyUI_windows_portable\ComfyUI\models\clip\qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" "g:\ComfyUI_windows_portable\ComfyUI\models\text_encoders\qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
  # Flux-Kontext + Z-Image-Turbo (UNet / CLIP)
  fsutil hardlink create "g:\ComfyUI_windows_portable\ComfyUI\models\unet\flux1-dev-kontext_fp8_scaled.safetensors" "g:\ComfyUI_windows_portable\ComfyUI\models\diffusion_models\flux1-dev-kontext_fp8_scaled.safetensors"
  fsutil hardlink create "g:\ComfyUI_windows_portable\ComfyUI\models\unet\z_image_turbo_bf16.safetensors" "g:\ComfyUI_windows_portable\ComfyUI\models\diffusion_models\z_image_turbo_bf16.safetensors"
  fsutil hardlink create "g:\ComfyUI_windows_portable\ComfyUI\models\clip\clip_l.safetensors" "g:\ComfyUI_windows_portable\ComfyUI\models\text_encoders\clip_l.safetensors"
  fsutil hardlink create "g:\ComfyUI_windows_portable\ComfyUI\models\clip\t5xxl_fp8_e4m3fn_scaled.safetensors" "g:\ComfyUI_windows_portable\ComfyUI\models\text_encoders\t5xxl_fp8_e4m3fn_scaled.safetensors"
  fsutil hardlink create "g:\ComfyUI_windows_portable\ComfyUI\models\clip\qwen_3_4b.safetensors" "g:\ComfyUI_windows_portable\ComfyUI\models\text_encoders\qwen_3_4b.safetensors"
  fsutil hardlink create "g:\ComfyUI_windows_portable\ComfyUI\models\loras\Z-Image-Turbo-Fun-Controlnet-Union.safetensors" "g:\ComfyUI_windows_portable\ComfyUI\models\model_patches\Z-Image-Turbo-Fun-Controlnet-Union.safetensors"
  ```
  之后重跑即可。

### 7.2 提交后 `/history` 里任务一直不在、队列为空、无产出
- 先 `curl http://localhost:8188/queue` 看 `queue_running`/`queue_pending` 是否真的进队。
- 若 `node_errors` 非空，按报错节点检查：模型文件名拼错、节点 id 引用错误、子图前缀写错。

### 7.3 500 错误
- 几乎都是把 **GUI 格式**（含 `nodes`/`links`）当 API 格式提交了。改用「Save (API Format)」导出的 JSON。

### 7.4 模型找不到（加载器报 missing）
- 先在本机 `models/` 下用 Everything/PowerShell 搜实际文件名，确认它在 `unet` 还是 `diffusion_models`、`clip` 还是 `text_encoders`，再用 7.1 的硬链接或改工作流文件名对齐。

### 7.5 显存不足（OOM）
- 视频类任务务必用 `int8_convrot`/`nvfp4`/`fp8` 量化模型；分辨率 `megapixels` 降到 0.4；不要并发提交多个重任务。
- 文本编码器 `type` 选对：Flux2 用 `flux2`，Qwen-Image 用 `qwen_image`，MiniMax 用 `minimax`。

---

## 8. Agent 操作 SOP（推荐流程）

1. **确认 ComfyUI 在线**：`curl http://localhost:8188/system_stats`。
2. **选工作流**：按需求从 §4 清单挑文件（生图→Flux2/Qwen；视频→MiniMax H3 i2v/t2v/r2v；多视角→Qwen 1-click）。
3. **准备输入图**（如需）：把用户给的图复制到 `ComfyUI/input/`，记录文件名。
4. **改入参**：方式 A 改 JSON（换图、写提示词、必要时改宽高/种子）；方式 B 用 `run_workflow(overrides=...)`。
5. **提交并轮询**：方式 A POST `/prompt` + 轮询 `/history`；方式 B 直接拿返回。
6. **校验产出**：检查 `output/`（或 `output/video/`）下新文件的时间戳与大小；若失败按 §7 自愈后重跑。
7. **回给用户**：列出**图片/视频的绝对路径**，必要时附一句效果说明。

---

## 9. 一句话提醒

> 工作流文件已在 `comfyui-mcp-server/workflows/`（及同源副本 `docs/examples/`）里**跑通验证过**。Agent 优先直接复用它们（只改 `image` / `prompt` 等少数入参），不要凭空手搓新工作流——手搓极易踩节点名/槽位/路径的坑。
