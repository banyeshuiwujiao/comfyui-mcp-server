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

不要用 `run_*.bat`（会卡在 `pause` 等待按键），也不要用带 `--windows-standalone-build`
而不带 `--disable-auto-launch` 的裸命令——standalone 模式会隐式 `--auto-launch` 弹浏览器。
Agent 只需要 HTTP 8188，直接使用仓库内置的无弹窗启动脚本：

```powershell
cd g:\ComfyUI_windows_portable\comfyui-mcp-server
pwsh -NoProfile -ExecutionPolicy Bypass -File .\start_comfyui_agent.ps1
```

等价命令（不弹 WebUI）：

```powershell
& g:\ComfyUI_windows_portable\python_embeded\python.exe -s g:\ComfyUI_windows_portable\ComfyUI\main.py `
  --windows-standalone-build --listen --disable-auto-launch
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
| `api_image_z_image_turbo.json` | **Z-Image-Turbo 文生图（轻量变体）** | `57:27` 的 `text`、`57:13` 的宽高、`57:3` 的 `seed` | `output/z-image-turbo_*.png` |
| `api_image_z_image_int8.json` | **Z-Image INT8 量化文生图**（低显存变体） | `76:67` 的 `text`、`76:69` 的 `seed` | `output/z-image-*.png` |
| `api_wan2.1_fun_control.json` | **Wan2.1 Fun-Control 控制图生视频**（Canny 控制） | `52` 的 `image`（控制图）、`6` 的 `text`、`3` 的 `seed` | `output/wan_*.mp4` |
| `api_vision_joy_caption_basic.json` | **JoyCaption 基础视觉反推**（图→详细描述文本） | `4` 的 `image`（输入图） | `output/joy_caption/basic_caption_*.txt` |
| `api_vision_joy_caption_advanced.json` | **JoyCaption 高级视觉反推**（带光照/构图/美学等附加选项） | `4` 的 `image`；`5` 的 `Joy_extra_options` 各项开关 | `output/joy_caption/advanced_caption_*.txt` |
| `api_vision_joy_caption_batch.json` | **JoyCaption 批量反推**（目录→训练用字幕 .txt） | `2` 的 `input_dir`（图片目录路径） | 各图同目录 `<原名>.txt` |
| `api_vision_joy_caption_flux_pipeline.json` | **JoyCaption→Flux 闭环**（反推描述→Flux 按描述重绘） | `190` 的 `image`（输入图） | `output/joy_caption_flux/pipeline_*.png` + 描述 txt |

### 4.1 视觉反推工作流（JoyCaptionAlpha Two 插件）

> **核心价值**：本机 ComfyUI 通过 `ComfyUI_SLK_joy_caption_two`（JoyCaptionAlpha Two 的 ComfyUI 实现）获得了**视觉理解/反推能力**——大模型（Agent）可以把「看不懂」的图片喂进来，得到高质量英文描述（caption），再用于文生图、训练数据标注、或作为 Flux 重绘的提示词。这是 Agent 补齐自身「视觉缺失」的关键工具。
>
> **插件安装位置**：`ComfyUI/custom_nodes/ComfyUI_SLK_joy_caption_two/`（已从 `g:\ComfyUI_windows_portable\ComfyUI_SLK_joy_caption_two` 复制安装）。
> **依赖模型**：`models/Joy_caption_two/`（clip_model.pt + image_adapter.pt + text_model/）、`models/LLM/Meta-Llama-3.1-8B-Instruct-bnb-4bit/`、`models/clip/siglip-so400m-patch14-384/`。

#### 4.1.1 四个实用反推工作流

| 工作流 | 场景 | MCP 暴露入参 | 调用示例（overrides） |
|---|---|---|---|
| `api_vision_joy_caption_basic.json` | 单图→长描述。适合「让 Agent 理解一张陌生图片的内容」 | `image` | `{"image": "photo_001.png"}` |
| `api_vision_joy_caption_advanced.json` | 单图→带可控维度（光照/构图/美学/水印/NSFW 等 17 项开关）的精细描述 | `image` | `{"image": "art_002.png"}`（选项在 `5` 节点预置，可按需改） |
| `api_vision_joy_caption_batch.json` | 整个目录→每张图生成训练格式字幕 `.txt`（支持触发词 `name`、前后缀、重命名） | `input_dir` | `{"input_dir": "E:/dataset/raw"}` |
| `api_vision_joy_caption_flux_pipeline.json` | 反推→Flux 按描述重新生成图像（视觉理解与生成的闭环） | `image` | `{"image": "ref.png"}` |

#### 4.1.2 Agent 调用 SOP（视觉反推）

1. **确认 ComfyUI 在线**：`curl http://localhost:8188/system_stats`。
2. **选工作流**：理解单图 → `basic`；要精细维度 → `advanced`；标数据集 → `batch`；反推后想重绘 → `flux_pipeline`。
3. **准备输入图**：单图放到 `ComfyUI/input/` 下（批量则放整个目录），记录文件名/路径。
4. **调用**：MCP 方式 `run_workflow(workflow_id, overrides={"image": "xxx.png"})`；或 REST 方式改 JSON 提交（见 §5）。
5. **取结果**：`basic/advanced/flux_pipeline` 的 caption 文本经 `SaveText` 节点在 `/history/<pid>` 的 `outputs["3" 或 "3"].text` 返回，同时也落盘 `output/joy_caption/*.txt`；`batch` 直接在各图同目录写 `<原名>.txt`。
6. **闭环用法**：把 `basic` 返回的 caption 串给 `api_image_z_image_turbo_t2i` 的 `prompt`，或用 `flux_pipeline` 一步到位重绘。

#### 4.1.3 实测验证记录（2026-08-16 本机 RTX 4070 Ti SUPER 16GB）

| 工作流 | 实测状态 | 端到端耗时 | 输出产物 | 评价 |
|---|---|---|---|---|
| `api_vision_joy_caption_basic.json` | OK | ~12s（首图含模型加载，后续 ~3s） | `basic_caption_00001.txt` 含准确描述 | 正确识别几何图形、文字、配色；SigLIP+LLM 反推质量高。 |
| `api_vision_joy_caption_advanced.json` | OK | ~12s | `advanced_caption_00001.txt` | 附加选项生效（光照/三分法/美学质量已出现在描述中）。 |
| `api_vision_joy_caption_batch.json` | OK | 2 图 ~25s | `red_circle.txt` + `green_triangle.txt` | Training Prompt 格式，适合直接做 LoRA/微调数据集标注。 |
| `api_vision_joy_caption_flux_pipeline.json` | OK | ~30s | `pipeline_00001_.png` + `caption_00001.txt` | 反推描述→Flux 重绘闭环跑通；cfg=1、fp8 模型稳定。 |

**兼容性修复记录**（安装时必须）：
- 插件 `requirements.txt` 锁 `numpy==1.26.4`，但本机 `scipy 1.18` 要求 `numpy>=2.0` → 已升 `numpy` 到 `2.2.6` 修复。
- torch 2.11 下 `SiglipVisionModel.device` 为只读，`comfy.model_patcher.ModelPatcher` 的 `model.device = x` setter 会抛 `AttributeError` → 已修改 `joy_caption_two_node.py`，改对 `JoyClipVisionModel` / `JoyImageAdapter` 用 `.to(device)` 直接管理显存，移除 ModelPatcher 包裹。
- `peft` 需 `>=0.20.0`（transformers 5.14.1 的 `integrations/peft.py` 引用 `_maybe_shard_state_dict_for_tp`，旧版 0.12/0.18 缺失）→ 已升 `peft-0.20.0`。
- 修改已同步回 `g:\ComfyUI_windows_portable\ComfyUI_SLK_joy_caption_two\` 源目录。

#### 4.1.4 中文输出与底座模型边界（重要）

- **不能直接把 `models/text_encoders` 里的 Qwen/Gemma 换进 JoyCaption**：JoyCaption 的图像适配器
  （image_adapter.pt）与文本投影、`<|image_start|>` 等特殊 token 都是按 **Llama-3.1-8B** 的
  hidden size/tokenizer 训练的；Qwen/Gemma 架构、hidden size、tokenizer 均不同，换底座会直接
  shape/token 不匹配。那些 `qwen_*.safetensors` 也是 ComfyUI 重打包的 **text encoder 格式**，
  不是可被 `Joy_caption_two_load` 加载的 HF 模型目录。
- **保证中文的正式路线已落地**：JoyCaption 先出英文 caption（质量稳定），再调
  `api_text_gemma_translate_zh`——ComfyUI 原生 `TextGenerate` + `CLIPLoader(type=ltxv)`
  加载**本机已有** `gemma_3_12B_it_fp4_mixed.safetensors`，零新模型下载、输出稳定简体中文。
  8 英雄中文打标与 CharacterVault 导入即走此链路（见 §4.1.5）。
- 若仍想 JoyCaption 原生中文，唯一兼容方向是找 **Llama-3.1-8B 同架构的中文指令微调模型**
  （另需 ~8GB 下载），当前未采用。

#### 4.1.5 中文打标 → CharacterVault 标准链路

```text
api_vision_joy_caption_batch (English, output_format=json)
  -> api_text_gemma_translate_zh (本地 Gemma，逐条/批量翻译)
  -> import_captions_to_character_vault (display_names/trigger_words/tags)
  -> apply_character_to_prompt (后续生成自动注入角色一致性关键词)
```

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
> **现已为每个工作流补上 `PARAM_` 占位符**（字符串与数值入参均支持：`image`/`prompt` 以及 `seed`/`width`/`height`），并把文件复制到 `comfyui-mcp-server/workflows/`，重启 MCP server 后即自动注册为参数化工具。`docs/examples/` 是同源副本，两者以字节级同步为准。

各工作流暴露的 MCP 入参（2026-08-16 实测版本）：

| 工作流 | 暴露入参 | 绑定节点 | 备注 |
|---|---|---|---|
| `api_flux_kontext_dev_image_edit.json` | `image`, `prompt`, `seed` | `190/image`, `192:6/text`, KSampler seed | 改图指令走 `prompt` |
| `api_image_flux2_klein_image_edit_9b_base.json` | `image`, `image2`, `prompt`, `seed` | `76/image`, `81/image`, `75:74/text`, noise_seed | 双图：主图 + 参考图 |
| `api_image_flux2_text_to_image_9b.json` | `prompt`, `seed` | `75:74/text`, `75:73/noise_seed` | 文生图 |
| `api_image_z_image_turbo_fun_union_controlnet.json` | `image`, `prompt`, `seed` | `58/image`, `70:45/text`, KSampler seed | 参考图 + 提示词 |
| `api_image_z_image_turbo_t2i.json` | `prompt`, `width`, `height`, `seed` | `57:27/text`, `57:13` 宽高, `57:3/seed` | 文生图（16:9 背景主力） |
| `api_qwen_Image_edit_subgraphed.json` | `image`, `prompt`, `seed` | `78/image`, `141:132/prompt`, KSampler seed | 编辑指令走 `prompt` |
| `api_qwen_image_edit_2511_1_click_multiple_character_angles-v1.0.json` | `image`, `prompt`, `seed` | `25/image`；`prompt` 统一覆盖 8 个分支 | 一键 6 角色视角 |
| `api_qwen_image_edit_2512_1_click_multiple_scene_angles-v1.0.json` | `image`, `prompt`, `seed` | `25/image`；`prompt` 统一覆盖 16 个 slot | 一键 9 场景视角 |
| `api_utility_z_image_turbo_2k_upscaler.json` | `image`, `prompt` | `77/image`, `87:67/text` | 待放大图 + 提示词 |
| `api_video_minimax_h3_i2v.json` | `image`, `seed` | `114/image`, noise_seed | **prompt 未标记**：原文含 `<Picture 1>` 首帧引用，整体替换会丢失语义，故保留 |
| `api_video_minimax_h3_t2v.json` | `prompt`, `seed` | `105:104/prompt`, noise_seed | t2v 无 `<Picture>` 占位符，可安全标记 |
| `api_video_minimax_h3_r2v.json` | `image`, `image2`, `seed` | `137/image`, `139/image`, noise_seed | **prompt 未标记**：原文含 `<Picture 1>/<Picture 2>` 参考图引用，保留 |

2026-08-16 追加参数化的 3 个工作流（强类型工具 12→15）：`api_image_z_image_turbo.json`（`prompt/width/height/seed`）、`api_image_z_image_int8.json`（`prompt/seed`）、`api_wan2.1_fun_control.json`（`image/prompt/seed`）。

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
- `list_models` 现在会聚合 `UNETLoader` / `DiffusionLoader` / `CLIPLoader` / `VAELoader` / `LoraLoader` / `CheckpointLoaderSimple` 全部 loader 的模型名（去重）。
- `regenerate(asset_id, param_overrides=...)` 的参数改写逻辑已泛化：提示词可改写任意 `prompt`/`text`/`positive` 输入；`seed` 同时覆盖 `seed` 与 `noise_seed`（兼容 Flux2 / MiniMax H3）；自动将原资产 ID 记录为 `parent_asset_id` 并维持血缘链路。

### 4.0.3 P0 进阶能力：自愈建议、SQLite 资产血缘、音视频多模态感知

**1. 结构化错误自愈建议（Error Diagnoser & Self-Healing）**
- 当 ComfyUI 执行失败时，MCP 统一返回结构化诊断字典：
  - `error_type`：`CUDA_OOM`、`DIMENSION_NOT_DIVISIBLE`、`MODEL_NOT_FOUND`、`PARAM_OUT_OF_BOUNDS`、`NODE_EXECUTION_ERROR`
  - `actionable_recommendations`：人类与 Agent 友好的清晰排障建议
  - `suggested_params`：可以直接解构并自动重试的修正参数（例如自动降级 75% 分辨率、步数钳位到 20、最相近的模型名称）
- **Agent 最佳实践**：如果捕获到含有 `suggested_params` 的错误响应，可自动带入 `suggested_params` 重新提交，无需人工介入！

**2. SQLite 资产持久化与血缘关联（Asset Lineage Graph）**
- 资产记录写入 SQLite（默认 `data/assets.db`），跨服务重启永久保留。
- **`get_asset_lineage(asset_id)`**：一键查询母本祖先链（`ancestors`）、直接衍生子资产（`children`）与全家族树（`family_tree`）。
- **`search_assets(query, tag, workflow_id, limit)`**：支持跨会话的长记忆按 Prompt 关键词或 Tag 检索历史资产。

**3. 视频多关键帧胶片图与动图感知（Video Summarizer）**
- **`view_video_preview(asset_id, mode="strip"|"gif"|"metadata", num_frames=4)`**：
  - `mode="strip"`：在内存中抽取 $N$ 个时间轴关键帧，自动拼接为带时间戳徽章（`0.0s`, `1.5s`, `3.0s`...）的 WebP 胶片图，Vision Agent 单次 Tool Call 即可看全片。
  - `mode="gif"`：生成轻量动图循环。
- `view_image` 对视频资产自动降级为胶片预览。

**4. 音频特征分析与波形感知（Audio Summarizer）**
- **`analyze_audio(asset_id)`**：返回精准的 BPM 节奏测速、RMS/Peak 响度（dBFS）、静音段区间（`silent_segments`）与歌词结构（`lyrics_sections`）时间对齐。
- **`view_audio_preview(asset_id, mode="waveform"|"analysis")`**：返回深色科技感波形图或特征诊断。
- `view_image` 对音频资产自动渲染波形图。

### 4.0.4 角色与画风一致性档案库（Character Vault）

在连续绘本、游戏角色资产、连贯分镜设计等场景中，Agent 可以将人物外貌、专有触发词、绑定 LoRA、参考图与画风预设保存为持久化档案：
- **`save_character_profile(character_id, display_name, trigger_words, ...)`**：保存人物档案（如 `detective_john`）。
- **`apply_character_to_prompt(character_id, prompt)`**：自动将人物触发词置顶注入、负向词补充、画风预设关键词展开（如 `anime`, `cyberpunk`, `pixel_art` 等），并返回 LoRA 绑定信息。
- **`list_character_profiles()`** / **`get_character_profile()`**：支持按标签或画风检索。
- **直接指定 `character_id`**：在 `run_pipeline` 步骤中直接指定 `character_id`，流水线自动完成注入。

### 4.0.5 游戏/Web 资产后处理（透明抠图与精灵图集打包）

- **一键抠图去底（`remove_background`）**：
  - `mode="auto"`：自动评估四角方差区分纯色演播室背景与自然场景；
  - `mode="color"`：精准色键识别（纯白、纯黑、绿幕等），配合高斯抗锯齿羽化；
  - `mode="grabcut"`：利用 OpenCV 图割算法智能提取主体；
  - 输出 32-bit 透明通道 RGBA PNG，自动记录 `generation_type="matting"` 与父级血缘。
- **Sprite Sheet 纹理图集打包（`generate_sprite_sheet`）**：
  - 将视频动作循环（I2V）或序列图切片按最优接近正方形排版；
  - 同步输出包含每一帧精确像素矩形 `{"x", "y", "w", "h"}` 的标准 JSON 元数据（兼容 TexturePacker / PixiJS / Phaser / Unity）。

### 4.0.6 模块化流水线连招（Modular Subgraph Pipelines）

- **`run_pipeline(steps=[...], pipeline_name=...)`**：单次 Tool Call 串联多阶段工作流，中间产物（`asset_id` / `filename`）自动在步骤间流动并串联血缘树。
- **`list_pipeline_recipes()`**：浏览预置经典连招：
  1. `t2i_to_2k_upscale`：文生图 + 2K 高清超分；
  2. `t2i_to_transparent_sticker`：文生图 + 自动抠图透明贴纸；
  3. `character_to_sprite_sheet`：角色生成 + 动作视频 + 精灵图集打包；
  4. `character_sheet_multiview`：角色生成 + Qwen 6 视角多角度输出。

### 4.0.7 深度拥抱 MCP 原生能力（Resources & Prompts）

- **MCP Resources（只读上下文注入）**：
  - `comfyui://system/gpu-health`：实时查询显存占用、剩余 GB 与队列状态（`healthy`/`busy`/`saturated`），避免盲目提交重任务导致 OOM。
  - `comfyui://models/checkpoints` / `comfyui://models/loras`：直读真实模型与 LoRA 清单，彻底杜绝模型名幻觉。
  - `comfyui://workflows` / `comfyui://characters`：直读已注册工作流目录与角色档案库。
- **MCP Prompts（专家提示词模板）**：
  - `flux_photo_prompt`：针对 FLUX 自然语言摄影长提示词偏好调优；
  - `cinematic_video_prompt`：针对 LTX-Video / MiniMax H3 电影运镜调优；
  - `character_sheet_prompt`：针对多视角角色设定表调优；
  - `music_generation_prompt`：针对 AceStep 音乐标签与结构化歌词调优。

### 4.0.8 2026-08-16 OneQi Godot 表现层实战迭代（数据飞轮落地）

本轮把 MCP 直接用于 `ET.Client` Godot 表现层（16:9 背景、QTE 特效、透明抠图、血缘检索），并修复了 6 项真实缺陷：

1. **输出根目录自动探测修复**：便携版布局 `comfyui-mcp-server/` 与 `ComfyUI/output` 同级，原候选表漏检 → 新增 `project_root.parent/ComfyUI/output`；`validate_comfyui_output_root` 放宽为接受 `video/audio` 子目录与输出标记文件。修复后启动日志：`Auto-detected ComfyUI output root: G:\ComfyUI_windows_portable\ComfyUI\output`。
2. **资产血缘持久化**：`server.py` 显式给 `AssetRegistry` 传 `data/assets.db`（库默认保持 `:memory:` 供单测隔离）。重启后 `search_assets` / `get_asset_lineage` 数据不再丢失。
3. **`fetch_asset_bytes` 对象化**：原实现只接受 URL 字符串，而 `remove_background`/`generate_sprite_sheet` 传的是 `AssetRecord` 对象 → 报 `No connection adapters were found for "AssetRecord(...)"`。现已支持两种形态，按 `filename/subfolder/folder_type` 拼 `/view` URL。
4. **后处理产物落盘**：matting / sprite-sheet 的字节原先只登记元数据、从不写盘，`asset_url` 悬空（`/view` 404）。新增 `persist_processed_bytes`，工具与 `run_pipeline` 步骤在登记前先写入 ComfyUI output 根。
5. **workflow 参数化补齐**：`workflows/api_utility_z_image_turbo_2k_upscaler.json` 补上 docs 已宣称的 `PARAM_IMAGE/PARAM_PROMPT`；`api_image_z_image_turbo_t2i.json` 新增 `PARAM_INT_WIDTH/PARAM_INT_HEIGHT`（实测 1024×576 直接可用）。
6. **新增 `comfy_mcp_cli.py`**：无 SDK 依赖的 streamable-http MCP CLI——`tools` / `call <tool> <json|@file>` / `read <uri>` / `prompts`，SSE 自动解析。脚本与 AI Agent 都可直接驱动 MCP。
7. **`comfyui://workflows` 资源修复**：`WorkflowManager.tool_definitions` 已是列表结构，资源端点仍按字典 `.items()` 遍历导致 `AttributeError: 'list' object has no attribute 'items'`；现已按 `WorkflowToolDefinition`（`workflow_id` + `parameters.values()` + `annotation`）正确序列化，全量 pytest 189/189 通过。

**Godot 表现层实战调用样例**（本仓库 `Config/mcp_gen_logs/` 有完整回包）：

```powershell
# 16:9 UI 背景（新 width/height 参数）
python comfy_mcp_cli.py call api_image_z_image_turbo_t2i '{"prompt":"dark fantasy arena background ...","width":1024,"height":576}'

# 特效图抠透明底（先 t2i 拿到 asset_id，再抠图，产物可直落 Resources/Fx/）
python comfy_mcp_cli.py call remove_background '{"asset_id":"<t2i_asset_id>","mode":"auto"}'

# 血缘检索（SQLite 持久化，跨重启）
python comfy_mcp_cli.py call get_asset_lineage '{"asset_id":"<t2i_asset_id>"}'
```

**经验**：给 Godot 生成 UI 资产时优先用 `z-image-turbo`（8 步 ≈ 16s/张）；需透明通道的资产走 `t2i → remove_background(auto)` 两段式；所有产物在 Godot 侧用 manifest.json 记录 `asset_id` 与血缘，形成可追溯的数据飞轮。

### 4.0.9 2026-08-16 默认模型告警清理（模型注入卫生）

- **问题**：`DefaultsManager._hardcoded_defaults` 曾为 `image`/`audio` 硬编码 `v1-5-pruned-emaonly.ckpt` / `ace_step_v1_3.5b.safetensors`。这两者不是所有机器都有的 checkpoint，导致启动即打 "not found in ComfyUI checkpoints" 告警；带 `model` 参数的工作流若调用方未显式传 model，还会被注入一个不存在的模型名直接报错。
- **修复**：删除硬编码 `model` 默认。工作流 JSON 自带模型节点，不再需要全局兜底模型；需要机器级默认时按优先级配置：`set_defaults` > `~/.config/comfy-mcp/config.json` > `COMFY_MCP_DEFAULT_{IMAGE,AUDIO,VIDEO}_MODEL`。
- **回归测试**：新增 `tests/test_defaults_manager.py`——无配置时启动零告警且 `get_default(..., "model")` 返回 None；显式配置了缺失模型时仍按原语义告警。

### 4.0.10 2026-08-16 新工作流参数化补强（12→15 个强类型工具）

- **发现**：`WorkflowManager._load_workflows` 对没有 `PARAM_*` 占位符的工作流直接跳过自动工具注册，只能走通用 `run_workflow` 裸调；上一轮入库的 11 个工作流全部处于该状态。
- **修复**：给与 Godot 表现层最相关的 3 个补齐占位符并验证注册：
  - `api_image_z_image_turbo.json`：`PARAM_PROMPT` + `PARAM_INT_WIDTH/HEIGHT` + `PARAM_INT_SEED`；
  - `api_image_z_image_int8.json`：`PARAM_PROMPT` + `PARAM_INT_SEED`（宽高保持分辨率选择器连线）；
  - `api_wan2.1_fun_control.json`：`PARAM_IMAGE` + `PARAM_PROMPT` + `PARAM_INT_SEED`。
- **约定**：入库新 workflow 时至少把 `prompt`（文本类）/`image`（图控类）/`seed` 参数化，否则等同"死资产"；重启 MCP 服务后强类型工具自动出现。

### 4.0.11 2026-08-16 数据飞轮健壮性修复（P0 崩溃点 + CLI 失败语义 + GPU 分级保护）

1. **修复 `run_pipeline` workflow 步骤崩溃**（`pipeline_orchestrator.py`）：`get_workflow`→`load_workflow`；`apply_workflow_overrides` 改为正确 4 参签名并移除 `__override_report__`；`diagnose_error`→`diagnose`。预置 recipes 中的 `generate_image` 统一替换为真实 `api_image_z_image_turbo_t2i`，并保留 `generate_image/generate_song/generate_video` 别名映射。
2. **修复 `comfyui://workflows` 资源崩溃**（`tools/mcp_resources.py`）：`tool_definitions` 是 list 而非 dict；参数类型字段是 `annotation` 而非 `type_hint`。现按真实 `WorkflowToolDefinition` 结构读取，返回 15 个工作流的完整参数清单。
3. **CLI 失败语义**（`comfy_mcp_cli.py`）：`tools/call` 的 `result.isError=true` 与业务 `{"error": ...}` 负载现在都使 CLI 退出码为 1（`--allow-error` 可恢复旧行为），shell 脚本 `&&` 链不再把生成失败当成功。
4. **GPU guard 分级保护**（`managers/gpu_guard.py`）：新增 `COMFY_MCP_GPU_MIN_FREE_GB`（默认 2GB）。轻量图像任务在空队列时仍可提交（ComfyUI 会换出常驻缓存），重负载（视频/音频/多视角）或队列非空时显存低于下限直接拒绝，防住 VAE decode access violation。
5. **资产血缘补强**：资产 metadata 新增 `workflow_hash`（工作流模板文件 SHA-256）；重复命中确定性文件名（`transparent_*`/`spritesheet_*`）时刷新全部内容字段，血缘/prompt/seed 始终描述最新字节。
6. **`docs/examples/` 重新与 `workflows/` 字节级同步**，消除"同源副本"漂移。
- **验证**：pytest 197/197；重启 MCP 后 47 工具在线；`comfyui://workflows` 读取成功；CLI 未知工具退出码 1。

### 4.0.12 2026-08-16 Godot 导出工具（数据飞轮第 3 步自动化）

- 新增 MCP 工具 **`export_to_godot(asset_id, target_dir, target_filename, category, overwrite)`**：
  - 直接从 ComfyUI `/view` 拉取资产字节，写入 Godot `Resources/<类别>/` 目录；
  - 自动维护目录旁 `manifest.json` 血缘台账（`asset_id`/`source_asset_id`/`matting_asset_id`/`workflow_id`/`prompt`/`workflow_hash`/像素尺寸/导出时间）；
  - `target_filename` 只接受纯文件名 + `.png/.webp/.jpg/.jpeg`，路径穿越一律拒绝；
  - 与 Godot 侧「AI 资产可选装饰 + try-catch 回落」约定匹配，失败返回结构化 `error_code`。
- 用法：
  ```powershell
  python comfy_mcp_cli.py call export_to_godot '{"asset_id":"<matting_asset_id>","target_dir":"E:/Desktop/MiniGame/OneQi-ET8.1/ET.Client/OneQi/MainPack/Resources/Fx","target_filename":"fx_victory.png","category":"Fx"}'
  ```
- **验证**：pytest 201/201；真实导出 `fx_victory.png`/`fx_defeat.png` 落盘并写入 5 条目 manifest（512×512、含完整血缘）。

### 4.0.13 2026-08-16 视频/音频工作流全量参数化 + publish/export 收敛

1. **24 个视频/音频工作流批量参数化**：AceStep 四套、MiniMax Music 3、LTX2 系列、WAN 2.1/2.2/VACE 系列全部补齐 `prompt`/`image`/`image2`/`seed`/`tags`/`lyrics`/`negative_prompt` 占位符；只动字符串与 seed，不参数化 steps/cfg/width/height（避免 0 值回落破坏采样）。自动注册强类型工具 **15→39**。唯一豁免：`api_video_minimax_h3_i2v` 的 prompt 含 `<Picture 1>` 首帧引用，保持不标记。
2. **`publish_asset` 与 `export_to_godot` 收敛**：二者共享 `_publish_asset_to_godot` 单一实现。`publish_asset` 新增 `target_dir`（Godot 模式），Godot 模式不要求 web publish root 就绪、拒绝 `web_optimize`；`export_to_godot` 保留为薄壳别名。命令行两条路径行为完全一致。
3. **Qwen 多视角实测结论（本机 16GB）**：`api_qwen_image_edit_2511` 引用 38GB bf16 模型，冷启动与 `--disable-async-offload` 均触发 comfy_aimdo `HostBuffer.read_file_slice failed`，**本机不可用**；`api_qwen_image_edit_2512`（fp8 19GB）冷启动成功并产出 9 分支。Godot 角色表当前推荐最快的 `api_image_z_image_turbo_t2i`（8 步）直接生成三视图 sheet。
- **验证**：pytest 205/205；重启后 72 行工具清单；`validate_workflow` 通过 minimax_music_3 / wan_vace_outpainting / wan2.2_i2v；`publish_asset(target_dir=...)` 实机重发布 `hero_1001_sheet.png` 成功（8 条目 lineage manifest）。

### 4.0.14 2026-08-16 音频工作流实测 + 预览工具修复 + 音视频导出

1. **`api_audio_minimax_music_3` 冷启动实测成功**：产出 30.8s mp3（-17.3 dBFS，158 BPM，含 4 段 0.35~0.6s 静音间隙）；`analyze_audio` 全特征可用；波形预览修复后返回 `image/webp`。
2. **预览工具返回类型修复**（`tools/asset.py`）：`view_image`/`view_video_preview`/`view_audio_preview` 标注 `-> dict` 但实际返回 `FastMCPImage`，触发 `outputSchema defined but no structured output returned`；统一改为 `-> Any`。
3. **`export_to_godot`/`publish_asset(Godot 模式)` 扩展音视频后缀**：`.mp3/.wav/.ogg/.mp4/.webm/.mov` 与图片同等走 basename 白名单 + lineage manifest；BGM 已实机落盘 OneQi `Resources/Audio/BGM/bgm_heroic_cue.mp3`。

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
