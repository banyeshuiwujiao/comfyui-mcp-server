# AI Agent 专用 ComfyUI 启动脚本（无 WebUI 弹窗）
#
# Windows portable 版的 --windows-standalone-build 会隐式开启
# --auto-launch（见 ComfyUI/comfy/cli_args.py），导致每次启动都弹浏览器。
# Agent 只需要 HTTP 8188，因此显式追加 --disable-auto-launch。
# 用法：pwsh -NoProfile -ExecutionPolicy Bypass -File .\start_comfyui_agent.ps1

$ErrorActionPreference = 'Stop'

$portableRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
$python = Join-Path $portableRoot 'python_embeded\python.exe'
$mainPy = Join-Path $portableRoot 'ComfyUI\main.py'

if (-not (Test-Path $python)) {
    throw "embedded python not found: $python"
}
if (-not (Test-Path $mainPy)) {
    throw "ComfyUI main.py not found: $mainPy"
}

Write-Host "[agent] starting ComfyUI without browser auto-launch"
& $python -s $mainPy --windows-standalone-build --listen --disable-auto-launch
exit $LASTEXITCODE
