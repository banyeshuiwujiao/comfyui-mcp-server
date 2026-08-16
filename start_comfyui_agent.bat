@echo off
REM AI Agent 启动 ComfyUI：禁用 --windows-standalone-build 隐含的浏览器弹窗。
pwsh -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_comfyui_agent.ps1"
