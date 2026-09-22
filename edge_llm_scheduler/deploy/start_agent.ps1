<#
SPDX-FileCopyrightText: Copyright (c) 2026
SPDX-License-Identifier: Apache-2.0

一台笔记本上启动 stage agent（Windows）。

用法（在 project 目录下）：
    powershell -ExecutionPolicy Bypass -File edge_llm_scheduler\deploy\start_agent.ps1 `
        -NodeId alpha -Port 9100 -Model .models\Qwen2.5-0.5B-Instruct -Device cuda:0

先用 -Device cpu 确认能起来，再换 cuda:0；若显存不足把 --dtype 设成 float16。
#>
param(
    [Parameter(Mandatory = $true)][string]$NodeId,
    [int]$Port = 9100,
    [string]$Model = ".models/Qwen2.5-0.5B-Instruct",
    [string]$Device = "cuda:0",
    [string]$Dtype = "auto",
    [string]$WireDtype = "float16",
    [int]$Threads = 4,
    [string]$LogLevel = "info"
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $root
$env:PYTHONPATH = "."
$env:PYTHONUNBUFFERED = "1"

Write-Host "启动 agent node=$NodeId port=$Port device=$Device model=$Model" -ForegroundColor Cyan
Write-Host "工作目录: $root"
Write-Host "别的机器用  python -m edge_llm_scheduler.experiments.measure_link --host <本机IP> --port 9200  测到本机的链路"
Write-Host ""

python -m edge_llm_scheduler.agents.stage_agent `
    --node-id $NodeId `
    --host 0.0.0.0 `
    --port $Port `
    --model $Model `
    --device $Device `
    --dtype $Dtype `
    --wire-dtype $WireDtype `
    --threads $Threads `
    --log-level $LogLevel
