<#
.SYNOPSIS
  Run the C++ live engine + Python brain (the two-terminal Phase 6 flow)
  from a single command.

.DESCRIPTION
  * Sets PYTHONUTF8 so the brain's emoji logs don't crash on Chinese-locale
    Windows (cp950).
  * Launches the Python brain in one new window and the C++ live engine in
    another. Both run until you close them.

  WARNING: this is LIVE on Binance Testnet — the brain will place real
  (paper) orders using the API keys in shared/config.json.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File run_live.ps1
#>

$ErrorActionPreference = "Stop"

$root   = $PSScriptRoot
$brain  = Join-Path $root "live_strategy\live_trend_bot.py"
$engine = Join-Path $root "live_engine\build_cmake\live_engine.exe"

if (-not (Test-Path $engine)) {
    throw "live_engine.exe not found. Run build.ps1 first.`nExpected: $engine"
}

# Python needs UTF-8 output on Chinese-locale Windows.
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

# Resolve the miniconda Python (not MSYS2's).
$python = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $python) { throw "python not found on PATH" }

Write-Host "1) Starting Python brain (new window)..." -ForegroundColor Cyan
Start-Process -FilePath $python -ArgumentList @($brain) -WorkingDirectory $root

Write-Host "   Giving the brain a moment to warm up..."
Start-Sleep -Seconds 3

Write-Host "2) Starting C++ live engine (new window)..." -ForegroundColor Cyan
Start-Process -FilePath "cmd.exe" -ArgumentList @('/k', ('"{0}"' -f $engine)) -WorkingDirectory (Split-Path $engine)

Write-Host ""
Write-Host "Live engine launched — both windows are long-running." -ForegroundColor Green
Write-Host "  * Engine window shows the Binance Testnet connection + klines." -ForegroundColor Green
Write-Host "  * WARNING: real (paper) orders on Binance Testnet. Close both windows to stop." -ForegroundColor Yellow
