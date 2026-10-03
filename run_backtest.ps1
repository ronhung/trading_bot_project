<#
.SYNOPSIS
  Run the C++ backtest + Python brain (the two-terminal Phase 5 flow)
  from a single command.

.DESCRIPTION
  * Sets PYTHONUTF8 so the brain's emoji logs don't crash on Chinese-locale
    Windows (cp950).
  * Launches the Python brain in one new window and the C++ backtest engine
    in another (the engine window stays open to show the BACKTEST REPORT).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File run_backtest.ps1
#>

$ErrorActionPreference = "Stop"

$root   = $PSScriptRoot
$brain  = Join-Path $root "live_strategy\live_trend_bot.py"
$engine = Join-Path $root "live_engine\build_cmake\backtest_engine.exe"

if (-not (Test-Path $engine)) {
    throw "backtest_engine.exe not found. Run build.ps1 first.`nExpected: $engine"
}

# Python needs UTF-8 output on Chinese-locale Windows.
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

# Resolve the miniconda Python (not MSYS2's).
$python = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $python) { throw "python not found on PATH" }

Write-Host "1) Starting Python brain (new window)..." -ForegroundColor Cyan
Start-Process -FilePath $python -ArgumentList @($brain, "--no-warmup") -WorkingDirectory $root

Write-Host "   Giving the brain a moment to connect..."
Start-Sleep -Seconds 3

Write-Host "2) Starting C++ backtest engine (new window)..." -ForegroundColor Cyan
Start-Process -FilePath "cmd.exe" -ArgumentList @('/k', ('"{0}"' -f $engine)) -WorkingDirectory (Split-Path $engine)

Write-Host ""
Write-Host "Backtest launched." -ForegroundColor Green
Write-Host "  * Engine window prints the BACKTEST REPORT and stays open." -ForegroundColor Green
Write-Host "  * When it finishes, close the Python brain window (Ctrl+C)." -ForegroundColor Yellow
