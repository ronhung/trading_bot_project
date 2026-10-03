<#
.SYNOPSIS
  Rebuild the C++ engines (backtest_engine.exe + live_engine.exe).

.DESCRIPTION
  Uses the MSYS2 UCRT64 toolchain (CMake + Ninja) to configure and build
  live_engine/CMakeLists.txt. FetchContent pulls IXWebSocket, httplib,
  libzmq, cppzmq and nlohmann-json automatically.

  Notes baked in (learned the hard way):
    * Disables libzmq IPC (ZMQ_HAVE_IPC=OFF) — the IPC code path includes
      POSIX <sys/socket.h> which does not exist on MinGW.
    * Bundles runtime DLLs next to live_engine.exe so it runs from any
      terminal without touching PATH.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File build.ps1
#>

$ErrorActionPreference = "Stop"

# ---- Configuration ---------------------------------------------------------
# Override via $env:MSYS2_ROOT if your MSYS2 lives elsewhere.
$MsysRoot  = if ($env:MSYS2_ROOT) { $env:MSYS2_ROOT } else { "C:\msys64" }
$UcrtBin   = Join-Path $MsysRoot "ucrt64\bin"
$UcrtPref  = (Join-Path $MsysRoot "ucrt64") -replace '\\', '/'
$EngineDir = Join-Path $PSScriptRoot "live_engine"
$BuildDir  = Join-Path $EngineDir "build_cmake"

# ---- Toolchain lookup ------------------------------------------------------
$Cmake = Join-Path $UcrtBin "cmake.exe"
$Ninja = Join-Path $UcrtBin "ninja.exe"
if (-not (Test-Path $Cmake)) { throw "cmake not found: $Cmake" }
if (-not (Test-Path $Ninja)) { throw "ninja not found: $Ninja" }

# Put UCRT64 bin first so the compiler (cc1plus/cc1) finds its DLLs and
# CMake finds the compiler + ninja. (Conda's older mingw DLLs would otherwise
# shadow them.)
$env:Path = "$UcrtBin;$env:Path"

# ---- Configure + build -----------------------------------------------------
New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
Push-Location $BuildDir
try {
    Write-Host "==> Configuring (CMake + Ninja) ..." -ForegroundColor Cyan
    & $Cmake -G Ninja `
        -DCMAKE_BUILD_TYPE=Release `
        -DCMAKE_PREFIX_PATH="$UcrtPref" `
        -DOPENSSL_ROOT_DIR="$UcrtPref" `
        -DZMQ_HAVE_IPC=OFF `
        -DZMQ_HAVE_STRUCT_SOCKADDR_UN=OFF `
        $EngineDir
    if ($LASTEXITCODE -ne 0) { throw "cmake configure failed (exit $LASTEXITCODE)" }

    Write-Host "==> Building ..." -ForegroundColor Cyan
    & $Ninja
    if ($LASTEXITCODE -ne 0) { throw "ninja build failed (exit $LASTEXITCODE)" }

    # ---- Bundle runtime DLLs next to live_engine.exe ------------------------
    Write-Host "==> Bundling runtime DLLs ..." -ForegroundColor Cyan

    $zmqDll = Get-ChildItem (Join-Path $BuildDir "_deps") -Recurse -Filter "libzmq.dll" |
        Select-Object -First 1
    if ($zmqDll) { Copy-Item $zmqDll.FullName -Destination $BuildDir -Force }

    # MinGW runtime (stable names)
    foreach ($d in @("libgcc_s_seh-1.dll", "libwinpthread-1.dll", "libstdc++-6.dll")) {
        $src = Join-Path $UcrtBin $d
        if (Test-Path $src) { Copy-Item $src -Destination $BuildDir -Force }
        else { Write-Warning "DLL not found (skipped): $src" }
    }
    # OpenSSL + its deps (versioned names -> glob)
    foreach ($pat in @("libssl-*.dll", "libcrypto-*.dll", "libbrotli*.dll", "zlib1.dll")) {
        Get-ChildItem $UcrtBin -Filter $pat | ForEach-Object {
            Copy-Item $_.FullName -Destination $BuildDir -Force
        }
    }

    Write-Host ""
    Write-Host "=== Build complete ===" -ForegroundColor Green
    foreach ($exe in @("live_engine.exe", "backtest_engine.exe")) {
        $p = Join-Path $BuildDir $exe
        if (Test-Path $p) {
            $f = Get-Item $p
            "{0,-22} {1,10:N0} KB  {2}" -f $f.Name, ($f.Length / 1KB), $f.LastWriteTime
        } else {
            Write-Warning "Expected output missing: $exe"
        }
    }
}
finally {
    Pop-Location
}
