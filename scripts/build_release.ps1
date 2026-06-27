<#
  Build & install the RELEASED vibeharness executable.

  - Freezes the CURRENT code (this repo) into a standalone vibeharness.exe (PyInstaller).
  - Installs it to "C:\Program Files\vibeharness" and adds that folder to the SYSTEM PATH.

  The released `vibeharness` command is decoupled from the dev repo's `vibe`, so you can
  test a stable build while development continues in the repo / worktrees.

  Usage:  powershell -ExecutionPolicy Bypass -File scripts\build_release.ps1
  Re-run anytime to cut a new release. The install step needs admin (a UAC prompt).
#>
param([switch]$InstallOnly)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$exe  = Join-Path $repo 'dist\vibeharness.exe'

if (-not $InstallOnly) {
    Write-Host "== Building vibeharness.exe (PyInstaller) from $repo ==" -ForegroundColor Cyan
    python -m pip install --quiet --disable-pip-version-check pyinstaller
    Remove-Item $exe -ErrorAction SilentlyContinue
    # --collect-submodules bundles the codec submodules into the frozen archive.
    # They are imported dynamically (vibeharness.codec.get_codec does
    # importlib.import_module on vibeharness.codecs.<name>_codec), so PyInstaller's
    # static analysis does not see them and would otherwise omit them entirely --
    # leaving available_codecs() empty and every command dead on arrival (issue #17).
    # benchmarks is collected too so its dynamically-discovered task/codec modules
    # are present in the frozen exe.
    python -m PyInstaller --onefile --noconfirm --name vibeharness `
        --paths $repo `
        --collect-submodules vibeharness.codecs `
        --collect-submodules benchmarks `
        --distpath (Join-Path $repo 'dist') `
        --workpath (Join-Path $repo 'build\pyi') `
        --specpath (Join-Path $repo 'build') `
        (Join-Path $repo 'run.py')
    if (-not (Test-Path $exe)) { throw "build failed: $exe not found" }
    Write-Host "built: $exe" -ForegroundColor Green

    # --- post-build SMOKE TEST: a dead exe must NEVER ship (issue #17) ---------- #
    # Run the freshly-built FROZEN exe (not the source tree) on commands that
    # exercise codec discovery AND codec loading, and fail the build if they break.
    Write-Host "== Smoke-testing frozen exe ==" -ForegroundColor Cyan

    # 1) --print-system builds the registry + system prompt via get_codec('json');
    #    it errors if codec loading is broken. Assert exit 0 and the prompt rendered.
    $printOut = & $exe --print-system 2>&1
    $printRc  = $LASTEXITCODE
    if ($printRc -ne 0) {
        Write-Host ($printOut | Out-String) -ForegroundColor Red
        throw "SMOKE TEST FAILED: '$exe --print-system' exited $printRc (expected 0). The frozen exe cannot load codecs."
    }
    if (($printOut | Out-String) -notmatch 'How the loop works') {
        Write-Host ($printOut | Out-String) -ForegroundColor Red
        throw "SMOKE TEST FAILED: '$exe --print-system' did not render the system prompt (missing 'How the loop works')."
    }
    Write-Host "  ok: --print-system rendered the system prompt" -ForegroundColor Green

    # 2) The same path under an explicit --codec json must also succeed, proving
    #    codec discovery resolves the default and get_codec('json') loads it.
    $jsonOut = & $exe --codec json --print-system 2>&1
    $jsonRc  = $LASTEXITCODE
    if ($jsonRc -ne 0 -or (($jsonOut | Out-String) -notmatch 'How the loop works')) {
        Write-Host ($jsonOut | Out-String) -ForegroundColor Red
        throw "SMOKE TEST FAILED: '$exe --codec json --print-system' exited $jsonRc; codec discovery/loading is broken in the frozen exe."
    }
    Write-Host "  ok: --codec json --print-system succeeded (codec discovery is non-empty)" -ForegroundColor Green
    Write-Host "Smoke test passed: the frozen exe can discover and load codecs." -ForegroundColor Green

    Write-Host "== Installing to Program Files (elevated) ==" -ForegroundColor Cyan
    Start-Process powershell -Verb RunAs -Wait -ArgumentList @(
        '-ExecutionPolicy','Bypass','-File', $PSCommandPath, '-InstallOnly')
    Write-Host "== Done. Open a NEW terminal and run: vibeharness --help ==" -ForegroundColor Green
    exit 0
}

# --- elevated install step ---
$dest = 'C:\Program Files\vibeharness'
New-Item -ItemType Directory -Force $dest | Out-Null
Copy-Item $exe (Join-Path $dest 'vibeharness.exe') -Force
$machinePath = [Environment]::GetEnvironmentVariable('Path', 'Machine')
if ($machinePath -notlike "*$dest*") {
    [Environment]::SetEnvironmentVariable('Path', ($machinePath.TrimEnd(';') + ';' + $dest), 'Machine')
    Write-Host "added to system PATH: $dest"
} else {
    Write-Host "system PATH already contains: $dest"
}
Write-Host "installed: $dest\vibeharness.exe"
