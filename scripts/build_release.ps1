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
    python -m PyInstaller --onefile --noconfirm --name vibeharness `
        --paths $repo `
        --distpath (Join-Path $repo 'dist') `
        --workpath (Join-Path $repo 'build\pyi') `
        --specpath (Join-Path $repo 'build') `
        (Join-Path $repo 'run.py')
    if (-not (Test-Path $exe)) { throw "build failed: $exe not found" }
    Write-Host "built: $exe" -ForegroundColor Green

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
