# run-tests.ps1 - wrapper around `cargo test --release`
#
# This script sets up a log file, runs the cargo test, and tail's the
# log to the console on exit. It's the equivalent of a `make test`
# for cargo projects on Windows PowerShell.
#
# Usage: .\scripts\run-tests.ps1

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogDir = Join-Path $ScriptDir "..\target\test-logs"
$LogFile = Join-Path $LogDir "test-$(Get-Date -Format 'yyyyMMdd-HHmmss').log"

if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

Write-Host "Running cargo test --release..."
Write-Host "Log: $LogFile"

Push-Location (Join-Path $ScriptDir "..")
try {
    cargo test --release 2>&1 | Tee-Object -FilePath $LogFile
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "Test log written to: $LogFile"
