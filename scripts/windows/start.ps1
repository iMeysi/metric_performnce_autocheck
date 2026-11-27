$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..\..')
$Py = Join-Path (Get-Location) '.venv\Scripts\python.exe'
if (-not (Test-Path $Py)) { throw 'Run setup.ps1 first.' }
& $Py app.py --config config.windows.yaml
if ($LASTEXITCODE -ne 0) { throw "Service exited: $LASTEXITCODE" }
