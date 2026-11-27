param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..\..')
function Check-Exit { if ($LASTEXITCODE -ne 0) { throw "Command failed: $LASTEXITCODE" } }
# Standard GIL build only.
& $Python -c "import sys,struct,sysconfig; assert sys.version_info[:2] == (3,15) and struct.calcsize('P') == 8 and not sysconfig.get_config_var('Py_GIL_DISABLED'), 'Standard Python 3.15 x64 required'"
Check-Exit
if (Test-Path '.venv') { throw 'Existing .venv: rename or remove it explicitly before migration.' }
& $Python -m venv .venv
Check-Exit
$Py = Join-Path (Get-Location) '.venv\Scripts\python.exe'
& $Py -m pip install --upgrade pip
Check-Exit
# Fail before installation when native wheels for this interpreter are absent.
& $Py -m pip download --only-binary=:all: -r requirements-windows.txt -d wheelhouse
Check-Exit
& $Py -m pip install --no-index --find-links=wheelhouse -r requirements-windows.txt
Check-Exit
& $Py -m pip check
Check-Exit
& $Py -m unittest discover -s tests -v
Check-Exit
& $Py scripts/smoke_real.py --epochs 1
Check-Exit
& $Py -m pip freeze | Out-File -Encoding utf8 requirements.resolved.windows.txt
Check-Exit
Write-Host 'Installation and smoke completed. Configure Prometheus before start.'
