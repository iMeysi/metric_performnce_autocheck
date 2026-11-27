Set-Location -LiteralPath 'D:\new project Claude\pad-omnianomaly-modern'

Expand-Archive -LiteralPath '.\smd-machine-1-1.zip' -DestinationPath . -Force

Get-ChildItem '.\data\smd\*\machine-1-1.txt' | Select-Object FullName, Length

& '.\.venv\Scripts\python.exe' '.\evaluate_smd.py'



