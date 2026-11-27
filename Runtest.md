$Py = ".\.venv\Scripts\python.exe"
& $Py -m pip install --only-binary=:all: "tqdm>=4,<5"


& $Py -m unittest discover -s tests -v




& $Py .\scripts\smoke_real.py --epochs 1
