import subprocess

try:
    result = subprocess.run(
        [r"venv\Scripts\pip.exe", "install", "-r", "requirements.txt"],
        capture_output=True,
        text=True
    )
    with open("pip_err.txt", "w", encoding="utf-8") as f:
        f.write(result.stdout + "\n" + result.stderr)
except Exception as e:
    with open("pip_err.txt", "w", encoding="utf-8") as f:
        f.write(str(e))
