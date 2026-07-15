import subprocess


def git_metadata():
    def run(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True, check=False).stdout.strip()

    return {"commit": run("rev-parse", "HEAD") or None, "dirty": bool(run("status", "--porcelain"))}
