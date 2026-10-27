"""Fetch a pinned official DIFFA source checkout without altering existing work."""
import argparse
from pathlib import Path
import subprocess

COMMIT = "0a87594e110984fbaf00ebf5bfe5840c212e4438"
REPOSITORY = "https://github.com/NKU-HLT/DIFFA.git"


def verify_checkout(root):
    root = Path(root).resolve()
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    if not (root / ".git").exists():
        raise ValueError(f"Missing upstream checkout: {root}; run setup.sh")
    if Path(git("rev-parse", "--show-toplevel")).resolve() != root:
        raise ValueError("DIFFA root must be the actual checkout root")
    if git("rev-parse", "HEAD") != COMMIT:
        raise ValueError(f"Expected upstream commit {COMMIT}; existing checkout left unchanged")
    if git("status", "--porcelain", "--untracked-files=all"):
        raise ValueError("Upstream checkout is dirty; use a clean checkout")
    return root / "DIFFA-2"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("external/DIFFA"))
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--no-checkout", REPOSITORY, str(root)], check=True)
        subprocess.run(["git", "-C", str(root), "checkout", "--detach", COMMIT], check=True)
    print(verify_checkout(root))


if __name__ == "__main__":
    main()
