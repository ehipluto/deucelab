import subprocess


def main() -> None:
    tracked = subprocess.check_output(["git", "ls-files", "-z"])
    ignored = subprocess.run(
        ["git", "check-ignore", "--no-index", "-z", "--stdin"],
        input=tracked,
        capture_output=True,
        check=False,
    )
    if ignored.returncode not in (0, 1):
        raise SystemExit(ignored.stderr.decode(errors="replace"))
    paths = sorted(
        path.decode(errors="backslashreplace") for path in ignored.stdout.split(b"\0") if path
    )
    if paths:
        raise SystemExit("Excluded files are tracked:\n" + "\n".join(paths))
    print("No tracked files match the publication exclusions.")


if __name__ == "__main__":
    main()
