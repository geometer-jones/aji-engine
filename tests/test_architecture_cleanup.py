from pathlib import Path


def test_abstractor_dependency_is_removed() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    package_dir = repo_root / "aji_engine"

    assert not (package_dir / "persistent_control.py").exists()

    offenders = []
    for path in package_dir.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "persistent_control" in text or "Abstractor" in text:
            offenders.append(path.name)

    assert offenders == []
