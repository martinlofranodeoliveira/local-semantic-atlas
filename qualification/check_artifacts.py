"""Verify real wheel/sdist contents, then rebuild a wheel from that source archive."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile

source, distributions, output = map(Path, sys.argv[1:])
assert len(sys.argv) == 4
output.mkdir(parents=True)
wheel = distributions / "local_semantic_atlas-0.1.0-py3-none-any.whl"
sdist = distributions / "local_semantic_atlas-0.1.0.tar.gz"
expected = {str(path.relative_to(source / "src")): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (source / "src/local_semantic_atlas").rglob("*") if path.is_file()}
with zipfile.ZipFile(wheel) as archive:
    names = archive.namelist()
    for name, digest in expected.items():
        assert hashlib.sha256(archive.read(name)).hexdigest() == digest, name
    assert not any(name.endswith((".onnx", ".whl", ".pyc")) for name in names)
    entry = archive.read("local_semantic_atlas-0.1.0.dist-info/entry_points.txt")
    assert b"local-semantic-atlas = local_semantic_atlas.cli:main" in entry
with tarfile.open(sdist) as archive:
    members = archive.getmembers()
    for name in ("README.md", "docs/README.pt-BR.md", "requirements-dev.lock", "requirements-runtime.lock",
                 "Dockerfile", "Dockerfile.prepare", "scripts/ci.sh", ".github/workflows/ci.yml",
                 "tests/test_core.py", "tests/test_io_boundary_original.py"):
        member = archive.extractfile("local_semantic_atlas-0.1.0/" + name)
        assert member is not None and member.read() == (source / name).read_bytes(), name
    for name, digest in expected.items():
        member = archive.extractfile("local_semantic_atlas-0.1.0/src/" + name)
        assert member is not None and hashlib.sha256(member.read()).hexdigest() == digest, name
    assert not any(item.name.endswith((".onnx", ".whl", ".pyc")) for item in members)
    with tempfile.TemporaryDirectory(prefix="atlas-sdist-") as temporary:
        archive.extractall(temporary, filter="data")
        rebuilt = output / "from-sdist"
        subprocess.run([sys.executable, "-m", "build", "--no-isolation", "--wheel", "--outdir", str(rebuilt),
                        str(Path(temporary) / "local_semantic_atlas-0.1.0")], cwd=temporary, check=True)
with zipfile.ZipFile(output / "from-sdist" / wheel.name) as archive:
    for name, digest in expected.items():
        assert hashlib.sha256(archive.read(name)).hexdigest() == digest, name
for artifact in (wheel, sdist): shutil.copyfile(artifact, output / artifact.name)
record = {"status": "PASS", "scope": "actual wheel and sdist code/data equality plus wheel rebuilt from sdist",
          "package_files_verified": len(expected), "artifacts": [
              {"path": str(path.relative_to(output)), "bytes": path.stat().st_size,
               "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
              for path in sorted(output.rglob("*")) if path.is_file()]}
(output / "artifact-verification.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record))
