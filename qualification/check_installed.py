"""Artifact-only installed package/data closure, outside checkout and build tools."""
from importlib import metadata, resources, util
import hashlib
import json
from pathlib import Path
import shutil
import sys

import local_semantic_atlas as atlas

installed = Path(atlas.__file__).resolve()
assert "site-packages" in installed.parts and installed.is_relative_to(Path(sys.prefix))
assert not Path("/build-source").exists() and not Path("/opt/preparation").exists()
for name in ("build", "setuptools", "wheel", "pyproject_hooks"):
    assert util.find_spec(name) is None, name
assert shutil.which("gcc") is None and shutil.which("cc") is None
assert shutil.which("local-semantic-atlas")
entrypoints = metadata.distribution("local-semantic-atlas").entry_points
assert any(entry.name == "local-semantic-atlas" and entry.value == "local_semantic_atlas.cli:main" for entry in entrypoints)
data = resources.files("local_semantic_atlas").joinpath("_data")
inventory = json.loads(data.joinpath("notice-inventory-all.json").read_bytes())
for item in inventory["files"]:
    payload = data.joinpath(item["path"]).read_bytes()
    assert len(payload) == item["bytes"] and hashlib.sha256(payload).hexdigest() == item["sha256"], item["path"]
for name in ("README.md", "README.pt-BR.md", "NOTICE.md", "CONTRIBUTING.md", "SECURITY.md"):
    assert data.joinpath("docs", name).is_file(), name
ready = json.loads(Path("/opt/model/READY.json").read_bytes())
assert ready["status"] == "COMPLETE" and ready["code"] == "MODEL_PREPARED" and len(ready["files"]) == 6
for item in ready["files"]:
    path = Path("/opt/model") / item["name"]
    payload = path.read_bytes()
    assert len(payload) == item["bytes"] and hashlib.sha256(payload).hexdigest() == item["sha256"]
print(json.dumps({"status": "PASS", "scope": "installed wheel rebuilt from sdist, no checkout/compiler/build backend; new downloader six-artifact closure",
                  "package": str(installed), "notice_files_verified": len(inventory["files"]),
                  "model_artifacts_verified": len(ready["files"]), "entrypoint": "local-semantic-atlas"}))
