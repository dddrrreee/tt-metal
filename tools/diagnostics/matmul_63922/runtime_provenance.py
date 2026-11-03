from pathlib import Path
import hashlib
import json
import os

def pytest_configure(config):
    import ttnn
    out = Path(os.environ["VALID_OUT"])
    side = os.environ["VALID_SIDE"]
    expected = json.loads((Path(os.environ["VALID_LIBRARIES"]) / side / "hashes.json").read_text())
    maps = [line for line in Path("/proc/self/maps").read_text().splitlines() if any(n in line for n in ["_ttnn", "libtt_metal"])]
    actual = {}
    for name in ["_ttnncpp.so", "_ttnn.so", "libtt_metal.so"]:
        path = next(Path(line.split()[-1]) for line in maps if line.endswith("/" + name))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == expected[name], name
        actual[name] = {"path": str(path), "sha256": digest}
    (out / "loaded-libraries.json").write_text(json.dumps(actual, indent=2) + "\n")
