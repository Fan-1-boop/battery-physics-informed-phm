from pathlib import Path
import hashlib, sys

EXPECTED = {
    "ResValData.zip": (250_042_781, "5898b06bf34550c2b57ea345c488d71fb24db4fbe6030225e260b062e22e840c"),
    "code_2025-06-05.zip": (3_791_887, "f9f681324ed460d2bee206fb41b22826dd48d80ed4601e7321c0c02f678f48fe"),
    "diagnostic_free_code.zip": (3_791_887, "f9f681324ed460d2bee206fb41b22826dd48d80ed4601e7321c0c02f678f48fe"),
}

def sha256(p: Path):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024), b''):
            h.update(b)
    return h.hexdigest()

root=Path(sys.argv[1] if len(sys.argv)>1 else Path(__file__).resolve().parents[1]/"data")
found=False
for name,(n,h) in EXPECTED.items():
    p=root/name
    if not p.exists():
        continue
    found=True
    actual_n=p.stat().st_size
    actual_h=sha256(p)
    print(f"{name}: bytes={actual_n} ({'OK' if actual_n==n else 'MISMATCH'}), sha256={actual_h} ({'OK' if actual_h==h else 'MISMATCH'})")
if not found:
    print(f"No official archives found in {root}")
    sys.exit(2)