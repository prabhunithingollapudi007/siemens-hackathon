"""download_topcow.py - download TopCoW data (CTA/MRA + Circle of Willis labels) from Zenodo."""
import hashlib, sys, zipfile
from pathlib import Path
import urllib.request

REC = "https://zenodo.org/records/15692630/files"
FILES = {   # name: md5  (sizes: see comments)
    "CTA_LargeIA.zip":         "83cf2f4df95738eb5f76c6eb6b78c3bf",  # 0.8 MB (masks only, see note)
    "MRA_Lausanne.zip":        "647dc1af8a97bb1c4b33f9522fbae0ed",  # 1.3 GB, CC0
    "MRA_IXI_HH.zip":          "bb3339b460b639be4a0a4d6901443284",  # 0.5 GB, CC BY-SA 3.0
    "CTA_ISLES2024_TUM.zip":   "885395780a96224a002f405a4b4b52e0",  # 2.0 GB, CC BY-NC
    "TopCoW2024_Data_Release.zip": "d248a932166316c6c906644b22296d6c",  # 10.5 GB, main CTA+MRA set
}

def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()

def download(name, out):
    dest = out / name
    if dest.exists() and md5(dest) == FILES[name]:
        print(f"{name}: already complete"); return dest
    print(f"{name}: downloading ...")
    urllib.request.urlretrieve(f"{REC}/{name}?download=1", dest,
        lambda b, bs, t: print(f"\r  {min(b*bs, t)/1e6:.0f}/{t/1e6:.0f} MB", end="") if t > 0 else None)
    print()
    if md5(dest) != FILES[name]:
        sys.exit(f"{name}: checksum mismatch - delete the file and retry")
    return dest

if __name__ == "__main__":
    out = Path("topcow_data"); out.mkdir(exist_ok=True)
    which = sys.argv[1:] or ["MRA_Lausanne.zip"]      # smallest useful real-data set by default
    for name in which:
        z = download(name, out)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(out / z.stem)
        print(f"{name}: extracted to {out / z.stem}")