"""List a project's training samples with what decides whether Studio uses them.

Read-only. The API key comes from the EI_API_KEY environment variable and is
only sent, as a header, to studio.edgeimpulse.com.

    EI_API_KEY=ei_... python3 scripts/check_samples.py <project id>
"""

import json
import os
import sys
import urllib.request

HOST = "https://studio.edgeimpulse.com"


def get(path: str, key: str) -> dict:
    req = urllib.request.Request(HOST + path, headers={"x-api-key": key, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as res:
        return json.load(res)


def samples(project: int, key: str, category: str):
    offset = 0
    while True:
        res = get(f"/v1/api/{project}/raw-data?category={category}&limit=100&offset={offset}", key)
        if not res.get("success", True):
            raise SystemExit(f"API error: {res.get('error')}")
        page = res.get("samples", [])
        yield from page
        offset += len(page)
        if not page or offset >= res.get("totalCount", 0):
            return


def box_problems(s: dict) -> str:
    dims = s.get("imageDimensions") or {}
    w, h = dims.get("width"), dims.get("height")
    bad = []
    for b in s.get("boundingBoxes") or []:
        if b["width"] <= 0 or b["height"] <= 0:
            bad.append("empty")
        elif w and h and (b["x"] < 0 or b["y"] < 0 or b["x"] + b["width"] > w or b["y"] + b["height"] > h):
            bad.append("outside")
    return ",".join(f"{bad.count(k)} {k}" for k in sorted(set(bad))) or "-"


def main() -> int:
    if len(sys.argv) != 2 or not sys.argv[1].isdigit():
        raise SystemExit(__doc__)
    key = os.environ.get("EI_API_KEY")
    if not key:
        raise SystemExit("Set EI_API_KEY (a project API key from Dashboard -> Keys)")
    project = int(sys.argv[1])

    rows = list(samples(project, key, "training"))
    print(f"{len(rows)} training samples\n")
    cols = ("id", "filename", "label", "boxes", "bad boxes", "size", "disabled", "processing", "error", "seg_status")
    print("\t".join(cols))
    for s in rows:
        dims = s.get("imageDimensions") or {}
        print("\t".join(str(v) for v in (
            s.get("id"), s.get("filename"), s.get("label"), len(s.get("boundingBoxes") or []),
            box_problems(s), f"{dims.get('width')}x{dims.get('height')}" if dims else "?",
            s.get("isDisabled"), s.get("isProcessing"), s.get("processingError") or "-",
            (s.get("metadata") or {}).get("seg_status", "-"),
        )))

    # Anything else that differs between samples, in case the cause is a field not listed above.
    skip = {"id", "filename", "boundingBoxes", "metadata", "created", "lastModified", "added",
            "sha256Hash", "signatureValidate", "signatureKey", "coldstorageFilename", "thumbnailVideo",
            "thumbnailVideoFull", "imageDimensions"}
    varying = {}
    for k in sorted({k for s in rows for k in s} - skip):
        values = {json.dumps(s.get(k), sort_keys=True)[:80] for s in rows}
        if len(values) > 1:
            varying[k] = sorted(values)[:6]
    print("\nOther fields that differ between samples:")
    for k, v in varying.items():
        print(f"  {k}: {', '.join(v)}")
    if not varying:
        print("  (none)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
