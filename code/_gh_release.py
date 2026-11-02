# -*- coding: utf-8 -*-
"""Create the v1.0.0 release and upload the checkpoint / details archives.

The token is read from Git Credential Manager in-process and never printed.
"""

from __future__ import annotations

import http.client
import json
import os
import subprocess
import urllib.error
import urllib.parse
import urllib.request

OWNER, REPO, TAG = "wanghl0921", "FCNTM", "v1.0.0"

ASSETS = [
    (r"C:\Users\Administrator\Downloads\FCNTM_检查点_20261014.zip",
     "FCNTM_检查点_20261014.zip"),
    (r"C:\Users\Administrator\Downloads\FCNTM_release_details_20261016.zip",
     "FCNTM_release_details_20261016.zip"),
]

BODY = "\n".join([
    "Reference implementation and archived artifacts for",
    "",
    "*Field-Cooperative Neural Topic Modeling of Civil Engineering Literature:",
    "Missing-Abstract Inference*.",
    "",
    "Assets in this release:",
    "",
    "- `FCNTM_检查点_20261014.zip` - 11 model-state files (49 states) used by the",
    "  released per-seed results; place them under `outputs/` and",
    "  `outputs/rerun/` before re-scoring.",
    "- `FCNTM_release_details_20261016.zip` - the large per-seed detail dumps",
    "  excluded from the repository itself.",
    "",
    "The manuscript and the submission files are not part of this repository.",
])


def token() -> str:
    out = subprocess.run(
        ["git", "credential", "fill"],
        input="protocol=https\nhost=github.com\n\n",
        capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        if line.startswith("password="):
            return line.split("=", 1)[1]
    raise SystemExit("no stored credential")


def api(path: str, method: str = "GET", payload=None):
    headers = {"User-Agent": "fcntm/1.0",
               "Accept": "application/vnd.github+json",
               "Authorization": "Bearer " + token()}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request("https://api.github.com" + path, data=data,
                                 headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=120) as fh:
        return json.loads(fh.read().decode())


def ensure_release() -> dict:
    try:
        rel = api("/repos/%s/%s/releases/tags/%s" % (OWNER, REPO, TAG))
        print("release exists:", rel["html_url"])
        return rel
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
    rel = api("/repos/%s/%s/releases" % (OWNER, REPO), "POST", {
        "tag_name": TAG,
        "target_commitish": "main",
        "name": "v1.0.0 - code, checkpoints and per-seed details",
        "body": BODY,
        "draft": False,
        "prerelease": False,
    })
    print("release created:", rel["html_url"])
    return rel


def upload(release_id: int, path: str, name: str) -> None:
    size = os.path.getsize(path)
    print("uploading %s (%.1f MB)" % (name, size / 1e6), flush=True)
    conn = http.client.HTTPSConnection("uploads.github.com", timeout=3600)
    url = ("/repos/%s/%s/releases/%d/assets?name=%s"
           % (OWNER, REPO, release_id, urllib.parse.quote(name)))
    conn.putrequest("POST", url)
    conn.putheader("Authorization", "Bearer " + token())
    conn.putheader("User-Agent", "fcntm/1.0")
    conn.putheader("Accept", "application/vnd.github+json")
    conn.putheader("Content-Type", "application/zip")
    conn.putheader("Content-Length", str(size))
    conn.endheaders()
    sent = 0
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(16 * 1024 * 1024)
            if not chunk:
                break
            conn.send(chunk)
            sent += len(chunk)
            print("   %.0f / %.0f MB" % (sent / 1e6, size / 1e6), flush=True)
    resp = conn.getresponse()
    body = resp.read().decode("utf-8", "replace")
    print("   status", resp.status)
    if resp.status >= 300:
        raise SystemExit("upload failed: " + body[:500])
    data = json.loads(body)
    print("   uploaded:", data.get("name"), data.get("state"),
          data.get("browser_download_url"))


def main() -> None:
    rel = ensure_release()
    existing = {a["name"] for a in api(
        "/repos/%s/%s/releases/%d/assets" % (OWNER, REPO, rel["id"]))}
    for path, name in ASSETS:
        if name in existing:
            print("asset already present:", name)
            continue
        upload(rel["id"], path, name)
    print("done:", rel["html_url"])


if __name__ == "__main__":
    main()
