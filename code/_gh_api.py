# -*- coding: utf-8 -*-
"""GitHub API helper that reuses the credential already stored by Git
Credential Manager (never prints the token)."""

from __future__ import annotations

import json
import subprocess
import sys
import urllib.request


def token() -> str:
    out = subprocess.run(
        ["git", "credential", "fill"],
        input="protocol=https\nhost=github.com\n\n",
        capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        if line.startswith("password="):
            return line.split("=", 1)[1]
    raise SystemExit("no stored credential")


def call(path: str, method: str = "GET", payload=None):
    data = None
    headers = {"User-Agent": "fcntm/1.0",
               "Accept": "application/vnd.github+json",
               "Authorization": "Bearer " + token()}
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request("https://api.github.com" + path, data=data,
                                 headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=60) as fh:
        return json.loads(fh.read().decode())


if __name__ == "__main__":
    me = call("/user")
    print("authenticated as:", me["login"])
    if len(sys.argv) > 1:
        print(json.dumps(call(sys.argv[1]), indent=1)[:1200])
