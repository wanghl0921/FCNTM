# -*- coding: utf-8 -*-
"""Delete the mis-named checkpoint asset and re-upload it under an ASCII name."""

import urllib.request

import _gh_release as R

OLD_ID = 655335640
NEW_NAME = "FCNTM_checkpoints_20261014.zip"
PATH = r"C:\Users\Administrator\Downloads\FCNTM_检查点_20261014.zip"


def delete_asset(asset_id: int) -> None:
    req = urllib.request.Request(
        "https://api.github.com/repos/%s/%s/releases/assets/%d"
        % (R.OWNER, R.REPO, asset_id),
        headers={"User-Agent": "fcntm/1.0",
                 "Accept": "application/vnd.github+json",
                 "Authorization": "Bearer " + R.token()},
        method="DELETE")
    with urllib.request.urlopen(req, timeout=120) as fh:
        print("delete status:", fh.status)


if __name__ == "__main__":
    delete_asset(OLD_ID)
    rel = R.api("/repos/%s/%s/releases/tags/%s" % (R.OWNER, R.REPO, R.TAG))
    R.upload(rel["id"], PATH, NEW_NAME)
