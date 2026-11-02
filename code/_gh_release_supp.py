# -*- coding: utf-8 -*-
"""Publish the supplementary PDF as a GitHub release asset (v1.0.1)."""

import urllib.error

import _gh_release as R

TAG = "v1.0.1"
PATH = r"C:\Users\Administrator\Downloads\FCNTM_submission_20261019" \
       r"\optional_supplementary\supplementary.pdf"
NAME = "supplementary.pdf"

BODY = "\n".join([
    "Supplementary material for",
    "",
    "*Field-Cooperative Neural Topic Modeling of Civil Engineering Literature:",
    "Missing-Abstract Inference*.",
    "",
    "`supplementary.pdf` contains Tables S1-S20, Figures S1-S3, the ablation",
    "details and the training details of the baselines cited in the manuscript.",
])


def ensure() -> dict:
    try:
        rel = R.api("/repos/%s/%s/releases/tags/%s" % (R.OWNER, R.REPO, TAG))
        print("release exists:", rel["html_url"])
        return rel
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
    rel = R.api("/repos/%s/%s/releases" % (R.OWNER, R.REPO), "POST", {
        "tag_name": TAG,
        "target_commitish": "main",
        "name": "v1.0.1 - supplementary material",
        "body": BODY,
        "draft": False,
        "prerelease": False,
    })
    print("release created:", rel["html_url"])
    return rel


if __name__ == "__main__":
    rel = ensure()
    existing = {a["name"] for a in R.api(
        "/repos/%s/%s/releases/%d/assets" % (R.OWNER, R.REPO, rel["id"]))}
    if NAME in existing:
        print("asset already present:", NAME)
    else:
        R.upload(rel["id"], PATH, NAME)
    print("download:", "https://github.com/%s/%s/releases/download/%s/%s"
          % (R.OWNER, R.REPO, TAG, NAME))
