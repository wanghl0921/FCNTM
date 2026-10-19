# -*- coding: utf-8 -*-
"""Verify every bibliography entry against Crossref / arXiv (2026-10-16).

Re-parses ``main_asce.bib``, queries the public Crossref API for every DOI
and the arXiv API for every arXiv identifier, and compares the returned
title / year / venue with the archived entry.  Entries that carry neither a
DOI nor an arXiv id are listed for manual checking.  Writes
``outputs/rerun/ref_check_20261016.txt``.
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path

PAPER = Path(r"C:\Users\Administrator\Downloads\FCNTM完整论文")
OUT = Path(__file__).resolve().parent / "outputs" / "rerun"
UA = {"User-Agent": "FCNTM-refcheck/1.0 (mailto:manqp@126.com)"}


def norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))


def similar(a: str, b: str) -> float:
    return SequenceMatcher(None, norm(a), norm(b)).ratio()


def get_json(url: str):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=40) as fh:
        return json.loads(fh.read().decode("utf-8", "replace"))


def get_text(url: str) -> str:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=40) as fh:
        return fh.read().decode("utf-8", "replace")


def parse_bib(raw: str):
    entries = []
    for block in raw.split("@")[1:]:
        head, _, body = block.partition(",")
        kind = head.split("{")[0].strip()
        key = head.split("{", 1)[1].strip() if "{" in head else ""
        fields = {}
        for m in re.finditer(r"(\w+)\s*=\s*\{", body):
            start = m.end() - 1          # index of the opening brace
            depth = 0
            for k in range(start, len(body)):
                if body[k] == "{":
                    depth += 1
                elif body[k] == "}":
                    depth -= 1
                    if depth == 0:
                        fields[m.group(1)] = body[start + 1:k]
                        break
        entries.append((kind, key, fields))
    return entries


def main() -> None:
    raw = (PAPER / "main_asce.bib").read_text(encoding="utf-8")
    entries = parse_bib(raw)
    lines = [f"reference check, {len(entries)} entries", ""]
    ok = notes = bad = manual = 0
    for kind, key, fields in entries:
        text = " ".join(fields.values())
        doi = fields.get("doi")
        arx = re.search(r"arXiv[:\s]*([0-9]{4}\.[0-9]{4,5})", text)
        title = fields.get("title", "")
        year = fields.get("year", "")
        venue = fields.get("journal") or fields.get("booktitle") or ""
        if doi:
            try:
                msg = get_json("https://api.crossref.org/works/"
                               + urllib.parse.quote(doi.replace("\\", ""))
                               + "?mailto=manqp@126.com")["message"]
            except Exception as exc:  # noqa: BLE001
                lines.append(f"[DOI-ERROR] {key}: {doi}: {exc}")
                manual += 1
                time.sleep(0.5)
                continue
            cr_title = (msg.get("title") or [""])[0]
            ratio = similar(title, cr_title)
            cr_year = ""
            for k in ("issued", "published-print", "published-online"):
                parts = (msg.get(k) or {}).get("date-parts") or []
                if parts and parts[0] and parts[0][0]:
                    cr_year = str(parts[0][0])
                    break
            cr_venue = (msg.get("container-title") or [""])[0]
            tag = "OK"
            details = [f"doi {doi}", f"title match {ratio:.2f}"]
            if ratio < 0.82:
                tag = "BAD"
                details.append(f"crossref title: {cr_title!r}")
            if cr_year and year and cr_year != year:
                details.append(f"year entry {year} vs crossref {cr_year}")
                if tag == "OK":
                    tag = "NOTE"
            if venue and cr_venue and similar(venue, cr_venue) < 0.55:
                details.append(f"venue entry {venue!r} vs crossref {cr_venue!r}")
                if tag == "OK":
                    tag = "NOTE"
            lines.append(f"[{tag}] {key}: " + "; ".join(details))
            ok += tag == "OK"
            notes += tag == "NOTE"
            bad += tag == "BAD"
            time.sleep(0.4)
        elif arx:
            try:
                xml = get_text("http://export.arxiv.org/api/query?id_list="
                               + arx.group(1))
                m = re.search(r"<entry>.*?<title>(.*?)</title>", xml, re.S)
                ax_title = re.sub(r"\s+", " ", m.group(1)).strip() if m else ""
            except Exception as exc:  # noqa: BLE001
                lines.append(f"[ARXIV-ERROR] {key}: arXiv:{arx.group(1)}: {exc}")
                manual += 1
                time.sleep(0.5)
                continue
            ratio = similar(title, ax_title)
            tag = "OK" if ratio >= 0.82 else "BAD"
            lines.append(f"[{tag}] {key}: arXiv:{arx.group(1)}; "
                         f"title match {ratio:.2f}"
                         + ("" if tag == "OK" else f"; arxiv title: {ax_title!r}"))
            ok += tag == "OK"
            bad += tag == "BAD"
            time.sleep(0.5)
        else:
            lines.append(f"[MANUAL] {key}: no DOI/arXiv id; "
                         f"{title[:80]!r} ({venue[:40]!r}, {year})")
            manual += 1
    summary = (f"summary: ok={ok} notes={notes} problems={bad} "
               f"manual={manual}")
    lines.append("")
    lines.append(summary)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ref_check_20261016.txt").write_text("\n".join(lines) + "\n",
                                                encoding="utf-8")
    print("\n".join(lines))
    print("report:", OUT / "ref_check_20261016.txt")


if __name__ == "__main__":
    main()
