# -*- coding: utf-8 -*-
"""The field-view FC-NTM numbers are prose now; verify them from the prose."""
from pathlib import Path
p = Path(r"C:\Users\Administrator\Downloads\FCNTM完整代码\check_paper_numbers.py")
raw = p.read_bytes().decode("utf-8")
crlf = "\r\n" in raw
text = raw.replace("\r\n", "\n")
s = text.index("def check_fieldview_table() -> None:")
e = text.index("def check_fieldview_baselines() -> None:")
new = '''def check_fieldview_table() -> None:
    """tab:fieldview: two tables were merged, so the FC-NTM rows are prose."""
    print("table (field view, FC-NTM rows in prose)")
    fc = fieldview()
    art = fc
    quote = r"(\\d+\\.\\d+)\\$\\\\pm\\$\\d+\\.\\d+"
    for corpus, label, marker in (("github", "GitHub", "在 GitHub 上为"),
                                  ("agnews", "AG News", "在 AG News 上为"),
                                  ("civil", "土木文献", "在土木文献上为")):
        pattern = (marker + r"\\s*" + quote + r"\\s*/\\s*" + quote +
                   r"\\s*\\n?（NPMI\\s*" + quote + r"\\s*/\\s*\\$-\\$" + quote)
        m = re.search(pattern, prose_text())
        if not m:
            FAILURES.append(f"field-view prose for {corpus} not found")
            print(f"  MISSING  field-view prose ({corpus})")
            continue
        bg_cv, sm_cv, bg_npmi, sm_npmi = m.groups()
        for view, cv, npmi in (("background", bg_cv, bg_npmi),
                               ("summary", sm_cv, sm_npmi)):
            g = fc[(fc["dataset"] == corpus) & (fc["view"] == view)]
            cmp(f"{label} {view} CV", g["CV"].mean(), cv)
            cmp(f"{label} {view} NPMI", g["NPMI"].mean(), npmi)
    support = re.search(
        r"概要视角的可评分主题数为 GitHub\\s*(\\d+)--(\\d+)、AG News\\s*"
        r"(\\d+)--(\\d+)、土木文献\\s*(\\d+)--(\\d+)", prose_text())
    if not support:
        FAILURES.append("field-view summary support range not found")
        print("  MISSING  field-view support range")
    else:
        got = [int(v) for v in support.groups()]
        for corpus, lo, hi in (("github", got[0], got[1]),
                               ("agnews", got[2], got[3]),
                               ("civil", got[4], got[5])):
            g = art[(art["dataset"] == corpus) & (art["view"] == "summary")]
            n = (int(g["CV_n"].min()), int(g["CV_n"].max()))
            global CHECKS
            CHECKS += 1
            if (lo, hi) != n:
                FAILURES.append(f"{corpus} summary support: artifact {n} "
                                f"vs paper {(lo, hi)}")
                print(f"  MISMATCH {corpus} summary support: {n} vs {(lo, hi)}")


'''
text = text[:s] + new + text[e:]
p.write_bytes((text.replace("\n", "\r\n") if crlf else text).encode("utf-8"))
print("checker updated")
