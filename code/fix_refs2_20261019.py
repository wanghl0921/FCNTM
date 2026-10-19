# -*- coding: utf-8 -*-
"""Second round of reference fixes (2026-10-19).

* [45] is still online-first (the December 2026 issue has not appeared), so
  the entry cites it as "in press" instead of a future volume/page.
"""

import io

PAPER = r"C:\Users\Administrator\Downloads\FCNTM完整论文"

OLD = ("Automation in Construction, 2026, 192: 107220. "
       "DOI: 10.1016/j.autcon.2026.107220.")
NEW = ("Automation in Construction, 2026: in press. "
       "DOI: 10.1016/j.autcon.2026.107220.")


def main() -> None:
    path = PAPER + r"\main_en.tex"
    text = io.open(path, encoding="utf-8").read()
    count = text.count(OLD)
    if count != 1:
        raise SystemExit("expected one match, found %d" % count)
    io.open(path, "w", encoding="utf-8").write(text.replace(OLD, NEW))
    print("main_en.tex: [45] now cited as in press")


if __name__ == "__main__":
    main()
