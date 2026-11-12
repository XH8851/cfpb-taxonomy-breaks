"""Verify every bibliography entry against its source before submission.

Usage: python -I scripts/check_citations.py paper/references.bib paper/main.tex
  * arXiv entries: title and first-author surname re-checked against the arXiv API
  * DOI entries: title re-checked against Crossref
  * other URLs: must return HTTP 200, unless the entry carries `verifiedmanually = {date, how}`
    (sites that block scripted access, e.g. BLS, JSTOR); those are still requested, the HTTP code
    is recorded, and they are listed as manual verifications instead of failures
  * every \\cite key in main.tex must exist in the bib
Writes paper/citation_check.json and prints a summary; exits 1 on any failure.
"""

from __future__ import annotations

import html
import json
import re
import sys
import time
import unicodedata
import urllib.request
from pathlib import Path


def norm(t: str) -> str:
    t = html.unescape(unicodedata.normalize("NFKD", t))
    t = re.sub(r"\\[a-zA-Z]+|[{}\\\"'`~^]", "", t)
    return re.sub(r"[^a-z0-9]", "", t.lower())


def entries(bib: str) -> dict:
    out = {}
    for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", bib, re.S):
        typ, key, body = m.groups()
        fields = {k.lower(): v.strip() for k, v in re.findall(r"(\w+)\s*=\s*\{(.*?)\}\s*,?\s*\n", body + "\n", re.S)}
        out[key.strip()] = (typ, fields)
    return out


def get(url: str, accept: str | None = None, browser: bool = False) -> tuple[int, str]:
    ua = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)" if browser else "citation-check/1.0 (mailto:none)"
    req = urllib.request.Request(url, headers={"User-Agent": ua,
                                               **({"Accept": accept} if accept else {})})
    try:
        with urllib.request.urlopen(req, timeout=40) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except Exception as e:  # noqa: BLE001
        return getattr(e, "code", 0) or 0, str(e)


def main() -> None:
    bib, tex = Path(sys.argv[1]).read_text(), Path(sys.argv[2]).read_text()
    ents = entries(bib)
    cited = set()
    for grp in re.findall(r"\\cite[pt]?\{([^}]+)\}", tex):
        cited |= {k.strip() for k in grp.split(",")}
    report, failures = {}, []
    missing = sorted(cited - ents.keys())
    if missing:
        failures.append(f"cited but not in bib: {missing}")
    arx = {k: f["eprint"] for k, (t, f) in ents.items() if "eprint" in f}
    arx |= {k: re.search(r"arxiv.org/abs/([\d.]+)", f["url"]).group(1)
            for k, (t, f) in ents.items() if "eprint" not in f and "arxiv.org/abs/" in f.get("url", "")}
    if arx:
        _, xml = get("https://export.arxiv.org/api/query?id_list=" + ",".join(arx.values()) + f"&max_results={len(arx) + 5}")
        api = {}
        for e in xml.split("<entry>")[1:]:
            aid = re.search(r"abs/([\d.]+)", e).group(1)
            api[aid] = (" ".join(re.search(r"<title>(.*?)</title>", e, re.S).group(1).split()),
                        re.findall(r"<name>(.*?)</name>", e))
        for k, aid in arx.items():
            title, authors = api.get(aid, ("", []))
            f = ents[k][1]
            ok_t = norm(title) == norm(f.get("title", ""))
            first = re.split(r"\s+and\s+", f.get("author", ""))[0]
            sur = norm(first.split(",")[0] if "," in first else first.split()[-1] if first.split() else "")
            ok_a = bool(authors) and (sur in norm(authors[0]) or sur in norm(" ".join(authors[:2])))
            report[k] = {"source": f"arXiv:{aid}", "title_ok": ok_t, "first_author_ok": ok_a}
            if not (ok_t and ok_a):
                failures.append(f"{k}: arXiv mismatch (api title '{title[:60]}', authors {authors[:1]})")
    for k, (typ, f) in ents.items():
        if k in arx:
            continue
        if "doi" in f:
            code, body = get("https://api.crossref.org/works/" + f["doi"])
            title = json.loads(body)["message"]["title"][0] if code == 200 else ""
            if code != 200:  # DataCite DOIs (e.g. UCI datasets) are not in Crossref; use doi.org metadata
                code, body = get("https://api.datacite.org/dois/" + f["doi"])
                title = json.loads(body)["data"]["attributes"]["titles"][0]["title"] if code == 200 else ""
            ok = code == 200 and norm(title) == norm(f.get("title", ""))
            report[k] = {"source": f"doi:{f['doi']}", "http": code, "title_ok": ok, "crossref_title": title}
            if not ok:
                failures.append(f"{k}: DOI check failed ({code}; crossref '{title[:60]}')")
            time.sleep(0.5)
        elif "url" in f:
            code, _ = get(f["url"], browser=True)
            report[k] = {"source": f["url"], "http": code}
            if code != 200 and "verifiedmanually" in f:
                report[k]["verified_manually"] = f["verifiedmanually"]
            elif code != 200:
                failures.append(f"{k}: URL returned {code}")
        else:
            report[k] = {"source": None, "note": "no identifier"}
            failures.append(f"{k}: no DOI/arXiv/URL to verify")
    Path(sys.argv[1]).with_name("citation_check.json").write_text(json.dumps(report, indent=1))
    print(f"{len(ents)} entries, {len(cited)} cited, {len(failures)} problems")
    for f_ in failures:
        print("  -", f_)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
