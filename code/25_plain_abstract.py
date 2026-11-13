r"""Plain-text abstract for the submission form, generated from paper/main.tex + paper/numbers*.tex (macros \N, \M, \R).
Usage: python -I scripts/25_plain_abstract.py <paper_dir>  ->  paper/abstract.txt"""
import re, sys
from pathlib import Path
paper = Path(sys.argv[1])
macros = {}
for f in sorted(paper.glob("numbers*.tex")):
    macros.update(re.findall(r"\\newcommand\{\\([NMR]\w+)\}\{(.*?)\}\n", f.read_text()))
tex = (paper / "main.tex").read_text()
ab = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S).group(1)
ab = re.sub(r"\\([NMR][A-Za-z]+)\{\}|\\([NMR][A-Za-z]+)(?![A-Za-z])", lambda m: macros[m.group(1) or m.group(2)], ab)
ab = ab.replace("{,}", ",").replace("\\%", "%").replace("--", "-").replace("~", " ")
ab = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", ab)
ab = " ".join(ab.split())
assert "\\" not in ab, ab
(paper / "abstract.txt").write_text(ab + "\n")
print(len(ab.split()), "words"); print(ab)
