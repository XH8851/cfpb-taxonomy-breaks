"""LaTeX crosswalk table generated from scripts/labels.py (the single source of truth).
Usage: python -I scripts/24_crosswalk_table.py <paper_dir>  ->  paper/tables/crosswalk.tex"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from labels import MAP, NEW_CLASSES, BREAKS
paper = Path(sys.argv[1])
esc = lambda s: s.replace("&", r"\&")
cols = []
for (a, b), m in MAP.items():
    groups = {}
    for old, (new, typ) in m.items():
        groups.setdefault((new, typ), []).append(old)
    lines = []
    for (new, typ), olds in sorted(groups.items(), key=lambda kv: (kv[0][1] == "same", kv[0][1] == "none", str(kv[0][0]))):
        lhs = ", ".join(esc(o) for o in olds)
        lines.append(f"{lhs} [{typ}]" if new is None or typ == "same" else f"{lhs} $\\to$ {esc(new)} [{typ}]")
    lines.append("New: " + ", ".join(r"\emph{" + esc(c) + "}" for c in NEW_CLASSES[b]))
    cols.append((f"{a} $\\to$ {b} ({BREAKS[b]})", lines))
n = max(len(c[1]) for c in cols)
rows = [" & ".join(c[1][i] if i < len(c[1]) else "" for c in cols) + r" \\" for i in range(n)]
(paper / "tables" / "crosswalk.tex").write_text(r"""\begin{table*}[t]
\caption{Product crosswalk between taxonomy regimes, generated from the released code. Edge types: \emph{same}; \emph{rename} (1:1); \emph{merge} (one or more old labels into a broader new label); \emph{successor} (an old label splits and the new label keeping its name takes the carry-over); \emph{none} (no name-continuous successor, no carry-over).}
\label{tab:crosswalk}
\small
\begin{tabular}{p{0.47\textwidth}p{0.47\textwidth}}
\toprule
%s & %s \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table*}
""" % (cols[0][0], cols[1][0], "\n".join(rows)))
print("crosswalk rows", n)
