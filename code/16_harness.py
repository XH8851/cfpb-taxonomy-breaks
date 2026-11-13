"""SPEC v1 policy-replay harness (CPU only, GPU-free; this is the released benchmark harness).

Input: long tier tables with key (period, deploy_month, model_regime, role, cid) and columns
score, pred, correct, plus item meta (month, regime, label, label_r, rank, eidx).
Roles: deploy (eval items of month m), cal (released audit_cal items of months m-6..m-1),
grid (eval items of month m-1, used unlabeled), static (eval-start model on every item).

Path. lam in L = {lam0, lam0+0.05, ..., 1.0}. For tier k, tau_k(lam) = (1-lam)-quantile of tier-k
scores on grid items reaching tier k under tau_1..tau_{k-1}; if < 50 grid items reach the tier,
a data-independent fallback tau = 1 - lam is used. Routing therefore never uses labels.

Policies (cell names)
  rolling_{cod|plugin}_{known|none|det}  monthly thresholds; calibration window W months
       known: window restricted to the item's regime (reset at the published break date; in the
              dev period D the pre-registered placebo dates act as "known" resets)
       none : earlier-regime calibration items kept, labels mapped (no carry-over -> dropped)
       det  : window restarts at the month a chi^2 test on tier-1 predicted labels
              (E_m vs E_{m-1}, p < 0.01) alarms
  static_{cod|plugin}        thresholds certified once at eval start (preaudit months 1-2 =
                             calibration, preaudit month 3 = grid), applied to scheduled models
  fullstatic_{cod|plugin}    the same thresholds on the eval-start models, never updated
                             (fullstatic_plugin with tiers=[t1] is the fatal baseline B1)
  rolling_sgr_{known|none|det}  SGR baseline (added 2026-11-13 after review): binary search on the
                             full-cascade diagonal, Bonferroni over ceil(log2 L) steps, CP bound
  crc      rolling CRC on joint risk P(err and accepted): largest lam with (k+1)/(n+1) <= alpha
  wcrc     weighted + recency CRC: weights exp(-age/2) * max(0, cos(e_i, centroid of E_m))
  aci      plug-in at an online level alpha_t, alpha_{t+1} = alpha_t + gamma*(alpha - r_t), with r_t
           the realized selective risk on month t's released audit (available at t+1)
cod: fixed sequence ('fs') from lam0 upward, or Bonferroni ('bf') over L; Clopper-Pearson at delta.

cod tests: fs (single diagonal path) or ms (one path per tier subset, Bonferroni across paths).
Usage (as a module): run(frame, tiers, period, policy, alpha, W, lam0, test, ...) -> monthly DataFrame
CLI: python -I scripts/16_harness.py <config.json>
"""
import sys, json, itertools
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import beta, chi2_contingency

DELTA = 0.05
BREAKS = {"A": {"R1": "2017-04-24"}, "B": {"R2": "2023-08-25"}, "D": {}}
PLACEBO = {"D": ["2019-10-01", "2020-10-01"]}
ORDER = ["R0", "R1", "R2"]


def mshift(m, k):
    return str(pd.Period(m, "M") + k)


def cp_upper(k, n, d):
    return 1.0 if n == 0 else (1.0 if k >= n else float(beta.ppf(1 - d, k + 1, n - k)))


def cp_lo(k, n):
    return np.nan if n == 0 else (0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1)))


def path(lam0):
    return np.round(np.arange(lam0, 1.0001, 0.05), 2)


def configs(tiers, lam0, test):
    """Candidate paths of per-tier acceptance fractions.
    'fs': one diagonal path (every tier accepts the same fraction lam of what reaches it).
    'ms': one diagonal path per non-empty subset of tiers (tiers outside the subset switched off),
          so a weak tier cannot block the cascade. Returns a list of paths (lists of configs)."""
    L = path(lam0)
    subsets = [tuple(range(len(tiers)))] if test == "fs" else [
        s for k in range(len(tiers), 0, -1) for s in itertools.combinations(range(len(tiers)), k)]
    return [[tuple(lam if i in s else 0.0 for i in range(len(tiers))) for lam in L] for s in subsets]


def taus_for(grid, tiers, lams):
    if np.isscalar(lams):
        lams = [lams] * len(tiers)
    taus, rem = [], grid
    for t, lam in zip(tiers, lams):
        s = rem[f"{t}_score"].values
        if lam <= 0:
            tau = np.inf
        elif lam >= 0.999:
            tau = -np.inf
        else:
            tau = float(np.quantile(s, 1 - lam, method="higher")) if len(s) >= 50 else 1.0 - lam
        taus.append(tau)
        rem = rem[rem[f"{t}_score"] < tau]
    return taus


def route(df, tiers, taus):
    """Return (accepted mask, correct-if-accepted mask, accepting tier index or -1)."""
    acc = np.zeros(len(df), bool)
    cor = np.zeros(len(df), bool)
    who = np.full(len(df), -1)
    left = np.ones(len(df), bool)
    for i, (t, tau) in enumerate(zip(tiers, taus)):
        a = left & (df[f"{t}_score"].values >= tau)
        acc |= a
        cor |= a & df[f"{t}_correct"].values.astype(bool)
        who[a] = i
        left &= ~a
    return acc, cor, who


def choose(cal, grid, tiers, alpha, lam0, rule, test="fs", weights=None, level=None):
    """Pick per-tier acceptance fractions for one month. rule in {cod, plugin, crc, wcrc}.
    cod: fixed-sequence test along each path at delta / #paths (Bonferroni across paths), keep the
         last passing config of each path; then the passing config with the largest acceptance on
         the unlabeled grid. plugin: the config (any path) with the largest grid acceptance whose
         empirical calibration risk <= alpha. crc/wcrc: the monotone full-cascade diagonal only.
    Returns (config or None, taus)."""
    a = alpha if level is None else level
    if rule == "sgr":
        # SGR (Geifman & El-Yaniv 2017): binary search for the largest acceptance on the full-cascade
        # diagonal whose Clopper-Pearson bound <= alpha, Bonferroni over its ceil(log2 L) steps; no lam0 knob
        P = configs(tiers, 0.05, "fs")[0]
        steps = int(np.ceil(np.log2(len(P))))
        lo, hi, best = -1, len(P) - 1, None
        while lo < hi:
            mid = (lo + hi + 1) // 2
            taus = taus_for(grid, tiers, P[mid])
            acc, cor, _ = route(cal, tiers, taus)
            if cp_upper(int((acc & ~cor).sum()), int(acc.sum()), DELTA / steps) <= a:
                lo, best = mid, (P[mid], taus)
            else:
                hi = mid - 1
        return best if best is not None else (None, [np.inf] * len(tiers))
    paths = configs(tiers, lam0, "fs" if rule in ("crc", "wcrc") else test)
    d_eff = DELTA / len(paths)
    cands = []
    for P in paths:
        last = None
        for c in P:
            taus = taus_for(grid, tiers, c)
            acc, cor, _ = route(cal, tiers, taus)
            n, k = int(acc.sum()), int((acc & ~cor).sum())
            if rule == "cod":
                ok = cp_upper(k, n, d_eff) <= a
            elif rule == "plugin":
                ok = n > 0 and k / n <= a
            elif rule == "crc":
                ok = (k + 1) / (len(cal) + 1) <= a
            else:
                loss = (acc & ~cor).astype(float)
                ok = (np.sum(weights * loss) + 1) / (np.sum(weights) + 1) <= a
            if ok:
                if rule == "plugin":
                    cands.append((c, taus))
                else:
                    last = (c, taus)
            elif rule != "plugin":
                break
        if last is not None:
            cands.append(last)
    if not cands:
        return None, [np.inf] * len(tiers)
    if len(grid):
        return max(cands, key=lambda ct: route(grid, tiers, ct[1])[0].mean())
    return max(cands, key=lambda ct: sum(ct[0]))


def regime_of_date(per, d):
    r = "R0" if d < "2017-04-24" else ("R1" if d < "2023-08-25" else "R2")
    return r


def run(F, tiers, per, policy, alpha, W=3, lam0=0.10, test="fs", gamma=0.01, months=None,
        deploy_filter=None, resample_seed=None, emb=None, new_classes=None):
    """F: merged frame for one (period, task) with columns per tier. Returns monthly rows."""
    D = F[F["role"] == "deploy"]
    if deploy_filter is not None:
        D = D[deploy_filter(D)]
    allm = sorted(D["deploy_month"].unique())
    months = [m for m in allm if (months is None or m in months)]
    rng = np.random.default_rng(resample_seed) if resample_seed is not None else None
    static_taus = None
    if policy.startswith("static") or policy.startswith("fullstatic"):
        S = F[F["role"] == "static"]
        pa = S[S["set"] == "preaudit"]
        pm = sorted(pa["month"].unique())
        cal, grid = pa[pa["month"].isin(pm[:2])], pa[pa["month"] == pm[2]]
        rule = "cod" if "cod" in policy else "plugin"
        _, static_taus = choose(cal, grid, tiers, alpha, lam0, rule, test)
        Sdep = S[S["set"] == "eval"]
        if deploy_filter is not None:
            Sdep = Sdep[deploy_filter(Sdep)]
    reset_month = {}
    a_t = alpha
    aci_hist = {}
    rows = []
    brk = {**BREAKS.get(per, {})}
    resets = {r: d[:7] for r, d in brk.items()}
    if per in PLACEBO:
        resets = {f"P{i}": d[:7] for i, d in enumerate(PLACEBO[per])}
    for m in months:
        Dm = D[D["deploy_month"] == m]
        if policy.startswith("fullstatic"):
            Dm = Sdep[Sdep["month"] == m]
        n_m = len(Dm["cid"].unique()) if not policy.startswith("fullstatic") else len(Dm)
        tot = {"n_auto": 0, "n_err": 0, "n_new": 0, "new_ok": 0, "new_bad": 0, "n_joint_err": 0}
        per_tier = np.zeros(len(tiers), int)
        lam_used = []
        for r in sorted(Dm["model_regime"].unique()):
            e = Dm[Dm["model_regime"] == r]
            if policy.startswith("static") or policy.startswith("fullstatic"):
                taus = static_taus
            else:
                Fm = F[(F["deploy_month"] == m) & (F["model_regime"] == r)]
                grid = Fm[Fm["role"] == "grid"]
                cal = Fm[Fm["role"] == "cal"]
                win = [mshift(m, -k) for k in range(1, W + 1)]
                cal = cal[cal["month"].isin(win)]
                mode = policy.split("_")[-1] if policy.startswith("rolling") else "known"
                if mode == "known":
                    cal = cal[cal["regime"] == r]
                    if per in PLACEBO:  # placebo resets in D: window starts at the placebo month
                        for d in PLACEBO[per]:
                            if m >= d[:7]:
                                cal = cal[cal["month"] >= d[:7]]
                elif mode == "none":
                    cal = cal[cal["label_r"].notna()]
                elif mode == "det":
                    cal = cal[cal["regime"] == r]
                    g1 = grid[f"{tiers[0]}_pred"]; e1 = e[f"{tiers[0]}_pred"]
                    if len(g1) >= 50 and len(e1) >= 50:
                        tab = pd.crosstab(np.r_[np.zeros(len(g1)), np.ones(len(e1))], np.r_[g1.values, e1.values])
                        if tab.shape[1] > 1 and chi2_contingency(tab)[1] < 0.01:
                            reset_month[r] = m
                    if r in reset_month:
                        cal = cal[cal["month"] >= reset_month[r]]
                if rng is not None and len(cal):
                    parts = [g.sample(len(g), replace=True, random_state=int(rng.integers(1e9))) for _, g in cal.groupby("month")]
                    cal = pd.concat(parts)
                if len(grid) < 50:
                    grid = grid.iloc[0:0]
                if policy.startswith("rolling"):
                    rule = "cod" if "_cod_" in policy else ("sgr" if "_sgr_" in policy else "plugin")
                    _, taus = choose(cal, grid, tiers, alpha, lam0, rule, test)
                elif policy == "crc":
                    _, taus = choose(cal, grid, tiers, alpha, lam0, "crc")
                elif policy == "wcrc":
                    if emb is not None and len(cal):
                        c = emb[e["eidx"].values].astype(np.float32).mean(0)
                        c /= np.linalg.norm(c) + 1e-9
                        sim = np.clip(emb[cal["eidx"].values].astype(np.float32) @ c, 0, None)
                        age = np.array([(pd.Period(m, "M") - pd.Period(x, "M")).n for x in cal["month"]])
                        w = np.exp(-age / 2.0) * sim
                        w = w * len(w) / max(w.sum(), 1e-9)
                    else:
                        w = np.ones(len(cal))
                    _, taus = choose(cal, grid, tiers, alpha, lam0, "wcrc", weights=w)
                elif policy == "aci":
                    _, taus = choose(cal, grid, tiers, alpha, lam0, "plugin", level=a_t)
                    # realized risk on this month's audit_cal (released next month) under taus
                    aud = F[(F["role"] == "cal") & (F["month"] == m) & (F["deploy_month"] == mshift(m, 1)) & (F["model_regime"] == r)]
                    if len(aud):
                        acc, cor, _ = route(aud, tiers, taus)
                        aci_hist[m] = (acc & ~cor).sum() / acc.sum() if acc.sum() else alpha
            acc, cor, who = route(e, tiers, taus)
            tot["n_auto"] += int(acc.sum()); tot["n_err"] += int((acc & ~cor).sum())
            for i in range(len(tiers)):
                per_tier[i] += int((who == i).sum())
            if new_classes is not None:
                isnew = e["label"].isin(new_classes.get(e["regime"].iloc[0], [])).values if len(e) else np.zeros(0, bool)
                tot["n_new"] += int(isnew.sum()); tot["new_ok"] += int((isnew & cor).sum())
                tot["new_bad"] += int((isnew & acc & ~cor).sum())
        if policy == "aci" and m in aci_hist:
            a_t = float(np.clip(a_t + gamma * (alpha - aci_hist[m]), 0.001, 0.5))
        na, ne = tot["n_auto"], tot["n_err"]
        rows.append({"period": per, "policy": policy, "alpha": alpha, "W": W, "lam0": lam0, "test": test,
                     "tiers": "+".join(tiers), "month": m, "n": n_m, "n_auto": na, "n_err": ne,
                     "risk": ne / na if na else np.nan, "joint_risk": ne / n_m if n_m else np.nan,
                     "ci_lo": cp_lo(ne, na), "automation": na / n_m if n_m else np.nan,
                     "vacuous": na < 50, "excess": max(0.0, ne - alpha * na),
                     **{f"auto_{t}": per_tier[i] / n_m for i, t in enumerate(tiers)},
                     "n_new": tot["n_new"], "new_correct_auto": tot["new_ok"], "new_wrong_auto": tot["new_bad"],
                     "aci_level": a_t if policy == "aci" else np.nan})
    return pd.DataFrame(rows)


def summarize(R, breaks_month=None):
    out = []
    for key, g in R.groupby(["period", "policy", "alpha", "W", "lam0", "test", "tiers"]):
        nv = g[~g["vacuous"]]
        exp_fp = 0.025 * len(nv)
        row = dict(zip(["period", "policy", "alpha", "W", "lam0", "test", "tiers"], key))
        row.update({"months": len(g), "excess_per_1k": 1000 * g["excess"].sum() / g["n"].sum(),
                    "ci_lo_over": int((nv["ci_lo"] > row["alpha"]).sum()), "ci_lo_expected_nominal": exp_fp,
                    "over_1pt": int(((nv["risk"] - row["alpha"]) > 0.01).sum()), "vacuous_months": int(g["vacuous"].sum()),
                    "automation": g["automation"].mean()})
        if breaks_month:
            b = breaks_month.get(key[0])
            if b:
                post = g[(g["month"] >= b) & (g["month"] <= mshift(b, 6))]
                row["automation_post6"] = post["automation"].mean()
                row["excess_per_1k_post6"] = 1000 * post["excess"].sum() / post["n"].sum()
        out.append(row)
    return pd.DataFrame(out)


def merge_tiers(tables):
    """tables: dict tier -> long DataFrame (same key structure). Returns one frame with
    <tier>_score/_pred/_correct columns joined on the key."""
    key = ["period", "deploy_month", "model_regime", "role", "cid"]
    meta = ["set", "month", "regime", "label", "label_r", "rank", "eidx"]
    base = None
    for t, df in tables.items():
        d = df[key + meta + ["score", "pred", "correct"]].rename(
            columns={"score": f"{t}_score", "pred": f"{t}_pred", "correct": f"{t}_correct"})
        d = d.drop_duplicates(key)
        base = d if base is None else base.merge(d.drop(columns=meta), on=key, how="inner")
    base["eidx"] = base["eidx"].astype(np.int64)  # concat upstream can upcast to float/object
    return base
