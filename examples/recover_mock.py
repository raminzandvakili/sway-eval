"""Recover the mock model's planted parameters from an audit's own records.

Input: a run folder written by `sway-eval audit` on examples/mock_audit.yaml or
examples/mock_exact.yaml. From its records alone (every layout, the formatting variants, the
option rotations) plus each section's rule points from the item metadata, one joint fit recovers
the five position weights, the carryover, the formatting effect and the option-order bias.
Features per record: the points of the section at each position, the sum of products of adjacent
sections' points, an indicator for a format without "- " bullets, an indicator for the target
option listed first, and an intercept.

The fit follows how answers are produced: noise on the logit, then rounding to the reported grid.
With a grid, each answer only says that the noisy logit lies in an interval (the grid cell; the
edge cells are open-ended), so the fit is interval-censored normal maximum likelihood, with the noise
SD estimated alongside; nothing is dropped. Without a grid or noise, least squares on the logit.
95% intervals: bootstrap over records (300 resamples).

Needs numpy, which the core package does not install. With uv:

    uv run --with numpy python examples/recover_mock.py examples/out/results/<run> OUT.json

With pip, in the activated environment:

    pip install numpy
    python examples/recover_mock.py examples/out/results/<run> OUT.json
"""

import json
import sys
from pathlib import Path

import numpy as np

SECTIONS = ["Summary", "Experience", "Education", "Skills", "Certifications"]
NAMES = [f"weight_{k + 1}" for k in range(5)] + ["carryover", "format_effect", "option_bias"]


def ncdf(x):
    """Standard normal CDF (Abramowitz and Stegun 7.1.26 for erf; |error| < 1.5e-7)."""
    t = 1 / (1 + 0.3275911 * np.abs(x) / np.sqrt(2))
    y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x / 2)
    return 0.5 * (1 + np.sign(x) * y)


def npdf(x):
    return np.exp(-x * x / 2) / np.sqrt(2 * np.pi)


def logit(p):
    return np.log(p / (1 - p))


def design(run):
    cfg = json.loads((run / "config.json").read_text())
    items = {json.loads(l)["id"]: json.loads(l) for l in open(run / "items.jsonl")}
    pts = {i: {s: it["meta"]["points"].get(s, 0) for s in SECTIONS} for i, it in items.items()}
    fv = cfg.get("format_variants", [])
    no_dash = {k + 1 for k, v in enumerate(fv) if "bullet" in v and v["bullet"] != "- "}
    X, P, owner = [], [], []
    for line in open(run / "records.jsonl"):
        r = json.loads(line)
        if r["kind"] not in ("layout", "format", "option_rotation"):
            continue
        i = r["example_id"]
        s = [pts[i][u.split(":", 1)[1]] for u in r["passage_ids"]]
        X.append(s + [sum(a * b for a, b in zip(s, s[1:])), float(r.get("format_index", 0) in no_dash),
                      float(r["options"].index("ADVANCE") == 0), 1.0])
        P.append(r["probs"][r["options"].index("ADVANCE")])
        owner.append(i)
    return cfg, np.array(X), np.array(P), np.array(owner)


def fit(X, P, grid, start=None):
    """Estimates of the coefficients, the noise SD (None without a grid) and the raw parameters."""
    b = np.linalg.lstsq(X, logit(np.clip(P, 1e-6, 1 - 1e-6)), rcond=None)[0]
    if not grid:
        return b, None, None
    lo = np.where(P <= grid + 1e-9, -np.inf, logit(np.clip(P - grid / 2, 1e-12, 1)))
    hi = np.where(P >= 1 - grid - 1e-9, np.inf, logit(np.clip(P + grid / 2, 0, 1 - 1e-12)))

    def parts(th):
        s = np.exp(th[-1])
        z = X @ th[:-1]
        a, c = (lo - z) / s, (hi - z) / s
        ia, ic = np.isinf(a), np.isinf(c)
        a0, c0 = np.where(ia, 0.0, a), np.where(ic, 0.0, c)
        Fa, Fc = np.where(ia, 0.0, ncdf(a0)), np.where(ic, 1.0, ncdf(c0))
        fa, fc = np.where(ia, 0.0, npdf(a0)), np.where(ic, 0.0, npdf(c0))
        L = np.maximum(Fc - Fa, 1e-300)
        return s, a0, c0, fa, fc, L

    def loglik(th):
        return float(np.sum(np.log(parts(th)[5])))

    def score(th):
        s, a0, c0, fa, fc, L = parts(th)
        gz = (fa - fc) / (s * L)
        return np.concatenate([X.T @ gz, [np.sum(-(c0 * fc - a0 * fa) / L)]])

    if start is None:  # start from least squares in probability space (Gauss-Newton)
        for _ in range(50):
            mu = 1 / (1 + np.exp(-(X @ b)))
            step = np.linalg.lstsq(X * (mu * (1 - mu))[:, None], P - mu, rcond=None)[0]
            b = b + step
            if np.abs(step).max() < 1e-10:
                break
        th = np.concatenate([b, [np.log(0.1)]])
    else:
        th = np.array(start, dtype=float)
    ll = loglik(th)
    for _ in range(200):
        g0 = score(th)
        H = np.column_stack([(score(th + e) - g0) / 1e-6 for e in np.eye(len(th)) * 1e-6])
        step = np.linalg.lstsq(H, -g0, rcond=None)[0]
        if not np.all(np.isfinite(step)) or g0 @ step <= 0:
            step = g0 / max(1.0, np.abs(g0).max())  # not an ascent direction: take a short gradient step
        t = 1.0
        while t > 1e-12:  # backtrack until the likelihood rises
            new = loglik(th + t * step)
            if np.isfinite(new) and new >= ll:
                break
            t /= 2
        th, ll_old, ll = th + t * step, ll, loglik(th + t * step)
        if abs(ll - ll_old) < 1e-10 and np.abs(t * step).max() < 1e-8:
            break
    return th[:-1], float(np.exp(th[-1])), th


def recover(run):
    run = Path(run)
    cfg, X, P, owner = design(run)
    m = cfg["model"]
    grid = m.get("grid")
    b, sigma, th = fit(X, P, grid)
    ids = sorted(set(owner))
    rows = {i: np.where(owner == i)[0] for i in ids}
    rng = np.random.default_rng(20261105)
    boots = []
    for _ in range(300):
        pick = np.concatenate([rows[ids[k]] for k in rng.choice(len(ids), len(ids))])
        boots.append(fit(X[pick], P[pick], grid, start=th)[0][:8])  # warm start at the full-sample fit
    lo, hi = np.percentile(np.array(boots), [2.5, 97.5], axis=0)
    planted = [m.get("scale", 1.0) * w for w in m["position_weights"]] + [
        m.get("carryover", 0.0), m.get("format_effect", 0.0), m.get("first_option_bias", 0.0)]
    return {"run": run.name, "noise": m.get("noise", 0.0), "grid": grid, "n_records_used": int(len(P)),
            "method": "interval-censored normal MLE" if grid else "least squares on the logit",
            "noise_sd_estimate": sigma,
            "parameters": {n: {"planted": pl, "estimate": float(e), "ci95": [float(a), float(c)]}
                           for n, pl, e, a, c in zip(NAMES, planted, b[:8], lo, hi)}}


if __name__ == "__main__":
    out = [recover(r) for r in sys.argv[1:-1]]
    Path(sys.argv[-1]).write_text(json.dumps(out, indent=1))
    for r in out:
        sd = "" if r["noise_sd_estimate"] is None else f", noise SD estimate {r['noise_sd_estimate']:.4f}"
        print(f"== {r['run']} (noise {r['noise']}, grid {r['grid']}; {r['method']}; {r['n_records_used']} records{sd})")
        for n, v in r["parameters"].items():
            inside = v["ci95"][0] - 1e-9 <= v["planted"] <= v["ci95"][1] + 1e-9
            print(f"  {n:14s} planted {v['planted']:+.3f}  estimate {v['estimate']:+.4f}  [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]  {'covers' if inside else 'MISSES'}")
