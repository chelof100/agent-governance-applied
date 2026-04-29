# -*- coding: utf-8 -*-
"""
Generate all figures for Paper 7.

Outputs (PDF, publication quality):
  plots/fig1_drift.pdf       -- Exp 1: D_hat vs enforcement signal (6 seeds)
  plots/fig2_coordination.pdf -- Exp 3: ACP coordination timeline (N=8)
  plots/fig3_ier_coverage.pdf -- Exp 2: IER vs coverage (3 strategies)
  plots/fig4_full_stack.pdf  -- Exp 4: D_hat + HALT events over time
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES     = os.path.join(_ROOT, "results")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        10,
    "axes.labelsize":   10,
    "axes.titlesize":   10,
    "legend.fontsize":  9,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

# B&W-friendly palette: distinguishable by both color and line style/marker
# Primary: solid black/dark gray; secondary: dashed; tertiary: dotted
COLORS = {
    "dhat":   "#1a1a1a",   # near-black  (solid line)
    "enf":    "#555555",   # dark gray   (fill)
    "ram":    "#1a1a1a",   # black       (solid, circle marker)
    "att":    "#888888",   # mid gray    (dashed, triangle marker)
    "always": "#555555",   # dark gray   (solid, square marker)
    "halt":   "#1a1a1a",   # black ticks
    "resume": "#666666",   # gray ticks
    "admit":  "#333333",   # dark
    "deny":   "#1a1a1a",   # black X
    "escal":  "#888888",   # gray
}

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Exp 1: D_hat vs Enforcement Signal (seed=42 main + 5 seeds shading)
# ─────────────────────────────────────────────────────────────────────────────
def fig1_drift():
    with open(os.path.join(RES, "exp1", "exp1_results.json")) as f:
        data = json.load(f)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 4.5),
                                   gridspec_kw={"height_ratios": [3, 1]},
                                   sharex=True)

    all_t, all_dhat = [], []
    for entry in data:
        recs = entry["records"]
        ts   = [r["t"] - 50 for r in recs]   # shift so drift starts at 0
        dhat = [r["D_hat"] for r in recs]
        all_t.append(ts)
        all_dhat.append(dhat)

    # Shading: min-max band across seeds
    n_steps = min(len(d) for d in all_dhat)
    t_common = all_t[0][:n_steps]
    dhat_arr = np.array([d[:n_steps] for d in all_dhat])
    d_min  = dhat_arr.min(axis=0)
    d_max  = dhat_arr.max(axis=0)
    d_mean = dhat_arr.mean(axis=0)

    ax1.fill_between(t_common, d_min, d_max, alpha=0.18, color=COLORS["dhat"], label="Range (seeds 1-5)")
    ax1.plot(t_common, dhat_arr[0], color=COLORS["dhat"], lw=1.5, label=r"$\hat{D}_t$ (seed 42)")
    ax1.axhline(0.20, color="gray", lw=0.8, ls="--", label=r"$\theta = 0.20$")

    # Mark T* for seed 42
    t_star = next((i for i, v in enumerate(dhat_arr[0]) if v >= 0.20), None)
    if t_star is not None:
        ax1.axvline(t_star, color="#666666", lw=0.8, ls=":")
        ax1.annotate(f"$T^*={t_star}$", xy=(t_star, 0.21), xytext=(t_star + 15, 0.23),
                     fontsize=8, color="#666666",
                     arrowprops=dict(arrowstyle="->", color="#666666", lw=0.8))

    ax1.set_ylabel(r"$\hat{D}_t$ (IML composite)")
    ax1.set_ylim(0, 0.42)
    ax1.legend(loc="upper left", framealpha=0.8)
    ax1.set_title("Experiment 1: IML Drift Detection Under Zero Enforcement")

    # Enforcement signal (bottom panel)
    enf = [r["enforcement"] for r in data[0]["records"]]
    ts_enf = [r["t"] - 50 for r in data[0]["records"]]
    ax2.fill_between(ts_enf, enf, alpha=0.7, color=COLORS["enf"],
                     label=r"$g(\tau)$ (enforcement signal)")
    ax2.set_ylim(-0.1, 1.5)
    ax2.set_yticks([0, 1])
    ax2.set_xlabel("Step (relative to burn-in end)")
    ax2.set_ylabel(r"$g(\tau)$")
    ax2.legend(loc="upper right", framealpha=0.8)

    fig.tight_layout(h_pad=0.5)
    out = os.path.join(OUT_DIR, "fig1_drift.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Exp 3: ACP Coordination Timeline (N=8)
# ─────────────────────────────────────────────────────────────────────────────
def fig2_coordination():
    with open(os.path.join(RES, "exp3", "exp3_results.json")) as f:
        data = json.load(f)

    # Find N=8 entry
    entry = next(e for e in data if e["n_agents"] == 8)
    log   = entry["log"]
    N     = entry["n_agents"]

    agents  = sorted(set(r["agent_id"] for r in log))
    a_idx   = {a: i for i, a in enumerate(agents)}
    color_map = {"ADMIT": COLORS["admit"], "ESCALATE": COLORS["escal"], "DENY": COLORS["halt"]}

    fig, ax = plt.subplots(figsize=(6.5, 3.5))

    for req_num, r in enumerate(log):
        ai   = a_idx[r["agent_id"]]
        dec  = r["decision"]
        col  = color_map.get(dec, "gray")
        ax.scatter(req_num + 1, ai, color=col, s=60, zorder=3,
                   marker="o" if dec == "ADMIT" else ("^" if dec == "ESCALATE" else "x"),
                   linewidths=1.5 if dec == "DENY" else 0.5)

    ax.set_yticks(range(N))
    ax.set_yticklabels([f"A{i}" for i in range(N)], fontsize=8)
    ax.set_xlabel("Request number")
    ax.set_ylabel("Agent")
    ax.set_title(f"Experiment 3: ACP Admission Timeline ($N={N}$, write\\_data, RS=45)")

    admit_patch  = mpatches.Patch(color=COLORS["admit"],  label="ADMIT")
    escal_patch  = mpatches.Patch(color=COLORS["escal"],  label="ESCALATE")
    deny_patch   = mpatches.Patch(color=COLORS["halt"],   label="DENY")
    ax.legend(handles=[admit_patch, escal_patch, deny_patch],
              loc="upper right", framealpha=0.85)

    # Annotate CW_appr
    cw = entry["cw_appr"]
    ax.axvline(cw + 0.5, color="gray", lw=0.8, ls="--")
    ax.text(cw + 1, N - 0.5, f"$CW_{{appr}}={cw}=2N$", fontsize=8, color="gray")

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig2_coordination.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Exp 2: IER vs Coverage (RAM vs Attestation vs Always-Execute)
# ─────────────────────────────────────────────────────────────────────────────
def fig3_ier_coverage():
    with open(os.path.join(RES, "exp2", "exp2_sweep.json")) as f:
        data = json.load(f)

    covs      = [e["coverage"]         for e in data]
    ram_ier   = [e["ram"]["IER"]       for e in data]
    att_ier   = [e["att"]["IER"]       for e in data]
    always_ier= [e["always"]["IER"]    for e in data]
    ram_ocr   = [e["ram"]["OCR"]       for e in data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.0))

    # Left: IER comparison — B&W distinguishable by line style + marker
    ax1.plot(covs, always_ier, color=COLORS["always"], lw=1.5, ls=(0,(3,1,1,1)),
             marker="s", ms=5, label="Always-execute")
    ax1.plot(covs, att_ier,    color=COLORS["att"],    lw=1.5, ls="--",
             marker="^", ms=5, label="Attestation")
    ax1.plot(covs, ram_ier,    color=COLORS["ram"],    lw=2.0, ls="-",
             marker="o", ms=6, label="RAM (this work)")
    ax1.axhline(0, color="gray", lw=0.5, ls=":")
    ax1.set_xlabel("Observation Coverage")
    ax1.set_ylabel("IER (Invalid Execution Rate)")
    ax1.set_title("(a) IER vs Coverage")
    ax1.set_ylim(-0.01, 0.25)
    ax1.legend(loc="upper right", framealpha=0.85)

    # Right: OCR (opportunity cost of RAM)
    ax2.plot(covs, ram_ocr, color=COLORS["ram"], lw=2.0, ls="-", marker="o", ms=5)
    ax2.set_xlabel("Observation Coverage")
    ax2.set_ylabel("OCR (Opportunity Cost Rate)")
    ax2.set_title("(b) RAM Opportunity Cost")
    ax2.set_ylim(-0.05, 1.1)
    ax2.axhline(1.0, color="gray", lw=0.5, ls=":")

    fig.suptitle("Experiment 2: RAM under Partial Observability (10k samples/point)", fontsize=9)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig3_ier_coverage.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Exp 4: Full Stack Timeline
# ─────────────────────────────────────────────────────────────────────────────
def fig4_full_stack():
    with open(os.path.join(RES, "exp4", "exp4_results.json")) as f:
        data = json.load(f)

    records = data["records"]
    ts      = [r["t"] - 50 for r in records]   # shift to start at 0
    dhat    = [r["D_hat"]   for r in records]
    enf     = [r["enforcement"] for r in records]

    halt_ts   = [r["t"] - 50 for r in records if r.get("ram") == "HALT" or
                 (r.get("ram") == "SKIP" and r.get("recovery") not in ("SKIP", None))]
    resume_ts = [r["t"] - 50 for r in records if r.get("recovery") == "RESUME"]
    halt_ts2  = [r["t"] - 50 for r in records if r.get("ram") == "HALT"]
    escal_ts  = [r["t"] - 50 for r in records if r.get("recovery") == "ESCALATE"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 4.8),
                                   gridspec_kw={"height_ratios": [2.5, 1.5]},
                                   sharex=True)

    # D_hat over time
    ax1.plot(ts, dhat, color=COLORS["dhat"], lw=1.2, alpha=0.9, label=r"$\hat{D}_t$")
    ax1.axhline(0.20, color="gray", lw=0.8, ls="--", label=r"$\theta = 0.20$")

    # Shade D_hat = 0 to 0.20 in green, above in orange
    ax1.fill_between(ts, dhat, 0.20, where=[v > 0.20 for v in dhat],
                     alpha=0.10, color="orange", interpolate=True)

    ax1.set_ylabel(r"$\hat{D}_t$ (IML composite)")
    ax1.set_ylim(0, 0.42)
    ax1.legend(loc="upper left", framealpha=0.85)
    ax1.set_title("Experiment 4: Full Stack (ACP + IML + RAM + Recovery Loop)")

    # HALT / RESUME events
    if halt_ts2:
        ax2.scatter(halt_ts2, [1.0] * len(halt_ts2), color=COLORS["halt"],
                    s=12, marker="|", lw=1.2, alpha=0.6, label=f"HALT ({len(halt_ts2)})")
    if resume_ts:
        ax2.scatter(resume_ts, [0.5] * len(resume_ts), color=COLORS["resume"],
                    s=12, marker="|", lw=1.2, alpha=0.6, label=f"RESUME ({len(resume_ts)})")
    if escal_ts:
        ax2.scatter(escal_ts, [0.0] * len(escal_ts), color=COLORS["escal"],
                    s=12, marker="|", lw=1.2, alpha=0.6, label=f"ESCALATE ({len(escal_ts)})")

    ax2.set_yticks([0.0, 0.5, 1.0])
    ax2.set_yticklabels(["ESC", "RES", "HLT"], fontsize=8)
    ax2.set_xlabel("Step (relative to burn-in end)")
    ax2.set_ylabel("Event type")
    ax2.legend(loc="upper right", fontsize=8, framealpha=0.85)
    ax2.set_ylim(-0.3, 1.4)

    fig.tight_layout(h_pad=0.4)
    out = os.path.join(OUT_DIR, "fig4_full_stack.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    print("Generating figures...")
    fig1_drift()
    fig2_coordination()
    fig3_ier_coverage()
    fig4_full_stack()
    print("Done.")
