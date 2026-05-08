# -*- coding: utf-8 -*-
"""
Experiment A — Governance Completeness (P8 Theorem 8.1).

Claim under test:
    T8.1: every HALT event resolves through either the Recovery Loop or
          a signed APB; no third path exists.

Empirical translation:
    Across 10 seeds x 1000 steps with the full stack
    (ACP + IML + RAM + Recovery Loop + GovernanceLayer), each HALT
    detected by RAM is classified as:
        - recovery_resume   if Recovery Loop returns RESUME
        - apb_signed        if Recovery returns HALT/ESCALATE and
                            GovernanceLayer produces a verified APB
        - neither           otherwise (must be 0)

Output:
    results/p8_apb/exp_a/exp_a_summary.json
    results/p8_apb/exp_a/exp_a_apb_log.jsonl    (one APB per persistent HALT)
    results/p8_apb/exp_a/exp_a_table.tex
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from agent.mock_llm import MockLLM
from agent.orchestrator import build_graph, AgentState
from agent.principal import Principal, PrincipalRegistry, generate_keypair
from baselines.enforcement import enforcement_signal
from iml.trace import Trace, Event
from stack.acp_gate import ACPGate, Decision
from stack.apb import APB
from stack.apb_verifier import verify_apb
from stack.governance_layer import GovernanceLayer, threshold_policy
from stack.iml_monitor import AdmissionSnapshotP7, IMLMonitor
from stack.ram_gate import RAMGate, Authority
from stack.recovery_loop import RecoveryLoop, ResumeDecision

BURN_IN  = 50
DRIFT    = 950
TOTAL    = BURN_IN + DRIFT
SEEDS    = list(range(42, 52))   # 10 seeds
COVERAGE = 0.70
THETA    = 0.20

OUT_DIR = os.path.join(_ROOT, "results", "p8_apb", "exp_a")


# ---------------------------------------------------------------------------
# Helpers: serializable views of runtime state for E_s construction
# ---------------------------------------------------------------------------

def _A0_view(A0: AdmissionSnapshotP7) -> dict:
    """Stable, hashable representation of A_0."""
    return {
        "P0": [float(p) for p in A0.P0],
        "depth_mean": float(A0.depth_mean),
        "depth_std": float(A0.depth_std),
    }


def _trace_view(trace: Trace, t: int) -> dict:
    return {
        "trace_id": trace.trace_id,
        "length_at_halt": t,
        "tail_5_tools": [e.tool for e in trace.events[-5:]],
    }


# ---------------------------------------------------------------------------
# Single-seed run
# ---------------------------------------------------------------------------

def run_seed(seed: int, registry: PrincipalRegistry, key_store: dict[str, bytes],
             apb_log: list) -> dict:
    llm  = MockLLM(seed=seed)
    app  = build_graph(llm)
    trace = Trace(trace_id=f"exp_a_s{seed}")

    acp = ACPGate(
        rule1_threshold=500, rule3_threshold=500,
        bonus_rule1=20.0, bonus_rule3=15.0,
        admit_threshold=200.0, deny_threshold=500.0,
    )
    ram = RAMGate(rs_threshold=45.0, coverage=COVERAGE, seed=seed)
    rec_loop = RecoveryLoop(max_attempts=5, base_coverage=0.30,
                            delta_coverage=0.15, seed=seed)
    gov = GovernanceLayer(registry, key_store)
    policy = threshold_policy(deny_above=0.4, recalibrate_above=0.7)

    iml: IMLMonitor = None
    A0: AdmissionSnapshotP7 = None

    counters = {
        "halt_events": 0,
        "recovery_resume": 0,
        "apb_signed": 0,
        "apb_invalid": 0,
        "neither": 0,
    }
    apb_decisions = {"RESUME": 0, "DENY": 0, "RECALIBRATE": 0}

    for t in range(TOTAL):
        phase    = "burn_in" if t < BURN_IN else "drift"
        progress = (t - BURN_IN) / DRIFT if phase == "drift" else 0.0

        init: AgentState = {
            "step": t, "phase": phase, "progress": progress,
            "task_intent": "", "tool": None,
            "risk_score": None, "depth": None, "execution_result": None,
        }
        out = app.invoke(init)
        tool, risk_score, depth = out["tool"], out["risk_score"], out["depth"]

        trace.add(Event(
            agent="A", action="tool_call",
            tool=tool, depth=depth,
            metadata={"risk_score": risk_score},
        ))

        if t == BURN_IN - 1:
            A0  = AdmissionSnapshotP7(trace)
            iml = IMLMonitor(A0)
            continue
        if t < BURN_IN:
            continue

        D_hat = iml.compute(trace)
        acp_rec = acp.evaluate(agent_id="A", tool=tool, rs_base=risk_score * 100)

        if acp_rec.decision != Decision.ADMIT:
            continue

        ram_dec = ram.check(tool=tool, risk_score=risk_score * 100, drift_level=D_hat)
        if ram_dec.authority != Authority.HALT:
            continue

        # ── HALT detected by RAM ─────────────────────────────────────────────
        counters["halt_events"] += 1
        rec_result = rec_loop.run(
            halt_decision=ram_dec, iml_D_hat=D_hat,
            tool=tool, risk_score=risk_score * 100,
            drift_level=D_hat,
        )

        if rec_result.decision == ResumeDecision.RESUME:
            counters["recovery_resume"] += 1
            continue

        # Recovery returned HALT (stuck) or ESCALATE → governance event
        cause = (
            "persistent_drift_unresolvable"
            if rec_result.decision == ResumeDecision.HALT
            else "ram_escalate_permanent"
        )
        try:
            apb = gov.resolve_halt(
                H_id="H_alice",
                A_0=_A0_view(A0),
                D_hat=D_hat,
                trace=_trace_view(trace, t),
                cause=cause,
                policy=policy,
            )
            report = verify_apb(apb, registry, max_age_seconds=3600.0)
            if report.is_valid:
                counters["apb_signed"] += 1
                apb_decisions[apb.D_h.decision] += 1
                apb_log.append({
                    "seed": seed, "t": t,
                    "recovery_outcome": rec_result.decision.value,
                    "apb": apb.to_dict(),
                })
            else:
                counters["apb_invalid"] += 1
                counters["neither"] += 1
        except Exception as e:
            counters["neither"] += 1
            print(f"  [seed={seed} t={t}] APB construction failed: {e}")

    return {
        "seed": seed,
        "counters": counters,
        "apb_decisions": apb_decisions,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Bootstrap: one principal H_alice (Exp A focuses on the Completeness
    # path, not multi-principal authorization).
    sk_alice, pk_alice = generate_keypair()
    registry = PrincipalRegistry()
    registry.add(Principal(H_id="H_alice", public_key=pk_alice, role="auditor"))
    key_store = {"H_alice": sk_alice}

    print("=" * 64)
    print("Exp A -- Governance Completeness (T8.1)")
    print(f"  Seeds: {SEEDS}")
    print(f"  Steps per seed: {TOTAL} (burn-in={BURN_IN}, drift={DRIFT})")
    print(f"  Coverage: {COVERAGE}  theta: {THETA}")
    print("=" * 64)

    apb_log: list = []
    per_seed: list = []
    for seed in SEEDS:
        print(f"\n[seed={seed}] running...")
        r = run_seed(seed, registry, key_store, apb_log)
        per_seed.append(r)
        c = r["counters"]
        print(f"  halts={c['halt_events']:4d}  "
              f"rec_resume={c['recovery_resume']:4d}  "
              f"apb={c['apb_signed']:4d}  "
              f"neither={c['neither']}")

    # Aggregate
    agg = {k: 0 for k in per_seed[0]["counters"]}
    for r in per_seed:
        for k, v in r["counters"].items():
            agg[k] += v

    apb_decision_agg = {"RESUME": 0, "DENY": 0, "RECALIBRATE": 0}
    for r in per_seed:
        for k, v in r["apb_decisions"].items():
            apb_decision_agg[k] += v

    completeness_holds = (agg["neither"] == 0)
    coverage_pct = (
        (agg["recovery_resume"] + agg["apb_signed"]) / agg["halt_events"] * 100
        if agg["halt_events"] > 0 else 100.0
    )

    summary = {
        "experiment": "p8_exp_a_governance_completeness",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seeds": SEEDS,
        "steps_per_seed": TOTAL,
        "burn_in": BURN_IN,
        "coverage": COVERAGE,
        "theta": THETA,
        "aggregate": agg,
        "apb_decision_breakdown": apb_decision_agg,
        "per_seed": per_seed,
        "completeness_holds": completeness_holds,
        "resolution_coverage_pct": round(coverage_pct, 2),
        "T8_1_assertion": "PASSED" if completeness_holds else "FAILED",
    }

    # Write outputs
    summary_path = os.path.join(OUT_DIR, "exp_a_summary.json")
    apb_path = os.path.join(OUT_DIR, "exp_a_apb_log.jsonl")
    table_path = os.path.join(OUT_DIR, "exp_a_table.tex")

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    with open(apb_path, "w") as f:
        for entry in apb_log:
            f.write(json.dumps(entry) + "\n")
    _write_latex_table(per_seed, agg, completeness_holds, table_path)

    # ── Final report ─────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("AGGREGATE RESULTS")
    print(f"  Total HALT events:        {agg['halt_events']}")
    print(f"  Recovery RESUME:          {agg['recovery_resume']}")
    print(f"  APB-signed (governance):  {agg['apb_signed']}")
    print(f"  NEITHER (T8.1 violation): {agg['neither']}")
    print(f"  Resolution coverage:      {coverage_pct:.2f}%")
    print(f"  APB decisions: RESUME={apb_decision_agg['RESUME']}  "
          f"DENY={apb_decision_agg['DENY']}  "
          f"RECALIBRATE={apb_decision_agg['RECALIBRATE']}")
    print(f"\n  T8.1 (Governance Completeness): "
          f"{'PASSED' if completeness_holds else 'FAILED'}")
    print("=" * 64)
    print(f"\nOutputs:\n  {summary_path}\n  {apb_path}\n  {table_path}")

    return 0 if completeness_holds else 1


def _write_latex_table(per_seed, agg, holds, path):
    rows = []
    for r in per_seed:
        c = r["counters"]
        rows.append(
            f"  {r['seed']} & {c['halt_events']} & {c['recovery_resume']} "
            f"& {c['apb_signed']} & {c['neither']} \\\\"
        )
    rows.append("  \\midrule")
    rows.append(
        f"  \\textbf{{Total}} & \\textbf{{{agg['halt_events']}}} & "
        f"\\textbf{{{agg['recovery_resume']}}} & "
        f"\\textbf{{{agg['apb_signed']}}} & "
        f"\\textbf{{{agg['neither']}}} \\\\"
    )
    body = "\n".join(rows)
    status = "PASSED" if holds else "FAILED"
    tex = f"""% Auto-generated by exp_a_governance_completeness.py
\\begin{{table}}[t]
\\centering
\\caption{{Governance Completeness (T8.1): every HALT resolves through Recovery
or a signed APB. NEITHER count is 0 across all seeds (assertion {status}).}}
\\label{{tab:exp_a_completeness}}
\\begin{{tabular}}{{rrrrr}}
\\toprule
Seed & HALTs & Recovery RESUME & APB-signed & NEITHER \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(tex)


if __name__ == "__main__":
    sys.exit(main())
