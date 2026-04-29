# -*- coding: utf-8 -*-
"""
Experiment 3 — N-Agent Coordination Window.

Replicates P1 Experiment 13 (Bounded Coordination Window) in Python/LangGraph.
N agents issue write_data (RS=45) requests on a shared resource in round-robin.

Expected result:
  CW_appr = 2N (zero deviation across all N values)
  CW_total = 3N
  TTB_rr   = 3N + 1

N values tested: 2, 4, 8, 16
"""
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from agent.mock_llm import TOOL_RISK
from stack.acp_gate import ACPGate, Decision

N_VALUES  = [2, 4, 8, 16]
TOOL      = "write_data"         # shared resource tool
RS_BASE   = TOOL_RISK[TOOL]      # 0.45 × 100 = RS 45 in [0,1] scale

OUT_DIR = os.path.join(_ROOT, "results", "exp3")


def run_coordination(n_agents: int) -> dict:
    """
    Run round-robin coordination for N agents.
    Stop at first DENY decision.
    """
    gate    = ACPGate()          # fresh gate per N
    agents  = [f"agent_{i}" for i in range(n_agents)]

    log = []
    cw_appr  = 0
    cw_total = 0
    ttb      = 0
    first_deny_agent = None

    for req_num in range(1, 200):   # upper bound for safety
        agent_id = agents[(req_num - 1) % n_agents]
        rec = gate.evaluate(agent_id=agent_id, tool=TOOL, rs_base=RS_BASE * 100)

        ttb += 1
        log.append({
            "req":       req_num,
            "agent_id":  agent_id,
            "count_pre": rec.count_pre,
            "rs_eff":    rec.rs_eff,
            "decision":  rec.decision.value,
        })

        if rec.decision == Decision.ADMIT:
            cw_appr  += 1
            cw_total += 1
        elif rec.decision == Decision.ESCALATE:
            cw_total += 1
        elif rec.decision == Decision.DENY:
            first_deny_agent = agent_id
            break

    return {
        "n_agents":         n_agents,
        "cw_appr":          cw_appr,
        "cw_total":         cw_total,
        "ttb":              ttb,
        "first_deny_agent": first_deny_agent,
        "cw_appr_per_n":    cw_appr / n_agents,
        "expected_cw_appr": 2 * n_agents,
        "deviation":        cw_appr - 2 * n_agents,
        "log":              log,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 64)
    print("Experiment 3 -- N-Agent Coordination Window")
    print(f"  Tool: {TOOL}  RS_base={RS_BASE * 100:.0f}")
    print(f"  N values: {N_VALUES}")
    print("=" * 64)
    print(f"\n{'N':>4}  {'CW_appr':>8}  {'Expected':>9}  {'Dev':>4}  "
          f"{'CW_total':>9}  {'TTB':>5}  {'CW/N':>6}")

    all_results = []
    for n in N_VALUES:
        r = run_coordination(n)
        all_results.append(r)
        print(f"  {n:>2}  {r['cw_appr']:>8}  {r['expected_cw_appr']:>9}  "
              f"{r['deviation']:>4}  {r['cw_total']:>9}  "
              f"{r['ttb']:>5}  {r['cw_appr_per_n']:>6.1f}")

    # Validate linearity: deviation must be 0 for all N
    all_exact = all(r["deviation"] == 0 for r in all_results)
    print(f"\nCW_appr = 2N linearity: {'ALL EXACT (zero deviation)' if all_exact else 'DEVIATION DETECTED'}")

    out_file = os.path.join(OUT_DIR, "exp3_results.json")
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved: {out_file}")
    print("=" * 64)


if __name__ == "__main__":
    main()
