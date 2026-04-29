# Agent Governance Applied — Paper 7

Empirical validation of the ACP+IML+RAM+RecoveryLoop governance stack
on real LangGraph agents.

**Series:** Agent Governance Series (P0–P7)
**Paper 7 companion repos:** iml-benchmark · acp-framework-en · reconstructive-authority-model

## Structure

```
stack/          ACP gate, IML monitor, RAM gate, Recovery Loop
agent/          LangGraph orchestrator, MockLLM, worker nodes
experiments/    exp1–exp4 runners
baselines/      OPA baseline, prompt guard heuristic
plots/          Figure generators
paper/          LaTeX source
```

## Experiments

| # | Name | Steps | Key metric |
|---|---|---|---|
| 1 | Drift Detection Under Zero Enforcement | 500 | D̂ vs g(τ)=0 |
| 2 | Partial Observability + RAM | 1000 | IER / SHR / OCR |
| 3 | N-Agent Coordination | variable | CW_appr vs 2N |
| 4 | Full Stack (ACP+IML+RAM+RecoveryLoop) | 2000 | convergence, liveness |

## Quick start

```bash
pip install -r requirements.txt
python experiments/exp1_drift.py
```

## Metrics

- **D̂** = 0.40·D_t + 0.35·D_c + 0.25·D_l (IML composite)
- **T\*** detection step (first t where D̂ ≥ θ)
- **IER** Invalid Execution Rate
- **SHR** Safe Halt Rate
- **OCR** Opportunity Cost Rate
- **CW_appr** Coordination Window (approved steps before denial)
