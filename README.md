# Closing the Execution Gap — Paper 7

**Agent Governance Series · Paper 7 of 8**

First empirical validation of the complete ACP+IML+RAM+RecoveryLoop governance stack on real LangGraph agents.

[![Zenodo](https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.19929771-blue)](https://doi.org/10.5281/zenodo.19929771)
[![arXiv](https://img.shields.io/badge/arXiv-pending-orange)](https://arxiv.org)
[![Series](https://img.shields.io/badge/Series-P0--P7-lightgrey)](https://agentcontrolprotocol.xyz/research.html)

---

## What this paper does

**Central finding:** Compliant Drift is real, measurable, and closeable.

**Compliant Drift** — g(τ)=0 throughout execution (every decision is approved by the governance stack) while D̂ grows monotonically toward the threshold θ=0.20. The agent escalates its risk profile without ever being denied. This structural gap between what governance validates and what agents actually do is the **Execution Gap**.

This paper provides:
1. First experimental proof that the Execution Gap exists and is measurable
2. Closure of the gap via the integrated ACP+IML+RAM+RecoveryLoop stack in LangGraph
3. Open-source implementation
4. 3 formal refinements to the theoretical framework

---

## Experiments

| # | Name | Setup | Key result |
|---|------|-------|-----------|
| 1 | Compliant Drift (MockLLM) | 500 steps, 6 seeds | T* ∈ [259, 403]; g(τ)=0 throughout |
| 1b | Compliant Drift (Real LLMs) | 150 steps, 2 models | Mistral T*=64 / DeepSeek T*=65; ΔT*=1 |
| 2 | Partial Observability + RAM | 10k Monte Carlo × 6 coverages | IER=0.000 at all coverage levels |
| 3 | Multi-agent coordination (ACP) | N ∈ {2, 4, 8, 16} | CW_appr=2N; deviation=0 |
| 4 | Full stack integration | 2000 steps | D̂ convergence, liveness, zero deadlock |

**Exp 1b — Multi-model results:**

| Model | Family | Size | D_final | T* | Σg(τ) |
|-------|--------|------|---------|-----|-------|
| mistral-small3.1 | Mistral | 15B | 0.367 | 64 | 0 |
| deepseek-r1:8b | DeepSeek | 8B | 0.362 | 65 | 0 |

T* differs by 1 step, D_final by 0.005 → drift is driven by the protocol, not the model family.

---

## Metrics

| Metric | Definition |
|--------|-----------|
| **D̂** | IML composite: 0.40·D_t + 0.35·D_c + 0.25·D_l (EMA α=0.15, θ=0.20) |
| **T\*** | Detection step — first τ where D̂(τ) ≥ θ |
| **g(τ)** | ACP gate signal — 0=approved, 1=denied |
| **IER** | Invalid Execution Rate |
| **SHR** | Safe Halt Rate |
| **OCR** | Opportunity Cost Rate |
| **CW_appr** | Coordination Window (approved steps before first denial) |

---

## Repository structure

```
stack/                  ACP gate, IML monitor, RAM gate, Recovery Loop
agent/
  live_llm.py           Multi-model LiveLLM (think=False auto-detect for reasoning models)
  mock_llm.py           Deterministic MockLLM for controlled experiments
experiments/
  exp1_drift.py         Compliant Drift — MockLLM, 6 seeds
  exp1b_multimodel.py   Compliant Drift — real LLMs (--model, --temperature)
  exp2_ram.py           Partial Observability + RAM, 10k Monte Carlo
  exp3_multiagent.py    Multi-agent ACP coordination
  exp4_full_stack.py    Full integrated stack
results/
  exp1/                 MockLLM seed results
  exp1b/                Real LLM results (Mistral, DeepSeek-R1)
plots/
  figures.py            Generates fig1–fig5 (run: python plots/figures.py)
paper/
  main.tex              LaTeX source (21 pp, 0 errors)
  references.bib        All series citations
  fig1_drift.pdf        Compliant Drift detection
  fig2_ram.pdf          RAM Monte Carlo results
  fig3_multiagent.pdf   ACP coordination
  fig4_fullstack.pdf    Full stack convergence
  fig5_live_llm.pdf     Real LLM Exp 1b (Mistral + DeepSeek + MockLLM baseline)
```

---

## Quick start

```bash
pip install -r requirements.txt

# Exp 1 — Compliant Drift with MockLLM (deterministic, 6 seeds)
python experiments/exp1_drift.py

# Exp 1b — Real LLMs (requires Ollama + model pulled)
ollama pull mistral-small3.1:latest
python experiments/exp1b_multimodel.py --model mistral-small3.1:latest --temperature 0.4

# Exp 2 — RAM under partial observability
python experiments/exp2_ram.py

# Exp 3 — Multi-agent coordination
python experiments/exp3_multiagent.py

# Exp 4 — Full integrated stack
python experiments/exp4_full_stack.py

# Generate all figures
python plots/figures.py
```

---

## Pending (next session — i9 with GPU)

- Exp 1b scaled: 3 models × 3 runs × 500 steps → mean ± std table
- Temperature sweep: Mistral × T ∈ {0.2, 0.4, 0.6, 0.8} × 500 steps
- Models: mistral-small3.1, deepseek-r1:8b, gemma4 (GPU required)

Scripts ready. See `experiments/exp1b_multimodel.py`.

---

## Series context

| Paper | Construct | Guarantee |
|-------|-----------|----------|
| P0 | Atomic boundary | No partial execution |
| P1 (ACP) | PatternKey | CW_appr = 2N |
| P2 (IML) | D̂ composite | Finite detection delay |
| P3/4 | Allocation + fairness | Irreducibility |
| P5 (RAM) | A(α) | IER = 0 |
| P6 (OpRAM) | Recovery Loop | Conditional liveness |
| **P7 (this)** | **Full stack empirical** | **Gap closure** |

Full series: [agentcontrolprotocol.xyz/research.html](https://agentcontrolprotocol.xyz/research.html)

---

## Citation

```bibtex
@misc{fernandez2026applied,
  author       = {Fernandez, Marcelo},
  title        = {Closing the Execution Gap in {LLM} Agent Systems:
                  Empirical Evidence for Compliant Drift, Partial Observability,
                  and Integrated Runtime Governance},
  year         = {2026},
  howpublished = {arXiv:TBD},
  note         = {Paper~7 of the Agent Governance Series.
                  \url{https://doi.org/10.5281/zenodo.19929771}}
}
```

Zenodo: [doi.org/10.5281/zenodo.19929771](https://doi.org/10.5281/zenodo.19929771)
