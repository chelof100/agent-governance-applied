# -*- coding: utf-8 -*-
"""
APB Verifier — signature verification, tamper detection, attribution.

Validates:
  - sigma_h against pk_i  (signature integrity)
  - hash(E_s || D_h) match (no post-construction tampering)
  - H_i in P at signing time, not revoked   (attribution)
  - t_e within acceptable window           (replay defense)

Used by: Exp B (APB Integrity) — runs 4 attack vectors per APB.

Sprint 1 scaffolding — implementation pending.
"""
raise NotImplementedError("stack/apb_verifier.py — implement in Sprint 1")
