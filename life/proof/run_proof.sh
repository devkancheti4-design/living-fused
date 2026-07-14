#!/bin/bash
# Reproduce the Required-Fusion Proof (PROOF.md).
# Core pillars: deterministic, stdlib-only, ~2 minutes.
#   bash run_proof.sh            # pillars I-IV (+ supporting exp5)
#   bash run_proof.sh --full     # + supporting experiments (exp1, exp7, exp8, exp9)
#   bash run_proof.sh --llm      # + pillar V on real LLMs (needs ollama running)
set -e
cd "$(dirname "$0")"

banner() { echo; echo "================================================================"; echo "$1"; echo "================================================================"; }

banner "PILLAR I — exact recall is optimizer-inaccessible (exp3)"
python3 experiments/exp3_exactness_untrainable.py

banner "PILLAR II — fixed-size distributed memory: cliff at K=d (exp4)"
python3 experiments/exp4_fastweight_capacity.py

banner "PILLAR III — softmax cannot abstain; table can (exp5b)"
python3 experiments/exp5b_confidence_on_garbage.py 2>&1 | tail -11

banner "PILLAR III+ — the priced gradient-side fix: ensemble disagreement (exp6)"
python3 experiments/exp6_trajectory_uncertainty.py 2>&1 | tail -14

banner "PILLAR IV — the Routing Law: agreement vs conflict regimes (exp2 + exp2b)"
python3 experiments/exp2_memory_conservation.py
python3 experiments/exp2b_conflict_control.py 2>&1 | tail -9

banner "SUPPORTING — curiosity coordinates: frozen models can't tell frontier from wall (exp5)"
python3 experiments/exp5_learnable_novelty.py

if [[ "$1" == "--full" || "$2" == "--full" ]]; then
  banner "SUPPORTING — paraphrase-resistant novelty gate (exp1)"
  python3 experiments/exp1_novelty_saturation.py
  banner "SUPPORTING — MDL as validation-free selector (exp7)"
  python3 experiments/exp7_mdl_selection.py
  banner "SUPPORTING — credit resolution law (exp8, several minutes)"
  python3 experiments/exp8_credit_assignment.py
  banner "SUPPORTING — Goodhart laws (exp9)"
  python3 experiments/exp9_goodhart_laws.py
fi

if [[ "$1" == "--llm" || "$2" == "--llm" ]]; then
  banner "PILLAR V — behavioral confirmation on real LLMs (exp10/exp10b, ~30-60 min)"
  python3 experiments/exp10_k_needle_law.py llama3.2:3b
  python3 experiments/exp10b_k_needle_load.py llama3.2:3b both
  python3 experiments/exp10b_k_needle_load.py llama3:8b both
fi

banner "PROOF REPRODUCED — see PROOF.md for the assembled argument"
