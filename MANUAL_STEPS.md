# Running HARPO stage-by-stage (without `reproduce.sh`)

`reproduce.sh` is just a wrapper: it runs the same `scripts/*.py` calls in a
fixed order, skips a stage once `logs/<stage>.done` exists, and (with two
GPUs) runs two independent chains in parallel. Everything below is the
single-GPU chain, unrolled into plain commands, for running by hand in your
`harpo` conda env.

If you'd rather not track `.done` files yourself: just run `reproduce.sh`
with `GPU0=GPU1` set to the same device — that already gets you this exact
sequential order, with resuming for free. Use the steps below only if you
want to inspect/tweak individual stages.

`reproduce.sh` itself does not create or manage any environment — it just
runs whatever `python`/`$PY` is on `PATH`. Create one yourself first:

```bash
conda create -n harpo python=3.10 -y
conda activate harpo
pip install -r requirements.txt
```

## 0. Setup

```bash
OUT=./harpo_out
DATA=$OUT/data/redial_data
RAW=$OUT/redial_raw
CK=$OUT/checkpoints
RES=$OUT/results
AG=$RES/agents
PROF=$RES/bridge/profiles.json
M05=Qwen/Qwen2.5-0.5B-Instruct
M7=Qwen/Qwen2.5-7B-Instruct
GPU=0

mkdir -p "$CK" "$RES/bridge" "$AG" "$OUT/logs"
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$GPU

# Unpack the exact data the published numbers came from
mkdir -p "$OUT/data" && tar -xzf data/redial_data.tar.gz -C "$OUT/data"
tar -xzf data/redial_raw.tar.gz -C "$OUT"
```

Shared flags used throughout:

```bash
S5=(--val-fraction 0.05 --charm-fraction 0.0)
BACKBONE=(--train-size 0 --test-size 0 --catalog-size 100000 --seq-len 256
           --batch-size 16 --grad-accum 1 --catalog-refresh 250)
CK7=$CK/7b/checkpoints/sft_final
READ=$RES/readings_0.json,$RES/readings_1.json
```

## 1. CHARM stage 1a — a 0.5B retriever, then CHARM warm-started on it

The 0.5B retriever never sees the conversations CHARM stage 1 and 2 later
train/eval on (`--charm-fraction 0.25` reserves them), so stage 1a can warm
up the CHARM adapter without contaminating later splits. Its checkpoint's
catalogue also produces the BRIDGE item profiles used everywhere below.

```bash
python scripts/run_experiment.py --data "$DATA" --model "$M05" \
    "${BACKBONE[@]}" --epochs 6 --val-fraction 0.05 --charm-fraction 0.25 \
    --output "$CK/05b" --results-dir "$RES/05b"

python scripts/train_charm_ce.py --data "$DATA" \
    --backbone "$CK/05b/checkpoints/sft_final" --base-model "$M7" --dtype bfloat16 \
    --val-fraction 0.05 --charm-fraction 0.25 --top-k 50 --group 16 --batch-groups 8 \
    --eval-top-k 50 --eval-every 1000 --score-dialogues 32 --no-test \
    --results-dir "$RES/charm_stage1a"

python scripts/bridge_profiles.py --model "$M7" \
    --catalog "$CK/05b/catalog.json" --out "$PROF"
```

## 2. Retriever — the 7B two-tower backbone, cold-start, and the two base agents

```bash
python scripts/run_experiment.py --data "$DATA" --model "$M7" \
    "${BACKBONE[@]}" --epochs 5 "${S5[@]}" --output "$CK/7b" --results-dir "$RES/7b"

python scripts/coldstart_probe.py --checkpoint "$CK7" --data "$DATA" \
    "${S5[@]}" --profiles "$PROF" --top 200 --out "$AG/coldstart_probe.json"

for split in test val; do
  python scripts/rerank_eval.py --checkpoint "$CK7" --data "$DATA" \
      --split $split --top-k 200 "${S5[@]}" --coldstart "$AG/coldstart_probe.json" \
      --profiles "$PROF" --results-dir "$AG"
done
```

`rerank_eval.py` writes `$AG/rerank_top200_raw.pt` (test) and
`$AG/rerank_top200_val_raw.pt` (val) — the shared shortlists every later
agent (CHARM, STAR, MAVEN) scores.

## 3. CHARM stage 1 (relevance), continued on the 7B retriever's shortlists

```bash
python scripts/train_charm_ce.py --data "$DATA" --backbone "$CK7" \
    --base-model "$M7" --dtype bfloat16 "${S5[@]}" --top-k 200 --eval-top-k 100 \
    --group 16 --batch-groups 8 --max-train-cases 20000 --eval-every 1250 \
    --score-dialogues 32 --init-adapter "$RES/charm_stage1a/charm_ce_adapter" \
    --no-test --results-dir "$RES/charm_stage1"
```

## 4. Preference readings + cross-fitted (2-fold) retrievers

Inputs for CHARM stage 2: an LLM's reading of what the seeker wants
(`pref_summaries.py`) and, for each training conversation, a shortlist from a
retriever that never trained on it (`run_experiment.py --cv-fold`).

```bash
for k in 0 1; do
  python scripts/pref_summaries.py --model "$M7" --data "$DATA" \
      --shard $k/2 --batch 64 --out "$RES/readings_$k.json"

  python scripts/run_experiment.py --data "$DATA" --model "$M7" \
      "${BACKBONE[@]}" --epochs 5 "${S5[@]}" --cv-folds 2 --cv-fold $k \
      --no-epoch-eval --output "$CK/fold_$k" --results-dir "$RES/fold_$k"
done
```

## 5. CHARM stage 2 — relevance + satisfaction + engagement, with the dialogue gate

```bash
python scripts/crossfit_shortlists.py \
    --folds "$CK/fold_0/checkpoints/sft_final,$CK/fold_1/checkpoints/sft_final" \
    --deployed "$CK7" --coldstart "$AG/coldstart_probe.json" --data "$DATA" \
    "${S5[@]}" --top-k 200 --out "$RES/shortlists_cv.pt"

python scripts/redial_labels.py --raw-dir "$RAW" --out "$RES/labels.json"

python scripts/train_charm_ce.py --data "$DATA" --backbone "$CK7" \
    --base-model "$M7" --dtype bfloat16 "${S5[@]}" --top-k 200 --eval-top-k 100 \
    --group 16 --batch-groups 4 --epochs 1 --eval-every 2347 \
    --score-dialogues 8 --max-length 512 --shortlists "$RES/shortlists_cv.pt" \
    --heads 3 --labels "$RES/labels.json" --summaries "$READ" --profiles "$PROF" \
    --denoise --ema 0.999 --head-lr 1e-4 --gate-lr 1e-3 --patience 2 \
    --init-adapter "$RES/charm_stage1/charm_ce_adapter" --no-test \
    --results-dir "$RES/charm_stage2"
```

(`--score-dialogues` here is `CHARM_VAL_BATCH` in `reproduce.sh`, default 8 —
lower it if you're short on GPU memory.)

## 6. Score both CHARM stages and STAR on val + test

```bash
for split in val test; do
  raw=$AG/rerank_top200_raw.pt
  [ "$split" = val ] && raw=$AG/rerank_top200_val_raw.pt

  # CHARM stage 1
  python scripts/score_shortlist_ce.py \
      --adapter "$RES/charm_stage1/charm_ce_adapter" --base-model "$M7" --dtype bfloat16 \
      --data "$DATA" --raw "$raw" --split $split "${S5[@]}" --pair-budget 800 \
      --out "$AG/charm1_$split.pt"

  # CHARM stage 2 (needs readings + BRIDGE profiles too)
  python scripts/score_shortlist_ce.py \
      --adapter "$RES/charm_stage2/charm_ce_adapter" --base-model "$M7" --dtype bfloat16 \
      --data "$DATA" --raw "$raw" --split $split "${S5[@]}" --pair-budget 800 \
      --summaries "$READ" --profiles "$PROF" --out "$AG/charm2_$split.pt"

  # STAR
  python scripts/star_search.py --lm "$M7" \
      --charm-adapter "$RES/charm_stage2/charm_ce_adapter" --base-model "$M7" --raw "$raw" \
      --summaries "$READ" --profiles "$PROF" --split $split "${S5[@]}" --data "$DATA" \
      --top-m 30 --beam 3 --branch 2 --depth 3 --dialogues-per-batch 16 \
      --out "$AG/star_$split.pt"
done
```

(`--pair-budget` = `SCORE_PAIRS` in `reproduce.sh` (800); `--dialogues-per-batch`
= `STAR_BATCH` (16) — lower either if you're short on GPU memory.)

## 7. MAVEN — consensus weights by cross-validation on val, test once

```bash
M=(--test-raw "$AG/rerank_top200_raw.pt" --val-raw "$AG/rerank_top200_val_raw.pt"
   --test-ce "$AG/charm2_test.pt" --val-ce "$AG/charm2_val.pt"
   --agent star="$AG/star_test.pt,$AG/star_val.pt" --diversity-profiles "$PROF")

# Headline result: every agent (both CHARM stages + STAR)
python scripts/maven_cv.py "${M[@]}" \
    --agent charm_stage1="$AG/charm1_test.pt,$AG/charm1_val.pt" --out "$RES/maven.json"

# Ablation: drop the CHARM stage-1 agent
python scripts/maven_cv.py "${M[@]}" --out "$RES/maven_pure.json"
```

`$RES/maven.json` is the reported result:
R@1 8.80 / R@10 30.45 / R@50 50.09 / NDCG@10 18.37 / MRR@10 14.65.

## Notes

- Every step reads from the previous ones' output paths above — run them in
  this order.
- GPU non-determinism and STAR's sampled readings mean a rerun lands close
  to, not identical with, the published numbers.
- Data is sha256-checked in `reproduce.sh`'s `check_data()`; if you unpack it
  yourself as above, that check is skipped — worth diffing against the
  hashes in `reproduce.sh` if your numbers look off.
- Run `python -m pytest tests` any time to sanity-check the modules in
  isolation, without any of the above.
