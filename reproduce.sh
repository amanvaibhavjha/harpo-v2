#!/bin/bash
# Reproduce HARPO on ReDial: standard test (4,975 cases), full 6,630-movie
# catalogue, all four modules, MAVEN weights chosen by cross-validation on
# validation:  R@1 8.80 / R@10 30.45 / R@50 50.09 / NDCG@10 18.37 / MRR@10 14.65
#
#   OUT=/path/to/out bash reproduce.sh
#   (detached: setsid nohup bash reproduce.sh > reproduce.log 2>&1 < /dev/null &)
#
# DATA  converted ReDial: sft_data.json, test_sft.json, movie_list.json (sha256-checked)
#                                             [data/redial_data.tar.gz, unpacked into OUT/data]
# RAW   the original ReDial release (train_data.jsonl, test_data.jsonl; sha256-checked):
#       seeker answers for CHARM's satisfaction and engagement labels
#                                             [data/redial_raw.tar.gz, unpacked into OUT]
# OUT   checkpoints, results, logs            [./harpo_out]
# M05   Qwen2.5-0.5B-Instruct                  [Qwen/Qwen2.5-0.5B-Instruct]
# M7    Qwen2.5-7B-Instruct                    [Qwen/Qwen2.5-7B-Instruct]
# GPU0 GPU1  80 GB GPUs; GPU1=GPU0 runs everything on one  [0 1]
# PY    python with requirements.txt installed [python]
# DRY=1 print the commands instead of running them
# Out of GPU memory? Smaller batches: CHARM_VAL_BATCH (dialogues per validation
# batch in CHARM training, 8), SCORE_PAIRS (candidates per scoring batch, 800),
# STAR_BATCH (dialogues per STAR batch, 16).
#
# Each stage is skipped once it has finished (OUT/logs/<stage>.done), so the
# script can simply be re-run after an interruption. About 27 h on one A100
# 80 GB, 16 h on two. GPU non-determinism and STAR's sampled readings make a
# rerun close to, not identical with, the result.

set -o pipefail
cd "$(dirname "$0")" || exit 1
OUT=$(mkdir -p "${OUT:-./harpo_out}" && cd "${OUT:-./harpo_out}" && pwd)
DATA=${DATA:-$OUT/data/redial_data}
RAW=${RAW:-$OUT/redial_raw}
M05=${M05:-Qwen/Qwen2.5-0.5B-Instruct}
M7=${M7:-Qwen/Qwen2.5-7B-Instruct}
GPU0=${GPU0:-0}; GPU1=${GPU1:-1}
PY=${PY:-python}
CK=$OUT/checkpoints; RES=$OUT/results; LOG=$OUT/logs; AG=$RES/agents
[ -n "$DRY" ] && LOG=$OUT/logs/dry_run          # never touches a real run's markers
mkdir -p "$CK" "$RES/bridge" "$AG" "$LOG"
export PYTHONUNBUFFERED=1 TQDM_MININTERVAL=60 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# The exact data the result was produced from.
check_data() {
  if [ ! -d "$DATA" ] && [ "$DATA" = "$OUT/data/redial_data" ]; then
    mkdir -p "$OUT/data" && tar -xzf data/redial_data.tar.gz -C "$OUT/data" || return 1
  fi
  if [ ! -f "$RAW/train_data.jsonl" ] || [ ! -f "$RAW/test_data.jsonl" ]; then
    if [ "$RAW" = "$OUT/redial_raw" ] && [ -f data/redial_raw.tar.gz ]; then
      tar -xzf data/redial_raw.tar.gz -C "$OUT" || return 1
    else                                        # ReDial's own release
      $PY -c "import sys; sys.path.insert(0, 'scripts')
from convert_redial import download_redial_from_github as get; get('$RAW')" || return 1
    fi
  fi
  sum() { if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1; }
  while read -r want file; do
    [ "$(sum "$file")" = "$want" ] || { echo "!! $file differs from the data the result used"; return 1; }
  done <<EOF
fcc1dc6889a8ca74dfc34219fbf922766c97bb83f385ef19aa53eddca43dabed $DATA/sft_data.json
73bf3cb74c90fa40311b83d03d97558a1a727c187a1f697b1e57e0667b26d63c $DATA/test_sft.json
fbc96491832a774824871ff256f5111b927579cd89006469796ca8d2dcf2eadf $DATA/movie_list.json
53dff9d8e3e175adf542635933ea1657d0108846ffb076cc54910e2497f4a89d $RAW/train_data.jsonl
d7781750787d104ac005e829cc7e6277d25b14644a223b0c2445aaaded6b19ac $RAW/test_data.jsonl
EOF
}

# stage NAME GPU command...: run once, log to OUT/logs/NAME.log, mark NAME.done.
stage() {
  local name=$1 gpu=$2; shift 2
  [ -f "$LOG/$name.done" ] && { echo "-- $name: done earlier, skipped"; return 0; }
  echo "== $name (GPU $gpu)  $(date)"
  if [ -n "$DRY" ]; then echo "   CUDA_VISIBLE_DEVICES=$gpu $*"; touch "$LOG/$name.done"; return 0; fi
  if CUDA_VISIBLE_DEVICES=$gpu "$@" > "$LOG/$name.log" 2>&1; then
    touch "$LOG/$name.done"
  else
    echo "!! $name failed, see $LOG/$name.log"; touch "$LOG/FAILED"; return 1
  fi
}
# after STAGE...: wait for stages run by the other lane.
after() {
  for s in "$@"; do
    while [ ! -f "$LOG/$s.done" ]; do
      [ -f "$LOG/FAILED" ] && return 1
      sleep 60
    done
  done
}

S5=(--val-fraction 0.05 --charm-fraction 0.0)
BACKBONE=(--train-size 0 --test-size 0 --catalog-size 100000 --seq-len 256 --batch-size 16
          --grad-accum 1 --catalog-refresh 250)
CK7=$CK/7b/checkpoints/sft_final
PROF=$RES/bridge/profiles.json
READ=$RES/readings_0.json,$RES/readings_1.json

# Retriever (7B two-tower, validation held out), cold-start item vectors chosen on
# validation, then the retriever and LM agents over its top-200.
retriever() {
  stage retriever "$GPU0" $PY scripts/run_experiment.py --data "$DATA" --model "$M7" \
      "${BACKBONE[@]}" --epochs 5 "${S5[@]}" --output "$CK/7b" --results-dir "$RES/7b" &&
  after bridge &&
  stage coldstart "$GPU0" $PY scripts/coldstart_probe.py --checkpoint "$CK7" --data "$DATA" \
      "${S5[@]}" --profiles "$PROF" --top 200 --out "$AG/coldstart_probe.json" || return 1
  for split in test val; do
    stage agents_$split "$GPU0" $PY scripts/rerank_eval.py --checkpoint "$CK7" --data "$DATA" \
        --split $split --top-k 200 "${S5[@]}" --coldstart "$AG/coldstart_probe.json" \
        --profiles "$PROF" --results-dir "$AG" || return 1
  done
}

# CHARM stage 1 (relevance): a 7B cross-encoder trained on the shortlists of a
# 0.5B retriever that never saw its conversations, then continued on the 7B
# retriever's top-200. BRIDGE profiles are written by the plain 7B model.
charm_stage1a() {
  stage retriever_05b "$GPU1" $PY scripts/run_experiment.py --data "$DATA" --model "$M05" \
      "${BACKBONE[@]}" --epochs 6 --val-fraction 0.05 --charm-fraction 0.25 \
      --output "$CK/05b" --results-dir "$RES/05b" &&
  stage charm_stage1a "$GPU1" $PY scripts/train_charm_ce.py --data "$DATA" \
      --backbone "$CK/05b/checkpoints/sft_final" --base-model "$M7" --dtype bfloat16 \
      --val-fraction 0.05 --charm-fraction 0.25 --top-k 50 --group 16 --batch-groups 8 \
      --eval-top-k 50 --eval-every 1000 --score-dialogues 32 --no-test \
      --results-dir "$RES/charm_stage1a" &&
  stage bridge "$GPU1" $PY scripts/bridge_profiles.py --model "$M7" \
      --catalog "$CK/05b/catalog.json" --out "$PROF"
}
charm_stage1() {  # lane GPU
  after retriever &&
  stage charm_stage1 "$1" $PY scripts/train_charm_ce.py --data "$DATA" --backbone "$CK7" \
      --base-model "$M7" --dtype bfloat16 "${S5[@]}" --top-k 200 --eval-top-k 100 --group 16 \
      --batch-groups 8 --max-train-cases 20000 --eval-every 1250 --score-dialogues 32 \
      --init-adapter "$RES/charm_stage1a/charm_ce_adapter" --no-test --results-dir "$RES/charm_stage1"
}

# Inputs of CHARM stage 2: preference readings, satisfaction/engagement labels,
# and cross-fitted shortlists (each training conversation ranked by a retriever
# that never trained on it: two fold retrievers).
readings_and_folds() {  # lane GPU fold
  local gpu=$1 k=$2
  stage readings_$k "$gpu" $PY scripts/pref_summaries.py --model "$M7" --data "$DATA" \
      --shard $k/2 --batch 64 --out "$RES/readings_$k.json" &&
  stage fold_$k "$gpu" $PY scripts/run_experiment.py --data "$DATA" --model "$M7" \
      "${BACKBONE[@]}" --epochs 5 "${S5[@]}" --cv-folds 2 --cv-fold $k --no-epoch-eval \
      --output "$CK/fold_$k" --results-dir "$RES/fold_$k"
}

# CHARM stage 2: relevance + satisfaction + engagement heads with the dialogue
# gate, readings and BRIDGE profiles, continued from stage 1.
charm_stage2() {
  after fold_0 fold_1 coldstart || return 1
  stage crossfit "$GPU0" $PY scripts/crossfit_shortlists.py \
      --folds "$CK/fold_0/checkpoints/sft_final,$CK/fold_1/checkpoints/sft_final" \
      --deployed "$CK7" --coldstart "$AG/coldstart_probe.json" --data "$DATA" "${S5[@]}" \
      --top-k 200 --out "$RES/shortlists_cv.pt" &&
  stage labels "$GPU0" $PY scripts/redial_labels.py --raw-dir "$RAW" --out "$RES/labels.json" &&
  after readings_0 readings_1 charm_stage1 &&
  stage charm_stage2 "$GPU0" $PY scripts/train_charm_ce.py --data "$DATA" --backbone "$CK7" \
      --base-model "$M7" --dtype bfloat16 "${S5[@]}" --top-k 200 --eval-top-k 100 --group 16 \
      --batch-groups 4 --epochs 1 --eval-every 2347 --score-dialogues "${CHARM_VAL_BATCH:-8}" \
      --max-length 512 --shortlists "$RES/shortlists_cv.pt" --heads 3 --labels "$RES/labels.json" \
      --summaries "$READ" --profiles "$PROF" --denoise --ema 0.999 --head-lr 1e-4 --gate-lr 1e-3 \
      --patience 2 --init-adapter "$RES/charm_stage1/charm_ce_adapter" --no-test \
      --results-dir "$RES/charm_stage2"
}

# CHARM agents (both stages) and STAR over the top-200, validation and test.
score() {  # lane GPU stage split
  local gpu=$1 st=$2 split=$3 raw=$AG/rerank_top200_raw.pt extra=()
  [ $split = val ] && raw=$AG/rerank_top200_val_raw.pt
  [ $st = 2 ] && extra=(--summaries "$READ" --profiles "$PROF")
  stage charm${st}_$split "$gpu" $PY scripts/score_shortlist_ce.py \
      --adapter "$RES/charm_stage$st/charm_ce_adapter" --base-model "$M7" --dtype bfloat16 \
      --data "$DATA" --raw "$raw" --split $split "${S5[@]}" --pair-budget "${SCORE_PAIRS:-800}" \
      "${extra[@]}" --out "$AG/charm${st}_$split.pt"
}
star() {  # lane GPU split
  local gpu=$1 split=$2 raw=$AG/rerank_top200_raw.pt
  [ $split = val ] && raw=$AG/rerank_top200_val_raw.pt
  stage star_$split "$gpu" $PY scripts/star_search.py --lm "$M7" \
      --charm-adapter "$RES/charm_stage2/charm_ce_adapter" --base-model "$M7" --raw "$raw" \
      --summaries "$READ" --profiles "$PROF" --split $split "${S5[@]}" --data "$DATA" \
      --top-m 30 --beam 3 --branch 2 --depth 3 --dialogues-per-batch "${STAR_BATCH:-16}" \
      --out "$AG/star_$split.pt"
}

# MAVEN: consensus weights chosen by cross-validation on validation, CHARM's
# diversity strength on validation, test once. Headline: every agent (both CHARM
# stages); maven_pure.json drops stage 1.
maven() {
  after charm1_val charm1_test charm2_val charm2_test star_val star_test || return 1
  local M=(--test-raw "$AG/rerank_top200_raw.pt" --val-raw "$AG/rerank_top200_val_raw.pt"
           --test-ce "$AG/charm2_test.pt" --val-ce "$AG/charm2_val.pt"
           --agent star="$AG/star_test.pt,$AG/star_val.pt" --diversity-profiles "$PROF")
  stage maven "$GPU0" $PY scripts/maven_cv.py "${M[@]}" \
      --agent charm_stage1="$AG/charm1_test.pt,$AG/charm1_val.pt" --out "$RES/maven.json" &&
  stage maven_pure "$GPU0" $PY scripts/maven_cv.py "${M[@]}" --out "$RES/maven_pure.json"
}

echo "HARPO reproduction  $(git rev-parse --short HEAD 2>/dev/null)  $(date)"
echo "DATA=$DATA  RAW=$RAW  OUT=$OUT  M05=$M05  M7=$M7  GPUs=$GPU0,$GPU1"
[ -n "$DRY" ] && rm -f "$LOG"/*.done
rm -f "$LOG/FAILED"
check_data || { [ -n "$ALLOW_OTHER_DATA" ] || exit 1; }

if [ "$GPU0" = "$GPU1" ]; then
  charm_stage1a && retriever && charm_stage1 "$GPU0" && readings_and_folds "$GPU0" 0 &&
  readings_and_folds "$GPU0" 1 && charm_stage2 && score "$GPU0" 1 val && score "$GPU0" 1 test &&
  score "$GPU0" 2 val && score "$GPU0" 2 test && star "$GPU0" val && star "$GPU0" test && maven
else
  { retriever && readings_and_folds "$GPU0" 0 && charm_stage2 && score "$GPU0" 2 val &&
    score "$GPU0" 2 test && star "$GPU0" val && maven; } & a=$!
  { charm_stage1a && charm_stage1 "$GPU1" && readings_and_folds "$GPU1" 1 &&
    after agents_val agents_test && score "$GPU1" 1 val && score "$GPU1" 1 test && after charm_stage2 && star "$GPU1" test; } & b=$!
  wait $a; wait $b
fi
[ -n "$DRY" ] && { rm -f "$LOG"/*.done; exit 0; }
[ -f "$LOG/maven_pure.done" ] || { echo "!! not finished; re-run to resume"; exit 1; }
echo; grep -E "CV MRR|^TEST" "$LOG/maven.log"
echo "result: $RES/maven.json"
