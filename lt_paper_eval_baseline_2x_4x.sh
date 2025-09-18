#!/bin/bash

OUTDIR=../../NeMo_LT/NeMo/test_paper_min4_frames_eos_all_make_sure_ar_has_min4
CODEC=/datap/misc/checkpoints/ml-model-INTERSPEECH_2025_abblations_21.5Hz_8_codebooks_2016_codes_enc_non_causal_dec_causal_1.89kbps.nemo
DATASETS="libritts_seen,libritts_test_clean_small" #,libritts_test_clean" #"libri_unseen_test_rfejgin" #,libri_dev_clean_eval_large"
#DATASETS="libritts_test_clean" #"libri_unseen_test_rfejgin" #,libri_dev_clean_eval_large"
REPEATS=5
CFG_SCALE=2.5 ##2.5
TEMPERATURE=0.6 #0.55
SAMPLING_TYPE="purity_default" #"default" #"causal"
#FIXED_SCHEDULE="0 1 16"
TOPK=80
mkdir -p $OUTDIR

echo `pwd`
# Function to map checkpoint path to hparams path
get_hparams_path() {
    local ckpt_path="$1"
    # Remove the /checkpoints/ part and everything after it
    local base_dir="${ckpt_path%/checkpoints/*}"
    # Append hparams.yaml
    echo "${base_dir}/hparams.yaml"
}

# Automatically build HPARAMS_ARRAY from CKPT_ARRAY
# Function to build hparams array from checkpoint array
build_hparams_array() {
    local -n ckpt_array_ref=$1
    local -n hparams_array_ref=$2
    hparams_array_ref=()
    for ckpt in "${ckpt_array_ref[@]}"; do
        hparams=$(get_hparams_path "$ckpt")
        hparams_array_ref+=("$hparams")
    done
}

# Function to validate hparams files exist
validate_hparams_files() {
    local -n hparams_array_ref=$1
    for HPARAMS in "${hparams_array_ref[@]}"; do
        if [ ! -f "$HPARAMS" ]; then
            echo "HPARAMS $HPARAMS does not exist"
            exit 1
        fi
    done
}

# Function to validate checkpoint files exist
validate_checkpoint_files() {
    local -n ckpt_array_ref=$1
    for CKPT in "${ckpt_array_ref[@]}"; do
        if [ ! -f "$CKPT" ]; then
            echo "CKPT $CKPT does not exist"
            exit 1
        fi
    done
}

# function that takes an array of strings and joins them with a ","
join_array_with_comma() {
    local -n array_ref=$1
    echo "${array_ref[@]}" | tr ' ' ','
}

BASE_DIR=/datap/cp/lt_paper
CKPT_BASELINE_2x=$BASE_DIR/lt_paper_k3_2x_baseline_16/checkpoints/magpie-tts--val_loss=5.4911-epoch=220-last.ckpt
HPARAMS_BASELINE_2x=$(get_hparams_path $CKPT_BASELINE_2x)
CKPT_BASELINE_4x=$BASE_DIR/lt_paper_k3_4x_baseline_16_batch16/checkpoints/magpie-tts--val_loss=5.9337-epoch=220-last.ckpt
HPARAMS_BASELINE_4x=$(get_hparams_path $CKPT_BASELINE_4x)

# Select checkpoints
CKPT_ARRAY_BASELINE=($CKPT_BASELINE_2x $CKPT_BASELINE_4x)
build_hparams_array CKPT_ARRAY_BASELINE HPARAMS_ARRAY_BASELINE
validate_hparams_files HPARAMS_ARRAY_BASELINE
CKPT_BASELINE_JOINED=$(join_array_with_comma CKPT_ARRAY_BASELINE)
HPARAMS_BASELINE_JOINED=$(join_array_with_comma HPARAMS_ARRAY_BASELINE)

#exit 0

echo
echo "Running experiments..."

set -x

echo "\n----------------------------------------\n"

# Parallel sampling 2x and 4x
python scripts/magpietts/infer_and_evaluate.py --hparams_files $HPARAMS_BASELINE_JOINED --checkpoint_files $CKPT_BASELINE_JOINED --datasets $DATASETS --temperature $TEMPERATURE --codecmodel_path $CODEC --out_dir $OUTDIR --num_repeats $REPEATS  --use_cfg --cfg_scale $CFG_SCALE --topk $TOPK --log_exp_name --batch_size 16
