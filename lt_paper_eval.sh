#!/bin/bash

OUTDIR=../../NeMo_LT/NeMo/test_paper_min4_frames_eos_all_make_sure_ar_has_min4
CODEC=/datap/misc/checkpoints/ml-model-INTERSPEECH_2025_abblations_21.5Hz_8_codebooks_2016_codes_enc_non_causal_dec_causal_1.89kbps.nemo
DATASETS="libritts_seen,libritts_test_clean_small" #,libritts_test_clean" #"libri_unseen_test_rfejgin" #,libri_dev_clean_eval_large"
#DATASETS="libritts_test_clean" #"libri_unseen_test_rfejgin" #,libri_dev_clean_eval_large"
REPEATS=5
CFG_SCALE=2.5 ##2.5
CFG_SCALE_MG=2.7
TEMPERATURE=0.6 #0.55
SAMPLING_TYPE="purity_default" #"default" #"causal"
#FIXED_SCHEDULE="0 1 16"
N_STEPS=3
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
CKPT_BASELINE=$BASE_DIR/lt_paper_k3_baseline/checkpoints/magpie-tts--val_loss=5.1236-epoch=220-last.ckpt
HPARAMS_BASELINE=$(get_hparams_path $CKPT_BASELINE)
CKPT_BASELINE_16=$BASE_DIR/lt_paper_k3_baseline_16/checkpoints/magpie-tts--val_loss=5.0996-epoch=220-last.ckpt
HPARAMS_BASELINE_16=$(get_hparams_path $CKPT_BASELINE_16)



CKPT_MG_1x=$BASE_DIR/lt_paper_k3_1x_maskgit/checkpoints/magpie-tts--val_loss=10.0375-epoch=220-last.ckpt
HPARAMS_MG_1x=$(get_hparams_path $CKPT_MG_1x)
CKPT_MG_2x=$BASE_DIR/lt_paper_k3_2x_maskgit/checkpoints/magpie-tts--val_loss=10.6545-epoch=220-last.ckpt
HPARAMS_MG_2x=$(get_hparams_path $CKPT_MG_2x)
CKPT_MG_4x=$BASE_DIR/lt_paper_k3_4x_maskgit/checkpoints/magpie-tts--val_loss=11.3289-epoch=220-last.ckpt
HPARAMS_MG_4x=$(get_hparams_path $CKPT_MG_4x)

CKPT_AR_1x=$BASE_DIR/lt_paper_k3_1x_ar/checkpoints/magpie-tts--val_loss=9.7647-epoch=220-last.ckpt
HPARAMS_AR_1x=$(get_hparams_path $CKPT_AR_1x)
CKPT_AR_2x=$BASE_DIR/lt_paper_k3_2x_ar/checkpoints/magpie-tts--val_loss=10.2695-epoch=220-last.ckpt
HPARAMS_AR_2x=$(get_hparams_path $CKPT_AR_2x)
CKPT_AR_4x=$BASE_DIR/lt_paper_k3_4x_ar/checkpoints/magpie-tts--val_loss=10.8828-epoch=220-last.ckpt
HPARAMS_AR_4x=$(get_hparams_path $CKPT_AR_4x)

# Select checkpoints
CKPT_ARRAY_MG=($CKPT_MG_1x $CKPT_MG_2x $CKPT_MG_4x)
CKPT_ARRAY_AR=($CKPT_AR_1x $CKPT_AR_2x $CKPT_AR_4x)
CKPT_ARRAY_BASELINE=($CKPT_BASELINE $CKPT_BASELINE_16)

build_hparams_array CKPT_ARRAY_MG HPARAMS_ARRAY_MG
build_hparams_array CKPT_ARRAY_AR HPARAMS_ARRAY_AR
build_hparams_array CKPT_ARRAY_BASELINE HPARAMS_ARRAY_BASELINE

validate_hparams_files HPARAMS_ARRAY_MG
validate_hparams_files HPARAMS_ARRAY_AR
validate_hparams_files HPARAMS_ARRAY_BASELINE

CKPT_MG_JOINED=$(join_array_with_comma CKPT_ARRAY_MG)
CKPT_AR_JOINED=$(join_array_with_comma CKPT_ARRAY_AR)
CKPT_BASELINE_JOINED=$(join_array_with_comma CKPT_ARRAY_BASELINE)

HPARAMS_MG_JOINED=$(join_array_with_comma HPARAMS_ARRAY_MG)
HPARAMS_AR_JOINED=$(join_array_with_comma HPARAMS_ARRAY_AR)
HPARAMS_BASELINE_JOINED=$(join_array_with_comma HPARAMS_ARRAY_BASELINE)

#exit 0

echo
echo "Running experiments..."

set -x


# MaskGit standard CFG scale (1x, 2x, 4x)
python scripts/magpietts/infer_and_evaluate.py --hparams_files $HPARAMS_MG_JOINED --checkpoint_files $CKPT_MG_JOINED --datasets $DATASETS --temperature $TEMPERATURE --codecmodel_path $CODEC --out_dir $OUTDIR --num_repeats $REPEATS  --use_cfg --cfg_scale $CFG_SCALE --maskgit_sampling_type $SAMPLING_TYPE --log_exp_name --topk $TOPK --use_local_transformer --maskgit_n_steps $N_STEPS

echo "\n----------------------------------------\n"

# AR (1x, 2x, 4x)
python scripts/magpietts/infer_and_evaluate.py --hparams_files $HPARAMS_AR_JOINED --checkpoint_files $CKPT_AR_JOINED --datasets $DATASETS --temperature $TEMPERATURE --codecmodel_path $CODEC --out_dir $OUTDIR --num_repeats $REPEATS  --use_cfg --cfg_scale $CFG_SCALE --topk $TOPK --use_local_transformer --log_exp_name

echo "\n----------------------------------------\n"

# Baseline
#python scripts/magpietts/infer_and_evaluate.py --hparams_files $HPARAMS_BASELINE --checkpoint_files $CKPT_BASELINE --datasets $DATASETS --temperature $TEMPERATURE --codecmodel_path $CODEC --out_dir $OUTDIR --num_repeats $REPEATS  --use_cfg --cfg_scale $CFG_SCALE --topk $TOPK --log_exp_name

echo "\n----------------------------------------\n"

# Baseline 16, with small batch size
python scripts/magpietts/infer_and_evaluate.py --hparams_files $HPARAMS_BASELINE_16 --checkpoint_files $CKPT_BASELINE_16 --datasets $DATASETS --temperature $TEMPERATURE --codecmodel_path $CODEC --out_dir $OUTDIR --num_repeats $REPEATS  --use_cfg --cfg_scale $CFG_SCALE --topk $TOPK --log_exp_name --batch_size 16




# old stuff before here

# echo "\n----------------------------------------\n"

# # MaskGit 2x but parallel sampling
# #python scripts/magpietts/infer_and_evaluate.py --hparams_files $HPARAMS_MG_2x --checkpoint_files $CKPT_MG_2x --datasets $DATASETS --temperature $TEMPERATURE --codecmodel_path $CODEC --out_dir $OUTDIR --num_repeats $REPEATS  --use_cfg --cfg_scale $CFG_SCALE --log_exp_name --topk $TOPK

# echo "\n----------------------------------------\n"

# # MaskGit 4x but parallel sampling
# #python scripts/magpietts/infer_and_evaluate.py --hparams_files $HPARAMS_MG_4x --checkpoint_files $CKPT_MG_4x --datasets $DATASETS --temperature $TEMPERATURE --codecmodel_path $CODEC --out_dir $OUTDIR --num_repeats $REPEATS  --use_cfg --cfg_scale $CFG_SCALE --log_exp_name --topk $TOPK

# echo "\n----------------------------------------\n"

# # AR 2x but parallel sampling
# #python scripts/magpietts/infer_and_evaluate.py --hparams_files $HPARAMS_AR_2x --checkpoint_files $CKPT_AR_2x --datasets $DATASETS --temperature $TEMPERATURE --codecmodel_path $CODEC --out_dir $OUTDIR --num_repeats $REPEATS  --use_cfg --cfg_scale $CFG_SCALE --log_exp_name --topk $TOPK

# echo "\n----------------------------------------\n"

# # AR 4x but parallel sampling
# #python scripts/magpietts/infer_and_evaluate.py --hparams_files $HPARAMS_AR_4x --checkpoint_files $CKPT_AR_4x --datasets $DATASETS --temperature $TEMPERATURE --codecmodel_path $CODEC --out_dir $OUTDIR --num_repeats $REPEATS  --use_cfg --cfg_scale $CFG_SCALE --log_exp_name --topk $TOPK

# echo "\n----------------------------------------\n"