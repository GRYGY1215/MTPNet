# Random Seeds
seeds=(1 2022 2023 2024 2025 2026)
# shellcheck disable=SC2068
for seed in ${seeds[@]}
do
    #----------------------------------predict length 24---------------------------------------
    python run.py --seed $seed --model MTPNet --data ILI --moving_avg '13, 17' \
    --seq_len 104 --label_len 24 --pred_len 24 --embed_dim 64 --dropout 0.4 \
    --learning_rate 5e-4 --patch_size '4, 6, 8, 12' --trend_patch_size '12, 24'

    #----------------------------------predict length 36---------------------------------------
    python run.py --seed $seed --model MTPNet --data ILI --moving_avg '13, 17' \
    --seq_len 104 --label_len 36 --pred_len 36 --embed_dim 64 --dropout 0.4 \
    --learning_rate 5e-4 --patch_size '4, 6, 8, 12' --trend_patch_size '12, 24'

    #----------------------------------predict length 48---------------------------------------
    python run.py --seed $seed --model MTPNet --data ILI --moving_avg '13, 17' \
    --seq_len 104 --label_len 48 --pred_len 48 --embed_dim 64 --dropout 0.4 \
    --learning_rate 5e-4 --patch_size '4, 6, 8, 12' --trend_patch_size '12, 24'

    #----------------------------------predict length 60---------------------------------------
    python run.py --seed $seed --model MTPNet --data ILI --moving_avg '13, 17' \
    --seq_len 104 --label_len 60 --pred_len 60 --embed_dim 64 --dropout 0.4 \
    --learning_rate 5e-4 --patch_size '4, 6, 8, 12' --trend_patch_size '12, 24'
done
