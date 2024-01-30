# Random Seeds
seeds=(1 2022 2023 2024 2025 2026)

# shellcheck disable=SC2068
for seed in ${seeds[@]}
do
    #----------------------------------predict length 96---------------------------------------
    python run.py --seed $seed --data Exchange --model MTPNet \
    --seq_len 96 --label_len 96 --pred_len 96 --embed_dim 8 --dropout 0.2 \
    --learning_rate 1e-3 --patch_size '6, 12, 24' --trend_patch_size '24, 48'

    #----------------------------------predict length 192---------------------------------------
    python run.py --seed $seed --data Exchange --model MTPNet \
    --seq_len 96 --label_len 96 --pred_len 192 --embed_dim 8 --dropout 0.2 \
    --learning_rate 1e-3 --patch_size '6, 12, 24' --trend_patch_size '24, 48'

    #----------------------------------predict length 336---------------------------------------
    python run.py --seed $seed --data Exchange --model MTPNet \
    --seq_len 96 --label_len 96 --pred_len 336 --embed_dim 8 --dropout 0.2 \
    --learning_rate 1e-3 --patch_size '24, 48, 96' --trend_patch_size '24, 48, 96'

    #----------------------------------predict length 720---------------------------------------
    python run.py --seed $seed --data Exchange --model MTPNet \
    --seq_len 192 --label_len 96 --pred_len 720 --embed_dim 8 --dropout 0.2 \
    --learning_rate 1e-3 --patch_size '24, 48, 96' --trend_patch_size '24, 48, 96'
done








