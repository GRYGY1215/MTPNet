# Random Seeds
seeds=(1 2022 2023 2024 2025 2026)
# shellcheck disable=SC2068
for seed in ${seeds[@]}
do
      echo $seed
      #----------------------------------predict length 96---------------------------------------
      python run.py --seed $seed --data Traffic --batch_size 4  --moving_avg '13, 17' \
      --seq_len 720 --label_len 96 --pred_len 96 --embed_dim 8 \
      --learning_rate 5e-4 --patch_size '8, 12, 24, 48'

      #----------------------------------predict length 192---------------------------------------
      python run.py --seed $seed --data Traffic --batch_size 4 --moving_avg '13, 17' \
      --seq_len 720 --label_len 96 --pred_len 192 --embed_dim 8 \
      --learning_rate 5e-4 --patch_size '8, 12, 24, 48'

      #----------------------------------predict length 336---------------------------------------
      python run.py --seed $seed --data Traffic --batch_size 4 --moving_avg '13, 17' \
      --seq_len 720 --label_len 96 --pred_len 336 --embed_dim 8 \
      --learning_rate 5e-4 --patch_size '8, 12, 24, 48'

      #----------------------------------predict length 720---------------------------------------
      python run.py --seed $seed --data Traffic --batch_size 8 --moving_avg '13, 17' \
      --seq_len 720 --label_len 96 --pred_len 720 --embed_dim 8 \
      --learning_rate 5e-4 --patch_size '12, 24, 48'
done


