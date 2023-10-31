
for se in '42' '56' '47' '2020' '2021' '2022' '2023' '26' '15' '25'
do
      python main.py --task version10000 \
              --dataset Twitter10000V2.0 \
              --data_dir ./T5_data \
              --model_name_or_path t5-base \
              --n_gpu 0 \
              --do_train \
              --do_direct_eval \
              --train_batch_size 16 \
              --gradient_accumulation_steps 1 \
              --eval_batch_size 32 \
              --learning_rate 2e-4 \
              --num_train_epochs 10 \
              --vinvl_region_number 32 \
              --img_path_vinvl ./twitterFMNERG_vinvl_extract36 \
              --image_annotation_path ./Twitter10000v2_final/xml \
              --use_visual_feats True \
              --num_beams 1 \
              --seed ${se} \
              --coarse_grained_auxiliary True \
              --output_pred
done