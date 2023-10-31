for vl in '0' #'36' '16' '8' '24' '4'
do
    for anp in '0' #'36' #'8' '4' '24' '4'
    do
        for lr in '1e-4' #'2e-4' #'3e-4'
        do
            for ds in 'Twitter10000V2.0' #'twitter17'
            do
                for se in '56' #'47' '2020' '3407' '2021' '2022' '2023' '26' '15' '25'
                do
                    for nb in '1' 
                    do
                        echo ${lr}
                        echo ${ds}
                        echo ${vl}
                        echo ${anp}
                        echo ${nb}
                        python main.py --task version10000 \
                                --dataset ${ds} \
                                --data_dir ./T5_data \
                                --model_name_or_path t5-base \
                                --n_gpu 0 \
                                --do_inference \
                                --checkpoint ./outputs/version \
                                --train_batch_size 16 \
                                --gradient_accumulation_steps 1 \
                                --eval_batch_size 16 \
                                --learning_rate ${lr} \
                                --num_train_epochs 20 \
                                --vinvl_region_number ${vl} \
                                --anp_number ${anp} \
                                --imgtag_path_vinvl ./twitterFMNERG_vinvl_extract36 \
                                --num_beams ${nb} \
                                --seed ${se} \
                                --has_tag \
                                --output_pred
                    done
                done               
            done
        done
    done
done
