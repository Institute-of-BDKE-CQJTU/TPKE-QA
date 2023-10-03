export MODEL="./model/origin_splinter"
export OUTPUT_DIR="output0"
export CUDA_VISIBLE_DEVICES=0

prop=1
for item in {bioasq,squad,searchqa,hotpotqa,naturalquestions,newsqa,textbookqa,triviaqa}; do
    for count in {16,32,64,128,256,512,1024}; do
        for seed in {42,43,44,45,46}; do
            echo "${item}-${count}-${seed}_p${prop}"
            python fine-tuning/run_mrqa.py \
                --model_type=bert \
                --model_name_or_path=$MODEL \
                --qass_head=True \
                --tokenizer_name=$MODEL \
                --output_dir=$OUTPUT_DIR \
                --train_file="./data/mrqa-few-shot/${item}/${item}-train-seed-${seed}-num-examples-${count}_delete_special_symbol_qass.jsonl" \
                --predict_file="./data/mrqa-few-shot/${item}/dev_delete_special_symbol_qass.jsonl" \
                --do_train \
                --do_eval \
                --max_seq_length=384 \
                --doc_stride=128 \
                --threads=8 \
                --save_steps=50000 \
                --per_gpu_train_batch_size=12 \
                --per_gpu_eval_batch_size=16 \
                --learning_rate=3e-5 \
                --max_answer_length=10 \
                --warmup_ratio=0.1 \
                --min_steps=200 \
                --dataset=${item} \
                --knowledge_loss_proportion=${prop} \
                --num_train_epochs=10 \
                --seed=${seed} \
                --use_cache=False \
                --load_pytorch_pretrained_model="path_of_tpke_model" \
                --evaluate_every_epoch=False \
                --overwrite_output_dir &>./logs/${item}_${count}_${seed}_prop_${prop}.txt
        done
    done
done