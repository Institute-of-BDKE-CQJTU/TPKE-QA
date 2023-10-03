export CUDA_VISIBLE_DEVICES=0
python post-training/run_post-training.py \
    --init_from_checkpoint \
    --pretrained_model_name_or_path=./model/origin_splinter \
    --train_data_file=ccnews,1-5_1-5 \
    --batch_size=48 \
    --scheduler=linear \
    --learning_rate=1e-5 \
    --save_steps=2000 \
    --logging_steps=200 \
    --weight_decay=0.01 \
    --epochs=1 \
    --overwrite_output_dir \
    --saved_dir="output/TPKE_1e5_ccnews_1-5_1-5"