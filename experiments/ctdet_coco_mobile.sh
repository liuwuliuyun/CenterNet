cd ../src
# train
python3 main.py ctdet --exp_id coco_mobile_v3_small --arch mobile --batch_size 32  --lr 5e-4 --gpus 0 --num_workers 4
