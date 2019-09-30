cd ../src
# train
python3 main.py ctdet --exp_id coco_msdcn --arch mobiledcn --batch_size 64 --lr 2e-3 --gpus 0,1 --num_workers 8 --save_all --input_res 512
