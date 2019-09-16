cd ../src
# train
python main.py ctdet --exp_id coco_msdcn --arch mobiledcn --batch_size 32 --lr 5e-4 --gpus 0 --num_workers 4 --save_all