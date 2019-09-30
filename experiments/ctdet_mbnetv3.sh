cd ../src
python main.py ctdet --exp_id coco_mobilenetv3 --arch mobile --batch_size 114 --master_batch 18 --lr 5e-4 --gpus 0,1 --num_workers 16
