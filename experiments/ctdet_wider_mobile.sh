cd ../src
# train
python main.py ctdet --exp_id coco_mobile_v3_small --arch mobile --batch_size 32 --master_batch 18 --lr 1.25e-4 --gpus 0 --num_workers 4 --dataset wider
# test
python test.py ctdet --exp_id coco_mobile_v3_small --arch mobile --keep_res --resume
# flip test
python test.py ctdet --exp_id coco_mobile_v3_small --arch mobile --keep_res --resume --flip_test 
# multi scale test
python test.py ctdet --exp_id coco_mobile_v3_small --arch mobile --keep_res --resume --flip_test --test_scales 0.5,0.75,1,1.25,1.5
cd ..