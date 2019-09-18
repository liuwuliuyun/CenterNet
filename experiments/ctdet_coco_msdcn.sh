cd ../src
# train
python3 main.py ctdet --exp_id coco_msdcn_2 --arch mobiledcn --batch_size 32 --lr 1e-4 --gpus 0 --num_workers 4 --save_all --lr_step 40,80 --input_res 512
# test
python3 test.py ctdet --exp_id coco_msdcn --arch mobiledcn --resume --flip_test --test_scales 0.5,0.75,1,1.25,1.5