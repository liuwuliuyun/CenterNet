cd ../src
# train
python3 main.py ctdet --exp_id exp_2_2 --arch mobile --batch_size 32 --lr 2e-3 --gpus 1 --num_workers 4 --save_all
