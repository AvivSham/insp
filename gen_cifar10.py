import os, glob

folder = '/cortex/data/images/cifar10_png/test/'
class_list = os.listdir(folder)
#ww = [i for i in range(8) if i not in [3, 5]]
ww = [0, 1, 2, 3]
for cls in class_list:
    li = glob.glob(folder + f'{cls}/*.png')
    li = sorted(li)[:4]
    GPU_NUM = 4

    for idx, cur in enumerate(li):
        print(f"CUDA_VISIBLE_DEVICES={ww[idx % GPU_NUM]} python experiment_scripts/train_img.py --model_type=sine --experiment_name cifar10_test_{cls}_{os.path.basename(cur)} --noise_level 0 --steps_til_summary 20000 --num_epochs 10001 --epochs_til_ckpt 10000 --img_path {cur} --hidden_features 32 --sz 32 &")
        if idx and idx % GPU_NUM == 0:
            print("sleep 5")
