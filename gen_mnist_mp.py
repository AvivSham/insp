import multiprocessing as mp
import subprocess
from pathlib import Path
import os


def command_call(command):
    subprocess.call(command.split())


if __name__ == '__main__':
    print("main process ID:", os.getpid())
    path = Path("/Users/shamsiav/Downloads/mnist_png/training")
    commands = []
    GPU_NUM = 6
    n_jobs_per_gpu = 4
    gpu_ = [0, 1, 2, 3, 6, 7]
    for idx, p in enumerate(path.glob("**/*.png")):
        cls = p.parts[-2]
        if os.path.exists(f'logs/mnist_train_{cls}_{os.path.basename(p)}/checkpoints/model_final.pth'):
            continue
        commands.append(
            f"CUDA_VISIBLE_DEVICES={gpu_[idx // n_jobs_per_gpu % GPU_NUM]} python experiment_scripts/train_img.py --model_type=sine --experiment_name mnist_train_{cls}_{os.path.basename(p)} --noise_level 0 --steps_til_summary 20000 --num_epochs 1001 --epochs_til_ckpt 1000 --img_path {p} --hidden_features 32 --sz 28 &"
        )
    with mp.Pool(n_jobs_per_gpu * GPU_NUM) as pool:
        pool.map_async(command_call, commands)
