import os
from itertools import product
import argparse
import GPUtil

from omegaconf import OmegaConf

from multiprocessing import Pool

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="RF")
parser.add_argument("--n_jobs", type=int, default=4)
parser.add_argument("--use_gpu", type=bool, default=True)
parser.add_argument("--test_run", type=bool, default=False)


args = parser.parse_args()

# Sort GPUs with increasing memory usage
if args.use_gpu:
    GPUs = GPUtil.getGPUs()
    GPUs.sort(key=lambda x: x.memoryUsed)
    assert len(GPUs) >= args.n_jobs, "Not enough GPUs available"
    GPUs = [gpu.id for gpu in GPUs[: args.n_jobs]]

# Load config
cfg = OmegaConf.load(f"config/config.yaml")
seed_fold = list(product(range(cfg.common.n_seeds), range(cfg.common.n_folds)))


# Main function
def main(gpu_id, seed, fold):
    command = f"python run.py model={args.model} common.seed={seed} common.fold={fold}"
    if args.use_gpu:
        command = f"CUDA_VISIBLE_DEVICES={gpu_id} " + command
    print("Running: ", command)
    os.system(command)


# Setup and run
if args.test_run:
    print("Running test run")
    pool = Pool(1)
    result = pool.apply_async(main, args=(1, 0, 0))
    result.get()
else:
    pool = Pool(args.n_jobs)
    results = []
    for job_i, seed_fold in enumerate(seed_fold):
        seed, fold = seed_fold
        gpu_id = GPUs[job_i % len(GPUs)]
        results.append(pool.apply_async(main, args=(gpu_id, seed, fold)))

    pool.close()
    pool.join()

    # Raise errors if any
    for result in results:
        result.get()
