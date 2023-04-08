# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# %%
from os.path import join
import subprocess

from itertools import product
from tqdm import tqdm
import yaml
from addict import Dict
import psutil
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np

# %%
N_ESTIMATORS = 1000
N_JOBS = psutil.cpu_count() // 2

REPO_ROOT = subprocess.check_output("git rev-parse --show-toplevel".split()).decode().strip()
CONFIG_ROOT = join(REPO_ROOT, "config")

DATA_CONFIG = Dict(yaml.load(open(join(CONFIG_ROOT, "data.yaml"), "r"), Loader=yaml.FullLoader))

EXP_PATH = join(DATA_CONFIG.artifacts_path, join("_".join(sorted(DATA_CONFIG.features) + [DATA_CONFIG.start_date, DATA_CONFIG.end_date])))

# %%
result_df = pd.DataFrame(columns=["rmse", "msll", "seed", "fold"])
result_df.set_index(["seed", "fold"], inplace=True)

def load_data(seed, fold):
    f = lambda name: pd.read_csv(join(EXP_PATH, f"seed_{seed}", f"fold_{fold}", f"{name}.csv"))
    train_x, train_y, test_x, test_y = map(f, ["train_x", "train_y", "test_x", "test_y"])
    
    if "time" in train_x:
        train_x["time"] = pd.to_datetime(train_x["time"]).astype(int)/1e18
        test_x["time"] = pd.to_datetime(test_x["time"]).astype(int)/1e18
        
    # print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
    train_y = train_y.dropna()
    test_y = test_y.dropna()
    train_x = train_x.loc[train_y.index]
    test_x = test_x.loc[test_y.index]
    # print("reduced", train_x.shape, train_y.shape, test_x.shape, test_y.shape)
    
    return map(lambda x: x.values.squeeze(), (train_x, train_y, test_x, test_y))
    
seeds = list(range(DATA_CONFIG.n_seeds))
folds = list(range(DATA_CONFIG.n_folds))

pbar = tqdm(product(seeds, folds))
for seed, fold in pbar:
    # log seed and fold in tqdm progress bar inplace
    pbar.set_description(f"seed: {seed}, fold: {fold}")
    
    model = RandomForestRegressor(n_estimators=N_ESTIMATORS, n_jobs=N_JOBS, random_state=seed)
    train_x, train_y, test_x, test_y = load_data(seed, fold)
    
    model.fit(train_x, train_y)
    pred_y = model.predict(test_x)
    pred_std_y = np.std([tree.predict(test_x) for tree in model.estimators_], axis=0)
    
    result_df.loc[(seed, fold), "rmse"] = rmse_fn(test_y, pred_y)
    result_df.loc[(seed, fold), "msll"] = msll_fn(test_y, pred_y, pred_std_y)

# %%
result_df.to_csv(join(EXP_PATH, f"metrics_{model.__class__.__name__}.csv"))
