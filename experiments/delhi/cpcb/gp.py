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

import torch
import gpytorch

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import norm
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

import argparse


# %%
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=2)
parser.add_argument("--config", type=str, default="data_jan")
parser.add_argument("--kernel", type=str, default="MaternKernel")

# add default args passed to jupyter
if 'get_ipython' in globals():
    parser.add_argument("--ip", type=str)
    parser.add_argument("--stdin", type=int)
    parser.add_argument("--control-port", type=int)
    parser.add_argument("--hb-port", type=int)
    parser.add_argument("--Session.signature_scheme", type=str)
    parser.add_argument("--Session.key", type=str)
    parser.add_argument("--shell", type=int)
    parser.add_argument("--transport", type=str)
    parser.add_argument("--iopub", type=int)
    parser.add_argument("--f", type=str)

args = parser.parse_args()

# %%
def rmse_fn(true_y, pred_y):
    return np.sqrt(np.mean((true_y.ravel() - pred_y.ravel()) ** 2))

def msll_fn(true_y, pred_y, pred_std_y):
    return -norm.logpdf(true_y.ravel(), loc=pred_y.ravel(), scale=pred_std_y).mean()


# %%
REPO_ROOT = subprocess.check_output("git rev-parse --show-toplevel".split()).decode().strip()
CONFIG_ROOT = join(REPO_ROOT, "config")

DATA_CONFIG = Dict(yaml.load(open(join(CONFIG_ROOT, f"{args.config}.yaml"), "r"), Loader=yaml.FullLoader))

EXP_PATH = join(REPO_ROOT, DATA_CONFIG.artifacts_path, join("_".join(sorted(DATA_CONFIG.features) + [DATA_CONFIG.start_date, DATA_CONFIG.end_date])))

DEVICE = f"cuda:{args.gpu}"

KERNEL_CLASS = eval(f"gpytorch.kernels.{args.kernel}")


# %%
class InbuiltExactGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(InbuiltExactGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        if KERNEL_CLASS.__name__ == "MaternKernel":
            self.covar_module = gpytorch.kernels.ScaleKernel(KERNEL_CLASS(nu=1.5, ard_num_dims=train_x.shape[1]))
        elif KERNEL_CLASS.__name__ == "SpectralMixtureKernel":  # takes huge amount of memory so not using this kernel
            self.covar_module = KERNEL_CLASS(num_mixtures=2, ard_num_dims=train_x.shape[1])
        else:
            self.covar_module = gpytorch.kernels.ScaleKernel(KERNEL_CLASS(ard_num_dims=train_x.shape[1]))
        
        if KERNEL_CLASS.__name__ == "SpectralMixtureKernel":
            # self.covar_module.initialize_from_data(train_x, train_y)
            pass

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class ExactGP:
    def __init__(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y
        
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = InbuiltExactGP(train_x, train_y, self.likelihood)
        
    def fit(self, train_x, train_y):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
        schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)
        
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model).to(train_x.device)
        
        self.mll.train()
        
        for i in range(20):
            optimizer.zero_grad()
            output = self.model(train_x)
            loss = -self.mll(output, train_y)
            loss.backward()
            optimizer.step()
            schedular.step(loss)
        
    def predict(self, test_x):
        self.mll.eval()
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(test_x))
            
        return observed_pred.mean.squeeze().cpu().numpy(), observed_pred.stddev.squeeze().cpu().numpy()


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
    X_scaler = MinMaxScaler()
    train_x[train_x.columns] = X_scaler.fit_transform(train_x[train_x.columns])
    test_x[test_x.columns] = X_scaler.transform(test_x[test_x.columns])
    
    y_scaler = StandardScaler()
    train_y[train_y.columns] = y_scaler.fit_transform(train_y[train_y.columns])
    test_y[test_y.columns] = y_scaler.transform(test_y[test_y.columns])
    
    return map(lambda x: torch.tensor(x.values.squeeze()).float(), (train_x, train_y, test_x, test_y)), X_scaler, y_scaler
    
seeds = list(range(DATA_CONFIG.n_seeds))
folds = list(range(DATA_CONFIG.n_folds))

pbar = tqdm(product(seeds, folds))
for seed, fold in pbar:
    # log seed and fold in tqdm progress bar inplace
    pbar.set_description(f"seed: {seed}, fold: {fold}")
    
    (train_x, train_y, test_x, test_y), X_scaler, y_scaler = load_data(seed, fold)
    model = ExactGP(train_x, train_y)
    
    model.fit(train_x.to(DEVICE), train_y.to(DEVICE))
    pred_y, pred_std_y = model.predict(test_x.to(DEVICE))
    
    with torch.no_grad():
        pred_y = y_scaler.inverse_transform(pred_y.reshape(-1, 1))
        test_y = y_scaler.inverse_transform(test_y.cpu().numpy().reshape(-1, 1))
        pred_std_y = pred_std_y * y_scaler.scale_
    
    result_df.loc[(seed, fold), "rmse"] = rmse_fn(test_y, pred_y)
    result_df.loc[(seed, fold), "msll"] = msll_fn(test_y, pred_y, pred_std_y)
    del model, train_x, train_y, test_x, test_y, pred_y, pred_std_y
    torch.cuda.empty_cache()

# %%
result_df.to_csv(join(EXP_PATH, f"metrics_{model.__class__.__name__}_{args.kernel}.csv"))
