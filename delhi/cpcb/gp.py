from time import time
import torch
import gpytorch

import pandas as pd
import numpy as np
from common import Model, rmse_fn, msll_fn


class ExactGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, model_cfg):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        kernel_name = model_cfg["kernel"]
        kernel_class = getattr(gpytorch.kernels, kernel_name)
        if kernel_name == "RBFKernel":
            self.covar_module = gpytorch.kernels.ScaleKernel(
                kernel_class(num_ard_dims=train_x.shape[1])
            )
        elif kernel_name == "MaternKernel":
            self.covar_module = gpytorch.kernels.ScaleKernel(
                kernel_class(nu=model_cfg["matern_nu"], ard_num_dims=train_x.shape[1])
            )
        else:
            raise NotImplementedError(f"Kernel {kernel_name} not implemented")

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GP(Model):
    def __init__(self, cfg, logger):
        super().__init__(cfg, logger)

        self.device = "cuda"

        # lazy load model in fit method

    def df_to_values_fn(self, df):
        return torch.tensor(df.values.squeeze()).float()

    def fit(self):
        init_time = time()

        model_cfg = self.cfg[self.cfg.model]

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = ExactGP(self.train_x, self.train_y, self.likelihood, model_cfg)

        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.likelihood, self.model
        ).to(self.device)

        optimizer = torch.optim.Adam(self.mll.parameters(), lr=model_cfg.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, verbose=True, factor=0.5
        )

        train_x = self.train_x.to(self.device)
        train_y = self.train_y.to(self.device)

        self.mll.train()
        losses = []
        for i in range(model_cfg.epochs):
            optimizer.zero_grad()
            output = self.model(train_x)
            loss = -self.mll(output, train_y)
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            scheduler.step(loss)

            if i % model_cfg.log_gap == 0:
                self.logger.info(f"Epoch {i}: loss {loss.item()}")

        self.logger.info(f"Training time (in minutes): {(time() - init_time) / 60}")

    def predict(self):
        self.mll.eval()

        test_x = self.test_x.to(self.device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred_dist = self.likelihood(self.model(test_x))
            pred_y = pred_dist.mean.cpu().numpy()
            pred_std_y = pred_dist.stddev.cpu().numpy()

            print(pred_y.shape, pred_std_y.shape)

        self.test_df[f"pred_{self.cfg.common.target}"] = pd.Series(
            pred_y, index=self.test_y_df.index
        )
        self.test_df[f"pred_std_{self.cfg.common.target}"] = pd.Series(
            pred_std_y, index=self.test_y_df.index
        )

        metrics = {
            "rmse": rmse_fn(self.test_y, pred_y),
            "msll": msll_fn(self.test_y, pred_y, pred_std_y),
        }
        return {"metrics": metrics}
