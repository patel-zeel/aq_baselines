import os
from os.path import join

from scipy.stats import norm
import numpy as np
import pandas as pd

import mlflow


ROOT = os.path.dirname(os.path.abspath(__file__))


def rmse_fn(y_true, y_pred):
    return np.sqrt(np.mean((y_true.ravel() - y_pred.ravel()) ** 2))


def msll_fn(y_true, y_pred, y_std):
    return -norm.logpdf(
        x=y_true.ravel(), loc=y_pred.ravel(), scale=y_std.ravel()
    ).mean()


class Model:
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger

        self.save_path = join(ROOT, "artifacts")

        assert self.__class__.__name__ == self.cfg.model

    def preprocess_and_load_data(self):
        f = lambda name: pd.read_csv(
            join(
                ROOT,
                "data",
                f"{self.cfg.common.start_date}_{self.cfg.common.end_date}",
                f"seed_{self.cfg.common.seed}",
                f"fold_{self.cfg.common.fold}",
                f"{name}.csv",
            )
        )
        self.train_df, self.test_df = map(f, ["train", "test"])

        if "time" in self.train_df:
            self.train_df["datetime"] = pd.to_datetime(self.train_df["time"])
            self.test_df["datetime"] = pd.to_datetime(self.test_df["time"])
            self.train_df["time"] = self.train_df["datetime"].astype(int) / 1e18
            self.test_df["time"] = self.test_df["datetime"].astype(int) / 1e18

        self.logger.info(f"self.train_df size: {self.train_df.shape[0]}")
        self.train_y_df = self.train_df[self.cfg.common.target].dropna()
        # log reduced self.train_df size
        self.logger.info(
            f"self.train_df size after dropping NaNs: {self.train_y_df.shape[0]}"
        )
        # log percentage of reduced self.train_df size
        self.logger.info(
            f"self.train_df reduction in size: {100*(1 - self.train_y_df.shape[0]/self.train_df.shape[0]):.2f}%"
        )

        self.logger.info(f"self.test_df size: {self.test_df.shape[0]}")
        self.test_y_df = self.test_df[self.cfg.common.target].dropna()
        # log reduced self.test_df size
        self.logger.info(
            f"self.test_df size after dropping NaNs: {self.test_y_df.shape[0]}"
        )
        # log percentage of reduced self.test_df size
        self.logger.info(
            f"self.test_df reduction in size: {100*(1 - self.test_y_df.shape[0]/self.test_df.shape[0]):.2f}%"
        )

        train_x_df = self.train_df[self.cfg.common.features].loc[self.train_y_df.index]
        test_x_df = self.test_df[self.cfg.common.features].loc[self.test_y_df.index]

        # convert based on convert_fn
        self.train_x, self.train_y, self.test_x = map(
            self.df_to_values_fn,
            [train_x_df, self.train_y_df, test_x_df],
        )
        self.test_y = self.test_y_df.values.squeeze()

    def fit(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def save(self):
        os.makedirs(self.save_path, exist_ok=True)

        # Add model hyparameters to test_df
        for k, v in self.cfg[self.__class__.__name__].items():
            self.test_df[k] = v

        model_name_path = self.__class__.__name__
        common_config_path = "_".join(
            [f"{k}={v}" for k, v in sorted(self.cfg.common.items())]
        )
        model_config_path = "_".join(
            [f"{k}={v}" for k, v in sorted(self.cfg[self.__class__.__name__].items())]
        )
        save_dir = join(
            self.save_path, model_name_path, common_config_path, model_config_path
        )

        os.makedirs(save_dir, exist_ok=True)
        path = join(save_dir, "test.csv")
        self.test_df.to_csv(path, index=False)
        self.logger.info(f"Saved {self.__class__.__name__} predictions to {path}")

    def run(self):
        self.preprocess_and_load_data()
        self.fit()
        result = self.predict()
        self.save()
        return result
