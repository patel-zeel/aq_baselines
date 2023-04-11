import psutil
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from common import Model, rmse_fn, msll_fn


class RF(Model):
    def __init__(self, cfg, logger):
        super().__init__(cfg, logger)
        self.model = RandomForestRegressor(
            n_jobs=psutil.cpu_count() // 2, **self.cfg[self.__class__.__name__]
        )

    def df_to_values_fn(self, df):
        return df.values.squeeze()

    def fit(self):
        self.model.fit(self.train_x, self.train_y)

    def predict_std(self):
        return np.std(
            [tree.predict(self.test_x) for tree in self.model.estimators_], axis=0
        )

    def predict(self):
        pred_y = self.model.predict(self.test_x)
        pred_std_y = self.predict_std()

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
