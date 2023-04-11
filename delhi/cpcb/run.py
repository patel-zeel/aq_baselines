from common import Model
import logging
import os
from rf import RF
from gb import GB
from gp import GP

import mlflow
import hydra


# Run with hydra
@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(cfg):
    model_class = globals()[cfg.model]
    assert issubclass(model_class, Model)
    logger = logging.getLogger(model_class.__name__)
    model = model_class(cfg, logger)
    result = model.run()
    mlflow.log_params({"model": model.__class__.__name__})
    mlflow.log_params({"seed": cfg.common.seed, "fold": cfg.common.fold})
    mlflow.log_params(cfg[model.__class__.__name__])
    mlflow.log_metrics(result["metrics"])


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Set the tracking URI to the path of the directory
    mlflow.set_tracking_uri(f"file://{script_dir}/mlruns")
    mlflow.start_run()
    main()
    mlflow.end_run()
