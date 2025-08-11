import os
import multiprocessing

from omegaconf import OmegaConf
from training import SelfPredictionTrainingRecipeDistributed


def main():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"

    multiprocessing.set_start_method("spawn", force=True)

    cfg = OmegaConf.load("configs/llama_0.1B_PHi.yaml")
    # cfg = OmegaConf.load("configs/transformer_pfa_0_1B_PHi.yaml")
    # cfg = OmegaConf.load("configs/lstm_pfa_0_1B_PHi.yaml")

    cfg.evaluate_every_n_steps = 10
    cfg.checkpoint_every_n_steps = 20
    cfg.compile = False
    # cfg.metric_logger._component_ = "torchtune.training.metric_logging.DiskLogger"

    recipe = SelfPredictionTrainingRecipeDistributed(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    main()
