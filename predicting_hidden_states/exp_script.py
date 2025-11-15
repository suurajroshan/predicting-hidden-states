import os
import multiprocessing
import sys

from omegaconf import OmegaConf
from training import SelfPredictionTrainingRecipeDistributed


def main():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"

    multiprocessing.set_start_method("spawn", force=True)

    cfg = OmegaConf.load("configs/llama_0.1B_PHi.yaml")
    # cfg = OmegaConf.load("configs/llama_3B_self_prediction.yaml")

    # override the cfg with cli parameters
    cli_cfg = OmegaConf.from_dotlist(sys.argv[1:])
    cli_debug_flag = cli_cfg.pop("debug", False)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    cfg.checkpoint_every_n_steps = 1000


    if cli_debug_flag:
        print('Set to debug mode')
        cfg.evaluate_every_n_steps = 5
        cfg.evaluate_n_datapoints = 5
        cfg.checkpoint_every_n_steps = 100000
        cfg.log_every_n_steps = 1

    # cfg.evaluate_n_datapoints = 10
    cfg.dataset.packed_sequence_length = 2048
    cfg.compile = False
    cfg.metric_logger._component_ = "torchtune.training.metric_logging.WandBLogger"
    recipe = SelfPredictionTrainingRecipeDistributed(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    main()
