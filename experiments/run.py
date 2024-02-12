from omegaconf import DictConfig
# include imports from template_repo

import hydra
import os
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
formatter = logging.Formatter('%(levelname)s - %(asctime)s - %(name)s: %(message)s')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

@hydra.main(config_path="conf", config_name="default")
def main(cfg: DictConfig):
  # Implement your script relying on the implementations in the library. 

if __name__ == "__main__":
    main()
