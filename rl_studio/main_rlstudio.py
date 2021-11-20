import argparse
import os

import yaml

from rl_studio.models.qlearn import QlearnModel
from rl_studio.agents.f1.train_qlearn import QlearnTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=argparse.FileType("r"), required=True, default="config.yml")
    parser.add_argument("-a", "--agent", type=str, required=True, default="config.yml")
    parser.add_argument("-e", "--environment", type=str, required=True, default="")
    parser.add_argument("-n", "--algorithm", type=str, required=True, default="")
    args = parser.parse_args()

    # READ YAML FILE
    print(args)
    config_file = yaml.load(args.file)
    print(f"INPUT CONFIGURATION FILE:\n{yaml.dump(config_file, indent=4)}")

    # config = read_config(args.config_file)
    # execute_algor = f"{config['Method']}_{config['Algorithm']}"

    # CHECK DIRS
    # os.makedirs("logs", exist_ok=True)
    # os.makedirs("images", exist_ok=True)

    print(f"\n\n{config_file['algorithm']}")

    # PARAMS
    params = QlearnModel(**config_file["algorithm"]["qlearn"])
    trainer = QlearnTrainer(params)
    trainer.main()


if __name__ == '__main__':
    main()
