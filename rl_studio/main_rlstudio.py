import json
import argparse

import yaml

from rl_studio.agents.f1.train_qlearn import QlearnTrainer
from rl_studio.models.trainer import TrainerValidator


def select_algorithm(config_file: dict, input_algorithm: str) -> dict:
    return config_file["algorithm"][input_algorithm]


def select_environment(config_file: dict, input_env: str) -> dict:
    return config_file["environments"][input_env]


def select_agent(config_file: dict, input_agent: str) -> dict:
    return config_file["agents"][input_agent]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=argparse.FileType("r"), required=True, default="config.yml")
    parser.add_argument("-a", "--agent", type=str, required=True, default="config.yml")
    parser.add_argument("-e", "--environment", type=str, required=True, default="")
    parser.add_argument("-n", "--algorithm", type=str, required=True, default="")
    args = parser.parse_args()

    config_file = yaml.load(args.file)
    print(f"INPUT CONFIGURATION FILE:\n{yaml.dump(config_file, indent=4)}")

    trainer_params = {
        args.algorithm: select_algorithm(config_file, args.algorithm),
        args.environment: select_environment(config_file, args.environment),
        args.agent: select_agent(config_file, args.agent)
    }
    print("\n\n---------------")
    print(json.dumps(trainer_params, indent=4))
    exit()

    # config = read_config(args.config_file)
    # execute_algor = f"{config['Method']}_{config['Algorithm']}"

    # CHECK DIRS
    # os.makedirs("logs", exist_ok=True)
    # os.makedirs("images", exist_ok=True)

    print(f"\n\n{config_file['algorithm']}")
    print(config_file)
    # PARAMS
    params = TrainerValidator(**config_file)
    trainer = QlearnTrainer(params)
    trainer.main()


if __name__ == '__main__':
    main()
