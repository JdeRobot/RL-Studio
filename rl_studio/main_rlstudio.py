import argparse
from distutils.command.config import config
import json
import yaml

from rl_studio.agents import TrainerFactory
from rl_studio.agents.trainer import TrainerValidator


def get_algorithm(config_file: dict, input_algorithm: str) -> dict:
    return {
        "name": input_algorithm,
        "params": config_file["algorithm"][input_algorithm],
    }

def get_environment(config_file: dict, input_env: str) -> dict:
    return {
        "name": input_env,
        "params": config_file["environments"][input_env],
        #"actions": config_file["actions"]["available_actions"][input_env],
        "actions": config_file["actions"]["available_actions"][config_file["actions"]["actions_set"]],
        "actions_set": config_file["actions"]["actions_set"],
        "actions_number": config_file["actions"]["actions_number"],
    }

def get_agent(config_file: dict, input_agent: str) -> dict:
    return {
        "name": input_agent,
        "params": config_file["agent"][input_agent],
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=argparse.FileType("r"), required=True, default="config.yml")
    parser.add_argument("-a", "--agent", type=str, required=True)
    parser.add_argument("-e", "--environment", type=str, required=True)
    parser.add_argument("-n", "--algorithm", type=str, required=True)
    args = parser.parse_args()

    config_file = yaml.load(args.file)
    # print(f"INPUT CONFIGURATION FILE:\n{yaml.dump(config_file, indent=4)}")

    trainer_params = {
        "settings": config_file["settings"],
        "algorithm": get_algorithm(config_file, args.algorithm),
        "environment": get_environment(config_file, args.environment),
        "agent": get_agent(config_file, args.agent),
    }

    # TODO: Create function to check dirs
    # os.makedirs("logs", exist_ok=True)
    # os.makedirs("images", exist_ok=True)

    # PARAMS
    params = TrainerValidator(**trainer_params)
    #print("PARAMS:\n")
    #print(json.dumps(dict(params), indent=2))
    trainer = TrainerFactory(params)
    trainer.main()

if __name__ == '__main__':
    main()
