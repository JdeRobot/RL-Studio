import argparse
import json

import yaml

from rl_studio.agents import TrainerFactory, InferenceExecutorFactory
from rl_studio.agents.trainer import TrainerValidator, InferenceExecutorValidator


def get_algorithm(config_file: dict, input_algorithm: str) -> dict:
    return {
        "name": input_algorithm,
        "params": config_file["algorithm"][input_algorithm],
    }


def get_environment(config_file: dict, input_env: str) -> dict:
    return {
        "name": input_env,
        "params": config_file["environments"][input_env],
        "actions": config_file["actions"]
        .get("available_actions", None)
        .get(config_file["actions"].get("actions_set", None), None),
        "actions_set": config_file["actions"].get("actions_set", None),
        "actions_number": config_file["actions"].get("actions_number", None),
    }


def get_agent(config_file: dict, input_agent: str) -> dict:
    return {
        "name": input_agent,
        "params": config_file["agent"][input_agent],
    }


def get_inference(config_file: dict, input_inference: str) -> dict:
    return {
        "name": input_inference,
        "params": config_file["inference"][input_inference],
    }


def get_settings(config_file: dict) -> dict:
    return {
        "name": "settings",
        "params": config_file["settings"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--file", type=argparse.FileType("r"), required=True, default="config.yml"
    )
    parser.add_argument("-a", "--agent", type=str, required=True)
    parser.add_argument("-e", "--environment", type=str, required=True)
    parser.add_argument("-n", "--algorithm", type=str, required=True)
    parser.add_argument("-m", "--mode", type=str, required=False, default="training")

    args = parser.parse_args()
    config_file = yaml.load(args.file, Loader=yaml.FullLoader)

    if args.mode == "inference":

        inference_params = {
            "settings": get_settings(config_file),
            "algorithm": get_algorithm(config_file, args.algorithm),
            "inference": get_inference(config_file, args.algorithm),
            "environment": get_environment(config_file, args.environment),
            "agent": get_agent(config_file, args.agent),
        }

        # TODO: Create function to check dirs
        # os.makedirs("logs", exist_ok=True)
        # os.makedirs("images", exist_ok=True)

        # PARAMS
        params = InferenceExecutorValidator(**inference_params)
        print("PARAMS:\n")
        print(json.dumps(dict(params), indent=2))
        inferenceExecutor = InferenceExecutorFactory(params)
        inferenceExecutor.main()

    else:

        trainer_params = {
            "settings": get_settings(config_file),
            "algorithm": get_algorithm(config_file, args.algorithm),
            "environment": get_environment(config_file, args.environment),
            "agent": get_agent(config_file, args.agent),
        }

        # TODO: Create function to check dirs
        # os.makedirs("logs", exist_ok=True)
        # os.makedirs("images", exist_ok=True)

        # PARAMS
        params = TrainerValidator(**trainer_params)
        print("PARAMS:\n")
        print(json.dumps(dict(params), indent=2))
        trainer = TrainerFactory(params)
        trainer.main()


if __name__ == "__main__":
    main()
