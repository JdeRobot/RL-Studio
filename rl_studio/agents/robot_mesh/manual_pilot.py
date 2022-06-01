import datetime
import sys as system
import time


class RobotMeshTrainer:
    def __init__(self, params):
        # TODO: Create a pydantic metaclass to simplify the way we extract the params
        # environment params
        self.environment_params = params.environment["params"]
        self.env_name = params.environment["params"]["env_name"]
        env_params = params.environment["params"]
        actions = params.environment["actions"]
        env_params["actions"] = actions
        self.env = gym.make(self.env_name, **env_params)
        # algorithm params
        self.alpha = params.algorithm["params"]["alpha"]
        self.epsilon = params.algorithm["params"]["epsilon"]
        self.gamma = params.algorithm["params"]["gamma"]


    def execute_action(self, env):
        for step in range(50000):

            input_order = input("provide action (0-up, 1-right, 2-down, 3-left): ")
            if input_order == "0" or input_order == "1" or input_order == "2" or input_order == "3":
                action = int(input_order)
                print("Selected Action!! " + str(action))
                # Execute the action and get feedback
                next_state, reward, env.done, lap_completed = env.step(action)

                env._flush(force=True)
                if not env.done:
                    state = next_state
                else:
                    break
            elif input_order == "q":
                system.exit(1)
            else:
                print("wrong action! Try again")

    def main(self):

        print(f"\t- Start hour: {datetime.datetime.now()}\n")
        print(f"\t- Environment params:\n{self.environment_params}")
        outdir = "./logs/robot_mesh_experiments/"
        env = gym.wrappers.Monitor(self.env, outdir, force=True)
        total_episodes = 20000
        env.done = False

        for episode in range(total_episodes):
            print("resetting")
            time.sleep(5)
            env.reset()

            self.execute_action(env)

        env.close()
