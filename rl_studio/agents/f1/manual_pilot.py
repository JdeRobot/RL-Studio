# coding: utf-8

import datetime
import pickle
import time

import gym

from rl_studio.agents.f1 import settings
from rl_studio.visual.ascii.text import MANUAL_PILOT

total_episodes = 200000


def save_times(checkpoints):
    file_name = "manual_pilot_checkpoints"
    file_dump = open("./logs/" + file_name + ".pkl", "wb")
    pickle.dump(checkpoints, file_dump)


if __name__ == "__main__":

    print(MANUAL_PILOT)
    print("    - Start hour: {}".format(datetime.datetime.now()))

    environment = settings.envs_params["manual"]
    env = gym.make(environment["env"])

    checkpoints = []  # "ID" - x, y - time
    time.sleep(5)

    previous = datetime.datetime.now()
    start_time = datetime.datetime.now()
    for episode in range(total_episodes):

        now = datetime.datetime.now()
        if now - datetime.timedelta(seconds=3) > previous:
            previous = datetime.datetime.now()
            x, y = env.get_position()
            checkpoints.append(
                [
                    len(checkpoints),
                    (x, y),
                    datetime.datetime.now().strftime("%M:%S.%f")[-4],
                ]
            )

        if (
            datetime.datetime.now() - datetime.timedelta(minutes=2, seconds=35)
            > start_time
        ):
            print("Finish. Saving parameters . . .")
            save_times(checkpoints)
            env.close()
            exit(0)


        env.execute()

        if episode % 500 == 0:
            print(episode)
