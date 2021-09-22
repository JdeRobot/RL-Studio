import datetime
import time

import gym

import numpy as np
from functools import reduce

from rl_studio.agents.robot_mesh import settings
from rl_studio.agents.robot_mesh import liveplot
from rl_studio.agents.robot_mesh import utils

from rl_studio.agents.robot_mesh.algorithms.qlearn import QLearn

if __name__ == '__main__':

    print(settings.title)
    print(settings.description)
    print(f"\t- Start hour: {datetime.datetime.now()}")

    environment = settings.envs_params["simple"]
    print(environment)
    env = gym.make(environment["env"], **environment)

    # TODO: Move to settings file
    outdir = './logs/robot_mesh_experiments/'
    stats = {}  # epoch: steps
    states_counter = {}
    states_reward = {}

    plotter = liveplot.LivePlot(outdir)

    last_time_steps = np.ndarray(0)

    actions = range(env.action_space.n)
    env = gym.wrappers.Monitor(env, outdir, force=True)
    counter = 0
    estimate_step_per_lap = environment["estimated_steps"]
    lap_completed = False
    total_episodes = 20000
    epsilon_discount = 0.999  # Default 0.9986

    qlearn = QLearn(actions=actions, alpha=0.9, gamma=0.95, epsilon=1)

    if settings.load_model:
        # TODO: Folder to models. Maybe from environment variable?
        file_name = "1_20210701_0848_act_set_simple_epsilon_0.19_QTABLE.pkl"
        utils.load_model(qlearn, file_name)
        qvalues = np.array(list(qlearn.q.values()), dtype=np.float64)
        print(qvalues)
        highest_reward = max(qvalues)
    else:
        highest_reward = 0
    initial_epsilon = qlearn.epsilon

    telemetry_start_time = time.time()
    start_time = datetime.datetime.now()
    start_time_format = start_time.strftime("%Y%m%d_%H%M")

    print(settings.lets_go)

    previous = datetime.datetime.now()
    checkpoints = []  # "ID" - x, y - time

    # START ############################################################################################################
    for episode in range(total_episodes):

        counter = 0
        done = False
        lap_completed = False

        cumulated_reward = 0
        state = env.reset()

        # state = ''.join(map(str, observation))

        for step in range(50000):

            counter += 1

            if qlearn.epsilon > 0.05:
                qlearn.epsilon *= epsilon_discount


            # Pick an action based on the current state
            action = qlearn.selectAction(state)

            print("Selected Action!! " + str(action))
            # Execute the action and get feedback
            nextState, reward, done, lap_completed = env.step(action)
            cumulated_reward += reward

            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            # nextState = ''.join(map(str, observation))

            try:
                states_counter[nextState] += 1
            except KeyError:
                states_counter[nextState] = 1

            qlearn.learn(state, action, reward, nextState, done)

            env._flush(force=True)

            if settings.save_positions:
                now = datetime.datetime.now()
                if now - datetime.timedelta(seconds=3) > previous:
                    previous = datetime.datetime.now()
                    x, y = env.get_position()
                    checkpoints.append([len(checkpoints), (x, y), datetime.datetime.now().strftime('%M:%S.%f')[-4]])

                if datetime.datetime.now() - datetime.timedelta(minutes=3, seconds=12) > start_time:
                    print("Finish. Saving parameters . . .")
                    utils.save_times(checkpoints)
                    env.close()
                    exit(0)

            if not done:
                state = nextState
            else:
                last_time_steps = np.append(last_time_steps, [int(step + 1)])
                stats[int(episode)] = step
                states_reward[int(episode)] = cumulated_reward
                print(f"EP: {episode + 1} - epsilon: {round(qlearn.epsilon, 2)} - Reward: {cumulated_reward}"
                      f"- Time: {start_time_format} - Steps: {step}")
                break

            if lap_completed:
                if settings.plotter_graphic:
                    plotter.plot_steps_vs_epoch(stats, save=True)
                utils.save_model(qlearn, start_time_format, stats, states_counter, states_reward)
                print(f"\n\n====> LAP COMPLETED in: {datetime.datetime.now() - start_time} - Epoch: {episode}"
                      f" - Cum. Reward: {cumulated_reward} <====\n\n")

            if counter > 1000:
                if settings.plotter_graphic:
                    plotter.plot_steps_vs_epoch(stats, save=True)
                qlearn.epsilon *= epsilon_discount
                utils.save_model(qlearn, start_time_format, episode, states_counter, states_reward)
                print(f"\t- epsilon: {round(qlearn.epsilon, 2)}\n\t- cum reward: {cumulated_reward}\n\t- dict_size: "
                      f"{len(qlearn.q)}\n\t- time: {datetime.datetime.now()-start_time}\n\t- steps: {step}\n")
                counter = 0

            if datetime.datetime.now() - datetime.timedelta(hours=2) > start_time:
                print(settings.eop)
                utils.save_model(qlearn, start_time_format, stats, states_counter, states_reward)
                print(f"    - N epoch:     {episode}")
                print(f"    - Model size:  {len(qlearn.q)}")
                print(f"    - Action set:  {settings.actions_set}")
                print(f"    - Epsilon:     {round(qlearn.epsilon, 2)}")
                print(f"    - Cum. reward: {cumulated_reward}")
                start_time = datetime.datetime.now()
                #env.close()
                #exit(0)

        if episode % 1 == 0 and settings.plotter_graphic:
            # plotter.plot(env)
            plotter.plot_steps_vs_epoch(stats)
            # plotter.full_plot(env, stats, 2)  # optional parameter = mode (0, 1, 2)

        if episode % 250 == 0 and settings.save_model and episode > 1:
            print(f"\nSaving model . . .\n")
            utils.save_model(qlearn, start_time_format, stats, states_counter, states_reward)

        m, s = divmod(int(time.time() - telemetry_start_time), 60)
        h, m = divmod(m, 60)

    print("Total EP: {} - epsilon: {} - ep. discount: {} - Highest Reward: {}".format(
            total_episodes,
            initial_epsilon,
            epsilon_discount,
            highest_reward
        )
    )

    l = last_time_steps.tolist()
    l.sort()

    # print("Parameters: a="+str)
    print("Overall score: {:0.2f}".format(last_time_steps.mean()))
    print("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

    plotter.plot_steps_vs_epoch(stats, save=True)

    env.close()
