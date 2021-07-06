
import time
from tqdm import tqdm
from cprint import cprint
from functools import reduce
import os

import gym

import numpy as np

import settings
import liveplot
import utils
from icecream import ic
from datetime import datetime, timedelta

from algorithms.Tabulars.qlearn import QLearn

from gym_gazebo.envs.gazebo_env import *


ic.enable()
#ic.disable()
#ic.configureOutput(prefix='Debug | ')
ic.configureOutput(prefix=f'{datetime.now()} | ')


def train_qlearning(config):

    #cprint.warn(f"\n [train_qlearning_f1] -> {config['Title']}")
    #cprint.ok(f"\n [train_qlearning_f1] -> {config['Description']}")
    #cprint.info(f"\n- [train_qlearning_f1] -> Start hour: {datetime.now()}")


    #--------------------- Init QLearning Vars
    algorithm_hyperparams = config['Hyperparams']
    model = config['Model']

    # ------------------ Init env
    environment = {}
    environment['agent'] = config['Agent']
    environment['env'] = config['envs_params'][model]['env']
    environment['training_type'] = config['envs_params'][model]['training_type']
    environment['circuit_name'] = config['envs_params'][model]['circuit_name']
    environment['actions'] = settings.AVAILABLE_ACTIONS[settings.actions_set]
    environment['launch'] = config['envs_params'][model]['launch']
    environment['gaz_pos'] = settings.GAZEBO_POSITIONS[config['envs_params'][model]['circuit_name']]
    environment['start_pose'] = [settings.GAZEBO_POSITIONS[config['envs_params'][model]['circuit_name']][1][1], settings.GAZEBO_POSITIONS[config['envs_params'][model]['circuit_name']][1][2]]
    environment['alternate_pose'] = config['envs_params'][model]['alternate_pose']
    environment['estimated_steps'] = config['envs_params'][model]['estimated_steps']
    environment['sensor'] = config['envs_params'][model]['sensor']
    environment['rewards'] = config['Rewards']
    environment['ROS_MASTER_URI'] = config['ROS_MASTER_URI']
    environment['GAZEBO_MASTER_URI'] = config['GAZEBO_MASTER_URI']


    #print(f"\n [train_qlearning_f1] -> environment: {environment}")
    ic(environment)

    env = gym.make(environment["env"], **environment)

    #cprint.info(f"\n ---- [train_qlearning_f1] -> come back train_qlearn_f1 ------------")

    # ---------------- Init hyperparams & Algorithm
    alpha = config['Hyperparams']['alpha']
    gamma = config['Hyperparams']['gamma']
    initial_epsilon = config['Hyperparams']['epsilon']
    epsilon = config['Hyperparams']['epsilon']
    epsilon_discount = config['Hyperparams']['epsilon_discount']
    total_episodes = config['Hyperparams']['total_episodes']
    estimated_steps = config['envs_params'][model]['estimated_steps']

    actions = range(env.action_space.n) # lo recibe de F1QlearnCameraEnv
    ic(actions)    

    # ---------------- Init vars 
    outdir = f"{config['Dirspace']}/logs/{config['Method']}_{config['Algorithm']}_{config['Agent']}"
    #print(f"\n outdir: {outdir}")    
    os.makedirs(f"{outdir}", exist_ok=True)
    ic(outdir)

    #env = gym.wrappers.Monitor(env, outdir, force=True)

    '''
        Init vars STATS
    '''
    stats = {}  # epoch: steps
    states_counter = {}
    states_reward = {}    
    ep_rewards = []
    aggr_ep_rewards = {
        'episode':[], 'avg':[], 'max':[], 'min':[], 'step':[], 'epsilon':[], 'time_training':[]
    } 
    last_time_steps = np.ndarray(0) 


    plotter = liveplot.LivePlot(outdir)
    
    counter = 0
    estimate_step_per_lap = environment["estimated_steps"]
    lap_completed = config['lap_completed']

    '''
        TIME
    '''
    start_time_training = time.time()
    telemetry_start_time = time.time()
    start_time = datetime.now()
    start_time_format = start_time.strftime("%Y%m%d_%H%M")
    previous = datetime.now()
    checkpoints = []  # "ID" - x, y - time 
    #print(f"\n [train_qlearning_f1] -> telemetry_start_time: {telemetry_start_time}, "
    #    f"start_time: {start_time}, start_time_format: {start_time_format}, previous: {previous}")

    #ic(start_time_training)
    #ic(telemetry_start_time)
    #ic(start_time)
    #ic(start_time_format)
    #ic(previous)       

    '''
        LOAD qtable to continue training. So we start from not empty table

    '''
    if config['load_qtable']:
        q_table = config['table_loaded']
        qlearn = QLearn(actions=actions, alpha=alpha, gamma=gamma, epsilon=epsilon, q_table=q_table)    

    else:
        qlearn = QLearn(actions=actions, alpha=alpha, gamma=gamma, epsilon=epsilon)    


    '''
        LOAD highest_reward, but then we dont use anymore.
        LOAD MODEL OR NOT. WE HERITATE THIS ONE
        BUT i NOT SURE IT WORKS PROPERLY
    '''
    if config['load_model']:
        file_name = config['file_load_pickle']
        utils.load_model(outdir, qlearn, file_name, config)
        highest_reward = max(qlearn.q.values(), key=stats.get)
    else:
        highest_reward = config['highest_reward'] # 0

    #cprint.warn(f"\n[train_qlearning_f1] -> {config['Lets_go']}")
    ic(config['Lets_go'])


    '''
        START TRAINING GOOOOOOOOOOO
    
    '''
    #start_time_training = time.time()
    for episode in range(total_episodes):

        counter = 0
        done = False
        lap_completed = False

        cumulated_reward = 0
        observation = env.reset()

        if epsilon > 0.05:
            epsilon *= epsilon_discount

        state = ''.join(map(str, observation))
        #print(f"\n -> START episode {episode}, counter: {counter}, observation: {observation}"
        #    f", done: {done}, lap_completed: {lap_completed}, cumulated_reward: {cumulated_reward}"
        #    f". state: {state}")

        ic('START EPISODE')
        ic(episode)    
        ic(counter)    
        ic(observation)    
        ic(lap_completed)    
        ic(cumulated_reward)    
        ic(done)    
        ic(state)  

        '''
            START STEPS in every episode
        '''

        for step in tqdm(range(estimated_steps)): #original range(500_000)

            counter += 1

            # Pick an action based on the current state
            action = qlearn.selectAction(state)

            # Execute the action and get feedback
            observation, reward, done, info = env.step(action)
            #print(f"\n ==> episode {episode}, step {step}: action: {action}"
            #    f", observation: {observation}, reward: {reward}"
            #    f", done: {done}, info: {info}")
            ic('init step')
            ic(episode)    
            ic(step)    
            ic(action)    
            ic(observation)    
            ic(reward)    
            ic(done)    
            ic(info)    
            
            cumulated_reward += reward

            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            nextState = ''.join(map(str, observation))

            try:
                states_counter[nextState] += 1
            except KeyError:
                states_counter[nextState] = 1

            qlearn.learn(state, action, reward, nextState)

            #env._flush(force=True)

            # ------------------ I dont really like this one
            if config['save_positions']:
                now = datetime.now()
                if now - timedelta(seconds=3) > previous:
                    previous = datetime.now()
                    x, y = env.get_position()
                    checkpoints.append([len(checkpoints), (x, y), datetime.now().strftime('%M:%S.%f')[-4]])

                if datetime.now() - timedelta(minutes=0, seconds=30) > start_time:
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
                print(f"Finish in: ---> Episode: {episode + 1} - Steps: {step} - epsilon: {round(qlearn.epsilon, 2)}"
                      f"- Reward: {cumulated_reward} - Time: {start_time_format}")
                break

            # Finish lap every "estimate_step_per_lap"        
            if step > estimate_step_per_lap and not lap_completed:
                lap_completed = True
                if config['plotter_graphic']:
                    plotter.plot_steps_vs_epoch(stats, save=True)
                utils.save_model(outdir, qlearn, start_time_format, stats, states_counter, states_reward)
                print(f"\n\n====> LAP COMPLETED in: {datetime.now() - start_time} - episode: {episode}"
                      f" - Cum. Reward: {cumulated_reward} - step: {step}<====\n\n")

            # every 100 steps
            if counter >= config['save_every_step']: #original 1000
                if config['plotter_graphic']:
                    #plotter.plot_steps_vs_epoch(stats, save=True)
                    plotter.plot(env)
                epsilon *= epsilon_discount
                utils.save_model(outdir, qlearn, start_time_format, episode, states_counter, states_reward)
                print(f"\tSAVING MODEL in counter {counter} in episode {episode}--------> \n")
                print(f"    - N epoch:     {episode}")
                print(f"    - Model size:  {len(qlearn.q)}")
                print(f"    - Action set:  {settings.actions_set}")
                print(f"    - Epsilon:     {round(qlearn.epsilon, 2)}")
                print(f"    - Cum. reward: {cumulated_reward}")      
                print(f"    - steps: {step}")      
                print(f"    - time: {datetime.now()-start_time}\n\t")      
                counter = 0

            #  Saving model in Training TIME (2 h)
            if datetime.now() - timedelta(hours=config['Train_hours']) > start_time:
                print(config['eop'])
                utils.save_model(outdir, qlearn, start_time_format, stats, states_counter, states_reward)
                print(f"\tSAVING MODEL in time {datetime.now() - timedelta(hours=config['Train_hours'])}\n")
                print(f"    - N epoch:     {episode}")
                print(f"    - Model size:  {len(qlearn.q)}")
                print(f"    - Action set:  {settings.actions_set}")
                print(f"    - Epsilon:     {round(qlearn.epsilon, 2)}")
                print(f"    - Cum. reward: {cumulated_reward}")
                print(f"    - steps: {step}")      
                print(f"    - time: {datetime.now()-start_time}\n\t")    
                env.close()
                exit(0)
  

        ### ----------- END STEPS
        # WE SAVE BEST VALUES IN EVERY EPISODE
        ep_rewards.append(cumulated_reward)
        #if highest_reward < cumulated_reward:
        #    highest_reward = cumulated_reward 
        if not episode % config['save_episodes'] and config['save_model'] and episode >= 1:
            average_reward = sum(ep_rewards[-config['save_episodes']:]) / len(ep_rewards[-config['save_episodes']:])

            aggr_ep_rewards['episode'].append(episode)
            aggr_ep_rewards['step'].append(step)
            aggr_ep_rewards['epsilon'].append(epsilon)
            aggr_ep_rewards['avg'].append(average_reward)
            aggr_ep_rewards['max'].append(max(ep_rewards[-config['save_episodes']:]))
            aggr_ep_rewards['min'].append(min(ep_rewards[-config['save_episodes']:]))
            aggr_ep_rewards['time_training'].append(datetime.now()-start_time)
            
            print(f"\tSAVING MODEL in time {datetime.now() - timedelta(hours=config['Train_hours'])}\n")
            print(f"    - N epoch:     {episode}")
            print(f"    - Model size:  {len(qlearn.q)}")
            print(f"    - Action set:  {settings.actions_set}")
            print(f"    - Epsilon:     {round(qlearn.epsilon, 2)}")
            print(f"    - Cum. reward: {cumulated_reward}")
            print(f"    - steps: {step}")      
            print(f"    - time: {datetime.now()-start_time}\n\t") 
            #print(f"\n\tSaving STATS of model every {config['save_episodes']} episodes, we are in episode {episode}. . .\n")
            #ic()
            #ic("Saving STATS of model every")
            #ic(episode)
            utils.save_stats_episodes(outdir, aggr_ep_rewards, config, episode)
            utils.save_model(outdir, qlearn, start_time_format, stats, states_counter, states_reward)
            utils.save_tables_npy_rewards(outdir, qlearn, config, episode)


        #if episode % config['save_every_episode'] == 0 and config['plotter_graphic']:
        #    plotter.plot(env)
        #    plotter.plot_steps_vs_epoch(stats)
            # plotter.full_plot(env, stats, 2)  # optional parameter = mode (0, 1, 2)

        #if episode % config['save_episodes'] == 0 and config['save_model'] and episode > 1:  #OJO: episode % 250 originally
        #    print(f"\nSaving model every {config['save_episodes']} episodes . . .\n")
        #    utils.save_model(outdir, qlearn, start_time_format, stats, states_counter, states_reward)
        #    print(f"\tSAVING MODEL in time {datetime.now() - timedelta(hours=config['train_hours'])}\n")
        #    print(f"    - N epoch:     {episode}")
        #    print(f"    - Model size:  {len(qlearn.q)}")
        #    print(f"    - Action set:  {settings.actions_set}")
        #    print(f"    - Epsilon:     {round(qlearn.epsilon, 2)}")
        #    print(f"    - Cum. reward: {cumulated_reward}")
        #    print(f"    - steps: {step}")      
        #    print(f"    - time: {datetime.now()-start_time}\n\t") 

        m, s = divmod(int(time.time() - telemetry_start_time), 60)
        h, m = divmod(m, 60)


    # ----- STATS at the end of EPISODES
    utils.draw_rewards(aggr_ep_rewards, config, episode)

    ## ----------------- END EPISODES    
    env.close()



    '''
       ################################ END STATS ZONE
    '''
    end_time_training = time.time()
    training_time = end_time_training - start_time_training
    #print(f"\n ========= FINISH TRAINING at episode {episode} in time: {datetime.now()-start_time} ========\n\t")
    ic("========= FINISH TRAINING at episode")
    ic(episode)
    ic("in time:")
    ic(datetime.now()-start_time)
    ic(start_time_training)
    ic(end_time_training)
    ic(training_time)
    #print(f"Start Time: {start_time_training}, end time: {end_time_training}, and exec time: {training_time} in seconds")




    #print(f"Total Episodes: {total_episodes} - initial epsilon: {initial_epsilon} "
    #        f"- epsilon discount: {epsilon_discount} - Highest Reward: {highest_reward}")

    ic(total_episodes)
    ic(initial_epsilon)
    ic(epsilon_discount)
    ic(highest_reward)

    # print("Parameters: a="+str)

    if done:
        print(f" ----- OVERALL STATS")
        print("Overall score: {:0.2f}".format(last_time_steps.mean()))
    
        l = last_time_steps.tolist()
        l.sort()
        print("Best 100 scores: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

        #plotter.plot_steps_vs_epoch(stats, save=True)


