######################################
#
# Common functions and classes for /algoritmhs: DDPG, Qlearn, DQN..
#
######################################
import time


def save_actorcritic_model(
    agent, global_params, algoritmhs_params, environment, cumulated_reward, episode, text
):

    agent.actor_model.save(
        f"{global_params.models_dir}/{time.strftime('%Y%m%d-%H%M%S')}_{algoritmhs_params.model_name}_{text}_ACTOR"
        f"Circuit-{environment['circuit_name']}_"
        f"States-{environment['states']}_"
        f"Actions-{environment['action_space']}_"
        f"BATCH_Rewards-{environment['reward_function']}_"
        f"MaxReward-{int(cumulated_reward)}_"
        f"Epoch-{episode}"
    )    
    agent.critic_model.save(
        f"{global_params.models_dir}/{time.strftime('%Y%m%d-%H%M%S')}_{algoritmhs_params.model_name}_{text}_CRITIC"
        f"Circuit-{environment['circuit_name']}_"
        f"States-{environment['states']}_"
        f"Actions-{environment['action_space']}_"
        f"BATCH_Rewards-{environment['reward_function']}_"
        f"MaxReward-{int(cumulated_reward)}_"
        f"Epoch-{episode}"
    )

    # save model in h5 format
    agent.actor_model.save(
        f"{global_params.models_dir}/{time.strftime('%Y%m%d-%H%M%S')}_{algoritmhs_params.model_name}_{text}_ACTOR"
        f"Circuit-{environment['circuit_name']}_"
        f"States-{environment['states']}_"
        f"Actions-{environment['action_space']}_"
        f"BATCH_Rewards-{environment['reward_function']}_"
        f"MaxReward-{int(cumulated_reward)}_"
        f"Epoch-{episode}.h5"
    )    
    agent.critic_model.save(
        f"{global_params.models_dir}/{time.strftime('%Y%m%d-%H%M%S')}_{algoritmhs_params.model_name}_{text}_CRITIC"
        f"Circuit-{environment['circuit_name']}_"
        f"States-{environment['states']}_"
        f"Actions-{environment['action_space']}_"
        f"BATCH_Rewards-{environment['reward_function']}_"
        f"MaxReward-{int(cumulated_reward)}_"
        f"Epoch-{episode}.h5"
    )

