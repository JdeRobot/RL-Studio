######################################
#
# Common functions and classes for /algoritmhs: DDPG, Qlearn, DQN..
#
######################################
import time


def save_actorcritic_model(
    agent, global_params, algoritmhs_params, environment, cumulated_reward, episode, text
):

    timestamp = time.strftime('%Y%m%d-%H%M%S')
    agent.actor_model.save(
        f"{global_params.models_dir}/{timestamp}_{text}_"
        f"S-{environment['states']}_"
        f"A-{environment['action_space']}_"
        f"MR-{int(cumulated_reward)}_"
        f"E-{episode}/ACTOR"
    )    
    agent.critic_model.save(
        f"{global_params.models_dir}/{timestamp}_{text}_"
        f"S-{environment['states']}_"
        f"A-{environment['action_space']}_"
        f"MR-{int(cumulated_reward)}_"
        f"E-{episode}/CRITIC"
    )


