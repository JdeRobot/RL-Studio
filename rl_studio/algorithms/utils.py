######################################
#
# Common functions and classes for /algoritmhs: DDPG, Qlearn, DQN..
#
######################################
import time


def save_actorcritic_model(
    agent, global_params, algoritmhs_params, cumulated_reward, episode, text
):
    agent.actor_model.save(
        f"{global_params.models_dir}/{time.strftime('%Y%m%d')}_{algoritmhs_params.model_name}_{text}_ACTOR_Max-"
        f"{int(cumulated_reward)}_Epoch-{episode}_State-{global_params.states}_Actions-{global_params.actions}_Rewards-"
        f"{global_params.rewards}_inTime-{time.strftime('%Y%m%d-%H%M%S')}"
    )
    agent.critic_model.save(
        f"{global_params.models_dir}/{time.strftime('%Y%m%d')}_{algoritmhs_params.model_name}_{text}_CRITIC_Max-"
        f"{int(cumulated_reward)}_Epoch-{episode}_State-{global_params.states}_Actions-{global_params.actions}_Rewards-"
        f"{global_params.rewards}_inTime-{time.strftime('%Y%m%d-%H%M%S')}"
    )
    # save model in format h5
    agent.actor_model.save(
        f"{global_params.models_dir}/{time.strftime('%Y%m%d')}_{algoritmhs_params.model_name}_{text}_ACTOR_Max-"
        f"{int(cumulated_reward)}_Epoch-{episode}_State-{global_params.states}_Actions-{global_params.actions}_Rewards-"
        f"{global_params.rewards}_inTime-{time.strftime('%Y%m%d-%H%M%S')}.h5"
    )
    agent.critic_model.save(
        f"{global_params.models_dir}/{time.strftime('%Y%m%d')}_{algoritmhs_params.model_name}_{text}_CRITIC_Max-"
        f"{int(cumulated_reward)}_Epoch-{episode}_State-{global_params.states}_Actions-{global_params.actions}_Rewards-"
        f"{global_params.rewards}_inTime-{time.strftime('%Y%m%d-%H%M%S')}.h5"
    )
