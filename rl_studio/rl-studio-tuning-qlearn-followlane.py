
import optuna
from rl_studio.agents import TrainerFactory, InferencerFactory


def objective(trial):

    config_file = {'settings': {'mode': 'training', 'task': 'follow_lane_gazebo', 
    'algorithm': 'qlearn', 'simulator': 'gazebo', 'environment_set': 'gazebo_environments', 
    'env': 'simple', 'agent': 'f1', 'actions': 'simple', 'states': 'sp1', 
    'rewards': 'follow_right_lane_only_center', 'framework': '_', 
    'total_episodes': 5, 'training_time': 6, 'models_dir': './checkpoints', 'logs_dir': './logs', 
    'metrics_dir': './metrics'}, 'ros': {'ros_master_uri': '11311', 'gazebo_master_uri': '11345'}, 
    'retraining': {'qlearn': {'retrain_qlearn_model_name':None}},
    'inference': {'qlearn': {'inference_qlearn_model_name': None}}, 
    'algorithm': {'qlearn': {'alpha': 0.2, 'epsilon': 0.95, 'epsilon_min': 0.05, 'gamma': 0.9}}, 
    'agents': {'f1': {'camera_params': {'width': 640, 'height': 480, 'center_image': 320, 'raw_image': False, 'image_resizing': 100, 
    'new_image_size': 32, 'num_regions': 16, 'lower_limit': 220}}}, 'states': {'image': {0: [3]}, 'sp1': {0: [10]}, 'sp3': {0: [5, 15, 22]}, 
    'sp5': {0: [3, 5, 10, 15, 20]}, 'spn': {0: [10]}}, 'actions': {'simple': {0: [3, 0], 1: [2, 1], 2: [2, -1]}, 
    'medium': {0: [3, 0], 1: [2, 1], 2: [2, -1], 3: [1, 1.5], 4: [1, -1.5]}, 
    'hard': {0: [3, 0], 1: [2, 1], 2: [2, -1], 3: [1.5, 1], 4: [1.5, -1], 5: [1, -1.5], 6: [1, -1.5]}, 
    'test': {0: [0, 0]}}, 
    'rewards': {'follow_right_lane_only_center': {'from_10': 10, 'from_02': 2, 'from_01': 1, 'penal': -100, 'min_reward': 5000, 'highest_reward': 100},
    'follow_right_lane_center_v_step': {'from_10': 10, 'from_02': 2, 'from_01': 1, 'penal': -100, 'min_reward': 5000, 'highest_reward': 100}, 
    'follow_right_lane_center_v_w_linear': {'beta_0': 3, 'beta_1': -0.1, 'penal': 0, 'min_reward': 1000, 'highest_reward': 100}}, 
    'gazebo_environments': {'simple': {'env_name': 'F1Env-v0', 'circuit_name': 'simple', 'launchfile': 'simple_circuit.launch', 
    'environment_folder': 'f1', 'robot_name': 'f1_renault', 'model_state_name': 'f1_renault', 'start_pose': 0, 'alternate_pose': False, 
    'estimated_steps': 100, 'sensor': 'camera', 'save_episodes': 5, 'save_every_step': 10, 'lap_completed': False, 'save_model': True, 
    'save_positions': True, 'debug_level': 'DEBUG', 'telemetry': False, 'telemetry_mask': False, 'plotter_graphic': False, 
    'circuit_positions_set': {0: [52.8, -12.734, 0.004, 0, 0, 1.57, -1.57], 1: [52.97, -42.06, 0.004, 0, 0, 1.57, -1.57], 
    2: [40.2, -30.741, 0.004, 0, 0, 1.56, 1.56], 3: [0, 31.15, 0.004, 0, 0.01, 0, 0.31], 4: [19.25, 43.5, 0.004, 0, 0.0, 1.57, -1.69], 
    5: [52.8, -35.486, 0.004, 0, 0, 1.57, -1.57]}}}}

    config_file['algorithm']['qlearn']['alpha'] = trial.suggest_float('alpha', 0.1, 1, log=True)
    config_file['algorithm']['qlearn']['gamma'] = trial.suggest_float('gamma', 0.1, 1, log=True)

    print(f"config_file: {config_file}")
    
    if config_file["settings"]["mode"] == "inference":
        inferencer = InferencerFactory(config_file)
        inferencer.main()
    else:
        trainer = TrainerFactory(config_file)
        trainer.main()


if __name__ == "__main__":

    study = optuna.create_study(direction='maximize')
    study.optimize(objective)
    trial = study.best_trial

    print(f"Accuracy = {trial.value}")
    print(f"best hypers = {trial.params}")