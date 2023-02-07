# CARTPOLE

TO know about the provided env by openAI refer to the [following url](https://www.gymlibrary.dev/environments/classic_control/cart_pole/)

## ENVIRONMENT MODIFICATIONS

In RL-Studio, Cartpole environment has been tuned to permit some fine grained experiments regarding the:
    - solidity of different algorithms when the difficulty of the problem arises
    - viability of the algorithms regarding the inference cycle iteration time
    - the performance improvement of implementing continuous actions with respect to discrete actions

The modifications are made in the environments cartpole_env_improved.py and cartpole_env_continuous_improved.py.
 
Use the following parameters in the section environments of the config.yaml in order to:
### configure the desired environment, 
    
    - To use the cartpole_env_continuous_improved.py
        - env_name: myCartpole-continuous-v1
        - environment_folder: cartpole
    - To use the cartpole_env_improved.py
        - env_name: myCartpole-v1
        - environment_folder: cartpole        

### configure the difficulty of the environment, use the following parameters in the config.yaml:
    
    - random_start_level: Indicates the standard deviation in radians of the initial pole angle
    - random_perturbations_level: Number between 0 and 1 that indicates the frequency of the random perturbations.
    A perturbation of 0.1 indicates that a perturbation will be added each 10 steps. 
    - perturbations_intensity_std: Number between 0 and 1 that indicates the standard deviation of perturbations intensity.
    A perturbations_instensity_std of 1 indicates that each perturbation instensity will have a standard deviation equal to 
    the intensity of the action applied to the pole.
    - initial_pole_angle: Indicates a fixed initial pole angle in radians

### configure an automatic launching of several inference experiments with different problem configurations
    
    - settings: mode: inference
    - experiments: Number of experiments for each of the different perturbations configurations to run. 
    A value of 20 indicates that 20 experiments of initial_pole, 20 of perturbation intensity and 20 of perturbation
    frequency will be run.
    - random_perturbations_level_step: step between one experiment and the following regarding the random_perturbation_level.
    A value of 0.1 means that the first experiment will be run with 0.1 random_perturbation_level, the second one with 0.2, the third one
    with 0.3...
    perturbations_intensity_std_step: step between one experiment and the following regarding the perturbations_intensity_std.
    A value of 0.1 means that the first experiment will be run with 0.1 perturbations_intensity_std, the second one with 0.2, the third one
    with 0.3...
    initial_pole_angle_steps: step between one experiment and the following regarding the initial_pole_angle.
    A value of 0.1 means that the first experiment will be run with 0.1 initial_pole_angle, the second one with 0.2, the third one
    with 0.3...

    Note that all experiments results and logs will be stored in the logs/<problem>/<algorithm>/inference folder. 