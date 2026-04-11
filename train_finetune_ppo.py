import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from my_carla_env_gymnasium.carla_env import CarlaEnv


class StateOnlyWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space.spaces['state']

    def observation(self, obs):
        return obs['state']


def main():
    log_dir = "./carla_tensorboard/"
    save_dir = "./ppo_checkpoints_finetuned/"
    os.makedirs(save_dir, exist_ok=True)

    params = {
        'number_of_vehicles': 0,
        'number_of_walkers': 0,
        'display_size': 256,
        'max_past_step': 1,
        'dt': 0.1,
        'discrete': False,
        'discrete_acc': [-3.0, 0.0, 3.0],
        'discrete_steer': [-0.2, 0.0, 0.2],
        'continuous_accel_range': [-3.0, 3.0],
        'continuous_steer_range': [-1.0, 1.0],
        'ego_vehicle_filter': 'vehicle.lincoln*',
        'port': 2000,
        'town': 'Town03',
        'task_mode': 'random',
        'max_time_episode': 1000,
        'max_waypt': 12,
        'obs_range': 32,
        'lidar_bin': 0.125,
        'd_behind': 12,
        'out_lane_thres': 2.0,
        'desired_speed': 8,
        'max_ego_spawn_times': 200,
        'display_route': True,
        'pixor_size': 64,
        'pixor': False
    }

    env = StateOnlyWrapper(CarlaEnv(params))

    # Save frequently because electricity cuts!
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=save_dir, name_prefix="finetune")

    print("Loading Golden Brain for Fine-Tuning...")
    # LOAD YOUR 780k CHECKPOINT HERE
    model = PPO.load(
        "./ppo_checkpoints/ppo_carla_780000_steps.zip",
        env=env,
        device="cuda",
        tensorboard_log=log_dir,
        # LOWER LEARNING RATE: We just want to gently adjust the steering, not rewrite the brain
        learning_rate=0.0001
    )

    print("Starting Fine-Tuning (200,000 steps)...")
    model.learn(
        total_timesteps=300_000,
        callback=checkpoint_callback,
        tb_log_name="PPO_FineTune_Smooth",
        reset_num_timesteps=True  # Set to True so it creates a fresh TensorBoard graph
    )

    model.save("baseline_perfect_smooth_model")


if __name__ == '__main__':
    main()