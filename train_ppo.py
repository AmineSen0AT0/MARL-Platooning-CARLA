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
    # Setup directories for logs and saving
    log_dir = "./carla_tensorboard/"
    save_dir = "./ppo_checkpoints/"
    os.makedirs(save_dir, exist_ok=True)

    params = {
        'number_of_vehicles': 0,  # Start with an empty road to learn driving first
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
        'pixor_size': 64,  # <--- ADD THIS LINE
        'pixor': False
    }

    print("1. Initializing Environment...")
    env = StateOnlyWrapper(CarlaEnv(params))

    print("2. Setting up PPO and Checkpoints...")
    # Save a checkpoint every 50,000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=save_dir,
        name_prefix="ppo_carla"
    )

    print("3. Loading PPO Brain from Checkpoint...")

    # 1. Point to your exact checkpoint file
    checkpoint_path = "./ppo_checkpoints/ppo_carla_450000_steps.zip"  # <-- Verify this name!

    # 2. LOAD the model instead of creating a new one
    model = PPO.load(
        checkpoint_path,
        env=env,
        device="cuda",  # or "cpu" if you changed it
        tensorboard_log=log_dir
    )

    print("4. Resuming Training...")
    try:
        model.learn(
            total_timesteps=550_000,  # Only run the remaining steps (1M - 450k)
            callback=checkpoint_callback,
            tb_log_name="PPO_GodMode_Run1",
            reset_num_timesteps=False  # CRUCIAL: This keeps your TensorBoard graph continuous!
        )

        print("5. Training Complete. Saving final model...")
        model.save("ppo_carla_final")

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving current progress...")
        model.save("ppo_carla_interrupted")


if __name__ == '__main__':
    main()