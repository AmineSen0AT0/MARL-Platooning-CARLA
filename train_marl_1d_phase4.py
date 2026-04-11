import os
import time
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

# IMPORT YOUR NEW PHASE 4 ENVIRONMENT
from my_carla_env_gymnasium.carla_env_phase4 import CarlaEnv


class CentralizedPlatoonWrapper(gym.Env):
    """
    Wraps the 6-agent CARLA environment into a single monolithic Gym Env.
    It takes a 5D action (Brakes for Agents 1-5) and flattens the 5x5 state into 25D.
    """

    def __init__(self, params):
        super().__init__()
        self.carla_env = CarlaEnv(params)

        print("🧠 Wrapper loading the Frozen Horse Brain for steering...")
        model_path = "./ppo_checkpoints_finetuned/baseline_perfect_smooth_model.zip"
        self.horse_model = PPO.load(model_path, device="cuda")

        # 5 actions (Gas/Brake for Agents 1-5)
        self.action_space = spaces.Box(low=-3.0, high=3.0, shape=(5,), dtype=np.float32)
        # 5 Agents * 5D Observation Space = 25D Input
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(25,), dtype=np.float32)

        self.step_count = 0

        # --- PACE CAR STATE MACHINE ---
        self.is_disturbing = False
        self.disturbance_end_step = 0
        self.current_brake_force = 0.0

    def reset(self, seed=None, options=None):
        obs, info = self.carla_env.reset(seed=seed, options=options)
        self.step_count = 0

        # Reset Pace Car State Machine
        self.is_disturbing = False
        self.disturbance_end_step = 0
        self.current_brake_force = 0.0

        # Flatten the 5 followers' states
        followers_state = obs['state'][1:6]
        flattened_state = followers_state.flatten()
        return flattened_state, info

    def step(self, action):
        self.step_count += 1
        full_actions = []
        current_state = self.carla_env._get_obs()['state']

        # ---------------------------------------------------------
        # 1. THE PACE CAR (AGENT 0) - SMART RANDOMIZED SHOCKWAVES
        # ---------------------------------------------------------
        # Get perfect steering AND default throttle from the Horse Brain
        horse_act_0, _ = self.horse_model.predict(current_state[0][:4], deterministic=True)
        pace_action = np.copy(horse_act_0)

        # Roll the dice every 80 steps (8.0 seconds)
        if self.step_count >= 50 and self.step_count % 80 == 0:
            self.is_disturbing = True

            # Roll for intensity
            self.current_brake_force = random.choice([-3.0, 0.0])

            # Roll for duration & prepare the print label
            if self.current_brake_force == 0.0:
                duration_s = random.choice([3.0, 4.0, 5.0])
                label = "COASTING"
            else:
                duration_s = random.choice([1.0, 2.0])
                label = "HARD SLAM"

            duration_steps = int(duration_s / 0.1)
            self.disturbance_end_step = self.step_count + duration_steps

            # PRINT TO TERMINAL SO AMINE KNOWS IT IS WORKING!
            print(f"🚨 [Step {self.step_count}] Pace Car executing {label} for {duration_s} seconds!")

        # Apply the disturbance if active
        if self.is_disturbing:
            if self.step_count < self.disturbance_end_step:
                pace_action[0] = self.current_brake_force  # Override the throttle
            else:
                self.is_disturbing = False
                print(f"🏎️ [Step {self.step_count}] Pace Car disturbance ended. Accelerating back to normal.")

        full_actions.append(pace_action)

        # ---------------------------------------------------------
        # 2. THE FOLLOWERS (AGENTS 1-5) - STITCHING THE BRAINS
        # ---------------------------------------------------------
        for i in range(1, 6):
            marl_brake = action[i - 1]

            # Get Steering from the Horse Brain
            horse_act, _ = self.horse_model.predict(current_state[i][:4], deterministic=True)
            horse_steer = horse_act[1]

            stitched_action = np.array([marl_brake, horse_steer])
            full_actions.append(stitched_action)

        # ---------------------------------------------------------
        # 3. STEP THE PHYSICS ENGINE
        # ---------------------------------------------------------
        obs, reward, terminated, truncated, info = self.carla_env.step(full_actions)

        followers_state = obs['state'][1:6]
        flattened_state = followers_state.flatten()

        return flattened_state, float(reward), terminated, truncated, info

    def close(self):
        self.carla_env.close()


def main():
    print("🚀 RESUMING PHASE 4 MARL TRAINING (CENTRALIZED PLATOON)...")

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
        'town': 'Town04',
        'task_mode': 'random',
        'max_time_episode': 2000,  # Give them plenty of time to drive on the highway
        'max_waypt': 12,
        'obs_range': 32,
        'lidar_bin': 0.125,
        'd_behind': 12,
        'out_lane_thres': 2.0,
        'desired_speed': 7.3,
        'max_ego_spawn_times': 200,
        'display_route': True,
    }

    env = CentralizedPlatoonWrapper(params)

    models_dir = "./marl_v2v_fixed_collision_models_50k_penality_collision"
    logdir = "./logs_marl"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)

    # Save a checkpoint every 5000 steps so we don't lose data if it crashes!
    checkpoint_callback = CheckpointCallback(save_freq=5000, save_path=models_dir,
                                             name_prefix='marl_v2v_fixed_collision_50k_penality_collision')

    # Load the 1.5M model
    checkpoint_path = f"{models_dir}/marl_v2v_fixed_collision_50k_penality_collision_1945032_steps.zip"

    # Fallback just in case it was saved as a step checkpoint instead of 'final'

    print(f"🧠 Loading the existing MARL Brain from {checkpoint_path}...")

    # Load model and pass env + tensorboard_log to reconnect them
    model = PPO.load(checkpoint_path, env=env, tensorboard_log=logdir, device="cuda")

    print("🔥 Resuming GPU Training Loop for another 55,000 steps (Target: 2M total)...")

    # reset_num_timesteps=False prevents TensorBoard from deleting your first 300k steps!
    model.learn(total_timesteps=55000, tb_log_name="PPO_V2V_Platoon_fixed_collision_50k_penality_collision",
                callback=checkpoint_callback, reset_num_timesteps=False)

    print("💾 Training Finished! Saving 2M model...")
    model.save(f"{models_dir}/final_marl_v2v_model_2M")
    env.close()


if __name__ == '__main__':
    main()