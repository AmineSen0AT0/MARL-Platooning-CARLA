
import os
import time
import random
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import matplotlib.pyplot as plt

# IMPORT PHASE 4 MARL ENVIRONMENT
from my_carla_env_gymnasium.carla_env_phase4 import CarlaEnv


class CentralizedPlatoonWrapper(gym.Env):
    def __init__(self, params):
        super().__init__()
        self.carla_env = CarlaEnv(params)

        print("🧠 Wrapper loading the Frozen Horse Brain for steering...")
        self.horse_model = PPO.load("./ppo_checkpoints_finetuned/baseline_perfect_smooth_model.zip", device="cuda")

        self.action_space = spaces.Box(low=-3.0, high=3.0, shape=(5,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(25,), dtype=np.float32)

        self.step_count = 0
        self.is_disturbing = False
        self.disturbance_end_step = 0
        self.current_brake_force = 0.0

    def reset(self, seed=None, options=None):
        obs, info = self.carla_env.reset(seed=seed, options=options)
        self.step_count = 0
        self.is_disturbing = False

        followers_state = obs['state'][1:6]
        return followers_state.flatten(), info

    def step(self, action):
        self.step_count += 1
        full_actions = []
        current_state = self.carla_env._get_obs()['state']

        # 1. PACE CAR LOGIC
        horse_act_0, _ = self.horse_model.predict(current_state[0][:4], deterministic=True)
        pace_action = np.copy(horse_act_0)

        if self.step_count >= 50 and self.step_count % 80 == 0:
            self.is_disturbing = True
            self.current_brake_force = random.choice([-3.0, 0.0])
            duration_s = random.choice([3.0, 4.0, 5.0]) if self.current_brake_force == 0.0 else random.choice(
                [1.0, 2.0])
            self.disturbance_end_step = self.step_count + int(duration_s / 0.1)
            label = "COASTING" if self.current_brake_force == 0.0 else "HARD SLAM"
            print(f"\n🚨 [Step {self.step_count}] Pace Car executing {label} for {duration_s} seconds!")

        if self.is_disturbing:
            if self.step_count < self.disturbance_end_step:
                pace_action[0] = self.current_brake_force
            else:
                self.is_disturbing = False

        full_actions.append(pace_action)

        # 2. FOLLOWERS LOGIC
        for i in range(1, 6):
            marl_brake = action[i - 1]
            horse_act, _ = self.horse_model.predict(current_state[i][:4], deterministic=True)
            stitched_action = np.array([marl_brake, horse_act[1]])
            full_actions.append(stitched_action)

        # 3. STEP
        obs, reward, terminated, truncated, info = self.carla_env.step(full_actions)
        followers_state = obs['state'][1:6]
        return followers_state.flatten(), float(reward), terminated, truncated, info

    def close(self):
        self.carla_env.close()


def main():
    print("👁️ INITIALIZING PHASE 4 DATA COLLECTION & VISUAL EVALUATION...")

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
        'max_time_episode': 1000,
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

    # 🛑 UPDATE THIS EXACT FILENAME IF NEEDED 🛑
    model_path = "./marl_v2v_fixed_collision_models_50k_penality_collision/marl_v2v_fixed_collision_50k_penality_collision_900032_steps.zip"

    print(f"🧠 Loading the 2M Step MARL Brain from: {model_path}")
    model = PPO.load(model_path, device="cuda")

    print("🌍 Resetting environment...")
    obs, _ = env.reset()

    terminated = False
    truncated = False

    # 🟢 THE RIGOROUS DATALOGGER
    data_log = []

    print("✅ Simulation running! Open Pygame/CARLA window to watch.")

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        current_time = env.step_count * params['dt']
        raw_state = env.carla_env._get_obs()['state']

        # Log data for all 6 cars for this exact millisecond
        for i in range(6):
            # 🟢 THE FIX: Speed is at Index [2], exactly like Phase 3!
            speed_ms = raw_state[i][2]

            data_log.append({
                "Time_s": current_time,
                "Agent_ID": i,
                "Speed_m_s": speed_ms,
                "Action_Throttle_Brake": action[i - 1] if i > 0 else env.current_brake_force
            })

        time.sleep(0.03)

    print("\n🛑 Data collection complete.")
    env.close()

    # 💾 SAVE THE DATASET FIRST!
    print("💾 Saving phase4_marl_freezing.csv...")
    df = pd.DataFrame(data_log)
    df.to_csv("phase4_marl_freezing.csv", index=False)
    print("✅ Data successfully saved! Phase 4 is officially recorded.")

    # 📈 THEN PLOT THE GRAPH DIRECTLY FROM THE DATAFRAME
    plt.figure(figsize=(10, 6))

    colors = {0: 'black', 1: 'green', 2: 'orange', 3: 'red', 4: 'darkred', 5: 'maroon'}
    labels = {
        0: 'Pace Car (Shockwave)',
        1: 'Agent 1 (Tracking)',
        2: 'Agent 2 (Risk-Averse)',
        3: 'Agent 3 (Frozen)',
        4: 'Agent 4 (Frozen)',
        5: 'Agent 5 (Frozen)'
    }
    line_styles = {0: '--', 1: '-', 2: '-', 3: '-', 4: '-', 5: '-'}

    for i in range(6):
        agent_data = df[df['Agent_ID'] == i]
        plt.plot(agent_data['Time_s'], agent_data['Speed_m_s'],
                 label=labels[i], color=colors[i], linestyle=line_styles[i], linewidth=2)

    plt.title('Phase 4: Velocity Profile Demonstrating Centralized Credit Assignment Bottleneck', fontsize=14)
    plt.xlabel('Time (Seconds)', fontsize=12)
    plt.ylabel('Velocity (m/s)', fontsize=12)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.7)

    filename = 'phase4_velocity.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"📈 Graph successfully saved as {filename}! Upload this.")
    plt.show()


if __name__ == '__main__':
    main()
