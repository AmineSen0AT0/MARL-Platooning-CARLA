# MARL Platooning in CARLA 🚗🤖

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![CARLA Simulator](https://img.shields.io/badge/CARLA-0.9.15-orange)](https://carla.readthedocs.io/)

**Official implementation of the centralized MARL baseline presented at AIR 2026.**

This repository provides a synchronous, Gymnasium-compliant framework designed to benchmark Deep Reinforcement Learning (DRL) algorithms for continuous-control autonomous platooning in high-fidelity 3D physics environments. It specifically explores the credit assignment bottleneck in centralized architectures.

---

## 📢 Acknowledgments & Core Foundation
This framework heavily extends the original single-agent [gym-carla](https://github.com/cjy1992/gym-carla) repository created by Jianyu Chen. We deeply appreciate their foundational work in bridging OpenAI Gym with the CARLA simulator. 

While the core server-client rendering logic remains inspired by `gym-carla`, this repository introduces substantial modifications for multi-agent V2V coordination, continuous longitudinal control, and hierarchical action-space decoupling.

## ✨ Core Features
* **Hierarchical Control Decoupling:** Isolates lateral lane-keeping (via a frozen, pre-trained controller) from longitudinal acceleration, mitigating multi-dimensional action-space noise.
* **V2V State-Space Formulation:** Features a centralized 25-dimensional observation matrix fusing standard radar bounding with explicit Vehicle-to-Vehicle (V2V) shockwave telemetry.
* **Synchronous Multi-Agent Support:** Designed specifically to handle 5-vehicle platoons without asynchronous physics lag.
* **Gymnasium Compliant:** Ready to be used seamlessly with modern RL libraries like Stable-Baselines3 (PPO, SAC, TD3).

## 🛠️ Installation

**1. Install the CARLA Simulator**
Download and install [CARLA](https://github.com/carla-simulator/carla/releases) (Recommended: v0.9.13). Ensure the CARLA server is running before executing any training scripts.

**2. Clone this repository**
```bash
git clone [https://github.com/AmineSen0AT0/MARL-Platooning-CARLA.git](https://github.com/AmineSen0AT0/MARL-Platooning-CARLA.git)
cd MARL-Platooning-CARLA
