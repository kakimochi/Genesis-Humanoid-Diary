import argparse
import os
import pickle
import shutil
from importlib import metadata

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from e
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from h1_env import H1Env


def get_train_cfg(exp_name, max_iterations):
    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
            "class_name": "ActorCritic",
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 24,
        "save_interval": 100,
        "empirical_normalization": None,
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 10,  # H1 leg joints only (5 per leg)
        # joint/link names - H1 leg joints only for walking
        "default_joint_angles": {  # [rad]
            "left_hip_yaw_joint": 0.0,
            "left_hip_roll_joint": 0.0,
            "left_hip_pitch_joint": -0.4,
            "left_knee_joint": 0.8,
            "left_ankle_joint": -0.4,
            "right_hip_yaw_joint": 0.0,
            "right_hip_roll_joint": 0.0,
            "right_hip_pitch_joint": -0.4,
            "right_knee_joint": 0.8,
            "right_ankle_joint": -0.4,
        },
        "joint_names": [
            "left_hip_yaw_joint",
            "left_hip_roll_joint",
            "left_hip_pitch_joint",
            "left_knee_joint",
            "left_ankle_joint",
            "right_hip_yaw_joint",
            "right_hip_roll_joint",
            "right_hip_pitch_joint",
            "right_knee_joint",
            "right_ankle_joint",
        ],
        # Joint limits from URDF
        "joint_limits": {
            "left_hip_yaw_joint": [-0.43, 0.43],
            "left_hip_roll_joint": [-0.43, 0.43],
            "left_hip_pitch_joint": [-3.14, 2.53],
            "left_knee_joint": [-0.26, 2.05],
            "left_ankle_joint": [-0.87, 0.52],
            "right_hip_yaw_joint": [-0.43, 0.43],
            "right_hip_roll_joint": [-0.43, 0.43],
            "right_hip_pitch_joint": [-3.14, 2.53],
            "right_knee_joint": [-0.26, 2.05],
            "right_ankle_joint": [-0.87, 0.52],
        },
        # PD control parameters
        "kp": 100.0,  # Higher for humanoid stability
        "kd": 2.0,
        # termination conditions
        "termination_if_roll_greater_than": 15,  # degree
        "termination_if_pitch_greater_than": 15,
        "termination_if_height_lower_than": 0.5,  # meters
        # base pose
        "base_init_pos": [0.0, 0.0, 1.0],  # H1 standing height
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.5,  # Larger scale for humanoid
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    obs_cfg = {
        "num_obs": 39,  # 3 + 3 + 3 + 10 + 10 + 10 = 39
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 1.0,  # Target standing height for H1
        "reward_scales": {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.5,
            "lin_vel_z": -2.0,
            "base_height": -10.0,
            "action_rate": -0.01,
            "similar_to_default": -0.1,
            "upright": -5.0,
            "joint_limits": -1.0,
        },
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.0, 1.0],  # Forward walking speed
        "lin_vel_y_range": [-0.5, 0.5],  # Side stepping
        "ang_vel_range": [-0.5, 0.5],  # Turning
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="h1-walking")
    parser.add_argument("-B", "--num_envs", type=int, default=2048)  # Smaller for humanoid
    parser.add_argument("--max_iterations", type=int, default=500)  # More iterations for humanoid
    parser.add_argument("--no_viewer", action="store_true", help="Disable viewer during training")
    args = parser.parse_args()

    gs.init(logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    env = H1Env(
        num_envs=args.num_envs, 
        env_cfg=env_cfg, 
        obs_cfg=obs_cfg, 
        reward_cfg=reward_cfg, 
        command_cfg=command_cfg,
        show_viewer=not args.no_viewer  # Default viewer ON, use --no_viewer to disable
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

"""
# training (with viewer by default)
python h1_train.py

# training without viewer
python h1_train.py --no_viewer

# training with custom parameters
python h1_train.py -e h1-walking-v2 -B 1024 --max_iterations 1000
"""
