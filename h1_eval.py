import argparse
import os
import pickle
from importlib import metadata

import torch

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="h1-walking")
    parser.add_argument("--ckpt", type=int, default=100)
    parser.add_argument("--no_viewer", action="store_true", help="Disable viewer during evaluation")
    args = parser.parse_args()

    gs.init(logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    
    # Check if config file exists
    config_path = f"logs/{args.exp_name}/cfgs.pkl"
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        print(f"Available experiments in logs/:")
        if os.path.exists("logs"):
            for exp in os.listdir("logs"):
                if os.path.isdir(f"logs/{exp}"):
                    print(f"  - {exp}")
        return
    
    # Load configuration
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(config_path, "rb"))
    
    # Disable reward computation for evaluation
    reward_cfg["reward_scales"] = {}

    # Create environment with single instance for evaluation
    env = H1Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=not args.no_viewer,  # Default viewer ON, use --no_viewer to disable
    )

    # Load trained model
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    
    if not os.path.exists(resume_path):
        print(f"Error: Model checkpoint not found at {resume_path}")
        print(f"Available checkpoints in {log_dir}:")
        if os.path.exists(log_dir):
            for file in os.listdir(log_dir):
                if file.startswith("model_") and file.endswith(".pt"):
                    print(f"  - {file}")
        return
    
    print(f"Loading model from: {resume_path}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=gs.device)

    print("Starting H1 humanoid evaluation...")
    print("Press Ctrl+C to stop evaluation")
    
    # Reset environment and start evaluation loop
    obs, _ = env.reset()
    step_count = 0
    
    with torch.no_grad():
        try:
            while True:
                # Get action from trained policy
                actions = policy(obs)
                
                # Step environment
                obs, rews, dones, infos = env.step(actions)
                step_count += 1
                
                # Print progress every 100 steps
                if step_count % 100 == 0:
                    print(f"Evaluation step: {step_count}")
                
                # Reset if episode is done
                if dones.any():
                    print(f"Episode completed at step {step_count}")
                    obs, _ = env.reset()
                    step_count = 0
                    
        except KeyboardInterrupt:
            print(f"\nEvaluation stopped at step {step_count}")
            print("Evaluation completed.")


if __name__ == "__main__":
    main()

"""
# evaluation with viewer (default)
python h1_eval.py -e h1-walking-with-viewer --ckpt 100

# evaluation without viewer
python h1_eval.py -e h1-walking-with-viewer --ckpt 100 --no_viewer

# evaluation with specific experiment and checkpoint
python h1_eval.py -e h1-walking-v2 --ckpt 200
"""
