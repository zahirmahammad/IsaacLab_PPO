import torch
import torch.nn as nn
import imageio
import numpy as np
import cv2
import os
from dataclasses import dataclass
from typing import Tuple
import gymnasium as gym
from pathlib import Path
import argparse
import matplotlib.pyplot as plt


from isaaclab.source.isaaclab.isaaclab.app import AppLauncher
# import isaaclab_tasks
# Launch Isaac Sim
app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

from isaaclab.source.isaaclab_tasks.isaaclab_tasks.utils import load_cfg_from_registry
import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from pxr import UsdGeom, Gf
import omni.usd
from isaaclab.assets import AssetBaseCfg
import BDXR.tasks  # noqa: F401

# ===================================================================================
# CONFIGURATION CLASS
# ===================================================================================
@dataclass
class TrainingConfig:
    # Environment Parameters
    env_name: str
    num_envs: int = 256
    timesteps: int = 256
    # num_eval_envs: int = 20
    device: str = "cuda"

    # Training Hyperparameters
    total_timesteps: int = int(2e7)
    num_epochs: int = 10
    batch_size: int = 1024
    lr: float = 5e-5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5

    # Regularization
    action_mag_penalty_coef: float = 0.5

    # outputs
    output_dir: str = None
    num_evals: int = 4
    test_freq: int = 4

    def __post_init__(self):
        self.output_dir = f"outputs/{self.env_name}"

# ===================================================================================
# AGENT NUERAL NETWORK
# ===================================================================================

class Agent(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()

        # observation normalization buffer
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_var", torch.ones(obs_dim))
        self.register_buffer("obs_count", torch.tensor(0.0))

        # Actor Network
        self.actor_net = nn.Sequential(
            self._layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            self._layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            self._layer_init(nn.Linear(256, 128)),
            nn.Tanh(),
            self._layer_init(nn.Linear(128, action_dim), std=0.01),
        )

        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

        # Critic Network
        self.critic_net = nn.Sequential(
            self._layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            self._layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            self._layer_init(nn.Linear(256, 128)),
            nn.Tanh(),
            self._layer_init(nn.Linear(128, 1), std=1.0),
        )

    @staticmethod
    def _layer_init(layer, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer

    def update_obs_stats(self, obs: torch.Tensor) -> None:
        '''Update the global mean and variance with the incoming new data'''
        batch_mean = obs.mean(dim=0)
        batch_var = obs.var(dim=0, unbiased=False)
        batch_count = obs.shape[0]

        total_count = self.obs_count + batch_count
        
        # mean = sum / N
        # new_mean = S1 + S2 / N1 + N2
        self.obs_mean = ((self.obs_mean * self.obs_count) + (batch_mean * batch_count)) / total_count

        # var = sum(x - u)^2 / N
        m_1 = self.obs_var * self.obs_count
        m_2 = batch_var * batch_count
        m = m_1 + m_2 + ((batch_mean - self.obs_mean)**2 * self.obs_count * batch_count) / (total_count)
        self.obs_var = m / total_count
        self.obs_count = total_count

    def normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return (obs - self.obs_mean) / torch.sqrt(self.obs_var + 1e-6)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        obs_norm = self.normalize_obs(obs)
        return self.critic_net(obs_norm)

    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor = None):
        obs_norm = self.normalize_obs(obs)

        mean = self.actor_net(obs_norm)
        log_std = torch.clamp(self.actor_logstd, -5, 2)
        log_std = log_std.expand_as(mean)
        std = torch.exp(log_std)

        dist = torch.distributions.Normal(mean, std)
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)

        return action, log_prob, entropy, self.get_value(obs)



# ===================================================================================
# TRAINING CLASS
# ===================================================================================

class PPOTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config

        self.env = self._create_environment()
        self.obs_dim = self.env.observation_space["policy"].shape[1]
        self.action_dim = self.env.action_space.shape[1]

        # agent
        self.agent = Agent(self.obs_dim, self.action_dim).to(self.config.device)
        self.optimizer = torch.optim.Adam(self.agent.parameters(), self.config.lr)

        print(f"{'-'*60}")
        print(f"✓ Environment: {config.env_name}")
        print(f"✓ Observation space: {self.obs_dim}")
        print(f"✓ Action space: {self.action_dim}")
        print(f"{'-'*60}")


    def _create_environment(self) -> gym.Env:
        cfg = load_cfg_from_registry(self.config.env_name, "env_cfg_entry_point")
        cfg.scene.num_envs = self.config.num_envs

        env = gym.make(self.config.env_name, cfg=cfg, render_mode='rgb_array')
        env = gym.wrappers.OrderEnforcing(env)
        return env

    def train(self) -> None:
        num_runs = int(self.config.total_timesteps // (self.config.num_envs * self.config.timesteps))
        divisor = max(1, int(num_runs // self.config.test_freq))

        done = torch.zeros(self.config.num_envs).to(self.config.device)

        # Track returns for plotting
        run_numbers = []
        returns_list = []

        # rollout storage
        obs_arr = torch.zeros(self.config.timesteps, self.config.num_envs, self.obs_dim).to(self.config.device)
        actions_arr = torch.zeros(self.config.timesteps, self.config.num_envs, self.action_dim).to(self.config.device)
        logprobs_arr = torch.zeros(self.config.timesteps, self.config.num_envs).to(self.config.device)
        rewards_arr = torch.zeros(self.config.timesteps, self.config.num_envs).to(self.config.device)
        done_arr = torch.zeros(self.config.timesteps, self.config.num_envs).to(self.config.device)
        values_arr = torch.zeros(self.config.timesteps, self.config.num_envs).to(self.config.device)

        # Get Rollout
        for run in range(num_runs):
            obs, _ = self.env.reset()
            obs = obs["policy"]

            for step in range(self.config.timesteps):
                with torch.no_grad():
                    action, log_prob, _, value = self.agent.get_action_and_value(obs)
                
                obs_arr[step] = obs
                done_arr[step] = done
                actions_arr[step] = action
                logprobs_arr[step] = log_prob
                values_arr[step] = value.flatten()

                obs, reward, terminated, truncated, _ = self.env.step(action)
                rewards_arr[step] = reward
                obs = obs["policy"]
                done = (terminated | truncated).float().to(self.config.device)


            # Compute advantages and returns
            advantages = torch.zeros_like(rewards_arr).to(self.config.device)
            lastgaelam = torch.zeros(self.config.num_envs).to(self.config.device)
            for step in reversed(range(self.config.timesteps)):
                if step == self.config.timesteps - 1:
                    next_not_done = 1 - done
                    with torch.no_grad():
                        next_value = self.agent.get_value(obs).flatten()
                else:
                    next_not_done = 1 - done_arr[step + 1]
                    next_value = values_arr[step + 1] 
                delta = rewards_arr[step] + self.config.gamma * next_not_done * next_value - values_arr[step]
                lastgaelam = delta + self.config.gamma * self.config.gae_lambda * next_not_done * lastgaelam
                advantages[step] = lastgaelam
            returns = advantages + values_arr

            # data preparation
            flat_obs = obs_arr.reshape(-1, self.obs_dim).to(self.config.device)
            flat_action = actions_arr.reshape(-1, self.action_dim).to(self.config.device)
            flat_logprobs = logprobs_arr.reshape(-1).to(self.config.device)
            flat_returns = returns.reshape(-1).to(self.config.device)
            flat_advantages = advantages.reshape(-1).to(self.config.device)

            # Update normalization statistics
            self.agent.update_obs_stats(flat_obs)

            advantages_norm = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)

            data_size = self.config.num_envs * self.config.timesteps
            losses = []

            # PPO update
            for epoch in range(self.config.num_epochs):
                indices = np.arange(data_size)
                np.random.shuffle(indices)

                for start_idx in range(0, data_size, self.config.batch_size):
                    end_idx = start_idx + self.config.batch_size
                    batch_indices = indices[start_idx:end_idx]

                    _, new_logprobs, entropy, new_values = self.agent.get_action_and_value(flat_obs[batch_indices], flat_action[batch_indices])

                    log_ratio = new_logprobs - flat_logprobs[batch_indices]
                    ratio = torch.exp(log_ratio)

                    batch_advantages = advantages_norm[batch_indices]

                    # policy gradient with clipping
                    objective_pg = torch.min(ratio * batch_advantages, torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * batch_advantages)
                    loss_pg = -objective_pg.mean()

                    # Value function loss
                    loss_vf = 0.5 * ((new_values.flatten() - flat_returns[batch_indices])**2).mean()

                    action_penalty = torch.mean(torch.abs(flat_action[batch_indices])) * self.config.action_mag_penalty_coef
            
                    total_loss = loss_pg + loss_vf - self.config.entropy_coef * entropy.mean() + action_penalty
                    losses.append(total_loss.item())

                    self.optimizer.zero_grad()
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()

            loss_mean = np.mean(losses)
            current_return = returns.mean().item()
            run_numbers.append(run + 1)
            returns_list.append(current_return)
            print(f"Run : {run+1:4d}/{num_runs} | Loss: {loss_mean:.4f} | Return: {current_return:.4f}")

            # Periodic evaluation
            if (run + 1) % divisor == 0:
                self.evaluate(run + 1)

        # Save training returns plot
        self._save_training_plot(run_numbers, returns_list)

        self.env.close()
        print(f"\n{'='*60}")
        print("Training complete!")
        print(f"{'='*60}\n")


    def evaluate(self, run) -> None:
        print(f"\n{'-'*60}")
        print(f"Evaluation at Run {run}")
        print(f"{'-'*60}")
        frames = []
        for episode in range(5):
            obs, _ = self.env.reset()
            obs = obs["policy"]
            total_reward = 0
            
            for _ in range(self.config.timesteps):
                with torch.no_grad():
                    obs_norm = self.agent.normalize_obs(obs)
                    action = self.agent.actor_net(obs_norm)
                
                obs, reward, terminated, truncated, _ = self.env.step(action)
                obs = obs["policy"]
                total_reward += reward
                
                # frame = self.env.render()
                # frame = (frame * 255).astype('uint8') if frame.dtype != 'uint8' else frame
                # frame = np.ascontiguousarray(frame)
                # cv2.putText(frame, f"Episode: {episode}", (20, 40),
                #            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
                # frames.append(frame)
            
            print(f"Episode {episode+1}: Reward = {total_reward.mean().item():.4f}")
        
        # Save video and checkpoint
        video_path = f"{self.config.output_dir}/videos"
        model_path = f"{self.config.output_dir}/models"

        Path(video_path).mkdir(parents=True, exist_ok=True)
        Path(model_path).mkdir(parents=True, exist_ok=True)

        # imageio.mimsave(f"{video_path}/{run}.mp4", frames, fps=30)
        torch.save(self.agent.state_dict(), f"{model_path}/{run}.pth")
        
        print(f"✓ Video saved: {video_path}")
        print(f"✓ Model saved: {model_path}\n")

    def _save_training_plot(self, run_numbers, returns_list) -> None:
        """Generate and save training returns plot as image"""
        plt.figure(figsize=(12, 6))
        plt.plot(run_numbers, returns_list, linewidth=2, marker='o', markersize=4)
        plt.xlabel('Run Number', fontsize=12)
        plt.ylabel('Return', fontsize=12)
        plt.title(f'Training Returns - {self.config.env_name}', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        normal_path = f"{self.config.output_dir}"
        Path(normal_path).mkdir(parents=True, exist_ok=True)
        plot_path = f"{normal_path}/training_returns.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Training plot saved: {plot_path}")


# ===================================================================================
# TESTING AND ROLLOUTS
# ===================================================================================

class PPOEvals:
    def __init__(self, config: TrainingConfig):
        self.config = config

        self.env = self._create_environment()

        self.obs_dim = self.env.observation_space["policy"].shape[1]
        self.action_dim = self.env.action_space.shape[1]

        # model_path = f"{self.config.output_dir}/models/760.pth"
        model_path = f"{self.config.output_dir}/models/304.pth"
        self.agent = Agent(self.obs_dim, self.action_dim).to(self.config.device)
        self.agent.load_state_dict(torch.load(model_path, map_location=self.config.device))




    def _create_environment(self) -> gym.Env:
        self.cfg = load_cfg_from_registry(self.config.env_name, "env_cfg_entry_point")
        self.cfg.scene.num_envs = 1

        self.cfg.scene.follow_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/follow_cam",   # 👈 NOT on robot
            update_period=0.0,
            height=1080,
            width=1080,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                # focal_length=50.0,          # 👈 tighter, less distortion
                # focus_distance=200.0,         # 👈 realistic (not 400)
                horizontal_aperture=20.955,
            ),
        )

        # ambient (dome)
        self.cfg.scene.dome_light = AssetBaseCfg(
            prim_path="/World/DomeLight",
            spawn=sim_utils.DomeLightCfg(
                intensity=200.0,
                color=(0.9, 0.9, 0.9),
            ),
        )

        # directional (sun)
        self.cfg.scene.sun_light = AssetBaseCfg(
            prim_path="/World/SunLight",
            spawn=sim_utils.DistantLightCfg(
                intensity=2000.0,
                angle=0.3,
            ),
        )
        # self.cfg.scene.dome_light.intensity = 3000.0
        # self.cfg.scene.dome_light.color = (1.0, 1.0, 1.0)

        env = gym.make(self.config.env_name, cfg=self.cfg, render_mode='rgb_array')
        env = gym.wrappers.OrderEnforcing(env)
        return env

    def _setup_camera(self, eye, target):
        """Camera will be handled by environment's built-in rendering"""
        self.cfg.viewer.eye = tuple(map(float, eye))
        self.cfg.viewer.lookat = tuple(map(float, target))



    def update_follow_cam(self, eye, target):
        stage = omni.usd.get_context().get_stage()
        cam_prim = stage.GetPrimAtPath("/World/envs/env_0/follow_cam")
        xform = UsdGeom.Xformable(cam_prim)
        xform.ClearXformOpOrder()
        eye = Gf.Vec3d(*map(float, eye))
        target = Gf.Vec3d(*map(float, target))
        up = Gf.Vec3d(0.0, 0.0, 1.0)
        view = Gf.Matrix4d().SetLookAt(eye, target, up)
        xform.AddTransformOp().Set(view.GetInverse())



    def _run_inference(self) -> None:
        print(f"\n{'-'*60}")
        print(f"Evaluation: {self.config.env_name}")
        print(f"{'-'*60}")
        frames = []
        for episode in range(5):
            obs, _ = self.env.reset()
            obs = obs["policy"]
            total_reward = 0

            robot = self.env.unwrapped.scene["robot"]
            pose = robot.data.root_state_w
            print(f"{'='*60}")
            print(f"Episode: {episode+1}")
            print(f"Current Robot Position: {pose[:, :3]}")
            print(f"{'-'*60}")

            for step in range(self.config.timesteps):
                with torch.no_grad():
                    obs_norm = self.agent.normalize_obs(obs)
                    action = self.agent.actor_net(obs_norm)

                obs, reward, terminated, truncated, info = self.env.step(action)
                obs = obs["policy"]
                total_reward += reward
            
                pose = robot.data.root_state_w
                target = pose[0, :3].cpu().numpy()
                eye = target + [-3.0, -2.0, 1.5]
                self.update_follow_cam(eye, target)

                cam = self.env.unwrapped.scene["follow_cam"]
                frame = cam.data.output["rgb"][0].cpu().numpy()

                # frame = self.env.render()
                frame = (frame * 255).astype('uint8') if frame.dtype != 'uint8' else frame
                frame = np.ascontiguousarray(frame)
                cv2.putText(frame, f"Episode: {episode}", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
                frames.append(frame)
            
            print(f"Episode {episode+1}: Reward = {total_reward.mean().item():.4f}")
        
        # Save video and checkpoint
        video_path = f"{self.config.output_dir}/videos"
        model_path = f"{self.config.output_dir}/models"

        Path(video_path).mkdir(parents=True, exist_ok=True)
        Path(model_path).mkdir(parents=True, exist_ok=True)

        imageio.mimsave(f"{video_path}/eval.mp4", frames, fps=30)
        torch.save(self.agent.state_dict(), f"{model_path}/eval.pth")
        
        print(f"✓ Video saved: {video_path}")
        print(f"✓ Model saved: {model_path}\n")

        self.env.close()





def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="PPO Training for Isaac Lab")
    parser.add_argument("--env", type=str, default="Isaac-Lift-Cube-Franka-v0",
                       help="Environment name (e.g., Isaac-Humanoid-v0)")
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train",
                       help="Run mode: train or test")
    args = parser.parse_args()
    
    # Create config with command-line env_name
    config = TrainingConfig(env_name=args.env)

    if args.mode == "train":
        print(f"\n Starting TRAINING mode with {args.env}")
        ppotrainer = PPOTrainer(config)
        ppotrainer.train()
    else:
        print(f"\n🎮 Starting TEST mode with {args.env}")
        ppotester = PPOEvals(config)
        ppotester._run_inference()
    
    simulation_app.close()

    


if __name__ == '__main__':
    main()
