import gymnasium as gym
import imageio
import torch
import torch.nn as nn
from dataclasses import dataclass
import cv2
import numpy as np
from pathlib import Path

from isaaclab.source.isaaclab.isaaclab.app import AppLauncher
# import isaaclab_tasks
# Launch Isaac Sim
app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

from isaaclab.source.isaaclab_tasks.isaaclab_tasks.utils import load_cfg_from_registry
import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCameraCfg, TiledCamera
# from pxr import UsdGeom, Gf
# import omni.usd
# from isaaclab.assets import AssetBaseCfg

# ===================================================================================
# CONFIGURATION CLASS
# ===================================================================================
@dataclass
class TrainConfig:
    env_name: str = "Isaac-Reach-Franka-v0"
    num_envs: int = 50 
    num_epochs: int = 10
    total_timesteps: int = 1e6
    timesteps: int = 100
    batch_size: int = 32
    eps: float = 0.2         # clipratio for ppo
    action_mag_pen_coef: float = 0.5

    gamma: float = 0.99
    device: str = "cuda"
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    lr: float = 3e-5              

# ===================================================================================
# AGENT NUERAL NETWORK
# ===================================================================================

class AgentImage(nn.Module):
    def __init__(self, image_shape, action_dim):
        super().__init__()

        c, h, w = image_shape

        # Shared CNN encoder
        self.encoder = nn.Sequential(
            self.layer_init_(nn.Conv2d(c, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            self.layer_init_(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            self.layer_init_(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten()
        )

        # Automatically compute feature dim
        with torch.no_grad():
            sample = torch.zeros(1, c, h, w)
            n_flatten = self.encoder(sample).shape[1]

        # Shared latent
        self.shared_fc = nn.Sequential(
            self.layer_init_(nn.Linear(n_flatten, 512)),
            nn.ReLU()
        )

        # Actor head
        self.actor_mean = self.layer_init_(nn.Linear(512, action_dim), std=0.01)

        # Critic head
        self.critic = self.layer_init_(nn.Linear(512, 1), std=1.0)

        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def layer_init_(self, layer, std: float = np.sqrt(2), bias_const: float = 0.0):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer
        
    def encode(self, x):
        x = self.encoder(x)
        x = self.shared_fc(x)
        return x
    
    def get_value(self, x):
        x = self.encode(x)
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        x = self.encode(x)
        mean = self.actor_mean(x)
        log_std = torch.clamp(self.actor_logstd, -5, 2)
        log_std = log_std.expand_as(mean)
        std = torch.exp(log_std)

        dist = torch.distributions.Normal(mean, std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action, log_prob, entropy, self.critic(x)

    

class Panda_PPO_Image:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.env = self._create_environment()

        self.obs_shape = (3, 224, 224)
        self.action_dim = self.env.action_space.shape
        print(f"Action Dim: {self.action_dim}")
        # self.agent = AgentImage(self.obs_shape, self.action_dim[0]).to(cfg.device)
        # self.optimizer = torch.optim.Adam(self.agent.parameters(), self.cfg.lr)

    def _create_environment(self) -> gym.Env:
        cfg = load_cfg_from_registry(self.cfg.env_name, "env_cfg_entry_point")
        cfg.scene.num_envs = self.cfg.num_envs

        cfg.scene.tiled_camera = TiledCameraCfg(
            prim_path="/World/envs/env_.*/Robot/Camera",
            offset=TiledCameraCfg.OffsetCfg(pos=(1.0, 0.0, 1.0), rot=(0, -0.3801884, 0, 0.9249091), convention="world"),
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
            ),
            width=80,
            height=80,
        )
        # tiled_camera = TiledCamera(my_tiled_camera)
        # data_type = "rgb"
        # env.reset()
        # data = tiled_camera.data.output[data_type]
        env = gym.make(self.cfg.env_name, cfg=cfg, render_mode='rgb_array')
        env = gym.wrappers.OrderEnforcing(env)
        return env
    
    def test_random_actions(self) -> None:
        """Test environment with random actions and save video"""
        print(f"\n{'-'*60}")
        print(f"Testing Random Actions: {self.cfg.env_name}")
        print(f"{'-'*60}\n")
        
        total_rewards_per_episode = []
        all_frames = []
        
        for episode in range(5):
            obs, _ = self.env.reset()
            cam = self.env.unwrapped.scene["tiled_camera"]
            frame = cam.data.output["rgb"][-1].cpu().numpy()
            cv2.imwrite("frame.png", frame)
            print(cam.data.output["rgb"].shape)
            print(f"Camera dimension: {frame.shape}")
            if isinstance(obs, dict):
                obs = obs["policy"]
            
            total_reward = torch.zeros(self.cfg.num_envs).to(self.cfg.device)
            episode_frames = []
            
            print(f"Episode {episode + 1}")
            
            for step in range(self.cfg.timesteps):
                # Sample random actions from action space
                action = torch.tensor(self.env.action_space.sample(), dtype=torch.float32, device=self.cfg.device)
                
                # Handle both single and multi-environment cases
                if self.cfg.num_envs > 1 and action.dim() == 1:
                    action = action.unsqueeze(0).repeat(self.num_envs, 1)
                
                # Step environment
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                if isinstance(obs, dict):
                    obs = obs["policy"]
                
                # Convert reward to tensor if needed
                if not isinstance(reward, torch.Tensor):
                    reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
                
                total_reward += reward
                
                # Capture frame from follow camera
                # try:
                #     robot = self.env.unwrapped.scene["robot"]
                #     pose = robot.data.root_state_w
                #     target = pose[0, :3].cpu().numpy()
                #     eye = target + np.array([-3.0, -2.0, 1.5])
                #     self.update_follow_cam(eye, target)

                #     cam = self.env.unwrapped.scene["follow_cam"]
                #     frame = cam.data.output["rgb"][0].cpu().numpy()
                #     frame = (frame * 255).astype('uint8') if frame.dtype != 'uint8' else frame
                #     frame = np.ascontiguousarray(frame)
                    
                #     # Add text to frame
                #     cv2.putText(frame, f"Episode: {episode + 1} | Step: {step + 1}", (20, 40),
                #                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                #     cv2.putText(frame, f"Reward: {total_reward.mean().item():.4f}", (20, 100),
                #                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
                #     episode_frames.append(frame)
                #     all_frames.append(frame)
                # except Exception as e:
                #     print(f"Warning: Could not capture frame at step {step}: {e}")
            
            avg_reward = total_reward.mean().item()
            total_rewards_per_episode.append(avg_reward)
            print(f"  Average Reward: {avg_reward:.4f}\n")
        
        self.env.close()
        
        # Save video
        # if all_frames:
        #     video_path = f"outputs/{self.env_name}/videos"
        #     Path(video_path).mkdir(parents=True, exist_ok=True)
            
        #     video_file = f"{video_path}/random_actions.mp4"
        #     imageio.mimsave(video_file, all_frames, fps=30)
        #     print(f"✓ Video saved: {video_file}\n")
        
        # Print summary
        print(f"\n{'='*60}")
        print("Test Summary")
        print(f"{'='*60}")
        print(f"Mean Reward across episodes: {np.mean(total_rewards_per_episode):.4f}")
        print(f"Std Reward: {np.std(total_rewards_per_episode):.4f}")
        print(f"Max Reward: {np.max(total_rewards_per_episode):.4f}")
        print(f"Min Reward: {np.min(total_rewards_per_episode):.4f}")
        print(f"{'='*60}\n")




    def get_img_obs(self):
        frames = self.env.render()
        frames = np.array(frames)        
        # Handle both single and multiple frames
        if frames.ndim == 3:  # Single frame (H, W, C)
            frames = frames[np.newaxis, ...]  # Add batch dimension
        # Resize all frames
        num_envs = frames.shape[0]
        resized_frames = []
        for i in range(num_envs):
            frame = cv2.resize(frames[i], (self.obs_shape[2], self.obs_shape[1]))
            resized_frames.append(frame)
        
        frames = np.stack(resized_frames)  # [num_envs, H, W, C]
        frames = torch.from_numpy(frames)
        frames = frames.float() / 255.0  # Normalize to [0, 1]
        frames = frames.permute(0, 3, 1, 2)  # [num_envs, C, H, W]
        
        return frames.to(self.cfg.device)
    
    def train(self):
        num_runs = int(self.cfg.total_timesteps // (self.cfg.timesteps * self.cfg.num_envs))

        # rollout storage
        obs_arr = torch.zeros(self.cfg.timesteps, self.cfg.num_envs, *self.obs_shape)
        actions_arr = torch.zeros(self.cfg.timesteps, self.cfg.num_envs, *self.action_dim)
        dones_arr = torch.zeros(self.cfg.timesteps, self.cfg.num_envs)
        log_prob_arr = torch.zeros(self.cfg.timesteps, self.cfg.num_envs)
        values_arr = torch.zeros(self.cfg.timesteps, self.cfg.num_envs)
        rewards_arr = torch.zeros(self.cfg.timesteps, self.cfg.num_envs)

        for run in range(num_runs):
            _, _ = self.env.reset()
            obs = self.get_img_obs()
            done = torch.zeros(self.cfg.num_envs).to(self.cfg.device)
            for t in range(self.cfg.timesteps):
                with torch.no_grad():
                    action, log_prob, _, value = self.agent.get_action_and_value(obs)

                obs_arr[t] = obs.cpu()
                actions_arr[t] = action.cpu()
                dones_arr[t] = done.cpu()
                log_prob_arr[t] = log_prob.cpu()
                values_arr[t] = value.flatten().cpu()
                
                # Step all environments with their corresponding actions
                _, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy())
                rewards_arr[t] = torch.from_numpy(reward).float()
                
                obs = self.get_img_obs()
                done = torch.from_numpy((terminated | truncated).astype(np.float32))
            
            # calculate returns
            advantages = torch.zeros_like(rewards_arr)
            lastgaelam = torch.zeros(self.cfg.num_envs)
            for step in reversed(range(self.cfg.timesteps)):
                if step == self.cfg.timesteps - 1:
                    next_not_done = 1 - done.cpu()
                    with torch.no_grad():
                        next_value = self.agent.get_value(obs).detach().cpu().flatten()
                else:
                    next_not_done = 1 - dones_arr[step+1]
                    next_value = values_arr[step+1]
                value_error = rewards_arr[step] + self.cfg.gamma * next_not_done * next_value - values_arr[step]
                lastgaelam = value_error + self.cfg.gamma * self.cfg.gae_lambda * next_not_done * lastgaelam
                advantages[step] = lastgaelam
            returns = advantages + values_arr

            # flatten
            flat_obs = obs_arr.reshape(-1, *self.obs_shape).to(self.cfg.device)
            flat_actions = actions_arr.reshape(-1, *self.action_dim).to(self.cfg.device)
            flat_advantages = advantages.reshape(-1).to(self.cfg.device)
            flat_log_probs = log_prob_arr.reshape(-1).to(self.cfg.device)
            flat_returns = returns.reshape(-1).to(self.cfg.device)

            # do the network update
            advatnages_norm = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)

            data_size = self.cfg.num_envs * self.cfg.timesteps
            losses = []

            for ep in range(self.cfg.num_epochs):        
                indices = np.arange(data_size)
                np.random.shuffle(indices)

                for start_idx in range(0, data_size, self.cfg.batch_size):
                    end_idx = start_idx + self.cfg.batch_size
                    b_inds = indices[start_idx:end_idx]

                    _, new_log_prob, new_ent, new_value = self.agent.get_action_and_value(flat_obs[b_inds], flat_actions[b_inds])

                    log_ratio = new_log_prob - flat_log_probs[b_inds]
                    ratio = torch.exp(log_ratio)

                    b_advantages = advatnages_norm[b_inds]

                    # maxmize objective = minimize negtive of objective
                    objective_pg = torch.min(ratio * b_advantages, 
                                        torch.clamp(ratio, 1 - self.cfg.eps, 1 + self.cfg.eps)*b_advantages)
                    loss_pg = -objective_pg.mean()

                    # value loss
                    loss_vf = 0.5 * ((new_value.flatten() - flat_returns[b_inds])**2).mean()

                    action_penalty = torch.mean(torch.abs(flat_actions[b_inds])) * self.cfg.action_mag_pen_coef

                    loss = loss_pg + loss_vf + action_penalty - new_ent.mean() * self.cfg.ent_coef
                    losses.append(loss.item())

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.cfg.max_grad_norm)
                    self.optimizer.step()
            
            loss_mean = np.mean(losses)
            print(f"Run : {run+1:4d}/{num_runs} | Loss: {loss_mean:.4f} | Return: {returns.mean():.4f}")

        self.env.close()
        print(f"\n{'='*60}")
        print("Training complete!")
        print(f"{'='*60}\n")
        torch.save(self.agent.state_dict(), "model.pth")
        print("Model Saved!!👀")    
            
        

    def evaluate(self):
        print(f"\n{'-'*60}")
        print(f"Evaluation")
        print(f"{'-'*60}")
        frames = []
        for episode in range(5):
            _, _ = self.env.reset()
            obs = self.get_img_obs()
            total_reward = 0
            
            for _ in range(self.config.timesteps):
                with torch.no_grad():
                    action = self.agent.actor(obs)
                
                _, reward, terminated, truncated, _ = self.env.step(action)
                obs = self.get_img_obs()
                total_reward += reward
                
                frame = self.env.render()
                frame = (frame * 255).astype('uint8') if frame.dtype != 'uint8' else frame
                frame = np.ascontiguousarray(frame)
                cv2.putText(frame, f"Episode: {episode}", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
                frames.append(frame)
            
            print(f"Episode {episode+1}: Reward = {total_reward.mean().item():.4f}")
        
        # Save video and checkpoint
        video_path = f"videos"
        model_path = f"models"

        Path(video_path).mkdir(parents=True, exist_ok=True)
        Path(model_path).mkdir(parents=True, exist_ok=True)

        imageio.mimsave(f"{video_path}/video.mp4", frames, fps=30)
        torch.save(self.agent.state_dict(), f"{model_path}/model.pth")
        
        print(f"✓ Video saved: {video_path}")
        print(f"✓ Model saved: {model_path}\n")

def main():
    print("Hello from gym-rl!")
    cfg = TrainConfig()
    ppo_image = Panda_PPO_Image(cfg)
    ppo_image.test_random_actions()
    # ppo_image.train()
    print("fuck man!!")
    simulation_app.close()



if __name__ == "__main__":
    main()