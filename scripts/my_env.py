import torch
import numpy as np
import cv2
import imageio
import gymnasium as gym
from pathlib import Path
import argparse

from isaaclab.source.isaaclab.isaaclab.app import AppLauncher

# Launch Isaac Sim
app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

from isaaclab.source.isaaclab_tasks.isaaclab_tasks.utils import load_cfg_from_registry
import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from pxr import UsdGeom, Gf
import omni.usd
from isaaclab.assets import AssetBaseCfg

# ===================================================================================
# RANDOM ACTION TEST
# ===================================================================================

class RandomActionTester:
    """Test environment with random actions"""
    
    def __init__(self, env_name: str, num_envs: int = 1, num_episodes: int = 5, timesteps: int = 256):
        self.env_name = env_name
        self.num_envs = num_envs
        self.num_episodes = num_episodes
        self.timesteps = timesteps
        self.device = "cuda"
        
        self.env = self._create_environment()
        self.action_dim = self.env.action_space.shape[1] if hasattr(self.env.action_space, 'shape') else self.env.action_space.shape[0]
        
        print(f"{'-'*60}")
        print(f"✓ Environment: {env_name}")
        print(f"✓ Number of parallel envs: {num_envs}")
        print(f"✓ Action dimension: {self.action_dim}")
        print(f"{'-'*60}\n")
    
    def _create_environment(self) -> gym.Env:
        """Create Isaac Lab environment with custom follow camera"""
        self.cfg = load_cfg_from_registry(self.env_name, "env_cfg_entry_point")
        self.cfg.scene.num_envs = self.num_envs

        self.cfg.scene.follow_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/follow_cam",
            update_period=0.0,
            height=1080,
            width=1080,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
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

        env = gym.make(self.env_name, cfg=self.cfg, render_mode='rgb_array')
        env = gym.wrappers.OrderEnforcing(env)
        return env
    
    def update_follow_cam(self, eye, target):
        """Update follow camera position and target"""
        stage = omni.usd.get_context().get_stage()
        cam_prim = stage.GetPrimAtPath("/World/envs/env_0/follow_cam")
        xform = UsdGeom.Xformable(cam_prim)
        xform.ClearXformOpOrder()
        eye = Gf.Vec3d(*map(float, eye))
        target = Gf.Vec3d(*map(float, target))
        up = Gf.Vec3d(0.0, 0.0, 1.0)
        view = Gf.Matrix4d().SetLookAt(eye, target, up)
        xform.AddTransformOp().Set(view.GetInverse())
    
    def test_random_actions(self) -> None:
        """Test environment with random actions and save video"""
        print(f"\n{'-'*60}")
        print(f"Testing Random Actions: {self.env_name}")
        print(f"{'-'*60}\n")
        
        total_rewards_per_episode = []
        all_frames = []
        
        for episode in range(self.num_episodes):
            obs, _ = self.env.reset()
            if isinstance(obs, dict):
                obs = obs["policy"]
            
            total_reward = torch.zeros(self.num_envs)
            episode_frames = []
            
            print(f"Episode {episode + 1}/{self.num_episodes}")
            
            for step in range(self.timesteps):
                # Sample random actions from action space
                action = torch.tensor(
                    self.env.action_space.sample(),
                    dtype=torch.float32,
                    device=self.device
                )
                
                # Handle both single and multi-environment cases
                if self.num_envs > 1 and action.dim() == 1:
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
                try:
                    robot = self.env.unwrapped.scene["robot"]
                    pose = robot.data.root_state_w
                    target = pose[0, :3].cpu().numpy()
                    eye = target + np.array([-3.0, -2.0, 1.5])
                    self.update_follow_cam(eye, target)

                    cam = self.env.unwrapped.scene["follow_cam"]
                    frame = cam.data.output["rgb"][0].cpu().numpy()
                    frame = (frame * 255).astype('uint8') if frame.dtype != 'uint8' else frame
                    frame = np.ascontiguousarray(frame)
                    
                    # Add text to frame
                    cv2.putText(frame, f"Episode: {episode + 1} | Step: {step + 1}", (20, 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(frame, f"Reward: {total_reward.mean().item():.4f}", (20, 100),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    episode_frames.append(frame)
                    all_frames.append(frame)
                except Exception as e:
                    print(f"Warning: Could not capture frame at step {step}: {e}")
            
            avg_reward = total_reward.mean().item()
            total_rewards_per_episode.append(avg_reward)
            print(f"  Average Reward: {avg_reward:.4f}\n")
        
        self.env.close()
        
        # Save video
        if all_frames:
            video_path = f"outputs/{self.env_name}/videos"
            Path(video_path).mkdir(parents=True, exist_ok=True)
            
            video_file = f"{video_path}/random_actions.mp4"
            imageio.mimsave(video_file, all_frames, fps=30)
            print(f"✓ Video saved: {video_file}\n")
        
        # Print summary
        print(f"\n{'='*60}")
        print("Test Summary")
        print(f"{'='*60}")
        print(f"Mean Reward across episodes: {np.mean(total_rewards_per_episode):.4f}")
        print(f"Std Reward: {np.std(total_rewards_per_episode):.4f}")
        print(f"Max Reward: {np.max(total_rewards_per_episode):.4f}")
        print(f"Min Reward: {np.min(total_rewards_per_episode):.4f}")
        print(f"{'='*60}\n")







def main():
    """Main function to test environment with random actions"""
    parser = argparse.ArgumentParser(description="Test Isaac Lab Environment with Random Actions")
    parser.add_argument(
        "--env",
        type=str,
        default="Isaac-Lift-Cube-Franka-v0",
        help="Environment name (e.g., Isaac-Lift-Cube-Franka-v0)"
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of parallel environments"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to test"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=256,
        help="Number of timesteps per episode"
    )
    
    args = parser.parse_args()
    
    print(f"\n🎮 Testing environment with random actions\n")
    
    tester = RandomActionTester(
        env_name=args.env,
        num_envs=args.num_envs,
        num_episodes=args.episodes,
        timesteps=args.timesteps
    )
    tester.test_random_actions()
    
    simulation_app.close()


if __name__ == '__main__':
    main()
