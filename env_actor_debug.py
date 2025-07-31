import os
import mani_skill.envs
import gymnasium as gym
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
import numpy as np
import traceback
import time

class EnvActor:
    def __init__(self):
        self.env = None
        self.env_ids = []
        self.env_unique_ids = []
    
    def init_venv(self, env_ids, env_unique_ids, task_instructions, is_valid, global_steps, max_steps):
        self.finished = np.zeros(len(env_unique_ids), dtype=bool)
        self.finish_step = np.zeros(len(env_unique_ids), dtype=int)
        # task_file_names = [f"{env_ids[venv_index]}_task_{env_unique_ids[venv_index]}_uid_{global_steps}" for venv_index in range(len(env_unique_ids))]
        task_file_names = {venv_index: f"{env_ids[venv_index]}_task_{env_unique_ids[venv_index]}_uid_{global_steps}" for venv_index in range(len(env_unique_ids))}
        try:
            if self.env is not None and (self.env_ids[0] != env_ids[0] or len(self.env_unique_ids) != len(env_unique_ids)):
                # reinit the env with new env_ids and env_unique_ids
                print("Closing existing environment and creating a new one...")
                self.env.close()
                self.env = None
            if self.env is None:
                env = gym.make(
                    env_ids[0], 
                    num_envs=len(env_unique_ids),
                    obs_mode="rgb",
                    control_mode="pd_ee_delta_pose",
                    sensor_configs={'height': 480, 'width': 480},
                    max_episode_steps=max_steps,
                    reward_mode='sparse',
                )
                self.env = ManiSkillVectorEnv(env, auto_reset=True, ignore_terminations=False)
            obs, _ = self.env.reset(seed=env_unique_ids)
            # print(obs['sensor_data']["base_camera"]["rgb"].shape, obs['sensor_data']["hand_camera"]["rgb"].shape)
            valid_images = np.concatenate([obs['sensor_data']["base_camera"]["rgb"].cpu().numpy(), obs['sensor_data']["hand_camera"]["rgb"].cpu().numpy()], axis=2) if is_valid else None
        except Exception as e:
            print(f"Error during environment initialization: {e}")
            traceback.print_exc()
            return {'error': f"{type(e).__name__}: {e}\n{traceback.format_exc()}"}
        self.env_ids = env_ids
        self.env_unique_ids = env_unique_ids        
        self.is_valid = is_valid

        return {
            'obs': obs,
            'task_instructions': task_instructions,
            'task_file_name': task_file_names,
            'complete': self.finished,
            'finish_step': self.finish_step,
            'valid_images': valid_images
        }  

    def step(self, action):
        try:
            if isinstance(action, list):
                action = np.array(action)
            obs, _, terminated, _, _ = self.env.step(action)
            terminated = terminated.cpu().numpy()
            self.finished = np.logical_or(self.finished, terminated)
            self.finish_step += 1
            valid_images = np.concatenate([obs['sensor_data']["base_camera"]["rgb"].cpu().numpy(), obs['sensor_data']["hand_camera"]["rgb"].cpu().numpy()], axis=2) if self.is_valid else None
        except Exception as e:
            print(f"Error during step execution: {e}")
            traceback.print_exc()
            return {'error': f"{type(e).__name__}: {e}\n{traceback.format_exc()}"}
        return {
            'obs': obs,
            'complete': self.finished,
            'finish_step': self.finish_step,
            'valid_images': valid_images
        }

    def close(self):
        self.env.close()
        self.env = None

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Set the visible CUDA devices to GPU 0
env = EnvActor()
env_ids = ['StackCube-v1'] * 2
env_unique_ids = list(range(len(env_ids)))
init_data = env.init_venv(
    env_ids=env_ids,
    env_unique_ids=env_unique_ids,
    task_instructions='pick and place',
    is_valid=True,
    global_steps=0,
    max_steps=200
)
print(init_data)
env_ids = ['StackCube-v1'] * 4
env_unique_ids = list(range(len(env_ids)))
init_data = env.init_venv(
    env_ids=env_ids,
    env_unique_ids=env_unique_ids,
    task_instructions='pick and place',
    is_valid=True,
    global_steps=0,
    max_steps=200
)
print(init_data)

# env = gym.make(
#     'StackCube-v1', 
#     num_envs=2,
#     obs_mode="rgb",
#     control_mode="pd_ee_delta_pose",
#     sensor_configs={'height': 480, 'width': 480},
#     max_episode_steps=200,
#     reward_mode='sparse',
# )
# env = ManiSkillVectorEnv(env, auto_reset=True, ignore_terminations=False)
# obs, _ = env.reset(seed=list(range(4)))
# print(0)
# print("Closing")
# env.close()
# # time.sleep(10)
# print("Closed")

# env = gym.make(
#     'StackCube-v1', 
#     num_envs=4,
#     obs_mode="rgb",
#     control_mode="pd_ee_delta_pose",
#     sensor_configs={'height': 480, 'width': 480},
#     max_episode_steps=200,
#     reward_mode='sparse',
# )
# env = ManiSkillVectorEnv(env, auto_reset=True, ignore_terminations=False)
# obs, _ = env.reset(seed=[0])
# print(1)
# print("Closing")
# env.close()
# print("Closed")

