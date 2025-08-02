import mani_skill.envs
import gymnasium as gym
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
import numpy as np
import traceback
from verl.workers.rollout.env_workers.utils import action_to_str, write_instruction_action


class EnvActor:
    def __init__(self):
        self.env = None
        self.env_ids = []
        self.env_unique_ids = []
        self.num_envs = 0
    
    def init_venv(self, env_ids, env_unique_ids, task_instructions, is_valid, global_steps, max_steps):
        assert len(env_ids) > 1, "Num venvs must be greater than 1 to avoid env re-initialization PHYSIX Errors."
        self.finished = np.zeros(len(env_unique_ids), dtype=bool)
        self.finish_step = np.zeros(len(env_unique_ids), dtype=int)
        # task_file_names = [f"{env_ids[venv_index]}_task_{env_unique_ids[venv_index]}_uid_{global_steps}" for venv_index in range(len(env_unique_ids))]
        task_file_names = {venv_index: f"{env_ids[venv_index].split('-')[0]}_seed_{env_unique_ids[venv_index]}_uid_{global_steps}" for venv_index in range(len(env_unique_ids))}
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
            # valid_images = np.concatenate([obs['sensor_data']["base_camera"]["rgb"].cpu().numpy(), obs['sensor_data']["hand_camera"]["rgb"].cpu().numpy()], axis=2) if is_valid else None
        except Exception as e:
            print(f"Error during environment initialization: {e}")
            traceback.print_exc()
            return {'error': f"{type(e).__name__}: {e}\n{traceback.format_exc()}"}
        self.env_ids = env_ids
        self.env_unique_ids = env_unique_ids        
        self.is_valid = is_valid
        self.num_envs = len(env_unique_ids)
        self.last_obs = obs
        return {
            'obs': obs,
            'task_instructions': task_instructions,
            'task_file_name': task_file_names,
            'complete': self.finished,
            'finish_step': self.finish_step,
        }  

    def step(self, action):
        try:
            action, string_response = action
            if isinstance(action, list):
                action = np.array(action)
            obs, _, terminated, _, _ = self.env.step(action)
            terminated = terminated.cpu().numpy()
            self.finished = np.logical_or(self.finished, terminated)
            self.finish_step += 1
            valid_images = None
            if self.is_valid:
                valid_images = np.concatenate([self.last_obs['sensor_data']["base_camera"]["rgb"].cpu().numpy(), self.last_obs['sensor_data']["hand_camera"]["rgb"].cpu().numpy()], axis=2)
                images = []
                for env_index in range(self.num_envs):
                    local_img = valid_images[env_index]
                    local_action = action[env_index]
                    local_raw_action = string_response[env_index]
                    action_str = f"A: {action_to_str(local_action, 3)}"
                    raw_action = "RA: " + local_raw_action
                    img = write_instruction_action(action_str, local_img, raw_action)
                    images.append(img)
                valid_images = np.stack(images, axis=0)
            self.last_obs = obs
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


def env_worker(task_name, env_ids, env_unique_ids, task_instruction, input_queue, output_queue, is_valid, global_steps, max_steps):
    import torch
    print(torch.cuda.current_device())
    if torch.cuda.is_available():
        torch.cuda.set_device(0)  # 根据需要设置 device
    # we assert all the venv has the same env_id (gym.make(id=env_id))
    env_id = env_ids[0]
    env = gym.make(env_id, num_envs=len(env_unique_ids))
    env = ManiSkillVectorEnv(env, auto_reset=True, ignore_terminations=False)
    obs, _  = env.reset(seed=env_unique_ids)
    valid_images = np.concatenate([obs['sensor_data']["base_camera"]["rgb"].cpu().numpy(), obs['sensor_data']["hand_camera"]["rgb"].cpu().numpy()], axis=2)
    is_first_finished = np.zeros(len(env_unique_ids), dtype=bool)
    finish_step = np.zeros(len(env_unique_ids))

    output_queue.put({
        'type': 'init',
        'obs': obs,
        "task_instructions": task_instruction,
        'valid_images': valid_images,
        'task_file_name': [f"{env_ids[venv_index]}_task_{env_unique_ids[venv_index]}_trial_{global_steps}" for venv_index in range(len(env_unique_ids))],
        'complete': is_first_finished,
        'finish_step': finish_step,
    })
    
    while True:
        action = input_queue.get()
        if action is None:
            output_queue.put({'type': 'terminate'})
            env.close()
            break
        # TODO: support multi step action
        # if len(action.shape) > len(env.action_space.shape):
        #     for i in range(len(action)):
        #         a = action[i]
        #         obs, rew, terminated, truncated, info = env.step(a)
        #         terminated = terminated.cpu().numpy()
        #         is_dones = np.logical_or(truncated.cpu().numpy(), terminated)
        #         if is_valid:
        #             img = np.concatenate([obs['sensor_data']["base_camera"]["rgb"].cpu().numpy(), obs['sensor_data']["hand_camera"]["rgb"].cpu().numpy()], axis=2)
        #             step_images.append(img)
        #         is_first_finished = np.logical_or(is_first_finished, is_dones)
        #         finish_step += 1
        #         finish_env_ids = np.where(is_first_finished)[0]
        # else:
        obs, rew, terminated, truncated, info = env.step(action)
        terminated = terminated.cpu().numpy()
        if is_valid:
            img = np.concatenate([obs['sensor_data']["base_camera"]["rgb"].cpu().numpy(), obs['sensor_data']["hand_camera"]["rgb"].cpu().numpy()], axis=2)
        is_first_finished = np.logical_or(is_first_finished, terminated)
        finish_step += 1
        output_data = {
            'type': 'step',
            'obs': obs,
            'complete': is_first_finished,
            'finish_step': finish_step,
            'valid_images': img if is_valid else None
        }
        output_queue.put(output_data)