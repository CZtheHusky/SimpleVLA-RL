import mani_skill.envs
import gymnasium as gym
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
import numpy as np


def env_worker(task_name, env_ids, env_unique_ids, task_instruction, input_queue, output_queue, is_valid, global_steps, max_steps):
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