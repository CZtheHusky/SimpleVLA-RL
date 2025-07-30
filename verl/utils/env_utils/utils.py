import numpy as np
import torch
import torchvision.transforms as T
from typing import Tuple
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import json
import os
from scipy.spatial.transform import Rotation as R
import re
import cv2
from typing import List
import roboticstoolbox as rtb
from enum import Enum
import string
import matplotlib.pyplot as plt
from verl.utils.vla_utils.internvl.utils import process_image_internvl


class ActionType(Enum):
    DELTA_EEF = 0
    ABS_EEF = 1
    ABS_JOINT = 2
    DELTA_JOINT = 3
    
class TaskSuite(Enum):
    MANISKILL = 0
    GRUTOPIA = 1
    LIBERO = 2
    
def action_to_str(action, num_floats: int = 4):
    return [np.round(a, num_floats) for a in action.values()] if isinstance(action, dict) else [np.round(a, num_floats) for a in action]

def parse_and_validate_vector(input_str: str):
    """
    解析并验证一个字符串，期望其格式为包含7个空格分隔的数字的花括号包围的向量。

    Args:
        input_str: 模型的原始输出字符串。

    Returns:
        如果格式完全正确，则返回一个包含7个整数的列表。
        如果格式有任何问题（缺少花括号、数字数量不对、包含非数字内容等），则返回 None。
    """
    # 1. 基础检查：确保输入是字符串
    if not isinstance(input_str, str):
        return None

    # 2. 预处理：去除首尾多余的空白字符
    s = input_str.strip()

    # 3. 验证格式：是否被花括号包围
    if not (s.startswith('{') and s.endswith('}')):
        return None

    # 4. 提取花括号内的内容
    content = s[1:-1].strip()
    
    # 如果内容为空（例如输入是 "{}" 或 "{ }"），也视为无效
    if not content:
        return None

    # 5. 分割内容
    parts = content.split()

    # 6. 验证数量：是否正好是7个数字
    if len(parts) != 7:
        return None

    # 7. 验证内容：尝试将所有部分转换为整数
    try:
        vector = [int(p) for p in parts]
        return np.array(vector, dtype=np.float32)  # 返回一个整数类型的NumPy数组
    except ValueError:
        # 如果任何一部分无法转换为整数（例如 "1.5", "abc"），则捕获异常
        print("Error response:", input_str)
        return None


# 统计并画图 (Updated to calculate on 99th percentile data)
def plot_and_print_stats(arr, name, save_parent):
    """
    Calculates and prints statistics on the central 99% of the data for each dimension,
    and plots a histogram of this filtered data.
    """
    print(f"--- Calculating 99th Percentile Statistics for: {name} ---")
    
    # We will store the stats for each dimension after filtering
    means, stds, maxs, mins = [], [], [], []
    save_parent = os.path.join(save_parent, name)
    os.makedirs(save_parent, exist_ok=True)
    # Iterate over each dimension (column) of the array
    for i in range(arr.shape[1]):
        column_data = arr[:, i]
        
        # 1. Calculate the 0.5 and 99.5 percentile to find the central 99% range
        lower_bound = np.percentile(column_data, 0.5)
        upper_bound = np.percentile(column_data, 99.5)
        
        # 2. Filter the data for the current dimension to be within this range
        filtered_column = column_data[(column_data >= lower_bound) & (column_data <= upper_bound)]
        # print(len(filtered_column), len(column_data), f"dim {i} ({name})")
        
        # 3. Calculate stats on the filtered data
        means.append(float(np.mean(filtered_column)))
        stds.append(float(np.std(filtered_column)))
        maxs.append(float(np.max(filtered_column)))
        mins.append(float(np.min(filtered_column)))

        # 4. Plot the histogram using only the filtered data
        plt.figure()
        plt.hist(filtered_column, bins=100, alpha=0.7)
        plt.title(f"{name} dim {i} (99th Percentile) histogram")
        plt.xlabel(f"{name}[{i}]")
        plt.ylabel("Count")
        # plt.show() 
        plt.savefig(os.path.join(save_parent, f"{name}_dim_{i}_histogram.png"))
        
        plt.figure()
        plt.hist(arr[:, i], bins=100, alpha=0.7)
        plt.title(f"{name} dim {i} histogram")
        plt.xlabel(f"{name}[{i}]")
        plt.ylabel("Count")
        # plt.show()
        plt.savefig(os.path.join(save_parent, f"{name}_dim_{i}_histogram_all.png"))
        
    # Assemble the final statistics dictionary in the required format
    statistics_dict = {
        f"{name}": {
            "shape_before_filtering": arr.shape,
            "p99_mean": means,
            "p99_std": stds,
            "p99_max": maxs,
            "p99_min": mins,
            "mean": np.mean(arr, axis=0).tolist(),
            "std": np.std(arr, axis=0).tolist(),
            "max": np.max(arr, axis=0).tolist(),
            "min": np.min(arr, axis=0).tolist(),
        }
    }
    
    # Print the final calculated statistics
    print(f"Original shape: {arr.shape}")
    print(f"Mean (99th perc): {np.array(means)}")
    print(f"Std (99th perc):  {np.array(stds)}")
    print(f"Max (99th perc):  {np.array(maxs)}")
    print(f"Min (99th perc):  {np.array(mins)}")
    print(f"Mean: {statistics_dict[name]['mean']}")
    print(f"Std:  {statistics_dict[name]['std']}")
    print(f"Max:  {statistics_dict[name]['max']}")
    print(f"Min:  {statistics_dict[name]['min']}")
    print("-" * (50 + len(name)))
    
    return statistics_dict

def action_decode(prompts, string_response, task_suite: TaskSuite):
    if task_suite == TaskSuite.GRUTOPIA:
        gripper_states = prompts['gripper_states']
        q_eefs = prompts['q_eefs']
        q_grippers = prompts['q_grippers']
        eef_transes = prompts['eef_transes']
        eef_quats = prompts['eef_quats']
        action_type = prompts['action_type']
        quatEEF = prompts['quatEEF']
        actions = []
        for idx in range(len(string_response)):
            gripper_state = gripper_states[idx]
            response = string_response[idx]
            q_eef = q_eefs[idx]
            q_gripper = q_grippers[idx]
            eef_trans = eef_transes[idx]
            eef_quat = eef_quats[idx]
            action_extracted = extract_action_vector(response, action_type=action_type, quatEEF=quatEEF)
            action = assemble_action_vla(
                action_extracted, 
                gripper_state, 
                q_eef, 
                q_gripper, 
                eef_trans, 
                eef_quat, 
                action_type, 
                quatEEF
            )
            actions.append(action)
    elif task_suite == TaskSuite.MANISKILL:
        qposes = prompts['qposes']
        actions = []
        for idx in range(len(string_response)):
            response = string_response[idx]
            action_extracted = parse_and_validate_vector(response)
            if action_extracted is None:
                action_extracted = np.zeros(7, dtype=np.float64)
                if qposes[idx][-1] >= 0.037:
                    action_extracted[-1] = 1
                else:
                    action_extracted[-1] = -1
            else:
                action_extracted[:-1] = action_extracted[:-1] / 1000
            actions.append(action_extracted)
    return actions
        


def assemble_action_vla(
    action_extracted: np.ndarray,
    gripper_state: int = None,  # 0: closed, 1: open
    q_eef: np.ndarray = None,   # joint pose of the arm
    q_gripper: np.ndarray = None,   # joint pose of the gripper
    eef_trans: np.ndarray = None,   # end-effector local translation
    eef_quat: np.ndarray = None,    # end-effector local quaternion
    action_type: ActionType = None,   # action_type
    quatEEF: bool = True,  # whether to use quaternion for end-effector pose action
):
    if action_type in [ActionType.ABS_EEF, ActionType.DELTA_EEF]:
        gripper_action = int(gripper_state) if action_extracted is None else int(action_extracted[-1])
        gripper_action = -1 if gripper_action == 1 else 1
        if action_extracted is None:
            target_action = (eef_trans, eef_quat)
        else:
            if action_type == ActionType.DELTA_EEF:
                if quatEEF:
                    target_action = [action_extracted[0] / 1000, action_extracted[1] / 1000]
                else:
                    target_action = [action_extracted[0] / 1000, action_extracted[1]]
                    target_action = rpy_to_quat(target_action)
                target_action = compute_target_pose_quat((eef_trans, eef_quat), target_action[:2])
            else:
                if quatEEF:
                    target_action = [action_extracted[0] / 1000, action_extracted[1] / 1000]
                else:
                    target_action = [action_extracted[0] / 1000, action_extracted[1]]
                    target_action = rpy_to_quat(target_action)
        action = {
            "eef_position": target_action[0],
            "eef_orientation": target_action[1],
            "gripper_action": gripper_action
        }
    elif action_type in [ActionType.ABS_JOINT, ActionType.DELTA_JOINT]:
        if action_extracted is None:
            target_action = np.concatenate([q_eef, q_gripper])
        elif action_type == ActionType.ABS_JOINT:
            target_action = np.concatenate([action_extracted, np.array([action_extracted[-1]])])
            target_action = target_action / 1000
        else:
            action_extracted = action_extracted / 1000
            target_action = np.concatenate([q_eef + action_extracted[:-1], action_extracted[-1:], action_extracted[-1:]])
        action = target_action.tolist()
    return action


def obs_process(inputs: List, task_descriptions, task_suite: TaskSuite, **kwargs):
    if task_suite == TaskSuite.GRUTOPIA:
        num_patches_list = []
        pixel_values = []
        questions = []
        gripper_states = []
        q_eefs = []
        q_grippers = []
        eef_transes = []
        eef_quats = []
        for idx in range(len(inputs)):
            obs = inputs[idx]
            instruction = task_descriptions[idx]
            obs = obs['franka_robot']
            q_eef = obs['joints_state']['positions'][:-2]
            q_gripper = obs['joints_state']['positions'][-2:]
            instruction = obs['instruction'].lower().rstrip(string.punctuation).lstrip()
            eef_trans = obs['eef_pose']['local_pose'][0]
            eef_quat = obs['eef_pose']['local_pose'][1]
            gripper_state = get_gripper_state(q_gripper)
            q_eefs.append(q_eef)
            q_grippers.append(q_gripper)
            gripper_states.append(gripper_state)
            eef_transes.append(eef_trans)
            eef_quats.append(eef_quat)
            query = generate_query_str(
                gripper_state=gripper_state,
                q_eef=q_eef,
                q_gripper=q_gripper,
                instruction=instruction,
                eef_trans=eef_trans,
                eef_quat=eef_quat,
                no_state=kwargs.get('no_state', False),
                action_type=kwargs.get('action_type', ActionType.DELTA_JOINT),
                quatEEF=kwargs.get('quatEEF', False),
            )
            query = "<image><image>" + query
            camera_0 = obs['sensors']['obs_camera']['rgb']
            wrist_camera = obs['sensors']['realsense']['rgb']
            pixel_0 = process_image_internvl(camera_0)
            pixel_1 = process_image_internvl(wrist_camera)
            patch_list = [pixel_0.size(0), pixel_1.size(0)]
            pixels = torch.cat((pixel_0, pixel_1), dim=0)
            questions.append(query)
            pixel_values.append(pixels)
            num_patches_list.append(patch_list)
        pixel_values = torch.cat(pixel_values, dim=0)
        return {
                    "pixel_values": pixel_values,
                    "questions": questions,
                    "num_patches_list": num_patches_list,
                    "gripper_states": gripper_states,
                    "q_eefs": q_eefs,
                    "q_grippers": q_grippers,
                    'eef_transes': eef_transes,
                    "eef_quats": eef_quats,
                    'action_type': kwargs.get('action_type', ActionType.DELTA_JOINT),
                    'quatEEF': kwargs.get('quatEEF', False),
                }
    elif task_suite == TaskSuite.MANISKILL:
        dual_cam = kwargs.get('dual_cam', True)
        instructions = kwargs.get("instructions", None)
        num_patches_list = []
        pixel_values = []
        questions = []
        qposes = []
        num_envs = inputs["agent"]["qpos"].shape[0]
        for env_id in range(num_envs):
            qpos = inputs["agent"]["qpos"][env_id].cpu().numpy()
            qposes.append(qpos)
            camera = inputs['sensor_data']["base_camera"]["rgb"][env_id].cpu().numpy()
            rescaled_qpos = np.round(qpos * 1000).astype(np.int32)
            query = f"The current position state of the robotic arm's end gripper is as follows: {{Joint_0: {rescaled_qpos[0]}, Joint_1: {rescaled_qpos[1]}, Joint_2: {rescaled_qpos[2]}, Joint_3: {rescaled_qpos[3]}, Joint_4: {rescaled_qpos[4]}, Joint_5: {rescaled_qpos[5]}, Joint_6: {rescaled_qpos[6]}, Joint_7: {rescaled_qpos[7]}, Joint_8: {rescaled_qpos[8]}}}. What action should the robot take to get better completion of instruction: {instructions[env_id]}?"
            pixel_0 = process_image_internvl(camera)
            patch_list = []
            pixels = []
            patch_list.append(pixel_0.size(0))
            pixels.append(pixel_0)
            if dual_cam:
                query = "<image><image>" + query
                hand_camera = inputs['sensor_data']["hand_camera"]["rgb"][env_id].cpu().numpy()
                pixel_1 = process_image_internvl(hand_camera)
                patch_list.append(pixel_1.size(0))
                pixels.append(pixel_1)
            else:
                query = "<image>" + query
            if len(pixels) == 1:
                pixels = pixels[0]
            else:
                pixels = torch.cat(pixels, dim=0)
            questions.append(query)
            pixel_values.append(pixels)
            num_patches_list.append(patch_list)
        pixel_values = torch.cat(pixel_values, dim=0)
        return {
            "pixel_values": pixel_values,
            "questions": questions,
            "num_patches_list": num_patches_list,
            "qposes": qposes,
        }
    else:
        raise NotImplementedError



def generate_query_str(
    gripper_state: int = None,  # 0: closed, 1: open
    q_eef: np.ndarray = None,   # joint pose of the arm
    q_gripper: np.ndarray = None,   # joint pose of the gripper
    instruction: str = None,    # instruction for the robot
    eef_trans: np.ndarray = None,   # end-effector local translation
    eef_quat: np.ndarray = None,    # end-effector local quaternion
    no_state: bool = False, # whether to add the state information in the query
    action_type: ActionType = None,   # action_type
    quatEEF: bool = True,  # whether to use quaternion for end-effector pose action
    traj_id: int = None,    # trajectory id for debugging
    step_idx: int = None,   # step index for debugging
):
    instruction = instruction.lower().rstrip(string.punctuation).lstrip()
    if no_state:
        query = f"What action should the robot take to get better completion of instruction: {instruction}?"
    elif action_type == ActionType.DELTA_EEF or action_type == ActionType.ABS_EEF:
        if quatEEF:
            eef_xyz, eef_quat = post_process_quat(eef_trans, eef_quat)
            if -1000 < eef_xyz[0] < 1000 and -1000 < eef_xyz[1] < 1000 and -1000 < eef_xyz[2] < 1000:
                print(f"Out of bound eef_xyz: {eef_xyz}, ep_id: {traj_id}, step_idx: {step_idx}")
            query = f"The current position state of the robotic arm's end gripper is as follows: {{x: {eef_xyz[0]}mm, y: {eef_xyz[1]}mm, z: {eef_xyz[2]}mm,  quat: {eef_quat[0]}, {eef_quat[1]}, {eef_quat[2]}, {eef_quat[3]}, open: {gripper_state}}}. What action should the robot take to get better completion of instruction: {instruction}?"
        else:
            eef_xyz, eef_rpy = quat_to_rpy([eef_trans, eef_quat])
            eef_xyz, eef_rpy = post_process_rpy(eef_xyz, eef_rpy)
            if -1000 < eef_xyz[0] < 1000 and -1000 < eef_xyz[1] < 1000 and -1000 < eef_xyz[2] < 1000:
                print(f"Out of bound eef_xyz: {eef_xyz}, ep_id: {traj_id}, step_idx: {step_idx}")
            query = f"The current position state of the robotic arm's end gripper is as follows: {{x: {eef_xyz[0]}mm, y: {eef_xyz[1]}mm, z: {eef_xyz[2]}mm, roll: {eef_rpy[0]} degrees, pitch: {eef_rpy[1]} degrees, yaw: {eef_rpy[2]} degrees, open: {gripper_state}}}. What action should the robot take to get better completion of instruction: {instruction}?"
    elif action_type == ActionType.ABS_JOINT or action_type == ActionType.DELTA_JOINT:
        qpos = np.concatenate([q_eef, q_gripper])
        qpos = post_process_qpos(qpos)
        query = f"The current joint state of the robotic arm's end gripper is as follows: {{joint_0: {qpos[0]}, joint_1: {qpos[1]}, joint_2: {qpos[2]}, joint_3: {qpos[3]}, joint_4: {qpos[4]}, joint_5: {qpos[5]}, joint_6: {qpos[6]}, joint_7: {qpos[7]}}}. What action should the robot take to get better completion of instruction: {instruction}?"
    else:
        raise ValueError("At least one of absEEF, deltaEEF, absQ, or deltaQ must be True.")
    return query
    
    

def generate_action_str(
    gripper_action: int = None, # action for the gripper, 0: close, 1: open
    q_eef: np.ndarray = None,   # joint pose of the arm
    joint_eef_action: np.ndarray = None,    # joint action of the arm
    joint_gripper_action: np.ndarray = None,    # joint action of the gripper
    eef_trans: np.ndarray = None,   # end-effector local translation
    eef_quat: np.ndarray = None,    # end-effector local quaternion
    eef_action_trans: np.ndarray = None,    # end-effector local translation action
    eef_action_quat: np.ndarray = None, # end-effector local quaternion action
    action_type: ActionType = None,   # action_type
    quatEEF: bool = True,  # whether to use quaternion for end-effector pose action
    traj_id: int = None,    # trajectory id for debugging
    step_idx: int = None,   # step index for debugging
):  
    if action_type == ActionType.ABS_EEF:
        if quatEEF:
            action_vector = np.concatenate([trans, quat, np.array([gripper_action])])
            trans, quat = post_process_quat(eef_action_trans, eef_action_quat)
            action_str = f"action: {{x: {trans[0]}mm, y: {trans[1]}mm, z: {trans[2]}mm, quat: {quat[0]}, {quat[1]}, {quat[2]}, {quat[3]}}}, open: {gripper_action}"
        else:
            trans, rpy = quat_to_rpy([eef_action_trans, eef_action_quat])
            action_vector = np.concatenate([trans, rpy, np.array([gripper_action])])
            trans, rpy = post_process_rpy(trans, rpy)
            assert -1000 < trans[0] < 1000 and -1000 < trans[1] < 1000 and -1000 < trans[2] < 1000, f"Invalid eef_action_trans: {trans}, ep_id: {traj_id}, step_idx: {step_idx}"
            action_str = f"action: {{x: {trans[0]}mm, y: {trans[1]}mm, z: {trans[2]}mm, roll: {rpy[0]} degrees, pitch: {rpy[1]} degrees, yaw: {rpy[2]} degrees, open: {gripper_action}}}"
    elif action_type == ActionType.DELTA_EEF:
        if quatEEF:
            delta_trans, delta_quat = compute_delta_eepose(
                [eef_action_trans, eef_action_quat], [eef_trans, eef_quat]
            )
            action_vector = np.concatenate([delta_trans, delta_quat, np.array([gripper_action])])
            delta_trans, delta_quat = post_process_quat(delta_trans, delta_quat)
            action_str = f"action: {{x: {delta_trans[0]}mm, y: {delta_trans[1]}mm, z: {delta_trans[2]}mm, quat: {delta_quat[0]}, {delta_quat[1]}, {delta_quat[2]}, {delta_quat[3]}}}, open: {gripper_action}"
        else:
            delta_trans, delta_rpy = compute_delta_eepose_rpy([eef_action_trans, eef_action_quat], [eef_trans, eef_quat])
            action_vector = np.concatenate([delta_trans, delta_rpy, np.array([gripper_action])])
            delta_trans, delta_rpy = post_process_rpy(delta_trans, delta_rpy)
            action_str = f"action: {{x: {delta_trans[0]}mm, y: {delta_trans[1]}mm, z: {delta_trans[2]}mm, roll: {delta_rpy[0]} degrees, pitch: {delta_rpy[1]} degrees, yaw: {delta_rpy[2]} degrees, open: {gripper_action}}}"
            
    elif action_type == ActionType.ABS_JOINT:
        qpos = np.concatenate([joint_eef_action, joint_gripper_action])
        action_vector = qpos[:-1]
        qpos = post_process_qpos(qpos)
        action_str = f"action: {{joint_0: {qpos[0]}, joint_1: {qpos[1]}, joint_2: {qpos[2]}, joint_3: {qpos[3]}, joint_4: {qpos[4]}, joint_5: {qpos[5]}, joint_6: {qpos[6]}, joint_7: {qpos[7]}}}"
    elif action_type == ActionType.DELTA_JOINT:
        arm_joints_delta = joint_eef_action - q_eef
        arm_joints_delta = np.concatenate([arm_joints_delta, joint_gripper_action])
        action_vector = arm_joints_delta[:-1]
        arm_joints_delta = np.round(1000 * arm_joints_delta).astype(np.int32)
        action_str = f"action: {{joint_0: {arm_joints_delta[0]}, joint_1: {arm_joints_delta[1]}, joint_2: {arm_joints_delta[2]}, joint_3: {arm_joints_delta[3]}, joint_4: {arm_joints_delta[4]}, joint_5: {arm_joints_delta[5]}, joint_6: {arm_joints_delta[6]}, joint_7: {arm_joints_delta[7]}}}"
    else:
        raise NotImplementedError()

    return action_str, action_vector


def joint_position_to_end_effector_pose(joint_position, panda=None):
    if panda is None:
        panda = rtb.models.Panda()
    hand_pose = panda.fkine(q=joint_position, end="panda_hand").A
    position = hand_pose[:3, 3]
    rotation = hand_pose[:3, :3]
    orientation = R.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]
    return position, orientation

def img_decode(img_bytes):
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # flags 参数必须指定
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def extract_action_vector(llm_output: str, action_type: ActionType = ActionType.DELTA_EEF, quatEEF: bool = False) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    从 LLM 的输出字符串中提取一个 7D 动作向量。

    Args:
        llm_output: LLM 的字符串输出。
                    示例: "action: {x: -129mm, y: 442mm, z: 183mm, roll: 2 degrees, pitch: 40 degrees, yaw: 0 degrees, open: 1}"
                    绝对动作示例: "action: {x: -11mm, y: 20mm, z: 33mm, quat: 10, 20, 30, 22, open: 1}"

    Returns:
        对于相对动作 (is_abs=False)，返回一个元组 (delta_trans, delta_r, gripper_open)。
        对于绝对动作 (is_abs=True)，返回一个元组 (abs_trans, abs_quat, gripper_open)。
        如果无法找到所有值，则返回 None。
    """
    if action_type in [ActionType.ABS_EEF, ActionType.DELTA_EEF]:
        if quatEEF:
            try:
                simple_pattern = r"(x|y|z|open):\s*(-?[\d.]+)"
                simple_matches = re.findall(simple_pattern, llm_output)
                data_dict = {key: float(value) for key, value in simple_matches}

                quat_pattern = r"quat:\s*(-?[\d.]+),\s*(-?[\d.]+),\s*(-?[\d.]+),\s*(-?[\d.]+)"
                quat_match = re.search(quat_pattern, llm_output)

                if not quat_match:
                    raise ValueError("未找到四元数 'quat' 值")

                abs_trans = np.array([data_dict['x'], data_dict['y'], data_dict['z']], dtype=np.float32)
                
                abs_quat = np.array([float(q) for q in quat_match.groups()], dtype=np.float32)

                gripper_open = float(data_dict['open'])

                return abs_trans, abs_quat, gripper_open

            except (KeyError, ValueError) as e:
                print(f"Error: {e} in the LLM output: {llm_output}")
                return None
        else:
            # Define the keys we want to extract in the desired final order
            keys_in_order = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'open']
            
            # This regex pattern finds all key-value pairs we are interested in.
            # It captures the key (e.g., 'x') and its corresponding numeric value.
            pattern = r"(x|y|z|roll|pitch|yaw|open):\s*(-?[\d.]+)"
            
            # re.findall will return a list of tuples, e.g., [('x', '-129'), ('y', '442'), ...]
            matches = re.findall(pattern, llm_output)
            
            # Convert the list of tuples into a dictionary for easy lookup.
            # The captured value is converted to a float.
            data_dict = {key: float(value) for key, value in matches}
            
            # Assemble the vector in the correct order using the dictionary.
            # We use .get() to safely handle cases where a key might be missing.
            try:
                vector = np.array([data_dict[key] for key in keys_in_order], dtype=np.float32)
                delta_trans = vector[:3]  # x, y, z
                delta_r = vector[3:6]  # roll, pitch, yaw
                gripper_open = vector[-1]  # open
                return delta_trans, delta_r, gripper_open
            except KeyError as e:
                print(f"Error: Missing key {e} in the LLM output: {llm_output}")
                return None
    elif action_type == ActionType.ABS_JOINT or action_type == ActionType.DELTA_JOINT:
        try:
            pattern = r"joint_(\d+):\s*(-?[\d.]+)"
            matches = re.findall(pattern, llm_output)

            # --- NEW: Check if the number of found joints is correct ---
            if len(matches) != 8:
                raise ValueError(f"Expected 8 joints, but found {len(matches)}.")

            # Sort the matches based on the joint index to ensure correct order
            sorted_matches = sorted(matches, key=lambda m: int(m[0]))
            
            # --- Optional but robust: Check for duplicate or skipped joints ---
            for i, match in enumerate(sorted_matches):
                if int(match[0]) != i:
                    raise ValueError(f"Joint index mismatch. Expected joint_{i} but found joint_{match[0]}. Check for duplicates or missing joints.")

            # Extract the numeric values from the sorted list
            joint_values = [float(match[1]) for match in sorted_matches]
            joint_vector = np.array(joint_values, dtype=np.float32)

            return joint_vector

        except (ValueError, IndexError) as e:
            # The error message will now be much more specific
            print(f"Error: {e} in the LLM output: {llm_output}")
            return None
    else:
        raise ValueError(f"Unsupported action type: {action_type}. Supported types are: {ActionType.DELTA_EEF}, {ActionType.ABS_EEF}, {ActionType.ABS_JOINT}.")


def compute_delta_eepose(pose1, pose2):
    """
    return the delta eepose between two poses: pose1 - pose2
    """
    pose1_transform = pose_to_transform(pose1)
    pose2_transform = pose_to_transform(pose2)
    delta_transform = pose1_transform @ np.linalg.inv(pose2_transform)
    delta_ee_pose = transform_to_pose(delta_transform)
    # tmp_pos1 = compute_target_pose_quat(pose2, delta_ee_pose)
    return delta_ee_pose

def transform_to_pose(transform):
    trans = transform[:3, 3]
    quat = R.from_matrix(transform[:3, :3]).as_quat()[[3, 0, 1, 2]] # from [x,y,z,w] to [w,x,y,z]
    return trans, quat


def pose_to_transform(pose):
    trans, quat = pose
    transform = np.eye(4)
    transform[:3, 3] = trans
    transform[:3, :3] = R.from_quat(quat[[1, 2, 3, 0]]).as_matrix() # from [w,x,y,z] to [x,y,z,w]
    return transform

def quat_to_rpy(pose: Tuple[np.ndarray, np.ndarray], degrees: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    translation, quaternion = pose
    rotation_object = R.from_quat([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
    rpy = rotation_object.as_euler('xyz', degrees=degrees)
    return translation, rpy

def rpy_to_quat(pose: Tuple[np.ndarray, np.ndarray], degrees: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    trans, rpy = pose
    rotation_object = R.from_euler('xyz', rpy, degrees=degrees)
    quat_scipy = rotation_object.as_quat()
    quaternion = np.array([quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]])
    pose = (trans, quaternion)
    return pose

def compute_delta_eepose_rpy(pose1, pose2, degrees: bool = True):
    delta_trans, delta_quat = compute_delta_eepose(pose1, pose2)
    delta_ee_pose_rpy = quat_to_rpy((delta_trans, delta_quat), degrees=degrees)
    return delta_ee_pose_rpy

def compute_target_pose_quat(pose2, delta_ee_pose):
    delta_transform = pose_to_transform(delta_ee_pose)
    pose2_transform = pose_to_transform(pose2)
    pose1_transform = delta_transform @ pose2_transform
    pose1 = transform_to_pose(pose1_transform)
    return pose1

def pad_to_square(image, resize_target=(480, 480)):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
    padding_value = np.array([103.53, 116.28, 123.675], dtype=np.uint8).tolist()    # bgr
    # padding_value = np.array([123.675, 116.28, 103.53], dtype=np.uint8).tolist()    # rgb
    h, w, _ = image.shape
    if h > w:
        pad_h = 0
        pad_w = (h - w) // 2
    else:
        pad_w = 0
        pad_h = (w - h) // 2
    padded_image = cv2.copyMakeBorder(image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=padding_value)
    resized_image = cv2.resize(padded_image, resize_target, interpolation=cv2.INTER_LINEAR)
    resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)  # Convert back to RGB
    return resized_image

def post_process_rpy(trans, rpy):
    return np.round(1000 * trans).astype(np.int32), np.round(rpy_bound(rpy)).astype(np.int32)

def post_process_quat(trans, quat):
    return np.round(1000 * trans).astype(np.int32), np.round(1000 * quat).astype(np.int32)

def post_process_qpos(qpos):
    return np.round(1000 * qpos).astype(np.int32)

def post_process_rpy_float(trans, rpy):
    return 1000 * trans, rpy_bound(rpy)

def get_ee_state(step_idx, data_dict):
    trans, rpy = quat_to_rpy(data_dict['ee_pose_state'][step_idx])
    return post_process_rpy(trans, rpy)

def get_delta_action(step_idx, data_dict, smooth_step=30):
    if step_idx + smooth_step >= len(data_dict['ee_pose_action']):
        trans, rpy = compute_delta_eepose_rpy(data_dict['ee_pose_action'][-1], data_dict['ee_pose_state'][step_idx])
        return post_process_rpy(trans, rpy)
    else:
        trans, rpy = compute_delta_eepose_rpy(data_dict['ee_pose_action'][step_idx + smooth_step], data_dict['ee_pose_state'][step_idx])
        return post_process_rpy(trans, rpy)

def get_gripper_state(qpos):
    return 1 if qpos[-1] >= 0.037 else 0

def get_gripper_action_smooth(step_idx, data_dict, smooth_step=13):
    sum_val = np.sum(data_dict['gripper_close'][step_idx:smooth_step + step_idx])
    if sum_val > 0:
        return 0
    else:
        return 1
    
def get_gripper_action(step_idx, data_dict):
    if data_dict['gripper_action'][step_idx][0] == 0.04 or data_dict['gripper_action'][step_idx][1] == 0.04:
        return 1
    else:
        return 0
    
def rpy_bound(rpy):
    # # boumd rpy to [-180, 180)
    rpy = np.array(rpy)
    rpy[rpy >= 180] -= 360
    rpy[rpy < -180] += 360
    return rpy



if __name__ == '__main__':
    # 定义测试用例
    test_cases = [
        # --- 相对动作测试 ---
        {
            "description": "1a: 相对动作 - 有效输入",
            "is_abs": False,
            "input": "action: {x: -129mm, y: 442mm, z: 183mm, roll: 2 degrees, pitch: 40 degrees, yaw: 0 degrees, open: 1}"
        },
        {
            "description": "1b: 相对动作 - 乱序有效输入",
            "is_abs": False,
            "input": "action: {roll: 2 degrees, open: 0, y: 442mm, pitch: 40 degrees, z: 183mm, yaw: 0 degrees, x: -129mm}"
        },
        {
            "description": "1c: 相对动作 - 无效输入 (缺少 'yaw')",
            "is_abs": False,
            "input": "action: {x: -129mm, y: 442mm, z: 183mm, roll: 2 degrees, pitch: 40 degrees, open: 1}"
        },
        # --- 绝对动作测试 ---
        {
            "description": "2a: 绝对动作 - 有效输入",
            "is_abs": True,
            "input": "action: {x: -11mm, y: 20mm, z: 33mm, quat: 0.5, -0.5, 0.5, 0.5, open: 1}"
        },
        {
            "description": "2b: 绝对动作 - 无效输入 (缺少 'quat')",
            "is_abs": True,
            "input": "action: {x: -11mm, y: 20mm, z: 33mm, open: 1}"
        },
        {
            "description": "2c: 绝对动作 - 无效输入 (缺少 'y')",
            "is_abs": True,
            "input": "action: {x: -11mm, z: 33mm, quat: 0.5, -0.5, 0.5, 0.5, open: 1}"
        },
        {
            "description": "2d: 绝对动作 - 无效输入 (quat 值不完整)",
            "is_abs": True,
            "input": "action: {x: -11mm, y: 20mm, z: 33mm, quat: 0.5, -0.5, 0.5, open: 1}"
        }
    ]

    # 遍历并执行所有测试用例
    for i, case in enumerate(test_cases):
        print(f"--- 测试用例 {case['description']} ---")
        print(f"输入: \"{case['input']}\"")
        print(f"调用: extract_action_vector(..., is_abs={case['is_abs']})")
        
        result = extract_action_vector(case['input'], is_abs=case['is_abs'])
        
        print("\n--- 输出 ---")
        if result:
            vec1, vec2, gripper = result
            if case['is_abs']:
                print(f"平移 (x, y, z): {vec1}")
                print(f"四元数 (quat):    {vec2}")
            else:
                print(f"相对平移 (x, y, z):   {vec1}")
                print(f"相对旋转 (r, p, y): {vec2}")
            print(f"夹爪状态 (open):        {gripper}")
        else:
            print("函数返回: None")
        
        print("-" * (len(case['description']) + 6) + "\n")