from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor, AutoModelForCausalLM
import torch

# import mani_skill.envs
# import gymnasium as gym
# from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
# N = 9
# env = gym.make("PickCube-v1", num_envs=N, reconfiguration_freq=None)
# env = ManiSkillVectorEnv(env, num_envs=N, ignore_terminations=False, auto_reset=True, record_metrics=True)
# env.action_space # shape (N, D)
# env.single_action_space # shape (D, )
# env.observation_space # shape (N, ...)
# env.single_observation_space # shape (...)
# obs, _ = env.reset()
# obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
# # obs (N, ...), rew (N, ), terminated (N, ), truncated (N, )



local_path = "/mnt/nfs_68/caozhe/workspace/vlav-project/maniskill_stack_cubes_dual/internvl2-2b/v0-20250729-171130/checkpoint-640"
torch_dtype = torch.bfloat16
actor_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=True)

# model = AutoModelForCausalLM.from_pretrained(
#     pretrained_model_name_or_path=local_path,
#     torch_dtype=torch_dtype,
#     config=actor_model_config,
#     trust_remote_code=True,
# ).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True, use_fast=False)
# img_context_token_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>',)
# # processor = AutoProcessor.from_pretrained(local_path, trust_remote_code=True)
# print(img_context_token_id)

import json

def read_jsonl_standard(file_path: str) -> list:
    """
    使用 Python 标准库逐行读取 JSONL 文件。
    
    :param file_path: JSONL 文件的路径。
    :return: 一个包含所有解析后的JSON对象（字典）的列表。
    """
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 移除行尾可能存在的空白字符（包括换行符）
                clean_line = line.strip()
                if clean_line:  # 确保不是空行
                    # 解析当前行
                    data.append(json.loads(clean_line))
    except FileNotFoundError:
        print(f"错误：文件未找到于 '{file_path}'")
    except json.JSONDecodeError as e:
        print(f"错误：文件 '{file_path}' 中存在JSON解析错误: {e}")
    
    return data

data = read_jsonl_standard("deltaQ_backbone/inference_log.jsonl")
for data_dict in data:
    response = data_dict['response']
    query = data_dict['question']
    # num tokens of the response
    num_tokens = len(tokenizer.encode(response))
    print(f"Response: {response}, Num Tokens: {num_tokens}")
    num_tokens = len(tokenizer.encode(query))
    print(f"Query: {query}, Num Tokens: {num_tokens}")
    break

current_strings = [
    # '{', 
    # ' ', 
    # '{ ', 
    # '-', 
    # '2', 
    # '-4', 
    # '1', 
    # '3', 
    # '0', 
    # '-2', 
    # '3', 
    # '40', 
    # '}', 
    # "{2 -4 1 3 0 -2 -3}", 
    # "{2 -4 1 3 0 -2 -3}{2 -4 1 3 0 -2 -3}{2 -4 1 3 0 -2 -3}{2 -4 1 3 0 -2 -3}", 
    # "2 -4 1 3 0 -2 -3", 
    "2 -4 1 3 0 -2 -3|", 
    "2 -4 1 3 0 -2 -3| 2 -4 1 3 0 -2 -3|", 
    "0 0 0 0 0 0 1|",
    "0 -1 0 0 0 0 1| 6 13 3 -3 7 -19 1|",
    "0 -1 0 0 0 0 1| 6 13 3 -3 7 -19 1| 6 14 3 -3 7 -20 1| 5 10 2 -3 5 -14 1| 3 6 1 -2 3 -9 1| 1 3 1 0 0 0 1| 0 0 0 0 0 0 1| 0 0 0 0 0 0 1|",
    "0 -1 0 0 0 0 1|",
    "0 -1 0 0 0 0 1| 6 13 3 -3 7 -19 1|", 
    "0 -1 0 0 0 0 1| 6 13 3 -3 7 -19 1| 6 14 3 -3 7 -20 1| 5 10 2 -3 5 -14 1| 3 6 1 -2 3 -9 1| 1 3 1 0 0 0 1| 0 0 0 0 0 0 1| 0 0 0 0 0 0 1|",
    "-18 -35 -28 8 -9 70 1|",
    "-18 -35 -28 8 -9 70 1| -30 -57 -44 13 -15 113 1|",
    "-18 -35 -28 8 -9 70 1| -30 -57 -44 13 -15 113 1| -42 -80 -62 19 -21 160 1|",
    "-18 -35 -28 8 -9 70 1| -30 -57 -44 13 -15 113 1| -42 -80 -62 19 -21 160 1| -56 -104 -80 24 -26 207 1| -70 -128 -99 30 -31 257 1| -86 -152 -117 36 -35 307 1| -104 -175 -137 42 -40 358 1| -123 -198 -156 48 -44 410 1|",
    # "2 -4 1 -3 0 -2 -3",
    # "2 -4 1 3 0 -2 -3 2 -4 1 3 0 -2 -3 2 -4 1 3 0 -2 -3 2 -4 1 3 0 -2 -3 2 -4 1 3 0 -2 -3"
]

for st_r in current_strings:
    tmp_response = st_r
    token = tokenizer.encode(tmp_response)
    num_tokens = len(token)
    print(f"Response: {tmp_response}\nNum Tokens: {num_tokens}\nToken: {token}\n--------------------------------")
    # 1) 拆成 token 字符串
    toks = tokenizer.tokenize(st_r)
    # 2) 编成 id
    ids  = tokenizer.convert_tokens_to_ids(toks)
    # print(f"原始字符串：{st_r}")
    # for tok, idx in zip(toks, ids):
    #     print(f"  Token: |{tok}|   →   ID: |{idx}|")
    # print()
tmp_response = "2, 4, 1, 3, 0, -2, 3, 40"
num_tokens = len(tokenizer.encode(tmp_response))
print(f"Response: {tmp_response}, Num Tokens: {num_tokens}")

tmp_response = "2 4 1 3 0 -2 3 40"
num_tokens = len(tokenizer.encode(tmp_response))
print(f"Response: {tmp_response}, Num Tokens: {num_tokens}")

tmp_response = "{2 -4 1 3 0 -2 -3}"
num_tokens = len(tokenizer.encode(tmp_response))
print(f"Response: {tmp_response}, Num Tokens: {num_tokens}")

response = ''
for i in range(3):
    response = tmp_response
    print(f"Num Tokens: {len(tokenizer.encode(response))}")

tmp_response = "{2 -4 1 3 0 -2 -3}{2 -4 1 3 0 -2 -3}{2 -4 1 3 0 -2 -3}"
num_tokens = len(tokenizer.encode(tmp_response))
print(f"Response: {tmp_response}, Num Tokens: {num_tokens}")