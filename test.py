from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor, AutoModelForCausalLM
import torch



local_path = "deltaQ_backbone"
torch_dtype = torch.bfloat16
actor_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=True)

# model = AutoModelForCausalLM.from_pretrained(
#     pretrained_model_name_or_path=local_path,
#     torch_dtype=torch_dtype,
#     config=actor_model_config,
#     trust_remote_code=True,
# ).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True, use_fast=False)

# processor = AutoProcessor.from_pretrained(local_path, trust_remote_code=True)

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
    # num tokens of the response
    num_tokens = len(tokenizer.encode(response))
    print(f"Response: {response}, Num Tokens: {num_tokens}")
    break
tmp_response = "2, 4, 1, 3, 0, -2, 3, 40"
num_tokens = len(tokenizer.encode(tmp_response))
print(f"Response: {tmp_response}, Num Tokens: {num_tokens}")

tmp_response = "2 4 1 3 0 -2 3 40"
num_tokens = len(tokenizer.encode(tmp_response))
print(f"Response: {tmp_response}, Num Tokens: {num_tokens}")

tmp_response = "2 -4 1 3 0 -2 -3 40"
num_tokens = len(tokenizer.encode(tmp_response))
print(f"Response: {tmp_response}, Num Tokens: {num_tokens}")