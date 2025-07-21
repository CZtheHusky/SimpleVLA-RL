from verl.utils.vla_utils.internvl.configuration_internvl_chat import InternVLChatConfig
from verl.utils.vla_utils.internvl.modeling_internvl_verl import InternVLForActionPrediction
from verl.utils.vla_utils.internvl.processing_internvl import InternVLImageProcessor, InternVLProcessor
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor, AutoModelForCausalLM
import torch

local_path = "deltaQ_backbone"
torch_dtype = torch.bfloat16
actor_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=local_path,
    torch_dtype=torch_dtype,
    config=actor_model_config,
    trust_remote_code=True,
).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True, use_fast=False)

processor = AutoProcessor.from_pretrained(local_path, trust_remote_code=True)
print(model)