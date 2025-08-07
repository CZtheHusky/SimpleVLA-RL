from transformers import AutoModelForCausalLM, AutoTokenizer
local_path = "/mnt/nfs_68/caozhe/workspace/vlav-project/maniskill_stack_cubes_dual/internvl2-2b/v0-20250729-171130/checkpoint-640"

tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)
model     = AutoModelForCausalLM.from_pretrained(local_path, trust_remote_code=True)

# 简单测试 prefix_allowed_tokens_fn
def dummy_fn(batch_id, input_ids):
    # 总是只允许 [0] 这个 token
    return [0]

input_ids = tokenizer("Hello", return_tensors="pt").input_ids
out = model.chat(
    tokenizer,
    pixel_values=None,
    prefix_allowed_tokens_fn=dummy_fn,
    do_sample=False,
).cuda()
print(tokenizer.decode(out[0]))
