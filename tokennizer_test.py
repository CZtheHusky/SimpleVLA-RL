from transformers import AutoTokenizer, AutoModelForCausalLM


local_path = "/mnt/nfs_68/caozhe/workspace/vlav-project/maniskill_stack_cubes_dual/internvl2-2b/v0-20250729-171130/checkpoint-640"
tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)
model     = AutoModelForCausalLM.from_pretrained(local_path, trust_remote_code=True)

# 1. Tokenizer 的 vocab
#    注意：对于 PreTrainedTokenizerFast，.vocab_size == len(get_vocab())
tokenizer_size = tokenizer.vocab_size
vocab_dict     = tokenizer.get_vocab()        # { token_str: token_id, … }

# 2. 模型实际的 embedding 大小
embed_size = model.get_input_embeddings().num_embeddings

print(f"Tokenizer.vocab_size = {tokenizer_size}")
print(f"Model embedding size  = {embed_size}")

# 3. 如果两者不一致，找出哪些 token_id 越界
if tokenizer_size != embed_size:
    # 所有在 tokenizer 里，但 id >= embed_size 的 token
    oob = [tok for tok, idx in vocab_dict.items() if idx >= embed_size]
    print(f"\n>>> 一共有 {len(oob)} 个 token_id 在模型 embedding 范围外：")
    print(oob[:50], "…")   # 只看前 50 个
else:
    print("两者一致，没有越界 token。")
