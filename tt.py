traj_len = 148
traj_batch = 30
traj_split_num = traj_len // traj_batch


for i in range(0, traj_len, traj_batch):
    print(i)


import torch

bs = 2
length = 10
desired = 12
generation_output = torch.ones([bs, length])
print(generation_output.shape)
generation_output = torch.concatenate([generation_output, torch.zeros([*generation_output.shape[:-1], 2], dtype=generation_output.dtype, device=generation_output.device)], dim=-1)
print(generation_output.shape)