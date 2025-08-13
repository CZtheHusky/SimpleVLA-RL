# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import itertools
from typing import Iterable, Tuple

from anyio import value
import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.workers.actor import BasePPOActor
from verl.utils.debug import gpu_memory
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits, log_probs_from_logits_all_rmpad
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
import verl.utils.torch_functional as verl_F
from codetiming import Timer
from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis
import os

__all__ = ['RobDataParallelPPOActor']

class RobDataParallelPPOActor(BasePPOActor):

    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
        logger=None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_remove_padding = self.config.get('use_remove_padding', False)
        print(f'Actor use_remove_padding={self.use_remove_padding}')
        print(f'PRM use dynamic bsz={self.config.get("use_dynamic_bsz", False)}')
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = False #self.ulysses_sequence_parallel_size > 1
        self.compute_entropy_from_logits = torch.compile(verl_F.entropy_from_logits, dynamic=True)
        self.logger = logger
        if torch.distributed.is_initialized():
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = 0
       
    # def process_tensor(self, tensor, pad_id):
    #     mask = tensor != pad_id
    #     if not torch.all(mask == mask[0:1], dim=1).all():
    #         raise ValueError("Padding error!")
    #     base_mask = mask[0]
    #     valid_len = base_mask.sum().item()
    #     return tensor[:, base_mask], valid_len
    
    def process_tensor(self, tensor: torch.Tensor, pad_id: int, no_padding=False):
        """
        裁剪 left padding 的 tensor，并返回有效区域和 mask。
        
        参数:
            tensor: [B, L] 的整数张量，left-padded。
            pad_id: padding 的 token ID。
            
        返回:
            cropped_tensor: [B, L_trimmed]，移除前导 pad 后的张量。
            mask: [B, L_trimmed]，布尔张量，True 表示有效 token。
        """
        mask = tensor != pad_id  # [B, L]，True 表示有效 token
        valid_starts = mask.float().argmax(dim=1)  # 每个 sample 第一个非 pad 的位置
        valid_len = valid_starts.min().item()  # 所有样本中最晚开始的有效 token
        max_start = valid_starts.max().item()  # 最早开始的有效 token
        consistent_shape = True
        if valid_len != max_start:
            consistent_shape = False
        if no_padding:
            cropped_mask = tensor[:, max_start:]
            cropped_tensor = tensor[:, max_start:]  # 去掉最前面的统一 pad
        else:
            cropped_tensor = tensor[:, valid_len:]  # 去掉最前面的统一 pad
            cropped_mask = mask[:, valid_len:]
        return cropped_tensor, (cropped_mask, consistent_shape, valid_len, max_start)

    
    def generate_traj_mask(self, end_step, traj_len):
        """
        Args:
            end_step: (batch_size,), 
            traj_len: 
        Returns:
            mask: (batch_size, traj_len),
        """
        steps = torch.arange(traj_len, device=end_step.device)  # (traj_len,)
        steps_expanded = steps.unsqueeze(0).expand(end_step.size(0), -1)
        mask = steps_expanded < end_step.unsqueeze(1)  # (batch_size, traj_len)
        return mask
    
    def apply_mask_with_grad_control(self, log_probs, entropy, mask):
        """
        Args:
            log_probs: (batch_size, traj_len, ...)
            entropy:   (batch_size, traj_len, ...)
            mask:      (batch_size, traj_len)
        Returns:
            log_probs_masked: 
            entropy_masked:   
        """
        mask_expanded = mask.unsqueeze(-1)  

        log_probs_masked = torch.where(
            mask_expanded,
            log_probs,
            torch.zeros_like(log_probs, requires_grad=False)  
        )

        entropy_masked = torch.where(
            mask_expanded,
            entropy,
            torch.zeros_like(entropy, requires_grad=False)   
        )

        return log_probs_masked, entropy_masked

    def _forward_micro_batch(self, micro_batch, temperature) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        micro_batch:
        
        Returns: 
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        
        batch_size = micro_batch['responses'].size(0)
        traj_len = micro_batch['responses'].size(1)
        tot_pad_len = micro_batch['input_ids'].size(2)
        
        assert all(micro_batch[key].size(0) == batch_size for key in ['responses', 'input_ids', 'attention_mask', 'pixel_values'])
        assert all(micro_batch[key].size(1) == traj_len for key in ['responses', 'input_ids', 'attention_mask', 'pixel_values'])
        assert all(micro_batch[key].size(2) == tot_pad_len for key in [ 'input_ids', 'attention_mask'])
        
            
        response_length = micro_batch['responses'].size(-1) # 7*8
        # self.logger.log(f"BS: {batch_size} TL: {traj_len} TOT: {tot_pad_len} RL: {response_length}")
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            attention_mask = micro_batch['attention_mask']
            pixel_values = micro_batch["pixel_values"]
            responses = micro_batch["responses"]

            input_ids = input_ids.reshape((batch_size * traj_len,) + input_ids.shape[2:])
            attention_mask = attention_mask.reshape((batch_size * traj_len,) + attention_mask.shape[2:])
            pixel_values = pixel_values.reshape((batch_size * traj_len,) + pixel_values.shape[2:])
            responses = responses.reshape((batch_size * traj_len,) + responses.shape[2:])
            input_ids_unpad, (cropped_mask, consistent_shape, valid_len, max_start) = self.process_tensor(input_ids, self.pad_token_id)
            attention_mask_unpad, (cropped_mask, consistent_shape, valid_len, max_start) = self.process_tensor(attention_mask, 0)
            if self.config.vla == "openvla-oft":
                # assert consistent_shape, "Input ids should have consistent shape after removing padding"
                logits = self.actor_module(input_ids=input_ids_unpad,
                                        attention_mask=attention_mask_unpad,
                                        pixel_values=pixel_values,
                                        )  # prevent model thinks we are generating
                
                assert self.actor_module.vocab_size == 32000
                start_index = self.actor_module.vocab_size - 256 
                logits = logits[..., -256-64:-64]  # Shape: [batch_size, seq_len, 256]
                responses = responses - start_index
                #assert (0<=responses<=255).all()
            
                logits = logits.div(temperature) 
                
                log_probs = logprobs_from_logits(logits, responses)
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
            
                assert len(log_probs.shape)==2 and len(entropy.shape)==2 
                log_probs = log_probs.reshape((batch_size, traj_len*8,7) )
                entropy = entropy.reshape((batch_size, traj_len*8,7) )

                mask = self.generate_traj_mask(micro_batch['finish_step'], traj_len*8)
                log_probs, entropy = self.apply_mask_with_grad_control(log_probs, entropy, mask)
                
                log_probs = log_probs.reshape((batch_size, traj_len*response_length))
                entropy = entropy.reshape((batch_size, traj_len*response_length)) 
                
            elif self.config.vla == "openvla":
                # assert consistent_shape, "Input ids should have consistent shape after removing padding"
                output = self.actor_module(input_ids=input_ids_unpad,
                                    attention_mask=attention_mask_unpad,
                                    pixel_values=pixel_values,
                                    use_cache=False)  # prevent model thinks we are generating
                logits = output.logits
                
                logits = logits[:, -response_length - 1:-1]  # (bsz, response_length)
                logits = logits.div(temperature) 
                
                log_probs = logprobs_from_logits(logits, responses)
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                #ADD
                
                log_probs = log_probs.reshape((batch_size, traj_len,) + log_probs.shape[1:])
                entropy = entropy.reshape((batch_size, traj_len,) + entropy.shape[1:])

                
                mask = self.generate_traj_mask(micro_batch['finish_step'], traj_len)
                log_probs, entropy = self.apply_mask_with_grad_control(log_probs, entropy, mask)
                
                log_probs = log_probs.reshape((batch_size, traj_len*response_length))
                entropy = entropy.reshape((batch_size, traj_len*response_length))

            elif self.config.vla == "internvl_chat":
                split_size = 30
                id_chunks   = input_ids_unpad.split(split_size, dim=0)
                attn_chunks = attention_mask_unpad.split(split_size, dim=0)
                pv_chunks   = pixel_values.split(split_size, dim=0)
                logits_list = []
                for id_ch, attn_ch, pv_ch in zip(id_chunks, attn_chunks, pv_chunks):
                    pv_ch = pv_ch.reshape(-1, *pv_ch.shape[-3:])  # Flatten the batch dimension
                    out = self.actor_module(
                        pixel_values=pv_ch,
                        input_ids=id_ch,
                        attention_mask=attn_ch,
                        use_cache=False  # pure forward
                    )
                    # Extract only the response-length slice and scale
                    lg = out.logits[:, -response_length - 1:-1].div(temperature)
                    logits_list.append(lg)

                # Concatenate all chunks: shape (B*T, response_length)
                logits_flat = torch.cat(logits_list, dim=0)
                # Compute log-probs and entropy
                log_probs = logprobs_from_logits(logits_flat, responses)
                entropy = verl_F.entropy_from_logits(logits_flat)

                log_probs = log_probs.reshape((batch_size, traj_len,) + log_probs.shape[1:])
                entropy = entropy.reshape((batch_size, traj_len,) + entropy.shape[1:])

                mask = self.generate_traj_mask(micro_batch['finish_step'], traj_len)
                log_probs, entropy = self.apply_mask_with_grad_control(log_probs, entropy, mask)
                # if torch.abs(log_probs).max() >= 15:
                #     breakpoint()

                # Reshape back: (B, T*response_length)
                log_probs = log_probs.reshape((batch_size, traj_len * response_length))
                entropy = entropy.reshape((batch_size, traj_len * response_length))

            return entropy, log_probs


    def validate_forward_micro_batch(self, micro_batch, temperature, actor_module) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        micro_batch:
        
        Returns: 
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        
        batch_size = micro_batch['responses'].size(0)
        traj_len = micro_batch['responses'].size(1)
        tot_pad_len = micro_batch['input_ids'].size(2)
        
        assert all(micro_batch[key].size(0) == batch_size for key in ['responses', 'input_ids', 'attention_mask', 'pixel_values'])
        assert all(micro_batch[key].size(1) == traj_len for key in ['responses', 'input_ids', 'attention_mask', 'pixel_values'])
        assert all(micro_batch[key].size(2) == tot_pad_len for key in [ 'input_ids', 'attention_mask'])
        
            
        response_length = micro_batch['responses'].size(-1) # 7*8
        # self.logger.log(f"BS: {batch_size} TL: {traj_len} TOT: {tot_pad_len} RL: {response_length}")
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            attention_mask = micro_batch['attention_mask']
            pixel_values = micro_batch["pixel_values"]
            responses = micro_batch["responses"]

            input_ids = input_ids.reshape((batch_size * traj_len,) + input_ids.shape[2:])
            attention_mask = attention_mask.reshape((batch_size * traj_len,) + attention_mask.shape[2:])
            pixel_values = pixel_values.reshape((batch_size * traj_len,) + pixel_values.shape[2:])
            responses = responses.reshape((batch_size * traj_len,) + responses.shape[2:])
            input_ids_unpad, (cropped_mask, consistent_shape, valid_len, max_start) = self.process_tensor(input_ids, self.pad_token_id)
            attention_mask_unpad, (cropped_mask, consistent_shape, valid_len, max_start) = self.process_tensor(attention_mask, 0)

            split_size = 30
            id_chunks   = input_ids_unpad.split(split_size, dim=0)
            attn_chunks = attention_mask_unpad.split(split_size, dim=0)
            pv_chunks   = pixel_values.split(split_size, dim=0)
            logits_list = []
            for id_ch, attn_ch, pv_ch in zip(id_chunks, attn_chunks, pv_chunks):
                pv_ch = pv_ch.reshape(-1, *pv_ch.shape[-3:])  # Flatten the batch dimension
                out = actor_module(
                    pixel_values=pv_ch,
                    input_ids=id_ch,
                    attention_mask=attn_ch,
                    use_cache=False  # pure forward
                )
                # Extract only the response-length slice and scale
                lg = out.logits[:, -response_length - 1:-1].div(temperature)
                logits_list.append(lg)

            # Concatenate all chunks: shape (B*T, response_length)
            logits_flat = torch.cat(logits_list, dim=0)
            # Compute log-probs and entropy
            log_probs = logprobs_from_logits(logits_flat, responses)
            entropy = verl_F.entropy_from_logits(logits_flat)

            log_probs = log_probs.reshape((batch_size, traj_len,) + log_probs.shape[1:])
            entropy = entropy.reshape((batch_size, traj_len,) + entropy.shape[1:])

            mask = self.generate_traj_mask(micro_batch['finish_step'], traj_len)
            log_probs, entropy = self.apply_mask_with_grad_control(log_probs, entropy, mask)
            # if torch.abs(log_probs).max() >= 15:
            #     breakpoint()

            # Reshape back: (B, T*response_length)
            log_probs = log_probs.reshape((batch_size, traj_len * response_length))
            entropy = entropy.reshape((batch_size, traj_len * response_length))

            return entropy, log_probs


    def _forward_micro_batch_update(self, input_ids, attention_mask, pixel_values, responses, temperature) -> Tuple[torch.Tensor, torch.Tensor]:
       
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            if self.config.vla == "openvla-oft":
                
                input_ids_unpad, (cropped_mask, consistent_shape, valid_len, max_start) = self.process_tensor(input_ids, self.pad_token_id)
                attention_mask_unpad, (cropped_mask, consistent_shape, valid_len, max_start) = self.process_tensor(attention_mask, 0)

                
                # assert consistent_shape, "Input ids should have consistent shape after removing padding"
                logits = self.actor_module(input_ids=input_ids_unpad,
                                                attention_mask=attention_mask_unpad,
                                                pixel_values=pixel_values,
                                                )  
                
                assert logits.requires_grad 
                
                assert self.actor_module.vocab_size == 32000
                start_index = self.actor_module.vocab_size - 256 
                logits = logits[..., -256-64:-64]  # Shape: [batch_size, seq_len, 256]
                responses = responses - start_index
                
                logits = logits.div(temperature) 
                
                log_probs = logprobs_from_logits(logits, responses)
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                
                log_probs = log_probs.reshape((1, -1))
                entropy = entropy.reshape((1, -1))
                
                return entropy, log_probs
            
            elif self.config.vla == "openvla":
                response_length = responses.size(-1)
                input_ids_unpad, (cropped_mask, consistent_shape, valid_len, max_start) = self.process_tensor(input_ids, self.pad_token_id)
                attention_mask_unpad, (cropped_mask, consistent_shape, valid_len, max_start) = self.process_tensor(attention_mask, 0)
                # assert consistent_shape, "Input ids should have consistent shape after removing padding"
                output = self.actor_module(input_ids=input_ids_unpad,
                                        attention_mask=attention_mask_unpad,
                                        pixel_values=pixel_values,
                                        use_cache=False)  # prevent model thinks we are generating
                logits = output.logits
                #
                
                logits = logits[:, -response_length - 1:-1]  # (bsz, response_length)
                logits = logits.div(temperature) 
                
                log_probs = logprobs_from_logits(logits, responses)
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                
                
                log_probs = log_probs.reshape((1, -1))
                entropy = entropy.reshape((1, -1))

                return entropy, log_probs
            
            elif self.config.vla == "internvl_chat":
                response_length = responses.size(-1)
                input_ids_unpad, (cropped_mask, consistent_shape, valid_len, max_start) = self.process_tensor(input_ids, self.pad_token_id)
                attention_mask_unpad, (cropped_mask, consistent_shape, valid_len, max_start) = self.process_tensor(attention_mask, 0)
                # if not consistent_shape:
                #     self.logger.log(f"_forward_micro_batch_update: warning: input_ids has inconsistent shape after removing padding, valid_len: {valid_len} max_start: {max_start}, this may cause issues in internvl_chat")
                pixel_values = pixel_values.reshape(-1, *pixel_values.shape[-3:])
                # assert consistent_shape, "Input ids should have consistent shape after removing padding"
                output = self.actor_module(input_ids=input_ids_unpad,
                                        attention_mask=attention_mask_unpad,
                                        pixel_values=pixel_values,
                                        use_cache=False)  # prevent model thinks we are generating
                logits = output.logits
                #
                
                logits = logits[:, -response_length - 1:-1]  # (bsz, response_length)
                logits = logits.div(temperature) 
                
                log_probs = logprobs_from_logits(logits, responses)
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                
                
                log_probs = log_probs.reshape((1, -1))
                entropy = entropy.reshape((1, -1))

                return entropy, log_probs
                

    def _forward_micro_batch_entropy(self, micro_batch, temperature) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = micro_batch['responses'].size(0)
        traj_len = micro_batch['responses'].size(1)
        tot_pad_len = micro_batch['input_ids'].size(2)
 
        assert all(micro_batch[key].size(0) == batch_size for key in ['responses', 'input_ids', 'attention_mask', 'pixel_values'])
        assert all(micro_batch[key].size(1) == traj_len for key in ['responses', 'input_ids', 'attention_mask', 'pixel_values'])
        assert all(micro_batch[key].size(2) == tot_pad_len for key in [ 'input_ids', 'attention_mask'])
            
        response_length = micro_batch['responses'].size(-1)
        #assert response_length == 7*8
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            #batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            pixel_values = micro_batch["pixel_values"]
            
            input_ids = input_ids.reshape((batch_size * traj_len,) + input_ids.shape[2:])
            attention_mask = attention_mask.reshape((batch_size * traj_len,) + attention_mask.shape[2:])
            pixel_values = pixel_values.reshape((batch_size * traj_len,) + pixel_values.shape[2:])
            
            
            input_ids_unpad, (cropped_mask, consistent_shape, valid_len, max_start) = self.process_tensor(input_ids, self.pad_token_id)
            attention_mask_unpad, (cropped_mask, consistent_shape, valid_len, max_start) = self.process_tensor(attention_mask, 0)

            if  self.config.vla == "openvla-oft":
                # assert consistent_shape, "Input ids should have consistent shape after removing padding"
                logits = self.actor_module(input_ids=input_ids_unpad,
                                                attention_mask=attention_mask_unpad,
                                                pixel_values=pixel_values,
                                                ) 
            
                assert self.actor_module.vocab_size == 32000
                start_index = self.actor_module.vocab_size - 256 
                logits = logits[..., -256-64:-64]  # Shape: [batch_size, seq_len, 256]
            
                logits = logits.div(temperature) 
            
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

                assert len(entropy.shape)==2 
                entropy = entropy.reshape((batch_size, traj_len*8,7) )
                mask = self.generate_traj_mask(micro_batch['finish_step'], traj_len*8)
                _, entropy = self.apply_mask_with_grad_control(entropy, entropy, mask)
                entropy = entropy.reshape((batch_size, traj_len*response_length))
                return entropy
            
            elif self.config.vla == "openvla":
                # assert consistent_shape, "Input ids should have consistent shape after removing padding"
                output = self.actor_module(input_ids=input_ids_unpad,
                                        attention_mask=attention_mask_unpad,
                                        pixel_values=pixel_values,
                                        use_cache=False)  # prevent model thinks we are generating
                logits = output.logits
                #
                
                
                logits = logits[:, -response_length - 1:-1]  # (bsz, response_length)
                logits = logits.div(temperature) 
                
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                #ADD

                entropy = entropy.reshape((batch_size, traj_len,) + entropy.shape[1:])
                mask = self.generate_traj_mask(micro_batch['finish_step'], traj_len)
                _, entropy = self.apply_mask_with_grad_control(entropy, entropy, mask)
                entropy = entropy.reshape((batch_size, traj_len*response_length))
                return entropy
            
            elif self.config.vla == "internvl_chat":
                # if not consistent_shape:
                    # self.logger.log(f"_forward_micro_batch_entropy: warning: input_ids has inconsistent shape after removing padding, valid_len: {valid_len} max_start: {max_start}, this may cause issues in internvl_chat")
                split_size = 30
                id_chunks   = input_ids_unpad.split(split_size, dim=0)
                attn_chunks = attention_mask_unpad.split(split_size, dim=0)
                pv_chunks   = pixel_values.split(split_size, dim=0)
                logits_list = []
                for id_ch, attn_ch, pv_ch in zip(id_chunks, attn_chunks, pv_chunks):
                    pv_ch = pv_ch.reshape(-1, *pv_ch.shape[-3:])  # Flatten the batch dimension
                    out = self.actor_module(
                        pixel_values=pv_ch,
                        input_ids=id_ch,
                        attention_mask=attn_ch,
                        use_cache=False  # pure forward
                    )
                    # Extract only the response-length slice and scale
                    lg = out.logits[:, -response_length - 1:-1].div(temperature)
                    logits_list.append(lg)

                # Concatenate all chunks: shape (B*T, response_length)
                logits_flat = torch.cat(logits_list, dim=0)
                # Compute log-probs and entropy
                entropy = verl_F.entropy_from_logits(logits_flat)

                entropy = entropy.reshape((batch_size, traj_len,) + entropy.shape[1:])

                mask = self.generate_traj_mask(micro_batch['finish_step'], traj_len)
                _, entropy = self.apply_mask_with_grad_control(entropy, entropy, mask)

                entropy = entropy.reshape((batch_size, traj_len * response_length))
                return entropy


    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        self.actor_optimizer.step()
        return grad_norm

    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        
        self.actor_module.eval()

        micro_batch_size = data.meta_info['micro_batch_size'] #256
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error # 1
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz'] #trues
        self.pad_token_id = data.meta_info['pad_token_id']
        
        select_keys = ['responses', 'input_ids', 'attention_mask', 'pixel_values',"finish_step"]
        batch = data.select(batch_keys=select_keys).batch

        if use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        # self.logger.log(f"Current GPU memory usage, before _forward_micro_batch: {gpu_memory()}")
        for micro_batch in micro_batches:
            # log current gpu memory
            with torch.no_grad():
                _, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature)
            log_probs_lst.append(log_probs)
        log_probs = torch.concat(log_probs_lst, dim=0)
        self.logger.log(f"Processed micro_batch with size {micro_batch.size()} and log_probs shape {log_probs.shape}, len of log_probs_lst: {len(log_probs_lst)} log_prob_shape: {log_probs.shape}")

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs


    def update_policy(self, data: DataProto):
        data = data.to('cuda')
        old_log_probs = self.compute_log_prob(data)
        data.batch['old_log_probs'] = old_log_probs
        self.actor_module.train()
        # if self.rank == 0: breakpoint()

        
        assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size == 0
        self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error

        select_keys = ['responses', 'input_ids', 'attention_mask', 'pixel_values', 'old_log_probs', 'advantages',"finish_step"]
        batch = data.select(batch_keys=select_keys).batch
        assert self.config.ppo_micro_batch_size == 1

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        dataloader = batch.split(self.config.ppo_mini_batch_size)
        metrics = {}

        # break_flag = False
        for batch_idx, data in enumerate(dataloader):
            # split batch into micro_batches
            mini_batch = data
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
            else:
                # split batch into micro_batches
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size)
            self.actor_optimizer.zero_grad()
            for test_idx, data in enumerate(micro_batches):
                data = data.cuda()  # actor device is cpu when using offload
                responses = data['responses']
                
                response_length = responses.size(1) *  responses.size(2)
                finish_step = data['finish_step'] * self.config.action_token_len
                steps = torch.arange(response_length, device=data['responses'].device)  # (traj_len,)
                steps_expanded = steps.unsqueeze(0).expand(data['responses'].size(0), -1)
                response_mask = steps_expanded < finish_step.unsqueeze(1)  # (batch_size, traj_len)
                
                response_mask_sum = response_mask.sum(axis=None)


                # bs traj_len * response_len
                old_log_prob = data['old_log_probs']
                advantages = data['advantages']
                
                #clip_ratio = self.config.clip_ratio
                clip_ratio_high = self.config.clip_ratio_high
                clip_ratio_low = self.config.clip_ratio_low
                entropy_coeff = self.config.entropy_coeff

                batch_size = data['responses'].size(0)
                traj_len = data['responses'].size(1)
                tot_pad_len = data['input_ids'].size(2)
                loss_info = {
                    #'actor/entropy_loss': entropy_loss.detach().item(),
                    'actor/pg_loss':0,
                    'actor/pg_clipfrac': 0,
                    'actor/ppo_kl': 0,
                }
                
                input_ids = data['input_ids']
                attention_mask = data['attention_mask']
                pixel_values = data["pixel_values"]
                responses = data["responses"]
                # 1 traj_len interact_len -> traj_len interact_len
                input_ids = input_ids.reshape((batch_size * traj_len,) + input_ids.shape[2:])
                attention_mask = attention_mask.reshape((batch_size * traj_len,) + attention_mask.shape[2:])
                pixel_values = pixel_values.reshape((batch_size * traj_len,) + pixel_values.shape[2:])
                responses = responses.reshape((batch_size * traj_len,) + responses.shape[2:])
                
                split_size = self.config.traj_mini_batch_size
                # split over traj_len, traj_len interact_len -> [(split_size, interact_len), ....]
                id_chunks   = input_ids.split(split_size, dim=0)
                attn_chunks = attention_mask.split(split_size, dim=0)
                pv_chunks   = pixel_values.split(split_size, dim=0)
                resp_chunks = responses.split(split_size, dim=0)
                # B L -> B * L
                for i, id_ch, attn_ch, pv_ch, res_ch in zip(range(0, traj_len, self.config.traj_mini_batch_size), id_chunks, attn_chunks, pv_chunks, resp_chunks):
                    # 1 split_size * response_len
                    # tmp_entropy, tmp_log_probs = self._forward_micro_batch({"responses": res_ch.unsqueeze(1), "input_ids": id_ch.unsqueeze(1), "attention_mask": attn_ch.unsqueeze(1), "pixel_values": pv_ch.unsqueeze(1), "finish_step": torch.ones((1, 1), dtype=id_ch.dtype, device=id_ch.device)}, temperature=1)
                    # tmp_entropy, tmp_log_probs = tmp_entropy.reshape(1, -1), tmp_log_probs.reshape(1, -1)
                    slice_id = i * self.config.action_token_len * self.config.action_chunks_len
                    actual_chunk = id_ch.size(0)
                    next_slice_id = (i + actual_chunk) * self.config.action_token_len * self.config.action_chunks_len
                    old_log_prob_tmp = old_log_prob[:, slice_id: next_slice_id]
                    advantages_tmp = advantages[:, slice_id: next_slice_id]
                    response_mask_tmp = response_mask[:, slice_id: next_slice_id]
                    response_mask_tmp_sum = response_mask_tmp.sum(axis=None)

                    entropy, log_prob = self._forward_micro_batch_update(
                        input_ids=id_ch, 
                        attention_mask=attn_ch, 
                        pixel_values=pv_ch, 
                        responses=res_ch, 
                        temperature=temperature
                    )
                    pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(old_log_prob=old_log_prob_tmp,
                                                                            log_prob=log_prob,
                                                                            advantages=advantages_tmp,
                                                                            eos_mask=response_mask_tmp,
                                                                            clip_ratio_high=clip_ratio_high,
                                                                            clip_ratio_low=clip_ratio_low)
                    # if pg_loss >= 500:
                    #     breakpoint()
                    #     from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor, AutoModelForCausalLM
                    #     local_path = "log_prob_compute_backup"
                    #     actor_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=True)
                    #     actor_module = AutoModelForCausalLM.from_pretrained(
                    #         pretrained_model_name_or_path=local_path,
                    #         torch_dtype=torch.bfloat16,
                    #         config=actor_model_config,
                    #         trust_remote_code=True,
                    #     ).cuda().eval()
                    #     with torch.no_grad():
                    #         validate_old_entropy, validate_old_log_prob = self.validate_forward_micro_batch(
                    #                 data, 
                    #                 temperature=temperature, 
                    #                 actor_module=actor_module
                    #         )


                    pg_loss = pg_loss * response_mask_tmp_sum / response_mask_sum
                    pg_clipfrac = pg_clipfrac * response_mask_tmp_sum / response_mask_sum
                    ppo_kl = ppo_kl * response_mask_tmp_sum / response_mask_sum
                    
                    policy_loss = pg_loss 
                    
                    loss = policy_loss / self.gradient_accumulation
                    loss_info['actor/pg_loss'] =  loss_info['actor/pg_loss'] + policy_loss.detach().item()
                    loss_info['actor/pg_clipfrac'] = loss_info['actor/pg_clipfrac'] + pg_clipfrac.detach().item()
                    loss_info['actor/ppo_kl'] = loss_info['actor/ppo_kl'] +  ppo_kl.detach().item()
                    msg = f"batch_idx: {batch_idx}, test_idx: {test_idx}\n"
                    for key, value in loss_info.items():
                        msg += f"{key}: {value:.4f}\n"
                    self.logger.log(msg)
                    # print(msg)
                    loss.backward()
                    append_to_dict(metrics, loss_info)
                # total_ppo_kl = torch.tensor(loss_info['actor/ppo_kl']).cuda()
                # torch.distributed.all_reduce(total_ppo_kl, op=torch.distributed.ReduceOp.AVG)
                # if total_ppo_kl >= 0.5:
                #     print("Terminating update policy, kl early stop")
                #     self.logger.log("Terminating update policy, kl early stop")
                #     self.actor_optimizer.zero_grad()
                #     break_flag = True
                #     break
            # torch.cuda.empty_cache()
            self.logger.log(f"before optimizer: {gpu_memory()}")
            grad_norm = self._optimizer_step()
            self.logger.log(f"after _optimizer_step: {gpu_memory()}")   
            data = {'actor/grad_norm': grad_norm.detach().item()}
            append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        self.logger.log(f"after final empty_cache: {gpu_memory()}")
        return metrics

    def compute_entropy(self, bacth_data: DataProto):
        
        if bacth_data.meta_info['train_mode'] ==True:
            self.actor_module.train()
            print("train mode")
        else:
            self.actor_module.eval()
            print("eval mode")

        assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size == 0
        self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size
        temperature = bacth_data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error

        select_keys = ['responses', 'input_ids', 'attention_mask', 'pixel_values', "finish_step"]
        batch = bacth_data.select(batch_keys=select_keys).batch

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        dataloader = batch.split(self.config.ppo_mini_batch_size)
        print("dataloader_length:", len(dataloader))
        
        metrics = {}
        for batch_idx, data in enumerate(dataloader):
            # split batch into micro_batches
            mini_batch = data
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
            else:
                # split batch into micro_batches
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size)

            for data in micro_batches:
                data = data.cuda()  # actor device is cpu when using offload
                responses = data['responses']
                response_length = responses.size(1) *  responses.size(2)
                finish_step = data['finish_step'] * self.config.action_token_len
                steps = torch.arange(response_length, device=data['responses'].device)  # (traj_len,)
                steps_expanded = steps.unsqueeze(0).expand(data['responses'].size(0), -1)
                response_mask = steps_expanded < finish_step.unsqueeze(1)  # (batch_size, traj_len)
                

                with torch.no_grad():
                    entropy = self._forward_micro_batch_entropy(micro_batch=data, temperature=temperature)
                    entropy_loss = verl_F.masked_mean(entropy, response_mask)

                if bacth_data.meta_info['is_filtered'] and bacth_data.meta_info['train_mode']:
                    data = {
                        'actor_after/entropy_loss_train': entropy_loss.detach().item(),
                    }
                    append_to_dict(metrics, data)
                elif bacth_data.meta_info['is_filtered'] and not bacth_data.meta_info['train_mode']:
                    data = {
                        'actor_after/entropy_loss_eval': entropy_loss.detach().item(),
                    }
                    append_to_dict(metrics, data)
                elif not bacth_data.meta_info['is_filtered'] and bacth_data.meta_info['train_mode']:
                    data = {
                        'actor_before/entropy_loss_train': entropy_loss.detach().item(),
                    }
                    append_to_dict(metrics, data)
                elif not bacth_data.meta_info['is_filtered'] and not bacth_data.meta_info['train_mode']:
                    data = {
                        'actor_before/entropy_loss_eval': entropy_loss.detach().item(),
                    }
                    append_to_dict(metrics, data)
                        
                
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return metrics


    def update_policy_legacy(self, data: DataProto):
        self.actor_module.train()

        assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size == 0
        self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error

        select_keys = ['responses', 'input_ids', 'attention_mask', 'pixel_values', 'old_log_probs', 'advantages',"finish_step"]
        batch = data.select(batch_keys=select_keys).batch
        assert self.config.ppo_micro_batch_size == 1

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        dataloader = batch.split(self.config.ppo_mini_batch_size)
        metrics = {}
        # self.logger.log(f"Current GPU memory usage, before update_policy: {gpu_memory()}")
        for batch_idx, data in enumerate(dataloader):
            # split batch into micro_batches
            mini_batch = data
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
            else:
                # split batch into micro_batches
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size)

            self.actor_optimizer.zero_grad()
            for test_idx, data in enumerate(micro_batches):
                data = data.cuda()  # actor device is cpu when using offload
                responses = data['responses']
                
                response_length = responses.size(1) *  responses.size(2)
                finish_step = data['finish_step'] * self.config.action_token_len
                steps = torch.arange(response_length, device=data['responses'].device)  # (traj_len,)
                steps_expanded = steps.unsqueeze(0).expand(data['responses'].size(0), -1)
                response_mask = steps_expanded < finish_step.unsqueeze(1)  # (batch_size, traj_len)
                
                response_mask_sum = response_mask.sum(axis=None)

                old_log_prob = data['old_log_probs']
                advantages = data['advantages']
                
                #clip_ratio = self.config.clip_ratio
                clip_ratio_high = self.config.clip_ratio_high
                clip_ratio_low = self.config.clip_ratio_low
                entropy_coeff = self.config.entropy_coeff

                batch_size = data['responses'].size(0)
                traj_len = data['responses'].size(1)
                tot_pad_len = data['input_ids'].size(2)
                
                
                input_ids = data['input_ids']
                attention_mask = data['attention_mask']
                pixel_values = data["pixel_values"]
                responses = data["responses"]
                
                input_ids = input_ids.reshape((batch_size * traj_len,) + input_ids.shape[2:])
                attention_mask = attention_mask.reshape((batch_size * traj_len,) + attention_mask.shape[2:])
                pixel_values = pixel_values.reshape((batch_size * traj_len,) + pixel_values.shape[2:])
                responses = responses.reshape((batch_size * traj_len,) + responses.shape[2:])
                loss_info = {
                    #'actor/entropy_loss': entropy_loss.detach().item(),
                    'actor/pg_loss':0,
                    'actor/pg_clipfrac': 0,
                    'actor/ppo_kl': 0,
                }
                
                assert traj_len % self.config.traj_mini_batch_size == 0, f"traj_len {traj_len} must be divisible by traj_mini_batch_size {self.config.traj_mini_batch_size}"
                traj_split_num = int(traj_len / self.config.traj_mini_batch_size)
                # B L -> B * L
                # if self.rank == 0: breakpoint()
                for i in range(0, traj_len, int(traj_len / traj_split_num)):
                    id_ch = input_ids[i:i+int(traj_len/traj_split_num)]
                    attention_mask_ch = attention_mask[i:i+int(traj_len/traj_split_num)]
                    pixel_values_ch = pixel_values[i:i+int(traj_len/traj_split_num)]
                    responses_ch = responses[i:i+int(traj_len/traj_split_num)]
                    entropy, log_prob = self._forward_micro_batch_update(
                        input_ids=id_ch,
                        attention_mask=attention_mask_ch,
                        pixel_values=pixel_values_ch,
                        responses=responses_ch,
                        temperature=temperature
                    )
                    slice_id = i * self.config.action_token_len * self.config.action_chunks_len
                    actual_chunk = id_ch.size(0)
                    next_slice_id = (i + actual_chunk) * self.config.action_token_len * self.config.action_chunks_len
                    assert next_slice_id <= old_log_prob.shape[-1], f"next_slice_id {next_slice_id} exceeds old_log_prob size {old_log_prob.shape[-1]}"
                    old_log_prob_tmp = old_log_prob[:, slice_id: next_slice_id]
                    advantages_tmp = advantages[:, slice_id: next_slice_id]
                    response_mask_tmp = response_mask[:, slice_id: next_slice_id]

                    pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(old_log_prob=old_log_prob_tmp,
                                                                            log_prob=log_prob,
                                                                            advantages=advantages_tmp,
                                                                            eos_mask=response_mask_tmp,
                                                                            clip_ratio_high=clip_ratio_high,
                                                                            clip_ratio_low=clip_ratio_low)

                    response_mask_tmp_sum = response_mask_tmp.sum(axis=None)
                    pg_loss = pg_loss * response_mask_tmp_sum
                    pg_clipfrac = pg_clipfrac * response_mask_tmp_sum / response_mask_sum
                    ppo_kl = ppo_kl * response_mask_tmp_sum / response_mask_sum
                    
                    policy_loss = pg_loss / response_mask_sum
                    
                    loss = policy_loss / self.gradient_accumulation
                    loss.backward()
                    loss_info['actor/pg_loss'] =  loss_info['actor/pg_loss'] + policy_loss.detach().item()
                    loss_info['actor/pg_clipfrac'] = loss_info['actor/pg_clipfrac'] + pg_clipfrac.detach().item()
                    loss_info['actor/ppo_kl'] = loss_info['actor/ppo_kl'] +  ppo_kl.detach().item()
                append_to_dict(metrics, loss_info)
            self.logger.log(f"after micro batch update: {gpu_memory()}")
            self.logger.log(f"after _optimizer_step: {gpu_memory()}")
            data = {'actor/grad_norm': grad_norm.detach().item()}
            append_to_dict(metrics, data)
            torch.cuda.empty_cache()
            self.logger.log(f"after empty_cache: {gpu_memory()}")
        grad_norm = self._optimizer_step()
        self.actor_optimizer.zero_grad()
        self.logger.log(f"after zero_grad: {gpu_memory()}")
        torch.cuda.synchronize()
        self.logger.log(f"after synchronize: {gpu_memory()}")
        torch.distributed.barrier()
        self.logger.log(f"after barrier: {gpu_memory()}")
        torch.cuda.empty_cache()
        self.logger.log(f"after final empty_cache: {gpu_memory()}")
        return metrics

    

def debug_tensor(t, name):
    return f"{name}: requires_grad={t.requires_grad}, grad_fn={t.grad_fn}, shape={t.shape}, dtype={t.dtype}"
