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
Rollout with huggingface models.
TODO: refactor this class. Currently, it will hang when using FSDP HybridShard. We should actually create a single GPU model.
Then, get full state_dict and bind the state_dict to the single GPU model. Then, use the single GPU model to perform generation.
"""
import contextlib
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.utils.rnn import pad_sequence

from verl import DataProto
from verl.utils.torch_functional import get_eos_mask
import verl.utils.torch_functional as verl_F
from .base import BaseRollout

from transformers import GenerationConfig, AutoProcessor

from verl.utils.libero_utils import get_libero_env, get_libero_dummy_action, get_image_resize_size, get_libero_image, get_libero_wrist_image, quat2axisangle, normalize_gripper_action, invert_gripper_action, save_rollout_video
import numpy as np
from PIL import Image
import tensorflow as tf
from verl import DataProto
from libero.libero import benchmark
from codetiming import Timer
from collections import deque
from copy import deepcopy
import random
import time
import multiprocessing
import gc
# from torch.multiprocessing import Process, Queue
from multiprocessing import Process, Queue
from collections import defaultdict
from enum import Enum
from verl.utils.env_utils.utils import obs_process, extract_action_vector, assemble_action_vla, action_decode, TaskSuite
from verl.workers.rollout.env_workers.libero_env_worker import center_crop_image
import ray
import os

__all__ = ['RobHFRollout']

OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

class ENV_TYPE(Enum):
    VENV = 0
    SINGLE_ENV = 1


class RobHFRollout(BaseRollout):

    def __init__(self, module: nn.Module, config):
        super().__init__()
        self.config = config
        self.module = module
        self.rank = int(os.environ.get("RANK", 0))
        self.processor = AutoProcessor.from_pretrained(config.pretrained_checkpoint, trust_remote_code=True)
        if config.vla == "internvl_chat":
            self.response_length = 16 if config.response_length is None else config.response_length
        self.vla_preprocess()
        self.process_kwargs = {}
        if config.task_suite_name == "grutopia":
            self.task_suite = TaskSuite.GRUTOPIA
            self.env_type = ENV_TYPE.VENV
            from verl.workers.rollout.env_workers.grutopia_env_worker import env_worker
        elif config.task_suite_name == "maniskill":
            self.task_suite = TaskSuite.MANISKILL
            if config.dual_cam:
                self.process_kwargs['dual_cam'] = True
            else:
                self.process_kwargs['dual_cam'] = False
            self.env_type = ENV_TYPE.VENV
            self.max_steps = {
                "StackCube-v1": 10,
            }
            from verl.workers.rollout.env_workers.maniskill_env_worker import env_worker, EnvActor
            self.env_actor = EnvActor()
        elif "libero" in config.task_suite_name:
            self.max_steps = {   
                "libero_spatial": 512,   # max step length 193
                "libero_object": 512,    # max step length 254
                "libero_goal": 512,      # max step length 270
                "libero_10": 512,        # max step length 505
                "libero_90": 512         # max step length 373 org 400 now change to 512
            }
            self.task_suite = TaskSuite.LIBERO
            self.env_type = ENV_TYPE.SINGLE_ENV
            from verl.workers.rollout.env_workers.libero_env_worker import env_worker
        self.env_worker = env_worker
        
        #oft add
        # unnorm_key=config.unnorm_key
        # if  unnorm_key not in self.module.norm_stats and f"{unnorm_key}_no_noops" in self.module.norm_stats:
        #     unnorm_key = f"{unnorm_key}_no_noops"
        # assert unnorm_key in self.module.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"
        # self.config.unnorm_key = unnorm_key
        #add end
        # gpus = tf.config.experimental.list_physical_devices('GPU')
        # if gpus:
        #     for gpu in gpus:  
        #         tf.config.experimental.set_memory_growth(gpu, True)
    
    def vla_preprocess(self):
        if self.config.vla in ["openvla","openvla-oft"]:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:  
                    tf.config.experimental.set_memory_growth(gpu, True)
        
        if self.config.vla in ["openvla-oft"]:
            if  self.config.unnorm_key not in self.module.norm_stats and f"{self.config.unnorm_key}_no_noops" in self.module.norm_stats:
                self.config.unnorm_key = f"{self.config.unnorm_key}_no_noops"
            assert self.config.unnorm_key in self.module.norm_stats, f"Action un-norm key {self.config.unnorm_key} not found in VLA `norm_stats`!"


    def generate_sequences(self, prompts):
        start_time = time.time()
        batch_size = prompts.batch.batch_size[0]
        
        if prompts.meta_info.get('n_samples') is None:  # validatino rollout
            micro_batch_size = self.config.val_micro_batch_size if self.config.val_micro_batch_size is not None else 1
        else:
            micro_batch_size = self.config.get('micro_batch_size', batch_size)  # batch size for training
        if self.task_suite == TaskSuite.MANISKILL:
            assert micro_batch_size > 1 and batch_size > 1, "Batch size (num venvs) must be greater than 1 to avoid env re-initialization PHYSIX Errors."
        num_chunks = max(batch_size // micro_batch_size, 1)
        # assert batch_size % micro_batch_size == 0, f"Batch size {batch_size} is not divisible by micro batch size {micro_batch_size}."    # avoid changing the num of venvs
        batch_prompts = prompts.chunk(chunks=num_chunks)
        # if self.rank == 0:
        #     breakpoint()
        output = [self._generate_minibatch(p) for p in batch_prompts]
        output = DataProto.concat(output)
        print("Batch generation time:", time.time() - start_time)
        return output
    
    
    def process_input(self, inputs:list, task_descriptions:list):
        if self.config.vla == "internvl_chat":
            batchdata = obs_process(inputs, task_descriptions, self.task_suite, **self.process_kwargs)
            return batchdata
        batchdata = {"input_ids":[],"attention_mask":[],"pixel_values":[]}  
        
        for i in range(len(inputs)):
            input = inputs[i]
            task_description = task_descriptions[i]
           
            image = Image.fromarray(input["full_image"]).convert("RGB")
            if self.config.center_crop:
                image = center_crop_image(image)
            prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"
            batch_feature  = self.processor(prompt, image)
            
            if "wrist_image" in input.keys():
                wrist_image = Image.fromarray(input["wrist_image"]).convert("RGB")
                if self.config.center_crop:
                    wrist_image = center_crop_image(wrist_image)
                wrist_batch_feature = self.processor(prompt, wrist_image)
                primary_pixel_values = batch_feature["pixel_values"]
                batch_feature["pixel_values"] = torch.cat([primary_pixel_values] + [wrist_batch_feature["pixel_values"]], dim=1)
                
            input_ids = batch_feature["input_ids"]
            attention_mask = batch_feature["attention_mask"]
            pixel_values = batch_feature["pixel_values"]
            
            if not torch.all(input_ids[:, -1] == 29871):
                input_ids = torch.cat(
                    (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
                )
                if self.config.vla in ["openvla-oft"]:
                    attention_mask = torch.cat(
                        (attention_mask, torch.unsqueeze(torch.Tensor([True]).bool(), dim=0).to(attention_mask.device)), dim=1
                    )
            
            batchdata["input_ids"].append(input_ids)    
            batchdata["attention_mask"].append(attention_mask)    
            batchdata["pixel_values"].append(pixel_values)    
        
        
        device = torch.device('cuda') 
        
        if self.config.vla in ["openvla-oft"]:
            batchdata["input_ids"] = [x.transpose(0, 1) for x in batchdata["input_ids"]]
            batchdata["attention_mask"] = [x.transpose(0, 1) for x in batchdata["attention_mask"]]
            batchdata["input_ids"] = pad_sequence(batchdata["input_ids"], batch_first=True, padding_value=self.processor.tokenizer.pad_token_id).squeeze(-1).to(device)
            batchdata["attention_mask"] = pad_sequence(batchdata["attention_mask"], batch_first=True, padding_value=0).squeeze(-1).to(device)
            
            padding_mask = batchdata["input_ids"].ne(self.processor.tokenizer.pad_token_id)
            assert  torch.all(padding_mask==batchdata["attention_mask"].ne(0))
            padding_mask = ~padding_mask
            padding_mask = padding_mask.int() 
            sorted_indices = torch.argsort(padding_mask, dim=1, descending=True, stable=True)
            batchdata["input_ids"] = torch.gather(batchdata["input_ids"], 1, sorted_indices)
            batchdata["attention_mask"] = torch.gather(batchdata["attention_mask"], 1, sorted_indices)
            batchdata["pixel_values"] = torch.cat(batchdata["pixel_values"] , dim=0).to(device)
            assert torch.all(batchdata["attention_mask"].ne(0) == batchdata["input_ids"].ne(self.processor.tokenizer.pad_token_id))
        else:
            for key in ["input_ids", "attention_mask", "pixel_values"]:
                batchdata[key] = torch.cat(batchdata[key], dim=0).to(device)

        return batchdata
   
    def _generate_minibatch(self, prompts):
        if self.env_type == ENV_TYPE.SINGLE_ENV:
            return self._generate_minibatch_libero(prompts)
        elif self.env_type == ENV_TYPE.VENV:
            return self._venv_generate_minibatch(prompts)
                           
    def _venv_generate_minibatch(self, prompts):
        self.module.eval()
        meta_info = prompts.meta_info
        n_samples = meta_info.get('n_samples', 1)
        env_unique_id = prompts.batch['env_unique_id'].repeat_interleave(n_samples, dim=0)
        task_suite_name = prompts.non_tensor_batch['task_suite_name']
        task_instruction = np.repeat(prompts.non_tensor_batch['task_instruction'], n_samples)
        env_id = np.repeat(prompts.non_tensor_batch['env_id'], n_samples)
        is_valid = meta_info.get('n_samples') is None
        global_steps = meta_info.get('global_steps', 0) if is_valid else 0
        # infos_print = f"-------------------------------------------------\nenv_unique_id: {prompts.batch['env_unique_id']}\nshape: {prompts.batch['env_unique_id'].shape}\ntask_instruction: {prompts.non_tensor_batch['task_instruction']}\ntask_suite_name: {task_suite_name}\nenv_id: {prompts.non_tensor_batch['env_id']}\nglobal_steps: {global_steps}\nEnv_id: {env_id[0]}\nNum of envs: {len(env_unique_id.cpu().numpy().squeeze(1))}\nis_valid: {is_valid}\n-------------------------------------------------"
        # print(infos_print)
        max_steps = self.max_steps[env_id[0]]
        batch_size = env_unique_id.size(0)    # Num of grpo samples
        init_data = self.env_actor.init_venv(
            env_id,
            env_unique_id.cpu().numpy().squeeze(1),
            task_instruction,
            is_valid,
            global_steps,
            max_steps
        )
        task_records = []
        valid_video = defaultdict(list)
        task_instructions = init_data["task_instructions"]
        inputs = init_data['obs']
        task_records = {
            "complete": init_data['complete'],
            "finish_step": init_data['finish_step'],
            "task_file_name": init_data['task_file_name']
        }
        if is_valid:
            for venv_index in init_data['task_file_name'].keys():
                valid_video[venv_index].append(init_data['valid_images'][int(venv_index)])
        vla_history = []
        step = 0
        is_already_done = np.zeros(len(env_unique_id), dtype=bool)
        while step < max_steps:
            current_inputs = inputs
            current_task_instructions = task_instructions
            # print(f"step: {step} processing input")
            vla_input = self.process_input(current_inputs, current_task_instructions)
            vla_input.update(meta_info)
            vla_output = self._generate_one_step(vla_input, step)
            actions = vla_output["action"]
            
            step_data = {
                    "responses": vla_output["responses"],
                    "input_ids": vla_output["input_ids"],
                    "attention_mask": vla_output["attention_mask"],
                    "pixel_values": vla_output["pixel_values"],
                    "action": actions,
                    "step": step
                }
            vla_history.append(step_data)
            output = self.env_actor.step(actions)
            is_complete = output['complete']
            finish_step = output['finish_step']
            for venv_index in range(len(is_already_done)):
                if not is_already_done[venv_index]:
                    if is_complete[venv_index]:
                        is_already_done[venv_index] = True
                    task_records['complete'][venv_index] = is_complete[venv_index]
                    task_records['finish_step'][venv_index] = finish_step[venv_index]
                    if is_valid:
                        valid_video[venv_index].append(output['valid_images'][venv_index])
            inputs = output['obs']
            step += self.config.action_chunks_len
        torch.cuda.empty_cache()
        if is_valid:
            for venv_idx, images in valid_video.items():
                complete = task_records['complete'][venv_idx]
                task_file = task_records['task_file_name'][venv_idx]
                save_rollout_video(
                    images,
                    self.config.experiment_name,
                    task_file,
                    global_steps,
                    complete
                )    
        self.module.train()
        
        batch = {
                'responses': [],
                'input_ids': [],  # here input_ids become the whole sentences
                'attention_mask': [],
                'pixel_values': []
            }
        for k in ["responses", "input_ids", "attention_mask", "pixel_values"]:
            for h in vla_history:
                batch[k].append(h[k]) 
                    
        for k,v in batch.items():
            batch[k] = torch.stack(v,dim=1) 
  
        
        for k in task_records.keys():
            batch[k] = task_records[k]
        del batch['task_file_name']
        
        batch["complete"] = torch.tensor(batch["complete"], dtype=torch.bool, device=batch['responses'].device)
        batch["finish_step"] = torch.tensor(batch["finish_step"], dtype=torch.int64, device=batch['responses'].device)
        # breakpoint()
        output_batch = TensorDict(
            batch,
            batch_size=batch_size)
        return DataProto(batch=output_batch)     
     
     
    def _generate_minibatch_libero(self, prompts):
        self.module.eval()
        meta_info = prompts.meta_info
        n_samples = meta_info.get('n_samples', 1)
        task_id = prompts.batch['task_id'].repeat_interleave(n_samples, dim=0)
        trial_id = prompts.batch['trial_id'].repeat_interleave(n_samples, dim=0)
        task_suite_name = np.repeat(prompts.non_tensor_batch['task_suite_name'], n_samples)
        max_steps = self.max_steps[self.config.task_suite_name]
        batch_size = task_id.size(0)    # Num of grpo samples
        is_valid = meta_info.get('n_samples') is None
        global_steps = meta_info.get('global_steps', 0) if is_valid else 0
        
        processes = []
        input_queues = []
        output_queues = []
        
        for idx in range(batch_size):
            task_name = task_suite_name[idx]
            t_id = task_id[idx][0].item()
            tr_id = trial_id[idx][0].item()
            input_q = Queue()
            output_q = Queue()
            p = Process(
                target=self.env_worker,
                args=(task_name, t_id, tr_id, self.config, input_q, output_q, is_valid, global_steps, max_steps)
            )
            p.start()
            processes.append(p)
            input_queues.append(input_q)
            output_queues.append(output_q)
        
        inputs = []
        task_descriptions = []
        task_records = []
        valid_video = defaultdict(list)
        for idx in range(batch_size):
            init_data = output_queues[idx].get(timeout=120)
            assert init_data['type'] == 'init'
            task_descriptions.append(init_data["task_description"])
            inputs.append(self._obs_to_input(init_data['obs']))
            task_records.append({
                "active": init_data['active'],
                "complete": init_data['complete'],
                "finish_step": init_data['finish_step'],
                "task_file_name": init_data['task_file_name']
            })
            if is_valid:
                valid_video[init_data['task_file_name']].extend(init_data['valid_images'])
        
        step = 0
        vla_history = []
        while step < max_steps:
            active_indices = [i for i, r in enumerate(task_records) if r['active']]
            
            current_inputs = inputs
            current_task_descriptions = task_descriptions
           
            vla_input = self.process_input(current_inputs, current_task_descriptions)
            vla_input.update(meta_info)
            vla_output = self._generate_one_step(vla_input)
            actions = vla_output["action"]
            
            step_data = {
                    "responses": vla_output["responses"],
                    "input_ids": vla_output["input_ids"],
                    "attention_mask": vla_output["attention_mask"],
                    "pixel_values": vla_output["pixel_values"],
                    "action": actions,
                    "step": step
                }
            vla_history.append(step_data)
            
            for idx in active_indices:
                input_queues[idx].put(actions[idx])
            
            new_inputs = inputs.copy()
            for idx in active_indices:
                result = output_queues[idx].get(timeout=30)
                assert result['type'] == 'step'
                new_inputs[idx] = self._obs_to_input(result['obs'])
                task_records[idx]['active'] = result['active']
                task_records[idx]['complete'] = result['complete']
                task_records[idx]['finish_step'] = result['finish_step']
                if is_valid:
                    valid_video[task_records[idx]['task_file_name']].extend(result['valid_images'])
            
            inputs = new_inputs
            step += self.config.action_chunks_len
            
        for q in input_queues:
            q.put(None)
        for p in processes:
            p.join(timeout=20)
            if p.is_alive():
                p.terminate()
        
        torch.cuda.empty_cache()
        
        if is_valid:
            for task_file, images in valid_video.items():
                complete = any(r['complete'] for r in task_records if r['task_file_name'] == task_file)
                save_rollout_video(
                    images,
                    self.config.experiment_name,
                    task_file,
                    global_steps,
                    complete
                )
        
        self.module.train()
        
        batch = {
                'responses': [],
                'input_ids': [],  # here input_ids become the whole sentences
                'attention_mask': [],
                'pixel_values': []
            }
        for k in ["responses", "input_ids", "attention_mask", "pixel_values"]:
            for h in vla_history:
                batch[k].append(h[k])
        
        for k,v in batch.items():
            batch[k] = torch.stack(v,dim=1) 
  
        batch["complete"] = []
        batch["finish_step"] = []
        
        for k in task_records:
            batch["complete"].append(k["complete"])
            batch["finish_step"].append(k["finish_step"])
        
        batch["complete"] = torch.tensor(batch["complete"], dtype=torch.bool, device=batch['responses'].device)
        batch["finish_step"] = torch.tensor(batch["finish_step"], dtype=torch.int64, device=batch['responses'].device)
        
        output_batch = TensorDict(
            batch,
            batch_size=batch_size)
        return DataProto(batch=output_batch)
    
    @torch.no_grad()
    def _generate_one_step(self, prompts: dict, step=None):
        if self.config.vla == "openvla-oft":
            idx = prompts['input_ids']  # (bs, prompt_length)
            attention_mask = prompts['attention_mask']  # left-padded attention_mask
            pixel_values = prompts["pixel_values"]
        
        
            param_ctx = contextlib.nullcontext()

            # make sampling args can be overriden by inputs
            do_sample = prompts.get('do_sample', self.config.do_sample)
        

            temperature = prompts.get('temperature', self.config.temperature)

            #generation_config = GenerationConfig(temperature=temperature, top_p=top_p, top_k=top_k)

            if isinstance(self.module, FSDP):
                # recurse need to set to False according to https://github.com/pytorch/pytorch/issues/100069
                param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=False)
            
            with param_ctx:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    actions, response = self.module.generate_action_verl(
                        input_ids=idx,
                        pixel_values=pixel_values,
                        attention_mask=attention_mask,
                        padding_idx = self.processor.tokenizer.pad_token_id,
                        do_sample=do_sample,
                        unnorm_key=self.config.unnorm_key,
                        temperature=temperature, )
            
            
            assert self.processor.tokenizer.pad_token_id is not None

            assert idx.ndim == 2
            idx = verl_F.pad_sequence_to_length(idx,max_seq_len=self.config.max_prompt_length,pad_token_id=self.processor.tokenizer.pad_token_id,left_pad=True)
            
            assert attention_mask.ndim == 2
            attention_mask = verl_F.pad_sequence_to_length(attention_mask,max_seq_len=self.config.max_prompt_length,pad_token_id=0,left_pad=True)
            
            
            assert idx.device.type == 'cuda'
            assert response.device.type == 'cuda'
            #assert seq.device.type == 'cuda'
            assert attention_mask.device.type == 'cuda'
            assert pixel_values.device.type == 'cuda'
            batch ={
                    'responses': response,
                    'input_ids': idx,
                    'attention_mask': attention_mask,
                    "pixel_values":pixel_values,
                    "action":actions,
                }

            return batch
        
        elif self.config.vla == "openvla": 
            idx = prompts['input_ids']  # (bs, prompt_length)
            attention_mask = prompts['attention_mask']  # left-padded attention_mask
            pixel_values = prompts["pixel_values"]
            
            # used to construct attention_mask
            eos_token_id = prompts['eos_token_id']
            pad_token_id = prompts['pad_token_id']

            batch_size = idx.size(0)
            prompt_length = idx.size(1)
            #self.module.eval()
            param_ctx = contextlib.nullcontext()

            do_sample = prompts.get('do_sample', self.config.do_sample)
            response_length =  self.module.get_action_dim(self.config.unnorm_key)
            top_p = prompts.get('top_p', self.config.get('top_p', 1.0))
            top_k = prompts.get('top_k', self.config.get('top_k', 0))
            if top_k is None:
                top_k = 0
            top_k = max(0, top_k)  # to be compatible with vllm

            temperature = prompts.get('temperature', self.config.temperature)
            generation_config = GenerationConfig(temperature=temperature, top_p=top_p, top_k=top_k)

            if isinstance(self.module, FSDP):
                # recurse need to set to False according to https://github.com/pytorch/pytorch/issues/100069
                param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=False)
            
            with param_ctx:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    
                    output = self.module.generate(
                        input_ids=idx,
                        pixel_values=pixel_values,
                        attention_mask=attention_mask,
                        do_sample=do_sample,
                        max_new_tokens=response_length,
                        # max_length=max_length,
                        eos_token_id=eos_token_id,
                        pad_token_id=pad_token_id,
                        generation_config=generation_config,
                        # renormalize_logits=True,
                        output_scores=False,  # this is potentially very large
                        return_dict_in_generate=True,
                        use_cache=True)
                    
           
            seq = output.sequences
            sequence_length = prompt_length + response_length
            delta_length = sequence_length - seq.shape[1]
            
            assert delta_length == 0
            assert seq.shape[1] == sequence_length

            prompt = seq[:, :prompt_length]  # (bs, prompt_length)
            response = seq[:, prompt_length:]  # (bs, response_length)

            response_length = response.size(1)
            #delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
            #delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)
            #response_position_ids = position_ids[:, -1:] + delta_position_id
            #position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

            response_attention_mask = get_eos_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
            attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

            # Extract predicted action tokens and translate into (normalized) continuous actions
            predicted_action_token_ids = response.detach().cpu().numpy()
            discretized_actions = self.module.vocab_size - predicted_action_token_ids
            discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.module.bin_centers.shape[0] - 1)
            normalized_actions = self.module.bin_centers[discretized_actions]

            # Unnormalize actions
            action_norm_stats = self.module.get_action_stats(self.config.unnorm_key)
            mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
            action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
            actions = np.where(
                mask,
                0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
                normalized_actions,
            )
            
            actions = np.expand_dims(actions, axis=1)
            
            assert self.processor.tokenizer.pad_token_id is not None
            assert prompt.ndim == 2
            prompt = verl_F.pad_sequence_to_length(prompt, max_seq_len=self.config.max_prompt_length,pad_token_id=self.processor.tokenizer.pad_token_id,left_pad=True)
            assert seq.ndim == 2
            seq = verl_F.pad_sequence_to_length(seq,max_seq_len=self.config.max_prompt_length,pad_token_id=self.processor.tokenizer.pad_token_id,left_pad=True)
            assert attention_mask.ndim == 2
            attention_mask = verl_F.pad_sequence_to_length(attention_mask,max_seq_len=self.config.max_prompt_length,pad_token_id=0,left_pad=True)
            
            batch ={
                    'prompts': prompt,
                    'responses': response,
                    'input_ids': seq,
                    'attention_mask': attention_mask,
                    "pixel_values":pixel_values,
                    "action":actions,
                    #'position_ids': position_ids
                }
            
            return batch
        
        elif self.config.vla == "internvl_chat":
            tokenizer = self.processor
            pixel_values = prompts["pixel_values"]
            questions = prompts['questions']
            num_patches_list = prompts['num_patches_list']
            
            img_context_token_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>',)
            self.module.set_img_context_token_id(img_context_token_id)
            # if verbose and pixel_values is not None:
            #     image_bs = pixel_values.shape[0]
            #     print(f'dynamic ViT batch size: {image_bs}')
            do_sample = self.config.do_sample
            top_p = prompts.get('top_p', self.config.get('top_p', 1.0))
            top_k = prompts.get('top_k', self.config.get('top_k', 0))
            temperature = prompts.get('temperature', self.config.temperature)
            queries = []
            # questions_str = ''
            # if step == 1:
            #     breakpoint()
            for idx, num_patches in enumerate(num_patches_list):
                question = questions[idx]
                # questions_str += f"idx: {idx}, Q: {question}\n"
                if pixel_values is not None and '<image>' not in question:
                    question = '<image>\n' + question
                # template = get_conv_template(self.module.template)
                template = deepcopy(self.module.conv_template)
                template.system_message = self.module.system_message
                template.append_message(template.roles[0], question)
                template.append_message(template.roles[1], None)
                query = template.get_prompt()
                # questions_str += f"idx: {idx}, query: {query}\n"
                for patches in num_patches:
                    image_tokens = '<img>' + '<IMG_CONTEXT>' * self.module.num_image_token * patches + '</img>'
                    query = query.replace('<image>', image_tokens, 1)
                queries.append(query)
            tokenizer.padding_side = 'left'
            model_inputs = tokenizer(queries, return_tensors='pt', padding=True)
            input_ids = model_inputs['input_ids'].to(self.module.device)
            attention_mask = model_inputs['attention_mask'].to(self.module.device)
            eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
            if isinstance(self.module, FSDP):
                # recurse need to set to False according to https://github.com/pytorch/pytorch/issues/100069
                param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=False)
            # questions_str += f"start inference\nlen inputs: {input_ids.shape}\npixel_values: {pixel_values.shape}"
            # print(questions_str)
            with param_ctx:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    pixel_values = pixel_values.to('cuda').to(torch.bfloat16)
                    generation_output = self.module.generate(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        eos_token_id=eos_token_id,
                        max_new_tokens=self.response_length,
                        do_sample=do_sample,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                    )
            # response_str = f"step: {step} inference done\n"
            full_seq = torch.concatenate((input_ids, generation_output), dim=-1)
            responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
            string_response = [response.split(template.sep.strip())[0].strip() for response in responses]
            # for idx, response in enumerate(string_response):
            #     response_str += f"idx: {idx}, R: {response}\n"
            # print(response_str)
            actions = action_decode(prompts, string_response, self.task_suite)
            response_attention_mask = get_eos_mask(response_id=generation_output, eos_token=eos_token_id, dtype=attention_mask.dtype)
            attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
            assert self.processor.pad_token_id is not None
            assert input_ids.ndim == 2
            input_ids = verl_F.pad_sequence_to_length(input_ids, max_seq_len=self.config.max_prompt_length, pad_token_id=self.processor.pad_token_id, left_pad=True)
            assert full_seq.ndim == 2
            full_seq = verl_F.pad_sequence_to_length(full_seq, max_seq_len=self.config.max_prompt_length, pad_token_id=self.processor.pad_token_id, left_pad=True)
            assert attention_mask.ndim == 2
            attention_mask = verl_F.pad_sequence_to_length(attention_mask,max_seq_len=self.config.max_prompt_length,pad_token_id=0,left_pad=True)
            return {
                'prompts': input_ids,
                'responses': generation_output,
                'input_ids': full_seq,
                'attention_mask': attention_mask,
                'pixel_values': torch.reshape(pixel_values, (input_ids.shape[0], -1) + pixel_values.shape[1:]),
                'action': actions,
            } 
            
    def _obs_to_input(self, obs):
        if self.config.num_images_in_input > 1:
            return {
                "full_image": get_libero_image(obs, 224),
                "wrist_image": get_libero_wrist_image(obs, 224),
                "state": np.concatenate([
                    obs["robot0_eef_pos"],
                    quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"]
                ])
            }
        else:
            return {
                "full_image": get_libero_image(obs, 224),
                "state": np.concatenate([
                    obs["robot0_eef_pos"],
                    quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"]
                ])
            }