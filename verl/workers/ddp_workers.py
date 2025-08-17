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
The main entry point to run the PPO algorithm
"""

import os
import logging
import warnings
import ray
import torch
import torch.distributed
from omegaconf import DictConfig, open_dict
from transformers import AutoModelForCausalLM

from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import register, Dispatch
import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.utils.model import compute_position_id_with_mask
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.import_utils import import_external_libs
from verl.utils.debug import log_gpu_memory_usage, gpu_memory
import verl.utils.hdfs_io as hdfs_io
from verl.utils import hf_tokenizer
from ..trainer.ppo import core_algos
from verl.utils.py_functional import append_to_dict
from codetiming import Timer
from verl.utils.logger.local_logger import LocalLogger, DummyLogger
from verl.utils.model import print_model_size, update_model_config
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor, AutoModelForCausalLM
from torch import optim
from peft import LoraConfig, PeftModel, get_peft_model, TaskType
from torch.nn.parallel import DistributedDataParallel as DDP
import json
from verl.utils.vla_utils.internvl.utils import prepare_logits_processor

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'WARN'))

def convert_to_regular_types(obj):
    from omegaconf import ListConfig, DictConfig
    if isinstance(obj, (ListConfig, DictConfig)):
        return {k: convert_to_regular_types(v) for k, v in obj.items()} if isinstance(obj, DictConfig) else list(obj)
    elif isinstance(obj, (list, tuple)):
        return [convert_to_regular_types(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_regular_types(v) for k, v in obj.items()}
    return obj

class RobActorRolloutRefWorker(Worker):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    def __init__(self, config: DictConfig, role: str, dt_flag: str=None):
        super().__init__()
        self.config = config
        import torch.distributed
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        self.local_device = self.rank % torch.cuda.device_count()
        self.logger = LocalLogger(log_name=f"{config.rollout.experiment_name}/{dt_flag}/RobActorRolloutRefWorker", world_size=world_size, rank=self._rank)
        self._is_lora = self.config.model.get('lora_rank', 0) > 0
        self.role = role
        assert self.role in ['actor', 'rollout', 'ref', 'actor_rollout', 'actor_rollout_ref']

        self._is_actor = self.role in ['actor', 'actor_rollout', 'actor_rollout_ref']
        self._is_rollout = self.role in ['rollout', 'actor_rollout', 'actor_rollout_ref']
        self._is_ref = self.role in ['ref', 'actor_rollout_ref']
        self.dt_flag = dt_flag
        # normalize config
        if self._is_actor:
            self.config.actor.ppo_mini_batch_size //= world_size
            self.config.actor.ppo_micro_batch_size //= world_size
        if self._is_rollout:
            self.config.rollout.log_prob_micro_batch_size //= world_size
        if self._is_ref:
            self.config.ref.log_prob_micro_batch_size //= world_size
        # self.logger.log("init_done")

    def _build_model_optimizer(self,
                               model_path,
                               optim_config,
                               override_model_config,
                               enable_gradient_checkpointing=False,
                               trust_remote_code=True):
        # self.logger.log("start init")
        log_gpu_memory_usage('Before init from HF AutoModel', logger=logger)
        # self.logger.log("gpu mem log, starting model init")
        local_path = copy_local_path_from_hdfs(model_path)
        #add oft
        if self.config.model.vla == "openvla-oft":
            from verl.utils.vla_utils.openvla_oft.configuration_prismatic import OpenVLAConfig
            from verl.utils.vla_utils.openvla_oft.modeling_prismatic import OpenVLAForActionPrediction
            from verl.utils.vla_utils.openvla_oft.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
            
            AutoConfig.register("openvla", OpenVLAConfig)
            AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
            AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
            AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
            if self.rank == 0:
                from verl.utils.openvla_utils import update_auto_map, check_model_logic_mismatch
                update_auto_map(local_path)
                check_model_logic_mismatch(local_path)
            torch.distributed.barrier()
            
        elif self.config.model.vla == "openvla":
            from verl.utils.vla_utils.openvla.configuration_prismatic import OpenVLAConfig
            from verl.utils.vla_utils.openvla.modeling_prismatic import OpenVLAForActionPrediction
            from verl.utils.vla_utils.openvla.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
            AutoConfig.register("openvla", OpenVLAConfig)
            AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
            AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
            AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
            if self.rank == 0:
                from verl.utils.openvla_utils import update_auto_map, check_model_logic_mismatch
                update_auto_map(local_path)
                check_model_logic_mismatch(local_path)
            torch.distributed.barrier()
        elif self.config.model.vla == "internvl_chat":
            from verl.utils.internvl_utils import replace_modeling_files
            if self.rank == 0: replace_modeling_files(local_path)
            torch.distributed.barrier() # prevent other workers init model before modeling files replaced
        elif self.config.model.vla == 'internvl_chat_head':
            pass
        
        #add end

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        # TODO(zhangchi.usc1992): 1. support create from random initialized model. 2. Support init with FSDP directly
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code, model=self.config.model.vla)
        torch_dtype = torch.float32 if self._is_actor else torch.bfloat16

        # override model kwargs
        actor_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        if self.config.model.use_remove_padding:
            from verl.models.registry import check_model_support_rmpad
            check_model_support_rmpad(actor_model_config.model_type)
        override_config_kwargs = {
            'bos_token_id': self.tokenizer.bos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config)
        update_model_config(actor_model_config, override_config_kwargs=override_config_kwargs)
        if self.rank == 0:
            print(f'Model config after override: {actor_model_config}')
        
        warnings.simplefilter("ignore")
        if self.config.model.vla == "openvla-oft":
            actor_module = AutoModelForVision2Seq.from_pretrained(
                                                    pretrained_model_name_or_path=local_path,
                                                    torch_dtype=torch_dtype,
                                                    #attn_implementation="flash_attention_2",
                                                    config=actor_model_config,              
                                                    trust_remote_code=True,
                                                )
            #oft add
            actor_module.vision_backbone.set_num_images_in_input(self.config.actor.num_images_in_input)
            
            dataset_statistics_path = os.path.join(local_path, "dataset_statistics.json")
            if os.path.isfile(dataset_statistics_path):
                with open(dataset_statistics_path, "r") as f:
                    norm_stats = json.load(f)
                actor_module.norm_stats = norm_stats
            else:
                print(
                    "WARNING: No local dataset_statistics.json file found for current checkpoint.\n"
                    "You can ignore this if you are loading the base VLA (i.e. not fine-tuned) checkpoint."
                    "Otherwise, you may run into errors when trying to call `predict_action()` due to an absent `unnorm_key`."
                )
        elif self.config.model.vla == "openvla":
            actor_module = AutoModelForVision2Seq.from_pretrained(
                                                pretrained_model_name_or_path=local_path,
                                                torch_dtype=torch_dtype,
                                                attn_implementation="flash_attention_2",
                                                config=actor_model_config,              
                                                trust_remote_code=True,
                                            )
        elif self.config.model.vla == "internvl_chat":
            # TODO: debug
            actor_module = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=local_path,
                torch_dtype=torch_dtype,
                config=actor_model_config,
                trust_remote_code=True,
            )
            processor_list, valid_list, response_token_num, numbers_index, valid_index2token_id_list = prepare_logits_processor(self.tokenizer)
            valid_token_index = []
            for num in numbers_index:
                valid_token_index.append(num)
                valid_token_index.append(num - 1)
            generation_config = dict(logits_processor=processor_list)
            # if self.config.model.get("mask_logits", False):
            #     print(f"Masking irrelevant logits")
            #     self.logger.log(f"Masking irrelevant logits")
            #     actor_module.set_action_allowed_list(valid_list)
            actor_module.set_action_generation_config(generation_config)
            self.internvl_help_kwargs = {
                "valid_index2token_id_list": valid_index2token_id_list,
                "valid_token_index": valid_token_index,
                # "top_k": self.config.rollout.get("top_k", 20),
                # "top_p": self.config.rollout.get("top_p", 0.7),
                "temperature": self.config.rollout.temperature,
                
            }
            
        actor_module.to(torch_dtype).cuda(self.local_device)
        if enable_gradient_checkpointing:
            actor_module.gradient_checkpointing_enable()
        # lora add
        if self._is_lora:
            print("Applying LoRA to actor module")
            
            lora_config = {
                #'task_type': TaskType.CAUSAL_LM,
                'r': self.config.model.lora_rank,
                'lora_alpha': self.config.model.lora_alpha,
                "lora_dropout": 0 ,
                'target_modules': convert_to_regular_types(self.config.model.target_modules),
                'init_lora_weights': "gaussian"
            }
            actor_module = get_peft_model(actor_module, LoraConfig(**lora_config))  
            actor_module.print_trainable_parameters()
            # lora end
        if self.rank == 0:
            print_model_size(actor_module)

        log_gpu_memory_usage('After init from HF AutoModel', logger=logger)
        
        
        actor_module_ddp = DDP(
                actor_module,
                device_ids=[self.local_device],
                # output_device=self.rank,
            )
        # TODO: add more optimizer args into config
        if self._is_actor:
            from verl.utils.torch_functional import get_constant_schedule_with_warmup
            if self.config.actor.optimizer_type == 'adamw':
                actor_optimizer = optim.AdamW(actor_module_ddp.parameters(),
                                            lr=optim_config.lr,
                                            betas=optim_config.get('betas', (0.9, 0.999)),
                                            weight_decay=optim_config.get('weight_decay', 1e-2))
                if os.path.exists(os.path.join(local_path, f'actor_optimizer_{self.rank}.pt')):
                    # load optimizer state from checkpoint
                    actor_optimizer.load_state_dict(torch.load(os.path.join(local_path, 'actor_optimizer.pt')))
                print(f'Loaded actor optimizer state from {local_path}')
            elif self.config.actor.optimizer_type == 'rms':
                actor_optimizer = optim.RMSprop(actor_module_ddp.parameters(),
                                            lr=optim_config.lr,
                                            momentum=0,
                                            weight_decay=0)
            else:
                raise NotImplementedError(f"Unsupported optimizer type: {self.config.actor.optimizer_type}")

            total_steps = optim_config.get('total_trddaining_steps', 0)
            num_warmup_steps_ratio = optim_config.get('lr_warmup_steps_ratio', 0.)
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

            print(f'Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}')

            actor_lr_scheduler = get_constant_schedule_with_warmup(optimizer=actor_optimizer,
                                                                   num_warmup_steps=num_warmup_steps)
        else:
            actor_optimizer = None
            actor_lr_scheduler = None

        log_gpu_memory_usage('After actor optimizer init', logger=logger)

        return actor_module_ddp, actor_optimizer, actor_lr_scheduler, actor_model_config

    def _build_rollout(self):
        if self.config.rollout.name == 'hf':
            from verl.workers.rollout import RobHFRollout
            from verl.workers.hybrid_engine import BaseShardingManager
            rollout = RobHFRollout(module=self.actor_module, config=self.config.rollout, dt_flag=self.dt_flag, logger=self.logger)
            sharding_manager = BaseShardingManager()
            # TODO: a sharding manager that do nothing?
        elif self.config.rollout.name == 'vllm':
            raise ValueError
            # from verl.workers.rollout.vllm_rollout import vLLMRollout
            # from verl.workers.hybrid_engine import FSDPVLLMShardingManager
            # log_gpu_memory_usage('Before building vllm rollout', logger=None)
            # rollout = vLLMRollout(actor_module=self.actor_module,
            #                       config=self.config.rollout,
            #                       tokenizer=self.tokenizer,
            #                       model_hf_config=self.actor_model_config)
            # log_gpu_memory_usage('After building vllm rollout', logger=None)
            # if torch.distributed.get_world_size() == 1:
            #     self.config.rollout.load_format = 'dummy_hf'
            # sharding_manager = FSDPVLLMShardingManager(module=self.actor_module,
            #                                            inference_engine=rollout.inference_engine,
            #                                            model_config=self.actor_model_config,
            #                                            full_params='hf' in self.config.rollout.load_format)
            # log_gpu_memory_usage('After building sharding manager', logger=None)

        return rollout, sharding_manager

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from verl.workers.actor import RobDataParallelPPOActor
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get('external_lib', None))
        from omegaconf import OmegaConf
        override_model_config = OmegaConf.to_container(self.config.model.get('override_config', OmegaConf.create()))

        if self._is_actor or self._is_rollout:
            # we need the model for actor and rollout
            if self._is_actor:
                optim_config = self.config.actor.optim
            else:
                optim_config = None
            self.actor_module_ddp, self.actor_optimizer, self.actor_lr_scheduler, self.actor_model_config = self._build_model_optimizer(
                model_path=self.config.model.path,
                optim_config=optim_config,
                override_model_config=override_model_config,
                enable_gradient_checkpointing=self.config.model.get('enable_gradient_checkpointing', False),
                trust_remote_code=self.config.model.get('trust_remote_code', False)) #
            self.actor_module = self.actor_module_ddp.module

        # load from checkpoint
        if self._is_actor:
            OmegaConf.set_struct(self.config.actor, True)
            self.actor = RobDataParallelPPOActor(config=self.config.actor,
                                              actor_module=self.actor_module_ddp,
                                              actor_optimizer=self.actor_optimizer,
                                              logger=self.logger,
                                              internvl_help_kwargs=getattr(self, "internvl_help_kwargs", None),
                                              )

        if self._is_rollout:
            self.rollout, self.sharding_manager = self._build_rollout()

        if self._is_ref:
            self.ref_module = self._build_model_optimizer(model_path=self.config.model.path,
                                                               optim_config=None,
                                                               override_model_config=override_model_config,
                                                               trust_remote_code=True)[0] #self.config.model.get('trust_remote_code', False)
                                                                   
            OmegaConf.set_struct(self.config.ref, True)
            self.ref_policy = RobDataParallelPPOActor(config=self.config.ref, actor_module=self.ref_module,
                                              logger=self.logger
                                              )

        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        
        
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def entropy_update_actor(self, data: DataProto):
        # self.save_checkpoint('log_prob_compute_backup')
        actor_out = self.update_actor(data)
        # entropy_out = self.compute_entropy(data)
        # entropy_out = {}
        return actor_out

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):

        #data = data.to('cuda')
        assert self._is_actor
        #data.batch = data.batch.cuda()

        log_gpu_memory_usage('Before update policy', logger=logger)
        self.logger.log(f"update_actor, before update policy, {gpu_memory()}")
        metrics = self.actor.update_policy(data=data)
        self.actor_lr_scheduler.step()
        lr = self.actor_lr_scheduler.get_last_lr()[0]
        metrics['actor/lr(1e-4)'] = lr * 1e4
        self.logger.log(f"update_actor, after update policy, {gpu_memory()}")
        log_gpu_memory_usage('After update policy', logger=logger)

        # TODO: here, we should return all metrics
        output = DataProto(meta_info={'metrics': metrics})
        output = output.to('cpu')

        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        self.logger.log(f"update_actor, after empty_cache, {gpu_memory()}")
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_entropy(self, data: DataProto):
        
        data = data.to('cuda')

        assert self._is_actor

        data.batch = data.batch.cuda()

        log_gpu_memory_usage('Before compute entropy', logger=logger)
        self.logger.log(f"compute_entropy, Before compute entropy, {gpu_memory()}")
        metrics = self.actor.compute_entropy(bacth_data=data)
        self.logger.log(f"compute_entropy, After compute entropy, {gpu_memory()}")
        log_gpu_memory_usage('After compute entropy', logger=logger)

        # TODO: here, we should return all metrics
        output = DataProto(meta_info={'metrics': metrics})
        output = output.to('cpu')
        self.logger.log(f"compute_entropy, After metrics move to cpu, {gpu_memory()}")
        
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        self.logger.log(f"compute_entropy, after empty cache, {gpu_memory()}")
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts):
        self.logger.log(f"generate_sequences, before empty cache, {gpu_memory()}")
        torch.cuda.empty_cache()
        self.logger.log(f"generate_sequences, after empty cache, {gpu_memory()}")
        prompts = prompts.to('cuda')
        # set to False if it is validation
        recompute_log_prob = prompts.meta_info.get('recompute_log_prob', True)

        assert self._is_rollout

        prompts.batch = prompts.batch.cuda()
        meta_info = {'eos_token_id': self.tokenizer.eos_token_id, 'pad_token_id': self.tokenizer.pad_token_id}
        prompts.meta_info.update(meta_info)
        
        #tmp_sample = prompts.meta_info.get('n_samples', -1)
        # with Timer(name=f'gen seq will start, and the num samples are: {tmp_sample}', text="{name}: {seconds:.1f} seconds") as timer:    
        #     print(f"gen seq will start, and the num samples are: {tmp_sample}")
    
        with self.sharding_manager:
            log_gpu_memory_usage('After entering sharding manager', logger=logger)    
            prompts = self.sharding_manager.preprocess_data(prompts)
            self.logger.log(f"generate_sequences, generate_sequences, {gpu_memory()}")
            output = self.rollout.generate_sequences(prompts=prompts)
            self.logger.log(f"generate_sequences, generate_sequences done, {gpu_memory()}")
            log_gpu_memory_usage('After rollout generation', logger=logger)
            # shape: BS * MAX_SEQ_LEN
            output = self.sharding_manager.postprocess_data(output)
            torch.cuda.synchronize()

        # with Timer(name=f'gen seq end ,  old log will begin', text="{name}: {seconds:.1f} seconds") as timer:    
        #     print("gen seq end ,  old log will begin")
        self.logger.log(f"generate_sequences, before compute log_prob: {gpu_memory()}, log_prob_micro_batch_size: {self.config.rollout.log_prob_micro_batch_size}")
        if self._is_actor and recompute_log_prob:
            # we should always recompute old_log_probs when it is HybridEngine
            
            output.meta_info['micro_batch_size'] = self.config.rollout.log_prob_micro_batch_size
            output.meta_info['temperature'] = self.config.rollout.temperature
            output.meta_info['use_dynamic_bsz'] = self.config.rollout.log_prob_use_dynamic_bsz
            output.meta_info['max_token_len'] = self.config.rollout.log_prob_max_token_len_per_gpu
            output.meta_info['pad_token_id'] = self.tokenizer.pad_token_id
            # old_log_probs = self.actor.compute_log_prob(data=output)
            # output.batch['old_log_probs'] = old_log_probs
            # self.logger.log(f"generate_sequences, log prob computed, shape; {old_log_probs.shape} {gpu_memory()}")
        output = output.to('cpu')
        self.logger.log(f"generate_sequences, move to cpu: {gpu_memory()}")
        # clear kv cache
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        self.logger.log(f"generate_sequences, after empty cache {gpu_memory()}")
        # log_gpu_memory_usage('After recompute log prob', logger=logger)
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_ref_log_prob(self, data: DataProto):
        assert self._is_ref

        data = data.to('cuda')

        micro_batch_size = self.config.ref.log_prob_micro_batch_size
        data.meta_info['micro_batch_size'] = micro_batch_size
        data.meta_info['temperature'] = self.config.rollout.temperature
        data.meta_info['max_token_len'] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info['use_dynamic_bsz'] = self.config.ref.log_prob_use_dynamic_bsz
        data.meta_info['pad_token_id'] = self.tokenizer.pad_token_id
        output = self.ref_policy.compute_log_prob(data=data)
        output = DataProto.from_dict(tensors={'ref_log_prob': output})

        output = output.to('cpu')

        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None):
        assert self._is_actor
        
        import torch.distributed as dist
        from peft import PeftModel
        import transformers

        #lora add
        if self._is_lora and isinstance(self.actor_module, PeftModel):
            if dist.get_rank() == 0:
                os.makedirs(local_path, exist_ok=True)

            lora_save_path = os.path.join(local_path, "lora_adapter")

            self.actor_module.save_pretrained(lora_save_path, safe_serialization=True)

            dist.barrier()
            if dist.get_rank() == 0:
                print(f"[rank-{self.rank}]: Saved LoRA adapter to: {lora_save_path}")
            
            # save total model
            base_vla = AutoModelForVision2Seq.from_pretrained(
                self.config.model.path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True, device_map="cpu"
            )
            merged_vla = PeftModel.from_pretrained(base_vla, lora_save_path)
            merged_vla = merged_vla.merge_and_unload()

            if dist.get_rank() == 0:
                merged_vla.save_pretrained(local_path)
                print(f"Saved merged model at: {local_path}")

            # Wait for merged model to be saved
            dist.barrier()    
                
        
        # TODO: support DCP and save sharded checkpoints
        else:
            import torch.distributed
            if self.rank == 0:
                print(f'Saving actor checkpoint to {local_path}')
                os.makedirs(local_path, exist_ok=True)
                self.actor_module.save_pretrained(local_path)
                self.tokenizer.save_pretrained(local_path)
                if hdfs_path is not None:
                    print(f'Uploading actor checkpoint to {hdfs_path}')
                    hdfs_io.makedirs(hdfs_path, exist_ok=True)
                    hdfs_io.copy(src=local_path, dst=hdfs_path)
            # torch.distributed.barrier()
            # if isinstance(self.actor_optimizer, optim.AdamW):
            #     torch.save(self.actor_optimizer.state_dict(), os.path.join(local_path, f'actor_optimizer_{self.rank}.pt'))
                
            # tmp_local_path = f"{local_path}_{self.rank}"
            # print(f'Saving actor checkpoint to {tmp_local_path}')
            # os.makedirs(tmp_local_path, exist_ok=True)
            # self.actor_module.save_pretrained(tmp_local_path)
            # self.tokenizer.save_pretrained(tmp_local_path)
            # if hdfs_path is not None:
            #     print(f'Uploading actor checkpoint to {hdfs_path}')
            #     hdfs_io.makedirs(hdfs_path, exist_ok=True)
            #     hdfs_io.copy(src=tmp_local_path, dst=hdfs_path)

        torch.distributed.barrier()


class CriticWorker(Worker):

    def __init__(self, config):
        super().__init__()
        import torch.distributed
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        self.config = config
        # normalize config
        self.config.ppo_mini_batch_size //= torch.distributed.get_world_size()
        self.config.ppo_micro_batch_size //= torch.distributed.get_world_size()

    def _build_critic_model_optimizer(self, config):
        # the following line is necessary
        from verl.utils.model import LambdaLayer, print_model_size, squeeze
        from torch import optim

        local_path = copy_local_path_from_hdfs(config.model.path)
        # note that the tokenizer between actor and critic may be different. So override tokenizer info with actor info
        # using random initialized model from any architecture. May not be the same as Actor.
        # TODO: support loading critic weights from RM. Support using AutoModelForTokenClassification
        from transformers import AutoTokenizer

        tokenizer_path = copy_local_path_from_hdfs(config.model.tokenizer_path)
        self.tokenizer = hf_tokenizer(tokenizer_path, trust_remote_code=config.model.get('trust_remote_code', False))

        from omegaconf import OmegaConf
        override_config = OmegaConf.to_container(self.config.model.get('override_config', OmegaConf.create()))
        override_config_kwargs = {
            'bos_token_id': self.tokenizer.bos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_config)
        if self.rank == 0:
            print(f'Critic overriding config {override_config_kwargs}')

        torch_dtype = 'fp32'

        from transformers import AutoConfig, AutoModelForCausalLM
        from torch import nn

        trust_remote_code = False
        critic_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)

        warnings.simplefilter("ignore")
        critic_module = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=local_path,
                                                            torch_dtype=torch_dtype,
                                                            config=critic_model_config,
                                                            attn_implementation='flash_attention_2',
                                                            trust_remote_code=trust_remote_code)
        critic_module.lm_head = nn.Sequential(nn.Linear(critic_model_config.hidden_size, 1, dtype=torch_dtype),
                                            LambdaLayer(fn=squeeze))

        # some parameters may not in torch_dtype
        critic_module.to(torch_dtype)

        if config.model.get('enable_gradient_checkpointing', False):
            critic_module.gradient_checkpointing_enable()
            
        if self.rank == 0:
            print_model_size(critic_module)

        critic_optimizer = optim.AdamW(critic_module.parameters(),
                                       lr=config.optim.lr,
                                       betas=config.optim.get('betas', (0.9, 0.999)),
                                       weight_decay=config.optim.get('weight_decay', 1e-2))

        total_steps = config.optim.get('total_training_steps', 0)
        num_warmup_steps_ratio = config.optim.get('lr_warmup_steps_ratio', 0.)
        num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

        print(f'Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}')

        from verl.utils.torch_functional import get_constant_schedule_with_warmup
        critic_lr_scheduler = get_constant_schedule_with_warmup(optimizer=critic_optimizer,
                                                                num_warmup_steps=num_warmup_steps)

        return critic_module, critic_optimizer, critic_lr_scheduler

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get('external_lib', None))

        from verl.workers.critic import DataParallelPPOCritic
        self.critic_module, self.critic_optimizer, self.critic_lr_scheduler = self._build_critic_model_optimizer(
            self.config)

        self.critic = DataParallelPPOCritic(config=self.config,
                                            critic_module=self.critic_module,
                                            critic_optimizer=self.critic_optimizer)
        torch.cuda.empty_cache()

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_values(self, data: DataProto):
        data = data.to('cuda')

        micro_batch_size = self.config.ppo_micro_batch_size
        data.meta_info['micro_batch_size'] = micro_batch_size
        data.meta_info['max_token_len'] = self.config.forward_max_token_len_per_gpu
        data.meta_info['use_dynamic_bsz'] = self.config.use_dynamic_bsz
        values = self.critic.compute_values(data=data)
        output = DataProto.from_dict(tensors={'values': values})
        output = output.to('cpu')
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_critic(self, data: DataProto):
        data = data.to('cuda')
        metrics = self.critic.update_critic(data=data)

        self.critic_lr_scheduler.step()
        lr = self.critic_lr_scheduler.get_last_lr()[0]
        metrics['critic/lr(1e-4)'] = lr * 1e4

        output = DataProto(batch=None, meta_info={'metrics': metrics})
        torch.cuda.empty_cache()
        output = output.to('cpu')
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None):
        import torch

        # TODO: support DCP and save sharded checkpoints
        import torch.distributed
        if self.rank == 0:
            print(f'Saving critic checkpoint to {local_path}')
            os.makedirs(local_path, exist_ok=True)
            self.critic_module.save_pretrained(local_path)
            self.tokenizer.save_pretrained(local_path)
            if hdfs_path is not None:
                print(f'Uploading critic checkpoint to {hdfs_path}')
                hdfs_io.makedirs(hdfs_path, exist_ok=True)
                hdfs_io.copy(src=local_path, dst=hdfs_path)

        torch.distributed.barrier()

