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

import logging
import os
from contextlib import contextmanager
from typing import Optional
import warnings

import torch

from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils.device import (
    get_device_name,
    get_torch_device,
)
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.distributed import initialize_global_process_group_ray
from verl.utils.profiler import DistProfiler, DistProfilerExtension, log_gpu_memory_usage
from verl.workers.config import HFModelConfig, RewardModelConfig, RewardModelDataProcessorConfig
from verl.workers.roles.reward_model_engine import get_reward_model_class

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()


class RewardModelWorker(Worker, DistProfilerExtension):
    def __init__(self, config: RewardModelConfig) -> None:
        self.config = config
        self.model_config = config.model_config
        self.input_model_config = config.input_model_config
        self.model_type = config.model_type
        assert self.model_type in ["discriminative", "generative"], f"model_type: {self.model_type} is not supported"
        Worker.__init__(self)
        self.profiler_config = self.config.profiler
        tool_config = self.profiler_config.tool_config
        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=self.profiler_config, tool_config=tool_config)
        )

        initialize_global_process_group_ray(timeout_second=None)

    def _build_reward_model(self):
        from torch.distributed.device_mesh import init_device_mesh

        # 1. parse reward model and huggingface model config
        reward_model_config: RewardModelConfig = self.config
        model_config: HFModelConfig = self.config.model_config
        data_processor_config: RewardModelDataProcessorConfig = self.config.data_processor_config
        self.tokenizer = self.model_config.get_processor()
        if self.input_model_config is None:
            self._do_switch_chat_template = False
            self.src_tokenizer = self.tokenizer
        else:
            self._do_switch_chat_template = True
            self.src_tokenizer = self.input_model_config.get_processor()
        self.preprocess_fn, self.postprocess_fn = data_processor_config.get_process_fn()
        if self.model_type == "generative":
            assert self.preprocess_fn is not None and self.postprocess_fn is not None, (
                "generative reward model must have preprocess_fn and postprocess_fn"
            )

        # 2. build reward model device mesh
        infer_tp = self.config.tensor_model_parallel_size
        dp = self.world_size // infer_tp
        assert self.world_size % infer_tp == 0, (
            f"reward model world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}"
        )
        reward_model_device_mesh = init_device_mesh(
            device_name, mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"]
        )
        is_collect = reward_model_device_mesh["infer_tp"].get_local_rank() == 0
        self._register_dispatch_collect_info(
            "reward_model", dp_rank=reward_model_device_mesh["dp"].get_local_rank(), is_collect=is_collect
        )

        # 3. init trainer and reward model random states
        self.torch_random_states = get_torch_device().get_rng_state()
        gen_dp_rank = reward_model_device_mesh["dp"].get_local_rank()
        get_torch_device().manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
        self.gen_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.torch_random_states)

        # 4. build reward model
        log_gpu_memory_usage("Before building sglang reward model", logger=logger)
        self.reward_model = get_reward_model_class(reward_model_config.name)(
            config=reward_model_config, model_config=model_config, device_mesh=reward_model_device_mesh
        )
        log_gpu_memory_usage("After building sglang reward model", logger=logger)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        self._build_reward_model()

    def _expand_to_token_level(self, data: DataProto, scores: torch.Tensor):
        batch_size = data.batch.batch_size[0]
        # expand as token_level_reward
        attention_mask = data.batch["attention_mask"]
        position_ids = data.batch["position_ids"]
        response_length = data.batch["responses"].shape[-1]
        if position_ids.dim() == 3:  # qwen2vl mrope [bs, 3, seq_len]
            position_ids = position_ids[:, 0, :]
        eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
        token_level_scores = torch.zeros_like(attention_mask, dtype=scores.dtype)  # (bsz, seqlen)
        token_level_scores[torch.arange(batch_size), eos_mask_idx] = scores

        # select the response part
        token_level_scores = token_level_scores[:, -response_length:]

        return token_level_scores

    def _preprocess_reward_inputs(self, data: DataProto):
        src_tokenizer = self.src_tokenizer
        tokenizer = self.tokenizer
        rm_inputs = []
        for i in range(len(data)):
            data_item = data[i]

            # get rollout question
            if "extra_infos" in data_item.non_tensor_batch and "question" in data_item.non_tensor_batch["extra_infos"]:
                rollout_question = data_item.non_tensor_batch["extra_infos"]["question"]
            else:
                # use prompt_str as a substitute for question
                prompt_ids = data_item.batch["prompts"]
                prompt_length = prompt_ids.shape[-1]
                valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]
                rollout_question = src_tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)

            # get rollout response
            response_ids = data_item.batch["responses"]
            response_length = response_ids.shape[-1]
            valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            rollout_response = src_tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            # get ground truth
            ground_truth = data_item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)

            if self.model_type == "discriminative":
                if self._do_switch_chat_template:
                    chats = [
                        {"role": "user", "content": rollout_question},
                        {"role": "assistant", "content": rollout_response},
                    ]
                    rm_input = tokenizer.apply_chat_template(chats, tokenize=True)
                else:
                    non_pad_indices = torch.nonzero(data_item.batch["attention_mask"], as_tuple=True)[0]
                    start_idx, end_idx = non_pad_indices[0], non_pad_indices[-1]
                    rm_input = data_item.batch["input_ids"][start_idx : end_idx + 1].tolist()
            else:
                assert self.preprocess_fn is not None, "generative reward model must have preprocess_fn"

                input_str = self.preprocess_fn(
                    rollout_question=rollout_question,
                    rollout_response=rollout_response,
                    ground_truth=ground_truth,
                )
                chats = [{"role": "user", "content": input_str}]
                rm_input = tokenizer.apply_chat_template(chats, add_generation_prompt=True, tokenize=True)

            rm_inputs.append(rm_input)

        return rm_inputs

    def _postprocess_reward_outputs(self, data: DataProto, output: list[float] | list[list[int]]):
        if self.model_type == "discriminative":
            scores = torch.tensor(output)
        else:
            assert self.postprocess_fn is not None, "generative reward model must have postprocess_fn"
            output_text = [self.tokenizer.decode(o) for o in output]
            # postprocess genrm responses to scores
            scores = [self.postprocess_fn(o) for o in output_text]
            scores = torch.tensor(scores)

        token_level_scores = self._expand_to_token_level(data, scores)
        return token_level_scores

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="reward_model"))
    @DistProfiler.annotate(color="brown")
    def compute_rm_score(self, data: DataProto):
        data = data.to("cpu")
        rm_data = self._preprocess_reward_inputs(data)

        output = self.reward_model.compute_reward(rm_data)
        token_level_scores = self._postprocess_reward_outputs(data, output)
        # Note that this is only the scores, may not be the final rewards used to train RL
        output = DataProto.from_dict(tensors={"rm_scores": token_level_scores})
        return output

import socket

def _is_port_free(port: int, host: str = "127.0.0.1") -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            return True
    except OSError:
        return False


def _find_free_port() -> int:
    candidates = (29500, 23456, 12355, 12345)
    for p in candidates:
        if _is_port_free(p):
            return p
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def ensure_master_addr_port(addr: Optional[str] = None, port: Optional[int] = None) -> None:
    """
    Ensure `MASTER_ADDR`/`MASTER_PORT` are set safely.

    - Respects existing environment variables.
    - Defaults `MASTER_ADDR` to localhost if unset.
    - Chooses a free TCP port if `MASTER_PORT` is unset to avoid collisions.
    - If `MASTER_PORT` is set to `"0"` or `"auto"`, it is resolved to a free port.
    """
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR") or addr or "localhost"

    env_port = os.environ.get("MASTER_PORT", "").strip().lower()
    if port is None and env_port not in {"", "0", "auto"}:
        try:
            port = int(env_port)
        except ValueError:
            pass

    os.environ["MASTER_PORT"] = str(_find_free_port() if port in (None, 0) else port)


def split_think(response: str, n: int = 4) -> list[str]:
    parts = response.split("\n\n")
    chunk_size = max(1, len(parts) // n)
    return [
        "\n\n".join(parts[: min(i + chunk_size, len(parts))]) + "\n\n"
        for i in range(0, len(parts), chunk_size)
    ]

from verl.utils.config import omega_conf_to_dataclass
from verl.utils.profiler import DistProfiler, DistProfilerExtension, ProfilerConfig, log_gpu_memory_usage, simple_timer

class CustomRewardModelWorker(Worker, DistProfilerExtension):

    def __init__(self, config):
        Worker.__init__(self)

        omega_profiler_config = config.get("profiler", {})
        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory"]:
            tool_config = omega_conf_to_dataclass(
                omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
            )
        else:
            tool_config = None
        DistProfilerExtension.__init__(
            self,
            DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config),
        )


        self.config = config
        
        infer_tp = self.config.tensor_model_parallel_size
        dp = self.world_size // infer_tp
        assert self.world_size % infer_tp == 0, (
            f"reward model world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}"
        )

        from torch.distributed.device_mesh import init_device_mesh

        reward_model_device_mesh = init_device_mesh(
            device_name, mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"]
        )
        is_collect = reward_model_device_mesh["infer_tp"].get_local_rank() == 0
        self._register_dispatch_collect_info(
            "reward", dp_rank=reward_model_device_mesh["dp"].get_local_rank(), is_collect=is_collect
        )

        # 3. init trainer and reward model random states
        self.torch_random_states = get_torch_device().get_rng_state()
        gen_dp_rank = reward_model_device_mesh["dp"].get_local_rank()
        get_torch_device().manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
        self.gen_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.torch_random_states)

        if self.config.micro_batch_size is not None:
            self.config.micro_batch_size //= torch.distributed.get_world_size()
            self.config.micro_batch_size_per_gpu = self.config.micro_batch_size

    def _build_vllm_model(self, config):
        from vllm import LLM, SamplingParams

        tensor_parallel_size = self.config.tensor_model_parallel_size

        if tensor_parallel_size != 1:
            assert tensor_parallel_size == self.world_size, "Only support tensor_parallel_size == world_size"
            
            import os
            os.environ["RANK"] = str(self.rank)
            os.environ["WORLD_SIZE"] = str(self.world_size)

            ensure_master_addr_port()
        self.tokenizer = hf_tokenizer(config.vllm.model_path)

        from functools import partial
        
        from verl.trainer.ppo.reward import get_custom_reward_fn
        from verl.workers.reward_manager import get_reward_manager_cls

        reward_manager_name = config.vllm.get("reward_manager", "naive")
        reward_manager_cls = get_reward_manager_cls(reward_manager_name)

        compute_fun = get_custom_reward_fn(config)
        compute_fun = partial(compute_fun, **config.get("reward_fn_args", {}))

        self.rewrad_manager = reward_manager_cls(
            tokenizer=self.tokenizer,
            num_examine=0,
            compute_score=compute_fun, 
        )
        self.rewrad_manager.rank = self.rank

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reward_module = LLM(
                model=config.vllm.model_path,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=config.vllm.get("gpu_memory_utilization", 0.8),
                max_num_seqs=self.config.micro_batch_size_per_gpu
                * tensor_parallel_size
                * config.vllm.get("re_generation_num", 1),
                max_model_len=config.vllm.get("max_model_len", 2048),
                distributed_executor_backend="external_launcher" if tensor_parallel_size > 1 else None,
                seed=self.rank // self.world_size,
                enable_sleep_mode=True
            )
            print("[DONE] reward model loaded")
            reward_module.sleep(level=1)
            print("[DONE] reward model sleep")

        self.sampling_params = SamplingParams(
            n=1,
            temperature=config.vllm.get("temperature", 0),
            top_p=config.vllm.get("top_p", 1.0),
            top_k=config.vllm.get("top_k", -1),
            min_p=config.vllm.get("min_p", 0.0),
            max_tokens=config.vllm.get("max_new_tokens", 512),
            repetition_penalty=config.vllm.get("repetition_penalty", 1.0),
        )

        return reward_module


    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems

        if self.config.get("use_vllm", True):
            self.reward_module = self._build_vllm_model(config=self.config)
        else:
            raise NotImplementedError("Only vllm reward model is supported now")

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    def _generate_micro_batch(self, micro_data: DataProto):
        prompts = micro_data
        querys = prompts.batch["prompts"]
        answer = prompts.batch["responses"]
        qur_texts = self.tokenizer.batch_decode(querys, skip_special_tokens=True)
        ans_texts = self.tokenizer.batch_decode(answer, skip_special_tokens=True)
        

        n = self.config.vllm.get("re_generation_num", 6)
        new_prompts = []
        re_generation_nums = []
        for qur, ans in zip(qur_texts, ans_texts):
            split_ans = split_think(ans, n=n)
            re_generation_nums.append(len(split_ans[-2::-1]))
            for sa in split_ans[-2::-1]:
                new_prompts.append(
                    self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": qur}, {"role": "assistant", "content": sa}],
                        add_generation_prompt=False,
                        continue_final_message=True,
                        tokenize=False,
                    )
                )

        outputs = self.reward_module.generate(
            new_prompts,
            sampling_params=self.sampling_params,
            use_tqdm=False,
        )

        resps = []
        offset = 0
        for num in re_generation_nums:
            group = []
            for j in range(num):
                output = outputs[offset + j]
                resp = output.outputs[0].token_ids
                group.append(resp)
            resps.append(group)
            offset += num

        assert len(resps) == len(qur_texts), f"{len(resps)} vs {len(qur_texts)} and resps: {resps}"
        import numpy as np

        non_tensor_batch = prompts.non_tensor_batch
        non_tensor_batch["re_generation"] = np.array(resps, dtype=object)
        
        return DataProto(batch=prompts.batch, non_tensor_batch=non_tensor_batch)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="reward"))
    @DistProfiler.annotate(color="brown")
    def compute_rm_score(self, data: DataProto):
        self.reward_module.wake_up()
        from verl.single_controller.base.decorator import _concat_data_proto_or_future
        # Support all hardwares

        from verl.utils.device import get_device_id

        data = data.to(get_device_id())

        micro_batches = data.split(self.config.micro_batch_size_per_gpu)
        output = []
        for micro_batch in micro_batches:
            sub_data = self._generate_micro_batch(micro_batch)
            output.append(sub_data)
        output = _concat_data_proto_or_future(output)

        token_level_scores = self.rewrad_manager(output, return_dict=False)
        self.reward_module.sleep(level=1)
        return DataProto.from_dict(tensors={"rm_scores": token_level_scores})
        
        