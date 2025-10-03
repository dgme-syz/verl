#!/bin/bash
set -e  
#export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_API_KEY=428f855d211a1e71e0dc27c8675469476d8c22a3
#export WANDB_MODE=offline 

export RAY_TMPDIR=/home/nfs05/shenyz/ray_tmp

GPUs=8
MODEL_NAME=Qwen2.5-0.5B
MODEL_PATH=$MODEL/$MODEL_NAME
DATA_PATH=/home/nfs06/shenyz/data
BATCH_SIZE=256
TEACHER_MODEL_NAME=Qwen3-8B
SPLIT=6

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_PATH/SimpleRL/hard/train_processed.parquet \
    data.val_files=$DATA_PATH/SimpleRL/hard/test_processed.parquet \
    data.train_batch_size=$BATCH_SIZE \
    data.max_prompt_length=4096 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=64 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=10240 \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.actor.kl_loss_coef=0.0001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=8 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger="[console,wandb]" \
    trainer.project_name=verl_grpo \
    trainer.experiment_name=$MODEL_NAME+n_$SPLIT+$TEACHER_MODEL_NAME+$BATCH_SIZE+gpus_$GPUs \
    trainer.n_gpus_per_node=$GPUs \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 \
    trainer.val_before_train=False \
    reward_model.reward_manager="naive" \
    custom_reward_function.path="verl/utils/reward_score/gsm8k_update.py" \
    custom_reward_function.name="eval_score" \
    reward_model.enable=True \
    reward_model.micro_batch_size_per_gpu=16 \
    +reward_model.use_legacy_worker_impl=vllm \
    +reward_model.tensor_model_parallel_size=1 \
    +reward_model.vllm.model_path=$MODEL/$TEACHER_MODEL_NAME \
    +reward_model.vllm.re_generation_num=$SPLIT \
    +reward_model.vllm.reward_manager="split" \
    +reward_model.reward_fn_args.score=2.0 \
    +reward_model.vllm.max_model_len=8192 \
    +reward_model.vllm.temperature=0.7 \
    +reward_model.vllm.top_p=0.8 \
    +reward_model.vllm.top_k=20 \
    +reward_model.vllm.repetition_penalty=1.05 \
    +reward_model.use_vllm=True \
    +reward_model.custom_reward_function.path="verl/utils/reward_score/gsm8k_update.py" \
    +reward_model.custom_reward_function.name="compute_score" \
    

