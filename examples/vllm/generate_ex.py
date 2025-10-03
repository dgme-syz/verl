import os

from vllm import LLM, SamplingParams


path = os.path.join(os.environ["MODEL"], "Qwen2.5-1.5B-Instruct")
llm_ = LLM(path, enable_sleep_mode=True)
llm_.sleep(level=1)
path = os.path.join(os.environ["MODEL"], "Qwen2.5-1.5B")
llm = LLM(path)

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(
    temperature=0.8, 
    top_p=0.95,
    n=2
)
outputs = llm.generate(prompts, sampling_params)

for i, output in enumerate(outputs):
    print(f"=== Prompt {i} ===")
    print(prompts[i])
    for sample_id in range(len(output.outputs)):
        print(f"--- Sample {sample_id} ---")
        print(output.outputs[sample_id].text)
        print()