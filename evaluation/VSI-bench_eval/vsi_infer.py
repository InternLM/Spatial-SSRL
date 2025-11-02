import argparse, json, os, copy, math
from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np
import decord
from PIL import Image
import torch.distributed as dist
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor
)
from qwen_vl_utils import process_vision_info
from datetime import timedelta
from datetime import timedelta

def fix_prompt(q, options, cot):
    choice = ['A. ', 'B. ', 'C. ', 'D. ']
    prompt = "Watch the given video and anser the following question.\n\n"
    prompt += q
    cnt = 0
    for i, obj in enumerate(options):
        if obj is None:
            break
        else:
            cnt += 1
            prompt += choice[i] + obj
    if not cot:
        if cnt==0:
            prompt += "\nAnswer directly with a number(integer or decimal)."
        else: #Multi-choice problem
            prompt += "\nAnswer directly with the option letter from the given choices."
    else:
        if cnt==0:
            prompt += "\nThe final answer should be a number(integer or decimal).\n"
        else: #Multi-choice problem
            prompt += "\nThe final answer should be the option letter from the given choices.\n"
        prompt += "You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}."
    return prompt

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--input_json",  required=True, type=str)
    parser.add_argument("--output_dir",  required=True, type=str)
    parser.add_argument("--output_json", required=True, type=str)
    parser.add_argument("--max_new_tokens", default=4096, type=int)
    parser.add_argument("--cot", default='false', type=str)
    return parser.parse_args()

def init_dist():
    """初始化分布式环境（torchrun 已自动设置 RANK LOCAL_RANK WORLD_SIZE）"""
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=5400))
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    return rank, world_size

def load_model_processor(model_path):
    """每个进程各自完整加载模型到单卡"""
    rank = int(os.environ["RANK"])
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",        # 关键：禁用 auto
        attn_implementation="flash_attention_2"  # 可选加速
    ) #use torch.bfloat16 for models based on qwen2.5-vl-3b
    processor = AutoProcessor.from_pretrained(model_path)
    model.eval()
    return model, processor

def uniform_sample_frames(video_path, max_frames=512):
    """
    均匀采样最多 max_frames 帧，返回 List[PIL.Image]
    """
    vr = decord.VideoReader(video_path, num_threads=1)
    total = len(vr)
    if total <= max_frames:
        idx = list(range(total))
    else:
        idx = np.linspace(0, total - 1, max_frames).round().astype(int).tolist()
    frames = vr.get_batch(idx).asnumpy()          # [N, H, W, 3]
    return [Image.fromarray(f) for f in frames]


@torch.inference_mode()
def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    cot = args.cot.lower()=='true'
    rank, world_size = init_dist()

    # 每个进程只加载自己那份数据
    with open(args.input_json, "r", encoding="utf-8") as f:
        all_lines = json.load(f)
    sample_indices = [i for i in range(rank, len(all_lines), world_size)]
    # 根据索引提取数据
    local_data = [all_lines[i] for i in sample_indices]

    model, processor = load_model_processor(args.model_path)

    local_output_file = f"{args.output_dir}/local_rank_{rank}.jsonl"
    local_answers = []
    if os.path.exists(local_output_file):
        print(f"{local_output_file} found, will be reused")
        with open(local_output_file, "r", encoding="utf-8") as f:
            for line in f:
                local_answers.append(json.loads(line))
    local_data = local_data[len(local_answers):]
    
    for sample in tqdm(local_data, desc=f"Rank {rank}/{world_size}"):
        mp4_path = sample["mp4_path"]
        question = sample["question"]
        frames = uniform_sample_frames(mp4_path, max_frames=128)
        options = [sample["A"], sample["B"], sample["C"], sample["D"]]
        messages = [{
            "role": "user",
            "content": [
                {"type": "video", "video": frames},
                {"type": "text",  "text": fix_prompt(question, options, cot)},
            ]
        }]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to('cuda')
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False
        )
        answer = processor.batch_decode(
            generated[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0].strip()

        out = copy.deepcopy(sample)
        out["pred"] = answer
        local_answers.append(out)
        with open(local_output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    # 收集到 rank0
    if rank == 0:
        gathered = [None] * world_size
    else:
        gathered = None
    dist.gather_object(local_answers, gathered, dst=0)

    if rank == 0:
        # 按原始顺序合并
        all_results = []
        for g in gathered:
            all_results.extend(g)
        all_results.sort(key=lambda x: all_lines.index({k: x[k] for k in x if k != "pred"}))
        with open(os.path.join(args.output_dir, args.output_json), "w", encoding="utf-8") as f:
            for item in all_results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Done! Result -> {args.output_json}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()