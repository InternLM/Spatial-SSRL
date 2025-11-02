import argparse
import os
import pandas as pd
import json
# 设置CUDA可见设备
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sys
import warnings


from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch
# from qwen_vl_utils import process_vision_info
from qwen_vl_utils.vision_process import process_vision_info

from torch.utils.data import DataLoader

from model_zoo import get_model
from dataset_zoo import get_dataset
from misc import seed_all, _default_collate, save_scores

from PIL import Image
import requests
import copy
from tqdm import tqdm
import numpy as np
import json
import re
# python main_aro.py --dataset=$dataset --model-name=$model_name
def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--num_workers", default=16, type=int)
    parser.add_argument("--model_name", default=None, type=str)
    parser.add_argument("--dataset", default="VG_Relation", type=str, \
            choices=["VG_Relation", "VG_Attribution", "COCO_Order", \
            "Flickr30k_Order", "Controlled_Images_A", "Controlled_Images_B", \
            "COCO_QA_one_obj", "COCO_QA_two_obj", "VG_QA_one_obj", "VG_QA_two_obj"])
    #parser.add_argument("--seed", default=1, type=int)
    
    parser.add_argument("--download", action="store_true", help="Whether to download the dataset if it doesn't exist. (Default: False)")
    parser.add_argument("--save-scores", action="store_true", help="Whether to save the scores for the retrieval to analyze later.")
    parser.add_argument("--output-dir", default="results", type=str)
    parser.add_argument("--CoT", action='store_true')
    parser.add_argument("--temperature", default=0.7, type=float)
    return parser.parse_args()

def extract(response):
    pattern = r"boxed\{([^}]*)\}"
    # 使用 re.findall 查找所有匹配的内容
    matches = re.findall(pattern, response)
    if len(matches) == 0:
        return ''
    return matches[-1]

def get_qwen_model(model_path):
# def load_model(image, caption_options):
    warnings.filterwarnings("ignore")
    device = 'cuda'
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, padding_side='left', use_fast=True)
    # model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)
    # model.eval()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",
            ).eval().to(device)
    return model, processor

def evaluate_scores(output):
    correct_pred = {
        "VG_Relation": 1,
        "VG_Attribution": 1,
        "COCO_Order": 0,
        "Flickr30k_Order": 0,
        "Controlled_Images_A": 0,
        "Controlled_Images_B": 0,
        "COCO_QA_one_obj": 0,
        "COCO_QA_two_obj": 0,
        "VG_QA_one_obj": 0,
        "VG_QA_two_obj": 0,
    }

    labels = ["A", "B", "C", "D"]
    correct_label = labels[correct_pred[args.dataset]]

    if correct_label in output:
        return True
    else:
        return False
    
def main(args):
    #seed_all(args.seed)
    file_path = 'model_config.json'

    # 打开并读取 JSON 文件
    with open(file_path, 'r', encoding='utf-8') as file:
        model_config = json.load(file)
    assert args.model_name in model_config
    model_path = model_config[args.model_name]
    response_dir = os.path.join(args.output_dir, "responses")
    acc_dir = os.path.join(args.output_dir, "accuracy")
    os.makedirs(response_dir, exist_ok=True)
    os.makedirs(acc_dir, exist_ok=True)
    # model, image_preprocess = get_model(args.model_name, args.device)
    image_preprocess = None
    dataset_name = args.dataset
    temperature = args.temperature
    if not args.CoT:
        temperature = 0.45
    device = "cuda"
    dataset = get_dataset(dataset_name, image_preprocess=image_preprocess, download=args.download)
    model, processor = get_qwen_model(model_path)
    correct_num = 0

    output_jsonl_path = os.path.join(response_dir, f"{args.dataset}_x_{args.model_name}.jsonl")

    for idx, item in enumerate(tqdm(dataset)):
        # if idx==5:
        #     break
        image_path = item.image_options[0] 
        caption_options = item.caption_options
        labels = ["A", "B", "C", "D"]

        caption_text = "\n".join([f"{labels[i]}: {opt}" for i, opt in enumerate(caption_options)])
        if args.CoT:
            format_prompt = "Based on the image, choose the correct option from the list below."
            format_prompt += "You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}."
            #format_prompt += "You FIRST think about the reasoning process and then provide the final answer in \\boxed{}." #used for qwen2.5-vl-7b as it fails to follow the prompt above
        else:
            format_prompt = "Based on the image, choose the correct option from the list below. Please only respond the corresponding letter (e.g., C)."
        question = (
            f"{format_prompt}"
            f"{caption_text}"
        )

        messages_query = [
            {
                "role": "user",
                "content": [
                    {"type": "image","image": image_path,"max_pixels": 512*28*28},
                    {"type": "text", "text": question},
                ],
            }
        ]

        image_inputs, _ = process_vision_info(messages_query)

        text_query = processor.apply_chat_template(
            messages_query,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = processor(
            text=[text_query],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)
        
        with torch.no_grad():
            if 'qwen2.5-vl-3b' in model_path.lower(): #Apply do_sample for better performance in small-size models
                output = model.generate(**inputs, max_new_tokens=4096, do_sample=True, temperature=temperature, top_k=50, top_p=0.9)
            else:
                output = model.generate(**inputs, max_new_tokens=4096, do_sample=False)
        response_text = processor.batch_decode(output[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]

        if args.CoT:
            ans = extract(response_text)
        else:
            ans = response_text
        
        hit = evaluate_scores(ans)
        if hit:
            correct_num += 1
        output_entry = {
            "index": idx,
            "question": question,
            "output": response_text,
            "answer": ans,
            "hit": hit
        }
        with open(output_jsonl_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(output_entry, ensure_ascii=False) + "\n")

    accuracy_mean = correct_num / len(dataset)

    jsonl_output_file = os.path.join(acc_dir, f"{args.dataset}_x_{args.model_name}.jsonl")
    accuracy_data = {
        "Model": args.model_name,
        "Mean_Accuracy": accuracy_mean,
        "Dataset": args.dataset
        }
    with open(jsonl_output_file, 'w') as file:
        json.dump(accuracy_data, file)

    print(accuracy_data)

    
if __name__ == "__main__":
    args = config()
    main(args)