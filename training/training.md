# Training Tutorials
We follow two stages in our training procedure: SFT coldstart + GRPO.

## Stage 1: SFT coldstart
Our code for this stage is built on <a href="https://github.com/hiyouga/LLaMAFactory">LLAMA-Factory</a >.

Download ```SFT-coldstart.json``` and ```images_coldstart.zip```
 from 🤗<a href="https://huggingface.co/datasets/internlm/Spatial-SSRL-81k">Spatial-SSRL-81k </a > to ```SFT-coldstart/data/``` and unzip the second file. The final organization of the folder should be like this:

```text
SFT-coldstart/
├── assets/  
├── data/                
│   ├── coldstart_SFT_images/    
│       ├──  img_0.jpg
        ├──  img_1.jpg
        ├──  ...
│   ├── SFT-coldstart.json       
│   └── dataset_info.json       
├── ...
···
```
Then prepare the environment and start your training!
```bash
cd SFT-coldstart
conda create -n coldstart 
conda activate coldstart
pip install -e .

#Qwen2.5-VL-3B
llamafactory-cli train sft_scripts/Qwen2.5-VL-3B.yaml
llamafactory-cli export sft_scripts/merge_2.5-3B.yaml

#Qwen2.5-VL-7B
llamafactory-cli train sft_scripts/Qwen2.5-VL-7B.yaml
llamafactory-cli export sft_scripts/merge_2.5-7B.yaml

#Qwen3-VL-3B
llamafactory-cli train sft_scripts/Qwen3-VL-4B.yaml
llamafactory-cli export sft_scripts/merge_3-4B.yaml
```
You can obtain the checkpoints for GRPO in ```./output```.

## Stage 2: GRPO
Our code for this stage is built on <a href="https://github.com/hiyouga/EasyR1">EasyR1</a >.


Download ```spatialssrl.parquet``` and ```images.zip```
 from 🤗<a href="https://huggingface.co/datasets/internlm/Spatial-SSRL-81k">Spatial-SSRL-81k </a >. Unzip ```images.zip``` to ```data/images```. Rename ```spatialssrl.parquet``` as ```train.parquet``` and put it to ```data/parquets/```. The final organization of the folder should be like this:

```text
SFT-coldstart/
├── assets/  
├── data/                
│   ├── images/    
│       ├──  crop/
        ├──  depth/
        ├──  flip/
        ├──  position/
        └──  shuffle/
│   ├── parquets/
│       ├──  train.parquet 
│   └── process_GRPO_val.py     
├── ...

```
Then process the validation parquet, prepare the environment and start your training!
```bash
#Generate val set from training set. (Or you can use other val set you want.)
cd GRPO/data
python3 process_GRPO_val.py
cd ..

conda create -n GRPO
conda activate GRPO
pip install -e .

#Qwen2.5-VL-3B
bash examples/spatial-ssrl_qwen2_5_vl_3b.sh 

#Qwen2.5-VL-7B
bash examples/spatial-ssrl_qwen2_5_vl_7b.sh 

#Qwen3-VL-3B
bash examples/spatial-ssrl_qwen3_vl_4b.sh

#Merge Checkpoint in Hugging Face Format
python3 scripts/model_merger.py --local_dir checkpoints/easy_r1/exp_name/global_step_1/actor
```


## Why SFT cold-start?
(You do not have to read this section.)

The fcold-start stage is crucial for the LVLMs to grasp the meanings of the self-supervised tasks constructed by us. Without this stage, they will fail to rollout responses with diversity in final answers during GRPO.

We have synthesized around 3.6k QA pairs with CoT responses for the SFT-coldstart stage. The data is provided in 🤗<a href="https://huggingface.co/datasets/internlm/Spatial-SSRL-81k">Spatial-SSRL-81k </a > for you to reproduce our result. Given the ground-truth answer of each problem, you can directly apply SFT, or synthesize SFT training data containing CoT yourself using any LVLM. 

Note that the cold-start phase is merely to let the LVLM know what possible output results there might be for each task, so whether there are CoT answers or the quality of CoT answers is not important, and we found SFT-coldstart that hardly improves the LVLM's spatial understanding.

