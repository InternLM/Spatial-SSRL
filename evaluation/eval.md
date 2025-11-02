## Evaluation Tutorials
- Benchmarks Supported by VLMEvalkit

    VLMEvalkit-supported benchmarks include Spatial457, 3DSRBench, SpatialEval, QSpatial_plus, MMBench_TEST_EN_V11, BLINK, HallusionBench, RealWorldQA, OCRBench, ChartQA_TEST, and SEEDBench2_Plus.
    We provide the necessary code for evaluation in `./Remaining_Bench_eval`, which is built based on the official repository of <a href="https://github.com/open-compass/VLMEvalKit">VLMEvalkit</a >. Run the following command for evaluation:
    ```bash
    cd Remaining_Bench_eval
    
    #Single GPU
    python3 run.py --data dataset_name --model Spatial-SSRL-7B --verbose

    #Multi-GPU (Distributed)
    torchrun --nproc-per-node=8 run.py --data dataset_name --model Spatial-SSRL-7B --verbose
    ```

- ViewSpatial

    We provide the necessary code for evaluation in `./ViewSpatial_eval/`, which is built based on the official repository of <a href="https://github.com/ZJU-REAL/ViewSpatial-Bench">ViewSpatial</a >. You can directly start your evaluation by running the scripts in `./ViewSpatial_eval/` or use the following command :
    ```bash
    cd ViewSpatial_eval
    
    #Single GPU
    python3 run.py --model_path internlm/Spatial-SSRL-7B --CoT

    #Multi-GPU (Distributed)
    torchrun --nproc-per-node=8 run.py --model_path internlm/Spatial-SSRL-7B --CoT
    ```
- VSI-Bench

    We provide the necessary code for evaluation in `./VSI-bench_eval/`. You can directly start your evaluation by running the following command :
    ```bash
    cd VSI-bench_eval
    bash run_cot.sh #Infer
    python3 check_accuracy.py #Eval
    ```

- What'sUp

    We provide the necessary code for evaluation in `./whatsup_eval`, which is built based on the official repository of <a href="https://github.com/amitakamath/whatsup_vlms">Whatsup_vlms</a >. You can directly start your evaluation by running the scripts in `./whatsup_eval/` or use the following command :
    ```bash
    cd whatsup_eval
    
    python3 main.py --model_name SpatialSSRL-7b --dataset Controlled_Images_A --CoT
    ```
