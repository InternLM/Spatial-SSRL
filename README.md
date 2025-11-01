<div align="center">
  <h1 align="center">
    <div style="display: flex; align-items: center; justify-content: center;">
      <div style="text-align: left; line-height: 1.3;">
        Spatial-SSRL: Enhancing Spatial Understanding via Self-Supervised
Reinforcement Learning
      </div>
    </div>
  </h1>
  <p align="center">
    <a href="https://scholar.google.com/citations?hl=en&user=yQ8U3tUAAAAJ"><strong>Yuhong Liu</strong></a >
    ·
    <a href="https://beichenzbc.github.io/"><strong>Beichen Zhang</strong></a >
    ·
    <a href="https://yuhangzang.github.io/"><strong>Yuhang Zang<sup>&dagger;</sup></strong></a >
    ·
    <a href="https://scholar.google.com/citations?user=sJkqsqkAAAAJ"><strong>Yuhang Cao</strong></a >
    ·
    <a href="https://github.com/Cooperx521"><strong>Long Xing</strong></a >
    </br>
    <a href="https://lightdxy.github.io/"><strong>Xiaoyi Dong</strong></a >
    ·
    <a href="https://github.com/kennymckormick"><strong>Haodong Duan</strong></a >
    ·
    <a href="http://dahua.site/"><strong>Dahua Lin</strong></a >
    ·
     <a href="https://myownskyw7.github.io/"><strong>Jiaqi Wang<sup>&dagger;</sup></strong></a >
  </p >
  <p align="center" style="font-size: 1em; margin-top: -1em">  <sup>&dagger;</sup>Corresponding authors. </p >
  <p align="center" style="font-size: 1.2em; margin-top: 0.5em">
    🏠<a href="https://github.com/InternLM/Spatial-SSRL">Homepage</a >
  | 🤗<a href="https://huggingface.co/internlm/Spatial-SSRL-7B">Spatial-SSRL-7B Model</a >
  | 🤗<a href="https://huggingface.co/datasets/internlm/Spatial-SSRL-81k">Spatial-SSRL-81k Dataset</a >
  </p > 
</div>

## 📢 News

## 🌈 Overview
We are thrilled to introduce <strong>Spatial-SSRL</strong>, a novel self-supervised RL paradigm aimed at enhancing LVLM spatial understanding. 
By optimizing Qwen2.5-VL-7B with Spatial-SSRL, the model exhibits stronger spatial intelligence across seven spatial understanding benchmarks in both image and video settings.
</p>
<p style="text-align: center;"> 
  <img src="assets/teaser_1029final.png" alt="Teaser" width="100%"> 
</p>
Spatial-SSRL is a <strong>lightweight</strong> tool-free framework that is natually compatible with the RLVR training paradigm and easy to extend to a multitude of pretext tasks.
Five tasks are currently formulated in the framework, requiring only ordinary RGB and RGB-D images. <strong>And we welcome you to join Spatial-SSRL with effective pretext tasks to further strengthen the capabilities of LVLMs!</strong>

<p style="text-align: center;"> 
  <img src="assets/pipeline_1029final.png" alt="Pipeline" width="100%"> 
</p>

## 💡 Highlights
- 🔥 **Highly Scalable:** Spatial-SSRL uses ordinary raw RGB and RGB-D images instead of richly-annotated public datasets or manual labels for data curation, making it highly scalable.
- 🔥 **Cost-effective:** Avoiding the need for human labels or API calls for general LVLMs throughout the entire pipeline endows Spatial-SSRL with cost-effectiveness.
- 🔥 **Lightweight:** Prior approaches for spatial understanding heavily rely on annotation of external tools, incurring inherent errors in training data and additional cost. In constrast, Spatial-SSRL is completely tool-free and can easily be extended to more self-supervised tasks. 
- 🔥 **Naturally Verifiable:** Intrinsic supervisory signals determined by pretext objectives are naturally verifiable, aligning Spatial-SSRL well with the RLVR paradigm.
<p style="text-align: center;"> 
  <img src="assets/comparison_1029final.png" alt="Teaser" width="100%"> 
</p>

## 📊 Results
We apply our Spatial-SSRL paradigm on Qwen2.5-VL-3B and Qwen2.5-VL-7B and the experimental results across seven spatial understanding benchmarks are shown below.
<p style="text-align: center;"> 
  <img src="assets/exp_result.png" alt="Pipeline" width="100%"> 
</p>

## ⭐️ Quick Start

