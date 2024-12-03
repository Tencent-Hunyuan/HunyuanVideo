<!-- ## **HunyuanVideo** -->

<p align="center">
  <img src="assets/logo.png"  height=100>
</p>

# HunyuanVideo: A Systematic Framework For Large Video Generation Model Training

<div align="center">
  <a href="https://github.com/Tencent/HunyuanVideo"><img src="https://img.shields.io/static/v1?label=HunyuanVideo Code&message=Github&color=blue&logo=github-pages"></a> &ensp;
  <a href="https://aivideo.hunyuan.tencent.com"><img src="https://img.shields.io/static/v1?label=Project%20Page&message=Web&color=green&logo=github-pages"></a> &ensp;
  <a href="https://video.hunyuan.tencent.com"><img src="https://img.shields.io/static/v1?label=Playground&message=Web&color=green&logo=github-pages"></a> &ensp;
  <a href="https://github.com/Tencent/HunyuanVideo/blob/main/assets/hunyuanvideo.pdf"><img src="https://img.shields.io/static/v1?label=Tech Report&message=Arxiv:HunyuanVideo&color=red&logo=arxiv"></a> &ensp;
  <a href="https://huggingface.co/tencent/HunyuanVideo"><img src="https://img.shields.io/static/v1?label=HunyuanVideo&message=HuggingFace&color=yellow"></a> &ensp; &ensp;
  <a href="https://huggingface.co/tencent/HunyuanVideo-PromptRewrite"><img src="https://img.shields.io/static/v1?label=HunyuanVideo-PromptRewrite&message=HuggingFace&color=yellow"></a> &ensp; &ensp;
</div>

-----

This repo contains PyTorch model definitions, pre-trained weights and inference/sampling code for our paper exploring HunyuanVideo. You can find more visualizations on our [project page](https://aivideo.hunyuan.tencent.com).

> [**HunyuanVideo: A Systematic Framework For Large Video Generation Model Training**](https://github.com/Tencent/HunyuanVideo/blob/main/assets/hunyuanvideo.pdf) <br>

## 🎥 Demo
<div align="center">
  <video src="https://github.com/user-attachments/assets/f37925a3-7d42-40c9-8a9b-5a010c7198e2" width="50%">
</div>

The video is heavily compressed due to complaince of GitHub policy. The high quality version can be downloaded from [here](https://aivideo.hunyuan.tencent.com/download/HunyuanVideo/material/demo.mov).

## 🔥🔥🔥 News!!
* Dec 3, 2024: 🤗 We release the inference code and model weights of HunyuanVideo.

## 📑 Open-source Plan

- HunyuanVideo (Text-to-Video Model)
  - [x] Inference 
  - [x] Checkpoints 
  - [ ] Penguin Video Benchmark
  - [ ] Web Demo (Gradio) 
  - [ ] ComfyUI
  - [ ] Diffusers 
- HunyuanVideo (Image-to-Video Model)
  - [ ] Inference 
  - [ ] Checkpoints 

## Contents
- [HunyuanVideo: A Systematic Framework For Large Video Generation Model Training](#hunyuanvideo--a-systematic-framework-for-large-video-generation-model-training)
  - [🎥 Demo](#-demo)
  - [🔥🔥🔥 News!!](#-news!!)
  - [📑 Open-source Plan](#-open-source-plan)
  - [Contents](#contents)
  - [**Abstract**](#abstract)
  - [**HunyuanVideo Overall Architecture**](#-hunyuanvideo-overall-architecture)
  - [🎉 **HunyuanVideo Key Features**](#-hunyuanvideo-key-features)
    - [**Unified Image and Video Generative Architecture**](#unified-image-and-video-generative-architecture)
    - [**MLLM Text Encoder**](#mllm-text-encoder)
    - [**3D VAE**](#3d-vae)
    - [**Prompt Rewrite**](#prompt-rewrite)
  - [📈 Comparisons](#-comparisons)
  - [📜 Requirements](#-requirements)
  - [🛠️ Dependencies and Installation](#-dependencies-and-installation)
    - [Installation Guide for Linux](#installation-guide-for-linux)
  - [🧱 Download Pretrained Models](#-download-pretrained-models)
  - [🔑 Inference](#-inference)
    - [Using Command Line](#using-command-line)
    - [More Configurations](#more-configurations)
  - [🔗 BibTeX](#-bibtex)
  - [Acknowledgements](#acknowledgements)
---

## **Abstract**
We present HunyuanVideo, a novel open-source video foundation model that exhibits performance in video generation that is comparable to, if not superior to, leading closed-source models. HunyuanVideo features a comprehensive framework that integrates several key contributions, including data curation, image-video joint model training, and an efficient infrastructure designed to facilitate large-scale model training and inference. Additionally, through an effective strategy for scaling model architecture and dataset, we successfully trained a video generative model with over 13 billion parameters, making it the largest among all open-source models. 

We conducted extensive experiments and implemented a series of targeted designs to ensure high visual quality, motion diversity, text-video alignment, and generation stability. According to professional human evaluation results, HunyuanVideo outperforms previous state-of-the-art models, including Runway Gen-3, Luma 1.6, and 3 top performing Chinese video generative models. By releasing the code and weights of the foundation model and its applications, we aim to bridge the gap between closed-source and open-source video foundation models. This initiative will empower everyone in the community to experiment with their ideas, fostering a more dynamic and vibrant video generation ecosystem. 

## **HunyuanVideo Overall Architecture**

HunyuanVideo is trained on a spatial-temporally
compressed latent space, which is compressed through Causal 3D VAE. Text prompts are encoded
using a large language model, and used as the condition. Gaussian noise and condition are taken as
input, our generate model generates an output latent, which is decoded to images or videos through
the 3D VAE decoder.
<p align="center">
  <img src="assets/overall.png"  height=300>
</p>

## 🎉 **HunyuanVideo Key Features**
### **Unified Image and Video Generative Architecture**
HunyuanVideo introduces the Transformer design and employs a Full Attention mechanism for unified image and video generation. 
Specifically, we use a "Dual-stream to Single-stream" hybrid model design for video generation. In the dual-stream phase, video and text
tokens are processed independently through multiple Transformer blocks, enabling each modality to learn its own appropriate modulation mechanisms without interference. In the single-stream phase, we concatenate the video and text
tokens and feed them into subsequent Transformer blocks for effective multimodal information fusion.
This design captures complex interactions between visual and semantic information, enhancing
overall model performance.
<p align="center">
  <img src="assets/backbone.png"  height=350>
</p>

### **MLLM Text Encoder**
Some previous text-to-video model typically use pretrained CLIP and T5-XXL as text encoders where CLIP uses Transformer Encoder and T5 uses a Encoder-Decoder structure. In constrast, we utilize a pretrained Multimodal Large Language Model (MLLM) with a Decoder-Only structure as our text encoder, which has following advantages: (i) Compared with T5, MLLM after visual instruction finetuning has better image-text alignment in the feature space, which alleviates the difficulty of instruction following in diffusion models; (ii)
Compared with CLIP, MLLM has been demonstrated superior ability in image detail description
and complex reasoning; (iii) MLLM can play as a zero-shot learner by following system instructions prepended to user prompts, helping text features pay more attention to key information. In addition, MLLM is based on causal attention while T5-XXL utilizes bidirectional attention that produces better text guidance for diffusion models. Therefore, we introduce an extra bidirectional token refiner for enhancing text features.
<p align="center">
  <img src="assets/text_encoder.png"  height=275>
</p>

### **3D VAE**
HunyuanVideo trains a 3D VAE with CausalConv3D to compress pixel-space videos and images into a compact latent space. We set the compression ratios of video length, space and channel to 4, 8 and 16 respectively. This can significantly reduce the number of tokens for the subsequent diffusion transformer model, allowing us to train videos at the original resolution and frame rate.
<p align="center">
  <img src="assets/3dvae.png"  height=150>
</p>

### **Prompt Rewrite**
To address the variability in linguistic style and length of user-provided prompts, we fine-tune the [Hunyuan-Large model](https://github.com/Tencent/Tencent-Hunyuan-Large) as our prompt rewrite model to adapt the original user prompt to model-preferred prompt.

We provide two rewrite modes: Normal mode and Master mode, which can be called using different prompts. The prompts are shown [here](hyvideo/prompt_rewrite.py). The Normal mode is designed to enhance the video generation model's comprehension of user intent, facilitating a more accurate interpretation of the instructions provided. The Master mode enhances the description of aspects such as composition, lighting, and camera movement, which leans towards generating videos with a higher visual quality. However, this emphasis may occasionally result in the loss of some semantic details. 

The Prompt Rewrite Model can be directly deployed and inferred using the [Hunyuan-Large original code](https://github.com/Tencent/Tencent-Hunyuan-Large). We release the weights of the Prompt Rewrite Model [here](https://huggingface.co/Tencent/HunyuanVideo-PromptRewrite).

## 📈 Comparisons

To evaluate the performance of HunyuanVideo, we selected five strong baselines from closed-source video generation models. In total, we utilized 1,533 text prompts, generating an equal number of video samples with HunyuanVideo in a single run. For a fair comparison, we conducted inference only once, avoiding any cherry-picking of results. When comparing with the baseline methods, we maintained the default settings for all selected models, ensuring consistent video resolution. Videos were assessed based on three criteria: Text Alignment, Motion Quality and Visual Quality. More than 60 professional evaluators performed the evaluation. Notably, HunyuanVideo demonstrated the best overall performance, particularly excelling in motion quality.

<p align="center">
<table> 
<thead> 
<tr> 
    <th rowspan="2">Model</th> <th rowspan="2">Open Source</th> <th>Duration</th> <th>Text Alignment</th> <th>Motion Quality</th> <th rowspan="2">Visual Quality</th> <th rowspan="2">Overall</th>  <th rowspan="2">Ranking</th>
</tr> 
</thead> 
<tbody> 
<tr> 
    <td>HunyuanVideo (Ours)</td> <td> ✔ </td> <td>5s</td> <td>68.5%</td> <td>64.5%</td> <td>96.4%</td> <td>44.7%</td> <td>1</td>
</tr> 
<tr> 
    <td>CNTopA (API)</td> <td> &#10008 </td> <td>5s</td> <td>68.8%</td> <td>57.5%</td> <td>95.8%</td> <td>38.8%</td> <td>2</td>
</tr> 
<tr> 
    <td>CNTopB (Web)</td> <td> &#10008</td> <td>5s</td> <td>64.5%</td> <td>59.3%</td> <td>97.7%</td> <td>37.6%</td> <td>3</td>
</tr> 
<tr> 
    <td>GEN-3 alpha (Web)</td> <td>&#10008</td> <td>6s</td> <td>49.3%</td> <td>48.3%</td> <td>97.1%</td> <td>24.6%</td> <td>4</td> 
</tr> 

<tr> 
    <td>CNTopC (Web)</td> <td>&#10008</td> <td>5s</td> <td>52.7%</td> <td>42.1%</td> <td>96.2%</td> <td>24.1%</td> <td>5</td>
</tr> 
<tr> 
    <td>Luma1.6 (API)</td><td>&#10008</td> <td>5s</td> <td>59.7%</td> <td>36.8%</td> <td>93.5%</td> <td>21.6%</td> <td>6</td>
</tr>
</tbody>
</table>
</p>

## 📜 Requirements

The following table shows the requirements for running HunyuanVideo model (batch size = 1) to generate videos:

|     Model    |  Setting<br/>(height/width/frame) | Denoising step | GPU Peak Memory  |
|:------------:|:--------------------------------:|:--------------:|:----------------:|
| HunyuanVideo   |        720px1280px129f          |       30       |       60G        |
| HunyuanVideo   |        544px960px129f           |       30       |       45G        |

* An NVIDIA GPU with CUDA support is required. 
  * We have tested on a single H800/H20 GPU.
  * **Minimum**: The minimum GPU memory required is 60GB for 720px1280px129f and 45G for 544px960px129f.
  * **Recommended**: We recommend using a GPU with 80GB of memory for better generation quality.
* Tested operating system: Linux

## 🛠️ Dependencies and Installation

Begin by cloning the repository:
```shell
git clone https://github.com/tencent/HunyuanVideo
cd HunyuanVideo
```

### Installation Guide for Linux

We provide an `environment.yml` file for setting up a Conda environment.
Conda's installation instructions are available [here](https://docs.anaconda.com/free/miniconda/index.html).

We recommend CUDA versions 11.8 and 12.0+.

```shell
# 1. Prepare conda environment
conda env create -f environment.yml

# 2. Activate the environment
conda activate HunyuanVideo

# 3. Install pip dependencies
python -m pip install -r requirements.txt

# 4. Install flash attention v2 for acceleration (requires CUDA 11.8 or above)
python -m pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.5.9.post1
```

Additionally, HunyuanVideo also provides a pre-built Docker image:
[docker_hunyuanvideo](https://hub.docker.com/repository/docker/hunyuanvideo/hunyuanvideo/general). 

```shell
# 1. Use the following link to download the docker image tar file (For CUDA 12).
wget https://aivideo.hunyuan.tencent.com/download/HunyuanVideo/hunyuan_video_cu12.tar

# 2. Import the docker tar file and show the image meta information (For CUDA 12).
docker load -i hunyuan_video.tar

docker image ls

# 3. Run the container based on the image
docker run -itd --gpus all --init --net=host --uts=host --ipc=host --name hunyuanvideo --security-opt=seccomp=unconfined --ulimit=stack=67108864 --ulimit=memlock=-1 --privileged  docker_image_tag
```


## 🧱 Download Pretrained Models

The details of download pretrained models are shown [here](ckpts/README.md).

## 🔑 Inference
We list the height/width/frame settings we support in the following table.

|      Resolution       |           h/w=9:16           |    h/w=16:9     |     h/w=4:3     |     h/w=3:4     |     h/w=1:1     |
|:---------------------:|:----------------------------:|:---------------:|:---------------:|:---------------:|:---------------:|
|         540p          |        544px960px129f        |  960px544px129f | 624px832px129f  |  832px624px129f |  720px720px129f |
| 720p (recommended)    |       720px1280px129f        | 1280px720px129f | 1104px832px129f | 832px1104px129f | 960px960px129f  |

### Using Command Line

```bash
cd HunyuanVideo

python3 sample_video.py \
    --video-size 720 1280 \
    --video-length 129 \
    --infer-steps 50 \
    --prompt "A cat walks on the grass, realistic style." \
    --flow-reverse \
    --flow-shift 7.0 \
    --seed 0 \
    --use-cpu-offload \
    --save-path ./results
```

### More Configurations

We list some more useful configurations for easy usage:

|        Argument        |  Default  |                Description                |
|:----------------------:|:---------:|:-----------------------------------------:|
|       `--prompt`       |   None    |   The text prompt for video generation    |
|     `--video-size`     | 720 1280  |      The size of the generated video      |
|    `--video-length`    |    129    |     The length of the generated video     |
|    `--infer-steps`     |    50     |     The number of steps for sampling      |
| `--embedded-cfg-scale` |    6.0    |    Embeded  Classifier free guidance scale       |
|     `--flow-shift`     |    7.0    | Shift factor for flow matching schedulers |
|     `--flow-reverse`   |    False  | If reverse, learning/sampling from t=1 -> t=0 |
|        `--seed`        |     0     |   The random seed for generating video    |
|  `--use-cpu-offload`   |   False   |    Use CPU offload for the model load to save more memory, necessary for high-res video generation    |
|     `--save-path`      | ./results |     Path to save the generated video      |


## 🔗 BibTeX
If you find [HunyuanVideo](https://github.com/Tencent/HunyuanVideo/blob/main/assets/hunyuanvideo.pdf) useful for your research and applications, please cite using this BibTeX:

```BibTeX
@misc{kong2024hunyuanvideo,
      title={HunyuanVideo: A Systematic Framework For Large Video Generative Models}, 
      author={Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, Kathrina Wu, Qin Lin, Aladdin Wang, Andong Wang, Changlin Li, Duojun Huang, Fang Yang, Hao Tan, Hongmei Wang, Jacob Song, Jiawang Bai, Jianbing Wu, Jinbao Xue, Joey Wang, Junkun Yuan, Kai Wang, Mengyang Liu, Pengyu Li, Shuai Li, Weiyan Wang, Wenqing Yu, Xinchi Deng, Yang Li, Yanxin Long, Yi Chen, Yutao Cui, Yuanbo Peng, Zhentao Yu, Zhiyu He, Zhiyong Xu, Zixiang Zhou, Zunnan Xu, Yangyu Tao, Qinglin Lu, Songtao Liu, Dax Zhou, Hongfa Wang, Yong Yang, Di Wang, Yuhong Liu, and Jie Jiang, along with Caesar Zhong},
      year={2024},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgements
We would like to thank the contributors to the [SD3](https://huggingface.co/stabilityai/stable-diffusion-3-medium), [FLUX](https://github.com/black-forest-labs/flux), [Llama](https://github.com/meta-llama/llama), [LLaVA](https://github.com/haotian-liu/LLaVA), [Xtuner](https://github.com/InternLM/xtuner), [diffusers](https://github.com/huggingface/diffusers) and [huggingface](https://huggingface.co) repositories, for their open research and exploration.
Additionally, we also thank the Tencent Hunyuan Multimodal team for their help with the text encoder. 

## Star History
<a href="https://star-history.com/#Tencent/HunyuanVideo&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=Tencent/HunyuanVideo&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=Tencent/HunyuanVideo&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=Tencent/HunyuanVideo&type=Date" />
 </picture>
</a>
