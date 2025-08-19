<!-- ## **HunyuanVideo** -->

[中文阅读](./README_zh.md)

<p align="center">
  <img src="https://raw.githubusercontent.com/Tencent-Hunyuan/HunyuanVideo/refs/heads/main/assets/logo.png"  height=100>
</p>

# HunyuanVideo: A Systematic Framework For Large Video Generation Model

<div align="center">
  <a href="https://github.com/Tencent-Hunyuan/HunyuanVideo"><img src="https://img.shields.io/static/v1?label=HunyuanVideo Code&message=Github&color=blue"></a> &ensp;
  <a href="https://aivideo.hunyuan.tencent.com"><img src="https://img.shields.io/static/v1?label=Project%20Page&message=Web&color=green"></a> &ensp;
  <a href="https://video.hunyuan.tencent.com"><img src="https://img.shields.io/static/v1?label=Playground&message=Web&color=green"></a>
</div>
<div align="center">
  <a href="https://arxiv.org/abs/2412.03603"><img src="https://img.shields.io/static/v1?label=Tech Report&message=Arxiv&color=red"></a> &ensp;
  <a href="https://aivideo.hunyuan.tencent.com/hunyuanvideo.pdf"><img src="https://img.shields.io/static/v1?label=Tech Report&message=High-Quality Version (~350M)&color=red"></a>
</div>
<div align="center">
  <a href="https://huggingface.co/tencent/HunyuanVideo"><img src="https://img.shields.io/static/v1?label=HunyuanVideo&message=HuggingFace&color=yellow"></a> &ensp;
  <a href="https://huggingface.co/docs/diffusers/main/api/pipelines/hunyuan_video"><img src="https://img.shields.io/static/v1?label=HunyuanVideo&message=Diffusers&color=yellow"></a> &ensp;
  <a href="https://huggingface.co/tencent/HunyuanVideo-PromptRewrite"><img src="https://img.shields.io/static/v1?label=HunyuanVideo-PromptRewrite&message=HuggingFace&color=yellow"></a>


 [![Replicate](https://replicate.com/zsxkib/hunyuan-video/badge)](https://replicate.com/zsxkib/hunyuan-video)
</div>

<p align="center">
    👋 Join our <a href="assets/WECHAT.md" target="_blank">WeChat</a> and <a href="https://discord.gg/tv7FkG4Nwf" target="_blank">Discord</a> 
</p>
<p align="center">

-----

This repo contains PyTorch model definitions, pre-trained weights and inference/sampling code for our paper exploring HunyuanVideo. You can find more visualizations on our [project page](https://aivideo.hunyuan.tencent.com).

> [**HunyuanVideo: A Systematic Framework For Large Video Generation Model**](https://arxiv.org/abs/2412.03603) <be>



## 🔥🔥🔥 News!!

* May 28, 2025: 💃 We release the [HunyuanVideo-Avatar](https://github.com/Tencent-Hunyuan/HunyuanVideo-Avatar), an audio-driven human animation model  based on HunyuanVideo.
* May 09, 2025: 🙆 We release the [HunyuanCustom](https://github.com/Tencent-Hunyuan/HunyuanCustom), a multimodal-driven architecture for customized video generation based on HunyuanVideo.
* Mar 06, 2025: 🌅 We release the [HunyuanVideo-I2V](https://github.com/Tencent-Hunyuan/HunyuanVideo-I2V), an image-to-video model based on HunyuanVideo.
* Jan 13, 2025: 📈 We release the [Penguin Video Benchmark](https://github.com/Tencent-Hunyuan/HunyuanVideo/blob/main/assets/PenguinVideoBenchmark.csv).
* Dec 18, 2024: 🏃‍♂️ We release the [FP8 model weights](https://huggingface.co/tencent/HunyuanVideo/blob/main/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8.pt) of HunyuanVideo to save more GPU memory.
* Dec 17, 2024: 🤗 HunyuanVideo has been integrated into [Diffusers](https://huggingface.co/docs/diffusers/main/api/pipelines/hunyuan_video).
* Dec 7, 2024: 🚀 We release the parallel inference code for HunyuanVideo powered by [xDiT](https://github.com/xdit-project/xDiT).
* Dec 3, 2024: 👋 We release the inference code and model weights of HunyuanVideo. [Download](https://github.com/Tencent-Hunyuan/HunyuanVideo/blob/main/ckpts/README.md).



## 🎥 Demo

<div align="center">
  <video src="https://github.com/user-attachments/assets/22440764-0d7e-438e-a44d-d0dad1006d3d" width="70%" poster="./assets/video_poster.png"> </video>
</div>


## 🧩 Community Contributions

If you develop/use HunyuanVideo in your projects, welcome to let us know.

- ComfyUI-Kijai (FP8 Inference, V2V and IP2V Generation): [ComfyUI-HunyuanVideoWrapper](https://github.com/kijai/ComfyUI-HunyuanVideoWrapper) by [Kijai](https://github.com/kijai)
- ComfyUI-Native (Native Support): [ComfyUI-HunyuanVideo](https://comfyanonymous.github.io/ComfyUI_examples/hunyuan_video/) by [ComfyUI Official](https://github.com/comfyanonymous/ComfyUI)

- FastVideo (Consistency Distilled Model and Sliding Tile Attention): [FastVideo](https://github.com/hao-ai-lab/FastVideo) and [Sliding Tile Attention](https://hao-ai-lab.github.io/blogs/sta/) by [Hao AI Lab](https://hao-ai-lab.github.io/)
- HunyuanVideo-gguf (GGUF Version and Quantization): [HunyuanVideo-gguf](https://huggingface.co/city96/HunyuanVideo-gguf) by [city96](https://huggingface.co/city96)
- Enhance-A-Video (Better Generated Video for Free): [Enhance-A-Video](https://github.com/NUS-HPC-AI-Lab/Enhance-A-Video) by [NUS-HPC-AI-Lab](https://ai.comp.nus.edu.sg/)
- TeaCache (Cache-based Accelerate): [TeaCache](https://github.com/LiewFeng/TeaCache) by [Feng Liu](https://github.com/LiewFeng)
- HunyuanVideoGP (GPU Poor version): [HunyuanVideoGP](https://github.com/deepbeepmeep/HunyuanVideoGP) by [DeepBeepMeep](https://github.com/deepbeepmeep)
- RIFLEx (Video Length Extrapolation): [RIFLEx](https://riflex-video.github.io/) by [Tsinghua University](https://riflex-video.github.io/)
- HunyuanVideo Keyframe Control Lora: [hunyuan-video-keyframe-control-lora](https://github.com/dashtoon/hunyuan-video-keyframe-control-lora) by [dashtoon](https://github.com/dashtoon)
- Sparse-VideoGen (Accelerate Video Generation with High Pixel-level Fidelity): [Sparse-VideoGen](https://github.com/svg-project/Sparse-VideoGen) by [University of California, Berkeley](https://svg-project.github.io/)
- FramePack (Packing Input Frame Context in Next-Frame Prediction Models for Video Generation): [FramePack](https://github.com/lllyasviel/FramePack) by [Lvmin Zhang](https://github.com/lllyasviel) 
- Jenga (Training-Free Efficient Video Generation via Dynamic Token Carving): [Jenga](https://github.com/dvlab-research/Jenga) by [DV Lab](https://github.com/dvlab-research)
- DCM (Dual-Expert Consistency Model for Efficient and High-Quality Video Generation): [DCM](https://github.com/Vchitect/DCM) by [Vchitect](https://github.com/Vchitect/DCM)


## 📑 Open-source Plan

- HunyuanVideo (Text-to-Video Model)
  - [x] Inference 
  - [x] Checkpoints
  - [x] Multi-gpus Sequence Parallel inference (Faster inference speed on more gpus)
  - [x] Web Demo (Gradio)
  - [x] Diffusers 
  - [x] FP8 Quantified weight
  - [x] Penguin Video Benchmark
  - [x] ComfyUI
- [HunyuanVideo (Image-to-Video Model)](https://github.com/Tencent/HunyuanVideo-I2V)
  - [X] Inference 
  - [X] Checkpoints 



## Contents
- [HunyuanVideo: A Systematic Framework For Large Video Generation Model](#hunyuanvideo-a-systematic-framework-for-large-video-generation-model)
  - [🎥 Demo](#-demo)
  - [🔥🔥🔥 News!!](#-news)
  - [🧩 Community Contributions](#-community-contributions)
  - [📑 Open-source Plan](#-open-source-plan)
  - [Contents](#contents)
  - [**Abstract**](#abstract)
  - [**HunyuanVideo Overall Architecture**](#hunyuanvideo-overall-architecture)
  - [🎉 **HunyuanVideo Key Features**](#-hunyuanvideo-key-features)
    - [**Unified Image and Video Generative Architecture**](#unified-image-and-video-generative-architecture)
    - [**MLLM Text Encoder**](#mllm-text-encoder)
    - [**3D VAE**](#3d-vae)
    - [**Prompt Rewrite**](#prompt-rewrite)
  - [📈 Comparisons](#-comparisons)
  - [📜 Requirements](#-requirements)
  - [🛠️ Dependencies and Installation](#️-dependencies-and-installation)
    - [Installation Guide for Linux](#installation-guide-for-linux)
  - [🧱 Download Pretrained Models](#-download-pretrained-models)
  - [🔑 Single-gpu Inference](#-single-gpu-inference)
    - [Using Command Line](#using-command-line)
    - [Run a Gradio Server](#run-a-gradio-server)
    - [More Configurations](#more-configurations)
  - [🚀 Parallel Inference on Multiple GPUs by xDiT](#-parallel-inference-on-multiple-gpus-by-xdit)
    - [Using Command Line](#using-command-line-1)
  - [🚀  FP8 Inference](#--fp8-inference)
    - [Using Command Line](#using-command-line-2)
  - [🔗 BibTeX](#-bibtex)
  - [Acknowledgements](#acknowledgements)
  - [Star History](#star-history)
---

## **Abstract**
We present HunyuanVideo, a novel open-source video foundation model that exhibits performance in video generation that is comparable to, if not superior to, leading closed-source models. In order to train HunyuanVideo model, we adopt several key technologies for model learning, including data curation, image-video joint model training, and an efficient infrastructure designed to facilitate large-scale model training and inference. Additionally, through an effective strategy for scaling model architecture and dataset, we successfully trained a video generative model with over 13 billion parameters, making it the largest among all open-source models. 

We conducted extensive experiments and implemented a series of targeted designs to ensure high visual quality, motion diversity, text-video alignment, and generation stability. According to professional human evaluation results, HunyuanVideo outperforms previous state-of-the-art models, including Runway Gen-3, Luma 1.6, and 3 top-performing Chinese video generative models. By releasing the code and weights of the foundation model and its applications, we aim to bridge the gap between closed-source and open-source video foundation models. This initiative will empower everyone in the community to experiment with their ideas, fostering a more dynamic and vibrant video generation ecosystem. 



## **HunyuanVideo Overall Architecture**

HunyuanVideo is trained on a spatial-temporally
compressed latent space, which is compressed through a Causal 3D VAE. Text prompts are encoded
using a large language model, and used as the conditions. Taking Gaussian noise and the conditions as
input, our generative model produces an output latent, which is then decoded to images or videos through
the 3D VAE decoder.
<p align="center">
  <img src="https://raw.githubusercontent.com/Tencent-Hunyuan/HunyuanVideo/refs/heads/main/assets/overall.png"  height=300>
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
  <img src="https://raw.githubusercontent.com/Tencent-Hunyuan/HunyuanVideo/refs/heads/main/assets/backbone.png"  height=350>
</p>

### **MLLM Text Encoder**
Some previous text-to-video models typically use pre-trained CLIP and T5-XXL as text encoders where CLIP uses Transformer Encoder and T5 uses an Encoder-Decoder structure. In contrast, we utilize a pre-trained Multimodal Large Language Model (MLLM) with a Decoder-Only structure as our text encoder, which has the following advantages: (i) Compared with T5, MLLM after visual instruction finetuning has better image-text alignment in the feature space, which alleviates the difficulty of the instruction following in diffusion models; (ii)
Compared with CLIP, MLLM has demonstrated superior ability in image detail description
and complex reasoning; (iii) MLLM can play as a zero-shot learner by following system instructions prepended to user prompts, helping text features pay more attention to key information. In addition, MLLM is based on causal attention while T5-XXL utilizes bidirectional attention that produces better text guidance for diffusion models. Therefore, we introduce an extra bidirectional token refiner to enhance text features.
<p align="center">
  <img src="https://raw.githubusercontent.com/Tencent-Hunyuan/HunyuanVideo/refs/heads/main/assets/text_encoder.png"  height=275>
</p>

### **3D VAE**
HunyuanVideo trains a 3D VAE with CausalConv3D to compress pixel-space videos and images into a compact latent space. We set the compression ratios of video length, space, and channel to 4, 8, and 16 respectively. This can significantly reduce the number of tokens for the subsequent diffusion transformer model, allowing us to train videos at the original resolution and frame rate.
<p align="center">
  <img src="https://raw.githubusercontent.com/Tencent-Hunyuan/HunyuanVideo/refs/heads/main/assets/3dvae.png"  height=150>
</p>

### **Prompt Rewrite**
To address the variability in linguistic style and length of user-provided prompts, we fine-tune the [Hunyuan-Large model](https://github.com/Tencent/Tencent-Hunyuan-Large) as our prompt rewrite model to adapt the original user prompt to model-preferred prompt.

We provide two rewrite modes: Normal mode and Master mode, which can be called using different prompts. The prompts are shown [here](hyvideo/prompt_rewrite.py). The Normal mode is designed to enhance the video generation model's comprehension of user intent, facilitating a more accurate interpretation of the instructions provided. The Master mode enhances the description of aspects such as composition, lighting, and camera movement, which leans towards generating videos with a higher visual quality. However, this emphasis may occasionally result in the loss of some semantic details. 

The Prompt Rewrite Model can be directly deployed and inferred using the [Hunyuan-Large original code](https://github.com/Tencent/Tencent-Hunyuan-Large). We release the weights of the Prompt Rewrite Model [here](https://huggingface.co/Tencent/HunyuanVideo-PromptRewrite).



## 📈 Comparisons

To evaluate the performance of HunyuanVideo, we selected five strong baselines from closed-source video generation models. In total, we utilized 1,533 text prompts, generating an equal number of video samples with HunyuanVideo in a single run. For a fair comparison, we conducted inference only once, avoiding any cherry-picking of results. When comparing with the baseline methods, we maintained the default settings for all selected models, ensuring consistent video resolution. Videos were assessed based on three criteria: Text Alignment, Motion Quality, and Visual Quality. More than 60 professional evaluators performed the evaluation. Notably, HunyuanVideo demonstrated the best overall performance, particularly excelling in motion quality. Please note that the evaluation is based on Hunyuan Video's high-quality version. This is different from the currently released fast version.

<p align="center">
<table> 
<thead> 
<tr> 
    <th rowspan="2">Model</th> <th rowspan="2">Open Source</th> <th>Duration</th> <th>Text Alignment</th> <th>Motion Quality</th> <th rowspan="2">Visual Quality</th> <th rowspan="2">Overall</th>  <th rowspan="2">Ranking</th>
</tr> 
</thead> 
<tbody> 
<tr> 
    <td>HunyuanVideo (Ours)</td> <td> ✔ </td> <td>5s</td> <td>61.8%</td> <td>66.5%</td> <td>95.7%</td> <td>41.3%</td> <td>1</td>
</tr> 
<tr> 
    <td>CNTopA (API)</td> <td> &#10008 </td> <td>5s</td> <td>62.6%</td> <td>61.7%</td> <td>95.6%</td> <td>37.7%</td> <td>2</td>
</tr> 
<tr> 
    <td>CNTopB (Web)</td> <td> &#10008</td> <td>5s</td> <td>60.1%</td> <td>62.9%</td> <td>97.7%</td> <td>37.5%</td> <td>3</td>
</tr> 
<tr> 
    <td>GEN-3 alpha (Web)</td> <td>&#10008</td> <td>6s</td> <td>47.7%</td> <td>54.7%</td> <td>97.5%</td> <td>27.4%</td> <td>4</td> 
</tr> 
<tr> 
    <td>Luma1.6 (API)</td><td>&#10008</td> <td>5s</td> <td>57.6%</td> <td>44.2%</td> <td>94.1%</td> <td>24.8%</td> <td>5</td>
</tr>
<tr> 
    <td>CNTopC (Web)</td> <td>&#10008</td> <td>5s</td> <td>48.4%</td> <td>47.2%</td> <td>96.3%</td> <td>24.6%</td> <td>6</td>
</tr> 
</tbody>
</table>
</p>


## 📜 Requirements

The following table shows the requirements for running HunyuanVideo model (batch size = 1) to generate videos:

|     Model    |  Setting<br/>(height/width/frame) | GPU Peak Memory  |
|:------------:|:--------------------------------:|:----------------:|
| HunyuanVideo   |        720px1280px129f          |       60GB        |
| HunyuanVideo   |        544px960px129f           |       45GB        |

* An NVIDIA GPU with CUDA support is required. 
  * The model is tested on a single 80G GPU.
  * **Minimum**: The minimum GPU memory required is 60GB for 720px1280px129f and 45G for 544px960px129f.
  * **Recommended**: We recommend using a GPU with 80GB of memory for better generation quality.
* Tested operating system: Linux



## 🛠️ Dependencies and Installation

Begin by cloning the repository:
```shell
git clone https://github.com/Tencent-Hunyuan/HunyuanVideo
cd HunyuanVideo
```

### Installation Guide for Linux

We recommend CUDA versions 12.4 or 11.8 for the manual installation.

Conda's installation instructions are available [here](https://docs.anaconda.com/free/miniconda/index.html).

```shell
# 1. Create conda environment
conda create -n HunyuanVideo python==3.10.9

# 2. Activate the environment
conda activate HunyuanVideo

# 3. Install PyTorch and other dependencies using conda
# For CUDA 11.8
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=11.8 -c pytorch -c nvidia
# For CUDA 12.4
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia

# 4. Install pip dependencies
python -m pip install -r requirements.txt

# 5. Install flash attention v2 for acceleration (requires CUDA 11.8 or above)
python -m pip install ninja
python -m pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.6.3

# 6. Install xDiT for parallel inference (It is recommended to use torch 2.4.0 and flash-attn 2.6.3)
python -m pip install xfuser==0.4.0
```

In case of running into float point exception(core dump) on the specific GPU type, you may try the following solutions:

```shell
# Option 1: Making sure you have installed CUDA 12.4, CUBLAS>=12.4.5.8, and CUDNN>=9.00 (or simply using our CUDA 12 docker image).
pip install nvidia-cublas-cu12==12.4.5.8
export LD_LIBRARY_PATH=/opt/conda/lib/python3.8/site-packages/nvidia/cublas/lib/

# Option 2: Forcing to explictly use the CUDA 11.8 compiled version of Pytorch and all the other packages
pip uninstall -r requirements.txt  # uninstall all packages
pip uninstall -y xfuser
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install ninja
pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.6.3
pip install xfuser==0.4.0
```

Additionally, HunyuanVideo also provides a pre-built Docker image. Use the following command to pull and run the docker image.

```shell
# For CUDA 12.4 (updated to avoid float point exception)
docker pull hunyuanvideo/hunyuanvideo:cuda_12
docker run -itd --gpus all --init --net=host --uts=host --ipc=host --name hunyuanvideo --security-opt=seccomp=unconfined --ulimit=stack=67108864 --ulimit=memlock=-1 --privileged hunyuanvideo/hunyuanvideo:cuda_12

# For CUDA 11.8
docker pull hunyuanvideo/hunyuanvideo:cuda_11
docker run -itd --gpus all --init --net=host --uts=host --ipc=host --name hunyuanvideo --security-opt=seccomp=unconfined --ulimit=stack=67108864 --ulimit=memlock=-1 --privileged hunyuanvideo/hunyuanvideo:cuda_11
```


## 🧱 Download Pretrained Models

The details of download pretrained models are shown [here](ckpts/README.md).



## 🔑 Single-gpu Inference

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
    --use-cpu-offload \
    --save-path ./results
```

### Run a Gradio Server
```bash
python3 gradio_server.py --flow-reverse

# set SERVER_NAME and SERVER_PORT manually
# SERVER_NAME=0.0.0.0 SERVER_PORT=8081 python3 gradio_server.py --flow-reverse
```

### More Configurations

We list some more useful configurations for easy usage:

|        Argument        |  Default  |                Description                |
|:----------------------:|:---------:|:-----------------------------------------:|
|       `--prompt`       |   None    |   The text prompt for video generation    |
|     `--video-size`     | 720 1280  |      The size of the generated video      |
|    `--video-length`    |    129    |     The length of the generated video     |
|    `--infer-steps`     |    50     |     The number of steps for sampling      |
| `--embedded-cfg-scale` |    6.0    |    Embedded  Classifier free guidance scale       |
|     `--flow-shift`     |    7.0    | Shift factor for flow matching schedulers |
|     `--flow-reverse`   |    False  | If reverse, learning/sampling from t=1 -> t=0 |
|        `--seed`        |     None  |   The random seed for generating video, if None, we init a random seed    |
|  `--use-cpu-offload`   |   False   |    Use CPU offload for the model load to save more memory, necessary for high-res video generation    |
|     `--save-path`      | ./results |     Path to save the generated video      |



## 🚀 Parallel Inference on Multiple GPUs by xDiT

[xDiT](https://github.com/xdit-project/xDiT) is a Scalable Inference Engine for Diffusion Transformers (DiTs) on multi-GPU Clusters.
It has successfully provided low-latency parallel inference solutions for a variety of DiTs models, including mochi-1, CogVideoX, Flux.1, SD3, etc. This repo adopted the [Unified Sequence Parallelism (USP)](https://arxiv.org/abs/2405.07719) APIs for parallel inference of the HunyuanVideo model.

### Using Command Line

For example, to generate a video with 8 GPUs, you can use the following command:

```bash
cd HunyuanVideo

torchrun --nproc_per_node=8 sample_video.py \
    --video-size 1280 720 \
    --video-length 129 \
    --infer-steps 50 \
    --prompt "A cat walks on the grass, realistic style." \
    --flow-reverse \
    --seed 42 \
    --ulysses-degree 8 \
    --ring-degree 1 \
    --save-path ./results
```

You can change the `--ulysses-degree` and `--ring-degree` to control the parallel configurations for the best performance. The valid parallel configurations are shown in the following table.

<details>
<summary>Supported Parallel Configurations (Click to expand)</summary>

|     --video-size     | --video-length | --ulysses-degree x --ring-degree | --nproc_per_node |
|----------------------|----------------|----------------------------------|------------------|
| 1280 720 or 720 1280 | 129            | 8x1,4x2,2x4,1x8                  | 8                |
| 1280 720 or 720 1280 | 129            | 1x5                              | 5                |
| 1280 720 or 720 1280 | 129            | 4x1,2x2,1x4                      | 4                |
| 1280 720 or 720 1280 | 129            | 3x1,1x3                          | 3                |
| 1280 720 or 720 1280 | 129            | 2x1,1x2                          | 2                |
| 1104 832 or 832 1104 | 129            | 4x1,2x2,1x4                      | 4                |
| 1104 832 or 832 1104 | 129            | 3x1,1x3                          | 3                |
| 1104 832 or 832 1104 | 129            | 2x1,1x2                          | 2                |
| 960 960              | 129            | 6x1,3x2,2x3,1x6                  | 6                |
| 960 960              | 129            | 4x1,2x2,1x4                      | 4                |
| 960 960              | 129            | 3x1,1x3                          | 3                |
| 960 960              | 129            | 1x2,2x1                          | 2                |
| 960 544 or 544 960   | 129            | 6x1,3x2,2x3,1x6                  | 6                |
| 960 544 or 544 960   | 129            | 4x1,2x2,1x4                      | 4                |
| 960 544 or 544 960   | 129            | 3x1,1x3                          | 3                |
| 960 544 or 544 960   | 129            | 1x2,2x1                          | 2                |
| 832 624 or 624 832   | 129            | 4x1,2x2,1x4                      | 4                |
| 624 832 or 624 832   | 129            | 3x1,1x3                          | 3                |
| 832 624 or 624 832   | 129            | 2x1,1x2                          | 2                |
| 720 720              | 129            | 1x5                              | 5                |
| 720 720              | 129            | 3x1,1x3                          | 3                |

</details>


<p align="center">
<table align="center">
<thead>
<tr>
    <th colspan="4">Latency (Sec) for 1280x720 (129 frames 50 steps) on 8xGPU</th>
</tr>
<tr>
    <th>1</th>
    <th>2</th>
    <th>4</th>
    <th>8</th>
</tr>
</thead>
<tbody>
<tr>
    <th>1904.08</th>
    <th>934.09 (2.04x)</th>
    <th>514.08 (3.70x)</th>
    <th>337.58 (5.64x)</th>
</tr>

</tbody>
</table>
</p>



## 🚀  FP8 Inference

Using HunyuanVideo with FP8 quantized weights, which saves about 10GB of GPU memory. You can download the [weights](https://huggingface.co/tencent/HunyuanVideo/blob/main/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8.pt) and [weight scales](https://huggingface.co/tencent/HunyuanVideo/blob/main/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8_map.pt) from Huggingface.

### Using Command Line

Here, you must explicitly specify the FP8 weight path. For example, to generate a video with fp8 weights, you can use the following command:

```bash
cd HunyuanVideo

DIT_CKPT_PATH={PATH_TO_FP8_WEIGHTS}/{WEIGHT_NAME}_fp8.pt

python3 sample_video.py \
    --dit-weight ${DIT_CKPT_PATH} \
    --video-size 1280 720 \
    --video-length 129 \
    --infer-steps 50 \
    --prompt "A cat walks on the grass, realistic style." \
    --seed 42 \
    --embedded-cfg-scale 6.0 \
    --flow-shift 7.0 \
    --flow-reverse \
    --use-cpu-offload \
    --use-fp8 \
    --save-path ./results
```



## 🔗 BibTeX

If you find [HunyuanVideo](https://arxiv.org/abs/2412.03603) useful for your research and applications, please cite using this BibTeX:

```BibTeX
@article{kong2024hunyuanvideo,
  title={Hunyuanvideo: A systematic framework for large video generative models},
  author={Kong, Weijie and Tian, Qi and Zhang, Zijian and Min, Rox and Dai, Zuozhuo and Zhou, Jin and Xiong, Jiangfeng and Li, Xin and Wu, Bo and Zhang, Jianwei and others},
  journal={arXiv preprint arXiv:2412.03603},
  year={2024}
}
```



## Acknowledgements

We would like to thank the contributors to the [SD3](https://huggingface.co/stabilityai/stable-diffusion-3-medium), [FLUX](https://github.com/black-forest-labs/flux), [Llama](https://github.com/meta-llama/llama), [LLaVA](https://github.com/haotian-liu/LLaVA), [Xtuner](https://github.com/InternLM/xtuner), [diffusers](https://github.com/huggingface/diffusers) and [HuggingFace](https://huggingface.co) repositories, for their open research and exploration.
Additionally, we also thank the Tencent Hunyuan Multimodal team for their help with the text encoder. 


## Github Star History
<a href="https://star-history.com/#Tencent-Hunyuan/HunyuanVideo&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=Tencent-Hunyuan/HunyuanVideo&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=Tencent-Hunyuan/HunyuanVideo&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=Tencent-Hunyuan/HunyuanVideo&type=Date" />
 </picture>
</a>
