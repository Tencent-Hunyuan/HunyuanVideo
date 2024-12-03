<!-- ## **HunyuanVideo** -->

[English](./README.md) | [日本語](./README_ja.md)

<p align="center">
  <img src="https://raw.githubusercontent.com/Tencent/HunyuanVideo/refs/heads/main/assets/logo.png"  height=100>
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

本仓库包含了 HunyuanVideo 项目的 PyTorch 模型定义、预训练权重和推理/采样代码。参考我们的项目页面 [project page](https://aivideo.hunyuan.tencent.com) 查看更多内容。

> [**HunyuanVideo: A Systematic Framework For Large Video Generation Model Training**](https://github.com/Tencent/HunyuanVideo/blob/main/assets/hunyuanvideo.pdf) <br>

## 🎥 作品展示
<div align="center">
  <video src="https://github.com/user-attachments/assets/f37925a3-7d42-40c9-8a9b-5a010c7198e2" width="50%">
</div>

注：由于 GitHub 的政策限制，上面的视频质量被大幅压缩。你可以从 [这里](https://aivideo.hunyuan.tencent.com/download/HunyuanVideo/material) 下载高质量版本。

## 🔥🔥🔥 更新!!
* 2024年12月03日: 🤗 开源 HunyuanVideo 文生视频的推理代码和模型权重。

## 📑 开源计划

- HunyuanVideo (文生视频模型)
  - [x] 推理代码
  - [x] 模型权重 
  - [ ] Penguin Video 基准测试集
  - [ ] Gradio
  - [ ] ComfyUI
  - [ ] Diffusers 
- HunyuanVideo (图生视频模型)
  - [ ] 推理代码 
  - [ ] 模型权重 

## 目录
- [HunyuanVideo: A Systematic Framework For Large Video Generation Model Training](#hunyuanvideo--a-systematic-framework-for-large-video-generation-model-training)
  - [🎥 作品展示](#-作品展示)
  - [🔥🔥🔥 更新!!](#-更新)
  - [📑 开源计划](#-开源计划)
  - [目录](#目录)
  - [**摘要**](#摘要)
  - [**HunyuanVideo 的架构**](#hunyuanvideo-的架构)
  - [🎉 **亮点**](#-亮点)
    - [**统一的图视频生成架构**](#统一的图视频生成架构)
    - [**MLLM 文本编码器**](#mllm-文本编码器)
    - [**3D VAE**](#3d-vae)
    - [**Prompt 改写**](#prompt-改写)
  - [📈 能力评估](#-能力评估)
  - [📜 运行配置](#-运行配置)
  - [🛠️ 安装和依赖](#-安装和依赖)
    - [Linux 安装指引](#linux-安装指引)
  - [🧱 下载预训练模型](#-下载预训练模型)
  - [🔑 推理](#-推理)
    - [使用命令行](#使用命令行)
    - [更多配置](#更多配置)
  - [🔗 BibTeX](#-bibtex)
  - [致谢](#致谢)
---

## **摘要**
HunyuanVideo 是一个全新的开源视频生成大模型，具有与领先的闭源模型相媲美甚至更优的视频生成表现。为了训练 HunyuanVideo，我们采用了一个全面的框架，集成了数据整理、图像-视频联合模型训练和高效的基础设施以支持大规模模型训练和推理。此外，通过有效的模型架构和数据集扩展策略，我们成功地训练了一个拥有超过 130 亿参数的视频生成模型，使其成为最大的开源视频生成模型之一。

我们在模型结构的设计上做了大量的实验以确保其能拥有高质量的视觉效果、多样的运动、文本-视频对齐和生成稳定性。根据专业人员的评估结果，HunyuanVideo 在综合指标上优于以往的最先进模型，包括 Runway Gen-3、Luma 1.6 和 3 个中文社区表现最好的视频生成模型。**通过开源基础模型和应用模型的代码和权重，我们旨在弥合闭源和开源视频基础模型之间的差距，帮助社区中的每个人都能够尝试自己的想法，促进更加动态和活跃的视频生成生态。**


## **HunyuanVideo 的架构**

HunyuanVideo 是一个隐空间模型，训练时它采用了 3D VAE 压缩时间维度和空间维度的特征。文本提示通过一个大语言模型编码后作为条件输入模型，引导模型通过对高斯噪声的多步去噪，输出一个视频的隐空间表示。最后，推理时通过 3D VAE 解码器将隐空间表示解码为视频。
<p align="center">
  <img src="https://raw.githubusercontent.com/Tencent/HunyuanVideo/refs/heads/main/assets/overall.png"  height=300>
</p>

## 🎉 **亮点**
### **统一的图视频生成架构**

HunyuanVideo 采用了 Transformer 和 Full Attention 的设计用于视频生成。具体来说，我们使用了一个“双流到单流”的混合模型设计用于视频生成。在双流阶段，视频和文本 token 通过并行的 Transformer Block 独立处理，使得每个模态可以学习适合自己的调制机制而不会相互干扰。在单流阶段，我们将视频和文本 token 连接起来并将它们输入到后续的 Transformer Block 中进行有效的多模态信息融合。这种设计捕捉了视觉和语义信息之间的复杂交互，增强了整体模型性能。
<p align="center">
  <img src="https://raw.githubusercontent.com/Tencent/HunyuanVideo/refs/heads/main/assets/backbone.png"  height=350>
</p>

### **MLLM 文本编码器**
过去的视频生成模型通常使用预训练的 CLIP 和 T5-XXL 作为文本编码器，其中 CLIP 使用 Transformer Encoder，T5 使用 Encoder-Decoder 结构。HunyuanVideo 使用了一个预训练的 Multimodal Large Language Model (MLLM) 作为文本编码器，它具有以下优势：
* 与 T5 相比，MLLM 基于图文数据指令微调后在特征空间中具有更好的图像-文本对齐能力，这减轻了扩散模型中的图文对齐的难度；
* 与 CLIP 相比，MLLM 在图像的细节描述和复杂推理方面表现出更强的能力；
* MLLM 可以通过遵循系统指令实现零样本生成，帮助文本特征更多地关注关键信息。

由于 MLLM 是基于 Causal Attention 的，而 T5-XXL 使用了 Bidirectional Attention 为扩散模型提供更好的文本引导。因此，我们引入了一个额外的 token 优化器来增强文本特征。
<p align="center">
  <img src="https://raw.githubusercontent.com/Tencent/HunyuanVideo/refs/heads/main/assets/text_encoder.png"  height=275>
</p>

### **3D VAE**
我们的 VAE 采用了 CausalConv3D 作为 HunyuanVideo 的编码器和解码器，用于压缩视频的时间维度和空间维度，其中时间维度压缩 4 倍，空间维度压缩 8 倍，压缩为 16 channels。这样可以显著减少后续 Transformer 模型的 token 数量，使我们能够在原始分辨率和帧率下训练视频生成模型。
<p align="center">
  <img src="https://raw.githubusercontent.com/Tencent/HunyuanVideo/refs/heads/main/assets/3dvae.png"  height=150>
</p>

### **Prompt 改写**
为了解决用户输入文本提示的多样性和不一致性的困难，我们微调了 [Hunyuan-Large model](https://github.com/Tencent/Tencent-Hunyuan-Large) 模型作为我们的 prompt 改写模型，将用户输入的提示词改写为更适合模型偏好的写法。

我们提供了两个改写模式：正常模式和大师模式。两种模式的提示词见[这里](hyvideo/prompt_rewrite.py)。正常模式旨在增强视频生成模型对用户意图的理解，从而更准确地解释提供的指令。大师模式增强了诸如构图、光照和摄像机移动等方面的描述，倾向于生成视觉质量更高的视频。注意，这种增强有时可能会导致一些语义细节的丢失。

Prompt 改写模型可以直接使用 [Hunyuan-Large](https://github.com/Tencent/Tencent-Hunyuan-Large) 部署和推理. 我们开源了 prompt 改写模型的权重，见[这里](https://huggingface.co/Tencent/HunyuanVideo-PromptRewrite).

## 📈 能力评估

为了评估 HunyuanVideo 的能力，我们选择了四个闭源视频生成模型作为对比。我们总共使用了 1,533 个 prompt，每个 prompt 通过一次推理生成了相同数量的视频样本。为了公平比较，我们只进行了一次推理以避免任何挑选。在与其他方法比较时，我们保持了所有选择模型的默认设置，并确保了视频分辨率的一致性。视频根据三个标准进行评估：文本对齐、运动质量和视觉质量。在 60 多名专业评估人员评估后，HunyuanVideo 在综合指标上表现最好，特别是在运动质量方面表现较为突出。

<p align="center">
<table> 
<thead> 
<tr> 
    <th rowspan="2">模型</th> <th rowspan="2">是否开源</th> <th>时长</th> <th>文本对齐</th> <th>运动质量</th> <th rowspan="2">视觉质量</th> <th rowspan="2">综合评价</th>  <th rowspan="2">排序</th>
</tr> 
</thead> 
<tbody> 
<tr> 
    <td>HunyuanVideo (Ours)</td> <td> ✔ </td> <td>5s</td> <td>61.8%</td> <td>66.5%</td> <td>95.7%</td> <td>41.3%</td> <td>1</td>
</tr> 
<tr> 
    <td>国内模型 A (API)</td> <td> &#10008 </td> <td>5s</td> <td>62.6%</td> <td>61.7%</td> <td>95.6%</td> <td>37.7%</td> <td>2</td>
</tr> 
<tr> 
    <td>国内模型 B (Web)</td> <td> &#10008</td> <td>5s</td> <td>60.1%</td> <td>62.9%</td> <td>97.7%</td> <td>37.5%</td> <td>3</td>
</tr> 
<tr> 
    <td>GEN-3 alpha (Web)</td> <td>&#10008</td> <td>6s</td> <td>47.7%</td> <td>54.7%</td> <td>97.5%</td> <td>27.4%</td> <td>4</td> 
</tr> 
<tr> 
    <td>Luma1.6 (API)</td><td>&#10008</td> <td>5s</td> <td>57.6%</td> <td>44.2%</td> <td>94.1%</td> <td>24.8%</td> <td>5</td>
</tr>
</tbody>
</table>
</p>

## 📜 运行配置

下表列出了运行 HunyuanVideo 模型使用文本生成视频的推荐配置（batch size = 1）：

|     模型      | 分辨率<br/>(height/width/frame) | 去噪步数 | 峰值显存  |
|:--------------:|:--------------------------------:|:--------------:|:----------------:|
| HunyuanVideo   |         720px1280px129f          |       30       |       60G        |
| HunyuanVideo   |          544px960px129f          |       30       |       45G        |

* 本项目适用于使用 NVIDIA GPU 和支持 CUDA 的设备
  * 本项目在单张 H800/H20 GPU 上进行了测试
  * 运行 720px1280px129f 的最小显存要求是 60GB，544px960px129f 的最小显存要求是 45GB。
* 测试操作系统：Linux

## 🛠️ 安装和依赖

首先克隆 git 仓库:
```shell
git clone https://github.com/tencent/HunyuanVideo
cd HunyuanVideo
```

### Linux 安装指引

我们提供了 `environment.yml` 文件来设置 Conda 环境。Conda 的安装指南可以参考[这里](https://docs.anaconda.com/free/miniconda/index.html)。

我们推理使用 CUDA 11.8 或 12.0+ 的版本。

```shell
# 1. 准备 conda 环境
conda env create -f environment.yml

# 2. 激活环境
conda activate HunyuanVideo

# 3. 安装 pip 依赖
python -m pip install -r requirements.txt

# 4. 安装 flash attention v2 用于加速 (要求 CUDA 11.8 或更高)
python -m pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.5.9.post1
```

另外，我们提供了一个预构建的 Docker 镜像 [docker_hunyuanvideo](https://aivideo.hunyuan.tencent.com/download/HunyuanVideo/hunyuan_video_cu12.tar)。

```shell
# 1.使用下面的链接下载 Docker 镜像 tar 包 (用于 CUDA 12).
wget https://aivideo.hunyuan.tencent.com/download/HunyuanVideo/hunyuan_video_cu12.tar

# 2. 导入 docker tar 包然后打印镜像的元信息 (用于 CUDA 12).
docker load -i hunyuan_video.tar

docker image ls

# 3. 启动容器
docker run -itd --gpus all --init --net=host --uts=host --ipc=host --name hunyuanvideo --security-opt=seccomp=unconfined --ulimit=stack=67108864 --ulimit=memlock=-1 --privileged  docker_image_tag
```


## 🧱 下载预训练模型

下载预训练模型参考[这里](ckpts/README.md)。

## 🔑 推理
我们在下表中列出了支持的高度/宽度/帧数设置。

|      分辨率       |           h/w=9:16           |    h/w=16:9     |     h/w=4:3     |     h/w=3:4     |     h/w=1:1     |
|:---------------------:|:----------------------------:|:---------------:|:---------------:|:---------------:|:---------------:|
|         540p          |        544px960px129f        |  960px544px129f | 624px832px129f  |  832px624px129f |  720px720px129f |
| 720p (推荐)    |       720px1280px129f        | 1280px720px129f | 1104px832px129f | 832px1104px129f | 960px960px129f  |

### 使用命令行

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

### 更多配置

下面列出了更多关键配置项：

|        参数        |  默认值  |                描述                |
|:----------------------:|:---------:|:-----------------------------------------:|
|       `--prompt`       |   None    |   用于生成视频的 prompt    |
|     `--video-size`     | 720 1280  |      生成视频的高度和宽度      |
|    `--video-length`    |    129    |     生成视频的帧数     |
|    `--infer-steps`     |    50     |     生成时采样的步数      |
| `--embedded-cfg-scale` |    6.0    |    文本的控制强度       |
|     `--flow-shift`     |    7.0    | 推理时 timestep 的 shift 系数，值越大，高噪区域采样步数越多 |
|     `--flow-reverse`   |    False  | If reverse, learning/sampling from t=1 -> t=0 |
|     `--neg-prompt`     |   None    | 负向词  |
|        `--seed`        |     0     |   随机种子    |
|  `--use-cpu-offload`   |   False   |    启用 CPU offload，可以节省显存    |
|     `--save-path`      | ./results |     保存路径      |


## 🔗 BibTeX
如果您认为 [HunyuanVideo](https://github.com/Tencent/HunyuanVideo/blob/main/assets/hunyuanvideo.pdf) 给您的研究和应用带来了一些帮助，可以通过下面的方式来引用:

```BibTeX
@misc{kong2024hunyuanvideo,
      title={HunyuanVideo: A Systematic Framework For Large Video Generative Models}, 
      author={Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, Kathrina Wu, Qin Lin, Aladdin Wang, Andong Wang, Changlin Li, Duojun Huang, Fang Yang, Hao Tan, Hongmei Wang, Jacob Song, Jiawang Bai, Jianbing Wu, Jinbao Xue, Joey Wang, Junkun Yuan, Kai Wang, Mengyang Liu, Pengyu Li, Shuai Li, Weiyan Wang, Wenqing Yu, Xinchi Deng, Yang Li, Yanxin Long, Yi Chen, Yutao Cui, Yuanbo Peng, Zhentao Yu, Zhiyu He, Zhiyong Xu, Zixiang Zhou, Zunnan Xu, Yangyu Tao, Qinglin Lu, Songtao Liu, Dax Zhou, Hongfa Wang, Yong Yang, Di Wang, Yuhong Liu, and Jie Jiang, along with Caesar Zhong},
      year={2024},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## 致谢
HunyuanVideo 的开源离不开诸多开源工作，这里我们特别感谢 [SD3](https://huggingface.co/stabilityai/stable-diffusion-3-medium), [FLUX](https://github.com/black-forest-labs/flux), [Llama](https://github.com/meta-llama/llama), [LLaVA](https://github.com/haotian-liu/LLaVA), [Xtuner](https://github.com/InternLM/xtuner), [diffusers](https://github.com/huggingface/diffusers) and [huggingface](https://huggingface.co) 的开源工作和探索。另外，我们也感谢腾讯混元多模态团队对 HunyuanVideo 适配多种文本编码器的支持。


## Star 趋势

<a href="https://star-history.com/#Tencent/HunyuanVideo&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=Tencent/HunyuanVideo&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=Tencent/HunyuanVideo&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=Tencent/HunyuanVideo&type=Date" />
 </picture>
</a>
