<!-- ## **HunyuanVideo** -->

[English](./README.md) | [中文](./README_zh.md)

<p align="center">
  <img src="https://raw.githubusercontent.com/Tencent/HunyuanVideo/refs/heads/main/assets/logo.png"  height=100>
</p>

# HunyuanVideo: 大規模ビデオ生成モデルトレーニングのための体系的フレームワーク

<div align="center">
  <a href="https://github.com/Tencent/HunyuanVideo"><img src="https://img.shields.io/static/v1?label=HunyuanVideo Code&message=Github&color=blue&logo=github-pages"></a> &ensp;
  <a href="https://aivideo.hunyuan.tencent.com"><img src="https://img.shields.io/static/v1?label=Project%20Page&message=Web&color=green&logo=github-pages"></a> &ensp;
  <a href="https://video.hunyuan.tencent.com"><img src="https://img.shields.io/static/v1?label=Playground&message=Web&color=green&logo=github-pages"></a> &ensp;
  <a href="https://github.com/Tencent/HunyuanVideo/blob/main/assets/hunyuanvideo.pdf"><img src="https://img.shields.io/static/v1?label=Tech Report&message=Arxiv:HunyuanVideo&color=red&logo=arxiv"></a> &ensp;
  <a href="https://huggingface.co/tencent/HunyuanVideo"><img src="https://img.shields.io/static/v1?label=HunyuanVideo&message=HuggingFace&color=yellow"></a> &ensp; &ensp;
  <a href="https://huggingface.co/tencent/HunyuanVideo-PromptRewrite"><img src="https://img.shields.io/static/v1?label=HunyuanVideo-PromptRewrite&message=HuggingFace&color=yellow"></a> &ensp; &ensp;
</div>

-----

このリポジトリには、HunyuanVideoのPyTorchモデル定義、事前トレーニング済みの重み、および推論/サンプリングコードが含まれています。詳細は[プロジェクトページ](https://aivideo.hunyuan.tencent.com)をご覧ください。

> [**HunyuanVideo: 大規模ビデオ生成モデルトレーニングのための体系的フレームワーク**](https://github.com/Tencent/HunyuanVideo/blob/main/assets/hunyuanvideo.pdf) <br>

## 🎥 デモ
<div align="center">
  <video src="https://github.com/user-attachments/assets/f37925a3-7d42-40c9-8a9b-5a010c7198e2" width="50%">
</div>

ビデオはGitHubのポリシーに準拠するために大幅に圧縮されています。高品質バージョンは[こちら](https://aivideo.hunyuan.tencent.com/download/HunyuanVideo/material/demo.mov)からダウンロードできます。

## 🔥🔥🔥 ニュース!!
* 2024年12月3日: 🤗 HunyuanVideoの推論コードとモデルの重みをリリースしました。

## 📑 オープンソース計画

- HunyuanVideo (テキストからビデオ生成モデル)
  - [x] 推論 
  - [x] チェックポイント 
  - [ ] Penguin Video Benchmark
  - [ ] Webデモ (Gradio) 
  - [ ] ComfyUI
  - [ ] Diffusers 
- HunyuanVideo (画像からビデオ生成モデル)
  - [ ] 推論 
  - [ ] チェックポイント 

## 目次
- [HunyuanVideo: 大規模ビデオ生成モデルトレーニングのための体系的フレームワーク](#hunyuanvideo--大規模ビデオ生成モデルトレーニングのための体系的フレームワーク)
  - [🎥 デモ](#-デモ)
  - [🔥🔥🔥 ニュース!!](#-ニュース)
  - [📑 オープンソース計画](#-オープンソース計画)
  - [目次](#目次)
  - [**概要**](#概要)
  - [**HunyuanVideoの全体アーキテクチャ**](#-hunyuanvideoの全体アーキテクチャ)
  - [🎉 **HunyuanVideoの主な特徴**](#-hunyuanvideoの主な特徴)
    - [**統一された画像とビデオ生成アーキテクチャ**](#統一された画像とビデオ生成アーキテクチャ)
    - [**MLLMテキストエンコーダ**](#mllmテキストエンコーダ)
    - [**3D VAE**](#3d-vae)
    - [**プロンプトリライト**](#プロンプトリライト)
  - [📈 比較](#-比較)
  - [📜 要件](#-要件)
  - [🛠️ 依存関係とインストール](#-依存関係とインストール)
    - [Linuxのインストールガイド](#linuxのインストールガイド)
  - [🧱 事前トレーニング済みモデルのダウンロード](#-事前トレーニング済みモデルのダウンロード)
  - [🔑 推論](#-推論)
    - [コマンドラインの使用](#コマンドラインの使用)
    - [その他の設定](#その他の設定)
  - [🔗 BibTeX](#-bibtex)
  - [謝辞](#謝辞)
---

## **概要**
HunyuanVideoは、ビデオ生成において、先進的なクローズドソースモデルに匹敵する、あるいはそれを上回る性能を示す新しいオープンソースのビデオ基盤モデルです。HunyuanVideoモデルをトレーニングするために、データキュレーション、画像-ビデオの共同モデルトレーニング、および大規模モデルのトレーニングと推論を容易にする効率的なインフラストラクチャを採用しています。さらに、モデルアーキテクチャとデータセットのスケーリング戦略を通じて、13億以上のパラメータを持つビデオ生成モデルを成功裏にトレーニングし、すべてのオープンソースモデルの中で最大のものとなりました。

高い視覚品質、動きの多様性、テキスト-ビデオの整合性、および生成の安定性を確保するために、広範な実験と一連のターゲットデザインを実施しました。専門家による評価結果によると、HunyuanVideoは、Runway Gen-3、Luma 1.6、および3つのトップパフォーマンスの中国のビデオ生成モデルを含む、以前の最先端モデルを上回っています。基盤モデルとそのアプリケーションのコードと重みを公開することで、クローズドソースとオープンソースのビデオ基盤モデルのギャップを埋めることを目指しています。このイニシアチブは、コミュニティのすべての人々が自分のアイデアを試すことを可能にし、よりダイナミックで活気のあるビデオ生成エコシステムを促進します。

## **HunyuanVideoの全体アーキテクチャ**

HunyuanVideoは、Causal 3D VAEを通じて空間的および時間的に圧縮された潜在空間でトレーニングされます。テキストプロンプトは大規模言語モデルを使用してエンコードされ、条件として使用されます。ガウスノイズと条件が入力として取り込まれ、生成モデルは出力潜在を生成し、3D VAEデコーダを通じて画像またはビデオにデコードされます。
<p align="center">
  <img src="https://raw.githubusercontent.com/Tencent/HunyuanVideo/refs/heads/main/assets/overall.png"  height=300>
</p>

## 🎉 **HunyuanVideoの主な特徴**
### **統一された画像とビデオ生成アーキテクチャ**
HunyuanVideoは、Transformer設計を導入し、統一された画像とビデオ生成のためにFull Attentionメカニズムを採用しています。
具体的には、ビデオ生成のために「デュアルストリームからシングルストリーム」ハイブリッドモデル設計を使用しています。デュアルストリームフェーズでは、ビデオとテキストのトークンが複数のTransformerブロックを通じて独立して処理され、各モダリティが干渉なしに適切な変調メカニズムを学習できるようにします。シングルストリームフェーズでは、ビデオとテキストのトークンを連結し、効果的なマルチモーダル情報融合のために後続のTransformerブロックに入力します。
この設計は、視覚情報と意味情報の複雑な相互作用をキャプチャし、全体的なモデル性能を向上させます。
<p align="center">
  <img src="https://raw.githubusercontent.com/Tencent/HunyuanVideo/refs/heads/main/assets/backbone.png"  height=350>
</p>

### **MLLMテキストエンコーダ**
以前のテキストからビデオ生成モデルは、通常、事前トレーニングされたCLIPとT5-XXLをテキストエンコーダとして使用していましたが、CLIPはTransformer Encoderを使用し、T5はEncoder-Decoder構造を使用しています。対照的に、我々はDecoder-Only構造を持つ事前トレーニングされたMultimodal Large Language Model (MLLM)をテキストエンコーダとして利用しています。これには以下の利点があります：(i) T5と比較して、視覚指示による微調整後のMLLMは、特徴空間での画像-テキストの整合性が向上し、拡散モデルでの指示追従の難易度を軽減します。(ii) CLIPと比較して、MLLMは画像の詳細な説明と複雑な推論において優れた能力を示しています。(iii) MLLMは、システム指示に従うことでゼロショット学習者として機能し、テキスト特徴が重要な情報により注意を払うのを助けます。さらに、MLLMは因果関係のある注意に基づいており、T5-XXLは双方向の注意を利用して拡散モデルに対してより良いテキストガイダンスを提供します。したがって、テキスト特徴を強化するために追加の双方向トークンリファイナーを導入しています。
<p align="center">
  <img src="https://raw.githubusercontent.com/Tencent/HunyuanVideo/refs/heads/main/assets/text_encoder.png"  height=275>
</p>

### **3D VAE**
HunyuanVideoは、CausalConv3Dを使用してピクセル空間のビデオと画像をコンパクトな潜在空間に圧縮する3D VAEをトレーニングします。ビデオの長さ、空間、およびチャネルの圧縮比をそれぞれ4、8、および16に設定します。これにより、後続の拡散トランスフォーマーモデルのトークン数を大幅に削減でき、元の解像度とフレームレートでビデオをトレーニングできます。
<p align="center">
  <img src="https://raw.githubusercontent.com/Tencent/HunyuanVideo/refs/heads/main/assets/3dvae.png"  height=150>
</p>

### **プロンプトリライト**
ユーザーが提供するプロンプトの言語スタイルと長さの変動に対処するために、[Hunyuan-Large model](https://github.com/Tencent/Tencent-Hunyuan-Large)を微調整してプロンプトリライトモデルとして使用し、元のユーザープロンプトをモデルに適したプロンプトに適応させます。

2つのリライトモードを提供しています：通常モードとマスターモードで、異なるプロンプトを使用して呼び出すことができます。プロンプトは[こちら](hyvideo/prompt_rewrite.py)に示されています。通常モードは、ビデオ生成モデルのユーザー意図の理解を強化し、提供された指示のより正確な解釈を促進するように設計されています。マスターモードは、構図、照明、カメラの動きなどの側面の説明を強化し、より高い視覚品質のビデオを生成する傾向があります。ただし、この強調は一部の意味的な詳細の喪失を引き起こすことがあります。

プロンプトリライトモデルは、[Hunyuan-Large original code](https://github.com/Tencent/Tencent-Hunyuan-Large)を使用して直接デプロイおよび推論できます。プロンプトリライトモデルの重みを[こちら](https://huggingface.co/Tencent/HunyuanVideo-PromptRewrite)で公開しています。

## 📈 比較

HunyuanVideoの性能を評価するために、クローズドソースのビデオ生成モデルから5つの強力なベースラインを選択しました。合計で1,533のテキストプロンプトを使用し、HunyuanVideoで同数のビデオサンプルを1回の実行で生成しました。公平な比較のために、推論は1回のみ実施し、結果の選別は行いませんでした。ベースラインメソッドと比較する際には、��べての選択されたモデルのデフォルト設定を維持し、一貫したビデオ解像度を確保しました。ビデオは、テキストの整合性、動きの品質、および視覚品質の3つの基準に基づいて評価されました。60人以上の専門家が評価を行いました。特に、HunyuanVideoは動きの品質で優れた性能を示し、全体的なパフォーマンスで最も優れていました。評価はHunyuan Videoの高品質バージョンに基づいています。これは現在リリースされている高速バージョンとは異なります。

<p align="center">
<table> 
<thead> 
<tr> 
    <th rowspan="2">モデル</th> <th rowspan="2">オープンソース</th> <th>期間</th> <th>テキストの整合性</th> <th>動きの品質</th> <th rowspan="2">視覚品質</th> <th rowspan="2">全体</th>  <th rowspan="2">ランキング</th>
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
    <td>Luma1.6 (API)</td><td>&#10008</td> <td>5s</td> <td>57.6%</td> <td>44.2%</td> <td>94.1%</td> <td>24.8%</td> <td>6</td>
</tr>
<tr> 
    <td>CNTopC (Web)</td> <td>&#10008</td> <td>5s</td> <td>48.4%</td> <td>47.2%</td> <td>96.3%</td> <td>24.6%</td> <td>5</td>
</tr> 
</tbody>
</table>
</p>

## 📜 要件

次の表は、HunyuanVideoモデルを実行してビデオを生成するための要件を示しています（バッチサイズ = 1）。

|     モデル    |  設定<br/>(高さ/幅/フレーム) | ノイズ除去ステップ | GPUピークメモリ  |
|:------------:|:--------------------------------:|:--------------:|:----------------:|
| HunyuanVideo   |        720px1280px129f          |       30       |       60GB        |
| HunyuanVideo   |        544px960px129f           |       30       |       45GB        |

* CUDAサポートを備えたNVIDIA GPUが必要です。
  * 単一のH800/H20 GPUでテストしました。
  * **最小**: 720px1280px129fの場合、必要な最小GPUメモリは60GB、544px960px129fの場合は45GBです。
  * **推奨**: より良い生成品質のために、80GBのメモリを持つGPUを使用することをお勧めします。
* テスト済みのオペレーティングシステム：Linux

## 🛠️ 依存関係とインストール

まず、リポジトリをクローンします：
```shell
git clone https://github.com/tencent/HunyuanVideo
cd HunyuanVideo
```

### Linuxのインストールガイド

Conda環境を設定するための`environment.yml`ファイルを提供しています。
Condaのインストール手順は[こちら](https://docs.anaconda.com/free/miniconda/index.html)で確認できます。

CUDAバージョン11.8および12.0+を推奨します。

```shell
# 1. Conda環境を準備する
conda env create -f environment.yml

# 2. 環境をアクティブにする
conda activate HunyuanVideo

# 3. pip依存関係をインストールする
python -m pip install -r requirements.txt

# 4. 加速のためにflash attention v2をインストールする（CUDA 11.8以上が必要）
python -m pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.5.9.post1
```

さらに、HunyuanVideoは事前構築されたDockerイメージも提供しています：
[docker_hunyuanvideo](https://aivideo.hunyuan.tencent.com/download/HunyuanVideo/hunyuan_video_cu12.tar)。

```shell
# 1. 次のリンクを使用してdockerイメージのtarファイルをダウンロードします（CUDA 12用）。
wget https://aivideo.hunyuan.tencent.com/download/HunyuanVideo/hunyuan_video_cu12.tar

# 2. docker tarファイルをインポートし、イメージのメタ情報を表示します（CUDA 12用）。
docker load -i hunyuan_video.tar

docker image ls

# 3. イメージに基づいてコンテナを実行します
docker run -itd --gpus all --init --net=host --uts=host --ipc=host --name hunyuanvideo --security-opt=seccomp=unconfined --ulimit=stack=67108864 --ulimit=memlock=-1 --privileged  docker_image_tag
```


## 🧱 事前トレーニング済みモデルのダウンロード

事前トレーニング済みモデルのダウンロードの詳細は[こちら](ckpts/README.md)に示されています。

## 🔑 推論
次の表に、サポートされている高さ/幅/フレームの設定を示します。

|      解像度       |           h/w=9:16           |    h/w=16:9     |     h/w=4:3     |     h/w=3:4     |     h/w=1:1     |
|:---------------------:|:----------------------------:|:---------------:|:---------------:|:---------------:|:---------------:|
|         540p          |        544px960px129f        |  960px544px129f | 624px832px129f  |  832px624px129f |  720px720px129f |
| 720p (推奨)    |       720px1280px129f        | 1280px720px129f | 1104px832px129f | 832px1104px129f | 960px960px129f  |

### コマンドラインの使用

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

### その他の設定

以下に、簡単に使用できるいくつかの設定を示します：

|        引数        |  デフォルト  |                説明                |
|:----------------------:|:---------:|:-----------------------------------------:|
|       `--prompt`       |   None    |   ビデオ生成のためのテキストプロンプト    |
|     `--video-size`     | 720 1280  |      生成されるビデオのサイズ      |
|    `--video-length`    |    129    |     生成されるビデオの長さ     |
|    `--infer-steps`     |    50     |     サンプリングのステップ数      |
| `--embedded-cfg-scale` |    6.0    |    埋め込まれたクラスフリーガイダンススケール       |
|     `--flow-shift`     |    7.0    | フロー一致スケジューラのシフト係数 |
|     `--flow-reverse`   |    False  | 逆の場合、t=1からt=0への学習/サンプリング |
|        `--seed`        |     None  |   ビデオ生成のためのランダムシード、Noneの場合、ランダムシードを初期化します    |
|  `--use-cpu-offload`   |   False   |    モデルロードのためにCPUオフロードを使用して、より多くのメモリを節約します。高解像度ビデオ生成に必要です。    |
|     `--save-path`      | ./results |     生成されたビデオを保存するパス      |


## 🔗 BibTeX
[HunyuanVideo](https://github.com/Tencent/HunyuanVideo/blob/main/assets/hunyuanvideo.pdf)が研究やアプリケーションに役立つ場合は、次のBibTeXを使用して引用してください：

```BibTeX
@misc{kong2024hunyuanvideo,
      title={HunyuanVideo: A Systematic Framework For Large Video Generative Models}, 
      author={Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, Kathrina Wu, Qin Lin, Aladdin Wang, Andong Wang, Changlin Li, Duojun Huang, Fang Yang, Hao Tan, Hongmei Wang, Jacob Song, Jiawang Bai, Jianbing Wu, Jinbao Xue, Joey Wang, Junkun Yuan, Kai Wang, Mengyang Liu, Pengyu Li, Shuai Li, Weiyan Wang, Wenqing Yu, Xinchi Deng, Yang Li, Yanxin Long, Yi Chen, Yutao Cui, Yuanbo Peng, Zhentao Yu, Zhiyu He, Zhiyong Xu, Zixiang Zhou, Zunnan Xu, Yangyu Tao, Qinglin Lu, Songtao Liu, Dax Zhou, Hongfa Wang, Yong Yang, Di Wang, Yuhong Liu, and Jie Jiang, along with Caesar Zhong},
      year={2024},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## 謝辞
[SD3](https://huggingface.co/stabilityai/stable-diffusion-3-medium)、[FLUX](https://github.com/black-forest-labs/flux)、[Llama](https://github.com/meta-llama/llama)、[LLaVA](https://github.com/haotian-liu/LLaVA)、[Xtuner](https://github.com/InternLM/xtuner)、[diffusers](https://github.com/huggingface/diffusers)、および[huggingface](https://huggingface.co)のリポジトリの貢献者に感謝します。
また、テキストエンコーダのサポートに対してTencent Hunyuan Multimodalチームにも感謝します。

## スター履歴
<a href="https://star-history.com/#Tencent/HunyuanVideo&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=Tencent/HunyuanVideo&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=Tencent/HunyuanVideo&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=Tencent/HunyuanVideo&type=Date" />
 </picture>
</a>
