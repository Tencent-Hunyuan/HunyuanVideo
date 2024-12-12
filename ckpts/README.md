# Download Pretrained Models

All models are stored in `HunyuanVideo/ckpts` by default, and the file structure is as follows
```shell
HunyuanVideo
  ├──ckpts
  │  ├──README.md
  │  ├──hunyuan-video-t2v-720p
  │  │  ├──transformers
  ├  │  ├──vae
  │  ├──text_encoder
  │  ├──text_encoder_2
  ├──...
```

## Download HunyuanVideo model
To download the HunyuanVideo model, first install the huggingface-cli. (Detailed instructions are available [here](https://huggingface.co/docs/huggingface_hub/guides/cli).)

```shell
python -m pip install "huggingface_hub[cli]"
```

Then download the model using the following commands:

```shell
# Switch to the directory named 'HunyuanVideo'
cd HunyuanVideo
# Use the huggingface-cli tool to download HunyuanVideo model in HunyuanVideo/ckpts dir.
# The download time may vary from 10 minutes to 1 hour depending on network conditions.
huggingface-cli download tencent/HunyuanVideo --local-dir ./ckpts
```

<details>
<summary>💡Tips for using huggingface-cli (network problem)</summary>

##### 1. Using HF-Mirror

If you encounter slow download speeds in China, you can try a mirror to speed up the download process. For example,

```shell
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download tencent/HunyuanVideo --local-dir ./ckpts
```

##### 2. Resume Download

`huggingface-cli` supports resuming downloads. If the download is interrupted, you can just rerun the download 
command to resume the download process.

Note: If an `No such file or directory: 'ckpts/.huggingface/.gitignore.lock'` like error occurs during the download 
process, you can ignore the error and rerun the download command.

</details>

---

## Download Text Encoder

HunyuanVideo uses an MLLM model and a CLIP model as text encoder.

1. MLLM model (text_encoder folder)

HunyuanVideo supports different MLLMs (including HunyuanMLLM and open-source MLLM models). At this stage, we have not yet released HunyuanMLLM. We recommend the user in community to use [llava-llama-3-8b](https://huggingface.co/xtuner/llava-llama-3-8b-v1_1-transformers) provided by [Xtuner](https://huggingface.co/xtuner), which can be downloaded by the following command

```shell
cd HunyuanVideo/ckpts
huggingface-cli download xtuner/llava-llama-3-8b-v1_1-transformers --local-dir ./llava-llama-3-8b-v1_1-transformers
```

In order to save GPU memory resources for model loading, we separate the language model parts of `llava-llama-3-8b-v1_1-transformers` into `text_encoder`.
```
cd HunyuanVideo
python hyvideo/utils/preprocess_text_encoder_tokenizer_utils.py --input_dir ckpts/llava-llama-3-8b-v1_1-transformers --output_dir ckpts/text_encoder
```

2. CLIP model (text_encoder_2 folder)

We use [CLIP](https://huggingface.co/openai/clip-vit-large-patch14) provided by [OpenAI](https://openai.com) as another text encoder, users in the community can download this model by the following command

```
cd HunyuanVideo/ckpts
huggingface-cli download openai/clip-vit-large-patch14 --local-dir ./text_encoder_2
```
