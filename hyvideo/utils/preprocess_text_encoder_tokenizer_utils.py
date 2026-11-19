import argparse
from pathlib import Path
import torch
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
)


def preprocess_text_encoder_tokenizer(args):

    processor = AutoProcessor.from_pretrained(args.input_dir)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.input_dir,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(0)

    model.language_model.save_pretrained(
        f"{args.output_dir}"
    )
    processor.tokenizer.save_pretrained(
        f"{args.output_dir}"
    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="The path to the llava-llama-3-8b-v1_1-transformers.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="The output path of the llava-llama-3-8b-text-encoder-tokenizer."
        "if '', the parent dir of output will be the same as input dir.",
    )
    args = parser.parse_args()

    if len(args.output_dir) == 0:
        args.output_dir = str(Path(args.input_dir).parent)

    preprocess_text_encoder_tokenizer(args)
