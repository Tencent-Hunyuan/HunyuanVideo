import torch
import time

from hyvideo.vae.unet_causal_3d_blocks import prepare_causal_attention_mask

def prepare_causal_attention_mask_original(n_frame: int, n_hw: int, dtype, device, batch_size: int = None):
    seq_len = n_frame * n_hw
    mask = torch.full((seq_len, seq_len), float("-inf"), dtype=dtype, device=device)
    for i in range(seq_len):
        i_frame = i // n_hw
        mask[i, : (i_frame + 1) * n_hw] = 0
    if batch_size is not None:
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    return mask


def test_mask_generation(n_frame: int, n_hw: int, batch_size: int = 1):
    dtype = torch.bfloat16
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    original_mask = prepare_causal_attention_mask_original(n_frame, n_hw, dtype, device, batch_size)
    new_mask = prepare_causal_attention_mask(n_frame, n_hw, dtype, device, batch_size)
    torch.testing.assert_close(original_mask, new_mask)

def benchmark_mask_generation(n_frame: int, n_hw: int, batch_size: int = 1):
    def run_benchmark(fun, n_warmup: int = 2, n_iter: int = 5):
        for _ in range(n_warmup):
            fun()
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_iter):
            fun()
        if device.type == "cuda":
            torch.cuda.synchronize()
        return (time.time() - t0) / n_iter
    
    dtype = torch.bfloat16
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    t_orig = run_benchmark(
        fun=lambda: prepare_causal_attention_mask_original(n_frame, n_hw, dtype, device, batch_size)
    )
    t_new = run_benchmark(
        fun=lambda: prepare_causal_attention_mask(n_frame, n_hw, dtype, device, batch_size)
    )

    print(f"Original mask generation time: {t_orig:.4f}s")
    print(f"New mask generation time: {t_new:.4f}s")
    print(f"Speedup: {t_orig / t_new:.2f}x")

if __name__ == "__main__":
    test_mask_generation(64, 128, 1)
    benchmark_mask_generation(32, 1024, 1) # 45x speedup on GPU
