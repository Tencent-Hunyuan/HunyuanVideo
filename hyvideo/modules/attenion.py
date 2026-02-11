import importlib.metadata
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import flash_attn
    from flash_attn.flash_attn_interface import _flash_attn_forward
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
except ImportError:
    flash_attn = None
    flash_attn_varlen_func = None
    _flash_attn_forward = None


MEMORY_LAYOUT = {
    "flash": (
        lambda x: x.view(x.shape[0] * x.shape[1], *x.shape[2:]),
        lambda x: x,
    ),
    "torch": (
        lambda x: x.transpose(1, 2),
        lambda x: x.transpose(1, 2),
    ),
    "vanilla": (
        lambda x: x.transpose(1, 2),
        lambda x: x.transpose(1, 2),
    ),
}


def get_cu_seqlens(text_mask, img_len):
    """Calculate cu_seqlens_q, cu_seqlens_kv using text_mask and img_len

    Args:
        text_mask (torch.Tensor): the mask of text
        img_len (int): the length of image

    Returns:
        torch.Tensor: the calculated cu_seqlens for flash attention
    """
    batch_size = text_mask.shape[0]
    text_len = text_mask.sum(dim=1)
    max_len = text_mask.shape[1] + img_len

    cu_seqlens = torch.zeros([2 * batch_size + 1], dtype=torch.int32, device="cuda")

    for i in range(batch_size):
        s = text_len[i] + img_len
        s1 = i * max_len + s
        s2 = (i + 1) * max_len
        cu_seqlens[2 * i + 1] = s1
        cu_seqlens[2 * i + 2] = s2

    return cu_seqlens


def attention(
    q,
    k,
    v,
    mode="flash",
    drop_rate=0,
    attn_mask=None,
    causal=False,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    max_seqlen_q=None,
    max_seqlen_kv=None,
    batch_size=1,
):
    """
    Perform QKV self attention.

    Args:
        q (torch.Tensor): Query tensor with shape [b, s, a, d], where a is the number of heads.
        k (torch.Tensor): Key tensor with shape [b, s1, a, d]
        v (torch.Tensor): Value tensor with shape [b, s1, a, d]
        mode (str): Attention mode. Choose from 'self_flash', 'cross_flash', 'torch', and 'vanilla'.
        drop_rate (float): Dropout rate in attention map. (default: 0)
        attn_mask (torch.Tensor): Attention mask with shape [b, s1] (cross_attn), or [b, a, s, s1] (torch or vanilla).
            (default: None)
        causal (bool): Whether to use causal attention. (default: False)
        cu_seqlens_q (torch.Tensor): dtype torch.int32. The cumulative sequence lengths of the sequences in the batch,
            used to index into q.
        cu_seqlens_kv (torch.Tensor): dtype torch.int32. The cumulative sequence lengths of the sequences in the batch,
            used to index into kv.
        max_seqlen_q (int): The maximum sequence length in the batch of q.
        max_seqlen_kv (int): The maximum sequence length in the batch of k and v.

    Returns:
        torch.Tensor: Output tensor after self attention with shape [b, s, ad]
    """
    pre_attn_layout, post_attn_layout = MEMORY_LAYOUT[mode]
    q = pre_attn_layout(q)
    k = pre_attn_layout(k)
    v = pre_attn_layout(v)

    if mode == "torch":
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.to(q.dtype)
        if cu_seqlens_q is None:
            x = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=drop_rate, is_causal=causal
            )
        else:
            # q shape after pre_attn_layout: [batch_size, heads, seq_len, dim]
            # cu_seqlens has 2*batch_size+1 entries encoding (valid, padding) pairs
            # per batch item. Iterate over all batch items to handle batch_size > 1.
            seq_len_q = q.shape[2]
            seq_len_kv = k.shape[2]
            x = torch.empty_like(q)
            for i in range(batch_size):
                # Per-batch valid lengths (cu_seqlens stores absolute offsets
                # into the flattened sequence; subtract the batch base offset)
                valid_q = (cu_seqlens_q[2 * i + 1] - cu_seqlens_q[2 * i]).item()
                valid_kv = (cu_seqlens_kv[2 * i + 1] - cu_seqlens_kv[2 * i]).item()

                # Attend over valid (image + text) tokens
                attn1 = F.scaled_dot_product_attention(
                    q[i : i + 1, :, :valid_q],
                    k[i : i + 1, :, :valid_kv],
                    v[i : i + 1, :, :valid_kv],
                    attn_mask=attn_mask,
                    dropout_p=drop_rate,
                    is_causal=causal,
                )
                x[i : i + 1, :, :valid_q] = attn1

                # Attend over padding tokens (no mask, no causal)
                pad_q = seq_len_q - valid_q
                pad_kv = seq_len_kv - valid_kv
                if pad_q > 0 and pad_kv > 0:
                    attn2 = F.scaled_dot_product_attention(
                        q[i : i + 1, :, valid_q:],
                        k[i : i + 1, :, valid_kv:],
                        v[i : i + 1, :, valid_kv:],
                        attn_mask=None,
                        dropout_p=drop_rate,
                        is_causal=False,
                    )
                    x[i : i + 1, :, valid_q:] = attn2
                elif pad_q > 0:
                    x[i : i + 1, :, valid_q:] = 0
    elif mode == "flash":
        x = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
        )
        # x with shape [(bxs), a, d]
        x = x.view(
            batch_size, max_seqlen_q, x.shape[-2], x.shape[-1]
        )  # reshape x to [b, s, a, d]
    elif mode == "vanilla":
        scale_factor = 1 / math.sqrt(q.size(-1))

        b, a, s, _ = q.shape
        s1 = k.size(2)
        attn_bias = torch.zeros(b, a, s, s1, dtype=q.dtype, device=q.device)

        # Build a block-diagonal mask from cu_seqlens so that tokens in
        # different segments (valid vs padding) cannot attend to each other.
        # Without this, vanilla attention bleeds across segment boundaries,
        # causing severe quality degradation compared to flash attention.
        if cu_seqlens_q is not None:
            for i in range(batch_size):
                # Per-batch valid lengths
                valid_q = (cu_seqlens_q[2 * i + 1] - cu_seqlens_q[2 * i]).item()
                valid_kv = (cu_seqlens_kv[2 * i + 1] - cu_seqlens_kv[2 * i]).item()
                # Valid query tokens must not attend to padding key tokens
                attn_bias[i, :, :valid_q, valid_kv:] = float("-inf")
                # Padding query tokens must not attend to valid key tokens
                attn_bias[i, :, valid_q:, :valid_kv] = float("-inf")

        if causal:
            # Only applied to self attention
            assert (
                attn_mask is None
            ), "Causal mask and attn_mask cannot be used together"
            temp_mask = torch.ones(b, a, s, s, dtype=torch.bool, device=q.device).tril(
                diagonal=0
            )
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(q.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask

        # TODO: Maybe force q and k to be float32 to avoid numerical overflow
        attn = (q @ k.transpose(-2, -1)) * scale_factor
        attn += attn_bias
        attn = attn.softmax(dim=-1)
        attn = torch.dropout(attn, p=drop_rate, train=True)
        x = attn @ v
    else:
        raise NotImplementedError(f"Unsupported attention mode: {mode}")

    x = post_attn_layout(x)
    b, s, a, d = x.shape
    out = x.reshape(b, s, -1)
    return out


def parallel_attention(
    hybrid_seq_parallel_attn,
    q,
    k,
    v,
    img_q_len,
    img_kv_len,
    cu_seqlens_q,
    cu_seqlens_kv
):
    attn1 = hybrid_seq_parallel_attn(
        None,
        q[:, :img_q_len, :, :],
        k[:, :img_kv_len, :, :],
        v[:, :img_kv_len, :, :],
        dropout_p=0.0,
        causal=False,
        joint_tensor_query=q[:,img_q_len:cu_seqlens_q[1]],
        joint_tensor_key=k[:,img_kv_len:cu_seqlens_kv[1]],
        joint_tensor_value=v[:,img_kv_len:cu_seqlens_kv[1]],
        joint_strategy="rear",
    )
    if flash_attn.__version__ >= '2.7.0':
        attn2, *_ = _flash_attn_forward(
            q[:,cu_seqlens_q[1]:],
            k[:,cu_seqlens_kv[1]:],
            v[:,cu_seqlens_kv[1]:],
            dropout_p=0.0,
            softmax_scale=q.shape[-1] ** (-0.5),
            causal=False,
            window_size_left=-1,
            window_size_right=-1,
            softcap=0.0,
            alibi_slopes=None,
            return_softmax=False,
        )
    else:
        attn2, *_ = _flash_attn_forward(
            q[:,cu_seqlens_q[1]:],
            k[:,cu_seqlens_kv[1]:],
            v[:,cu_seqlens_kv[1]:],
            dropout_p=0.0,
            softmax_scale=q.shape[-1] ** (-0.5),
            causal=False,
            window_size=(-1, -1),
            softcap=0.0,
            alibi_slopes=None,
            return_softmax=False,
        )
    attn = torch.cat([attn1, attn2], dim=1)
    b, s, a, d = attn.shape
    attn = attn.reshape(b, s, -1)

    return attn
