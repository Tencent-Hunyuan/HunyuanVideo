from pathlib import Path, PureWindowsPath

from einops import rearrange

import torch
import torchvision
import numpy as np
import imageio

CODE_SUFFIXES = {
    ".py",  # Python codes
    ".sh",  # Shell scripts
    ".yaml",
    ".yml",  # Configuration files
}

def contained_file(directory, filename: str) -> Path:
    """Join a filename under directory, ignoring any directory part in the name.

    PureWindowsPath.name treats both '/' and '\\' as separators, so a prompt
    such as '..\\\\..\\\\evil' cannot escape on Windows or POSIX. resolve()
    then relative_to() rejects symlink escapes.
    """
    root = Path(directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    leaf = PureWindowsPath(str(filename)).name
    if leaf in {"", ".", ".."}:
        raise ValueError(f"Refusing unsafe filename: {filename}")
    path = (root / leaf).resolve()
    path.relative_to(root)
    return path


def safe_dir(path):
    """
    Create a directory (or the parent directory of a file) if it does not exist.

    Args:
        path (str or Path): Path to the directory.

    Returns:
        path (Path): Path object of the directory.
    """
    path = Path(path)
    path.mkdir(exist_ok=True, parents=True)
    return path


def safe_file(path):
    """
    Create the parent directory of a file if it does not exist.

    Args:
        path (str or Path): Path to the file.

    Returns:
        path (Path): Path object of the file.
    """
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    return path

def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=1, fps=24):
    """save videos by video tensor
       copy from https://github.com/guoyww/AnimateDiff/blob/e92bd5671ba62c0d774a32951453e328018b7c5b/animatediff/utils/util.py#L61

    Args:
        videos (torch.Tensor): video tensor predicted by the model
        path (str): path to save video. Parent directory must already exist.
        rescale (bool, optional): rescale the video tensor from [-1, 1] to  . Defaults to False.
        n_rows (int, optional): Defaults to 1.
        fps (int, optional): video save fps. Defaults to 8.
    """
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = torch.clamp(x, 0, 1)
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    imageio.mimsave(path, outputs, fps=fps)
