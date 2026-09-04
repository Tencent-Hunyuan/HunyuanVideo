"""Temporal VAE tiling helpers.

Causal 3D tiles request ``tile_min + 1`` frames so neighbors share one frame of
context. The extra frame must not be treated as a standalone final tile: after
the encoder/decoder runs, subsequent tiles drop that first frame, which would
leave an empty leftover (the ~192-frame / 25th-tile failure).
"""


def iter_temporal_tile_ranges(length, overlap_size, tile_extent):
    """Return clamped ``(start, end)`` ranges along a temporal axis.

    Args:
        length: Number of frames on the tiled tensor (sample or latent).
        overlap_size: Step between tile starts.
        tile_extent: Requested slice length, typically ``tile_min + 1``.

    A trailing start that only contains the last frame already covered by the
    previous tile is omitted so the causal first-frame drop cannot yield an
    empty tile.
    """
    if length <= 0:
        return []
    if overlap_size <= 0:
        raise ValueError(f"overlap_size must be positive, got {overlap_size}")
    if tile_extent <= 0:
        raise ValueError(f"tile_extent must be positive, got {tile_extent}")

    ranges = []
    for start in range(0, length, overlap_size):
        end = min(start + tile_extent, length)
        if start >= end:
            break
        if (
            ranges
            and start > 0
            and end - start <= 1
            and ranges[-1][1] >= length
        ):
            break
        ranges.append((start, end))
    return ranges
