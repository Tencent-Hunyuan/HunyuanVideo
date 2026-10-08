"""Regression tests for temporal VAE tile bounds.

No pretrained weights. Tensor checks use a 5D index tensor so encode/decode
tiling can be exercised without constructing the full causal 3D VAE.
"""
import os
import sys
import unittest

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

import importlib.util

_tile_path = os.path.join(project_root, "hyvideo", "vae", "temporal_tile.py")
_tile_spec = importlib.util.spec_from_file_location("hyvideo_vae_temporal_tile", _tile_path)
_tile_mod = importlib.util.module_from_spec(_tile_spec)
_tile_spec.loader.exec_module(_tile_mod)
iter_temporal_tile_ranges = _tile_mod.iter_temporal_tile_ranges

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _naive_ranges(length, overlap_size, tile_extent):
    """Pre-fix loop: ``range(0, T, overlap)`` + unclamped ``i:i+extent``."""
    ranges = []
    for start in range(0, length, overlap_size):
        ranges.append((start, start + tile_extent))
    return ranges


def _apply_ranges(z, ranges, drop_first=True):
    tiles = []
    for i, (start, end) in enumerate(ranges):
        tile = z[:, :, start:end, :, :]
        if drop_first and i > 0:
            tile = tile[:, :, 1:, :, :]
        tiles.append(tile)
    return tiles


@unittest.skipIf(torch is None, "torch is required")
class TemporalTileRangeTests(unittest.TestCase):
    def test_issue_223_25th_tile_is_not_a_one_frame_leftover(self):
        # SkyReels/diffusers naming: 8 latent frames/tile, stride 8, T=193.
        # Naive loop's 25th start is 192; after the causal drop the tile is empty.
        length = 193
        overlap_size = 8
        tile_extent = 8 + 1

        naive = _naive_ranges(length, overlap_size, tile_extent)
        self.assertEqual(len(naive), 25)
        self.assertEqual(naive[-1], (192, 201))

        z = torch.arange(length, dtype=torch.float32).view(1, 1, length, 1, 1)
        naive_last = z[:, :, naive[-1][0] : naive[-1][1], :, :]
        self.assertEqual(int(naive_last.shape[2]), 1)
        self.assertEqual(int(naive_last[:, :, 1:, :, :].shape[2]), 0)

        fixed = iter_temporal_tile_ranges(length, overlap_size, tile_extent)
        self.assertLess(len(fixed), 25)
        self.assertEqual(fixed[-1][1], length)
        self.assertGreater(fixed[-1][1] - fixed[-1][0], 1)

        tiles = _apply_ranges(z, fixed)
        self.assertTrue(all(int(t.shape[2]) > 0 for t in tiles))
        self.assertEqual(int(tiles[-1][:, :, -1, :, :]), length - 1)

    def test_default_hunyuan_decode_omits_latent_leftover(self):
        # sample_tsize=64, time_compression=4, overlap_factor=0.25
        # T_latent = 49 -> 193 sample frames. Last naive start is 48.
        length = 49
        overlap_size = 12
        tile_extent = 16 + 1
        z = torch.arange(length, dtype=torch.float32).view(1, 1, length, 1, 1)

        naive_last_start = list(range(0, length, overlap_size))[-1]
        self.assertEqual(naive_last_start, 48)
        leftover = z[:, :, naive_last_start : naive_last_start + tile_extent, :, :]
        self.assertEqual(int(leftover.shape[2]), 1)

        ranges = iter_temporal_tile_ranges(length, overlap_size, tile_extent)
        self.assertNotIn((48, 49), ranges)
        self.assertEqual(ranges[-1][1], length)
        tiles = _apply_ranges(z, ranges)
        self.assertTrue(all(int(t.shape[2]) > 0 for t in tiles))
        covered = torch.cat(tiles, dim=2)
        self.assertEqual(int(covered[:, :, -1, :, :]), length - 1)

    def test_slices_never_exceed_tensor_and_do_not_wrap(self):
        for length in (17, 33, 49, 65, 193):
            ranges = iter_temporal_tile_ranges(length, overlap_size=8, tile_extent=9)
            z = torch.arange(length, dtype=torch.float32).view(1, 1, length, 1, 1)
            for start, end in ranges:
                self.assertGreaterEqual(start, 0)
                self.assertLessEqual(end, length)
                tile = z[:, :, start:end, :, :]
                self.assertEqual(int(tile.shape[2]), end - start)
                self.assertTrue(torch.equal(tile[0, 0, :, 0, 0], torch.arange(start, end, dtype=z.dtype)))

    def test_long_sequence_covers_past_192_without_looping(self):
        length = 257
        ranges = iter_temporal_tile_ranges(length, overlap_size=12, tile_extent=17)
        z = torch.arange(length, dtype=torch.float32).view(1, 1, length, 1, 1)
        tiles = _apply_ranges(z, ranges)
        tail = tiles[-1][0, 0, :, 0, 0]
        self.assertGreater(int(tail[-1]), 192)
        # Unique increasing ids: looping would repeat an earlier prefix.
        self.assertTrue(torch.all(tail[1:] > tail[:-1]))
        self.assertEqual(int(tail[-1]), length - 1)

    def test_empty_and_invalid_inputs(self):
        self.assertEqual(iter_temporal_tile_ranges(0, 8, 9), [])
        with self.assertRaises(ValueError):
            iter_temporal_tile_ranges(16, 0, 9)
        with self.assertRaises(ValueError):
            iter_temporal_tile_ranges(16, 8, 0)


@unittest.skipIf(torch is None, "torch is required")
class TemporalTiledMockVAETests(unittest.TestCase):
    """Run the public tiling methods with identity time compress/expand."""

    def _make_vae(self, sample_tsize=8, time_compression_ratio=4):
        from types import MethodType
        try:
            from hyvideo.vae.autoencoder_kl_causal_3d import AutoencoderKLCausal3D
        except ImportError as exc:  # pragma: no cover
            raise unittest.SkipTest(f"VAE imports unavailable: {exc}") from exc

        vae = AutoencoderKLCausal3D.__new__(AutoencoderKLCausal3D)
        vae.time_compression_ratio = time_compression_ratio
        vae.tile_sample_min_tsize = sample_tsize
        vae.tile_latent_min_tsize = sample_tsize // time_compression_ratio
        vae.tile_overlap_factor = 0.25
        vae.use_spatial_tiling = False
        vae.tile_sample_min_size = 32
        vae.tile_latent_min_size = 4

        ratio = time_compression_ratio

        def encoder(x):
            return x[:, :, ::ratio, :, :]

        def decoder(z):
            b, c, t, h, w = z.shape
            out_t = (t - 1) * ratio + 1 if t > 0 else 0
            out = z.new_zeros(b, c, out_t, h, w)
            if t == 0:
                return out
            out[:, :, 0] = z[:, :, 0]
            for i in range(1, t):
                out[:, :, (i - 1) * ratio + 1 : i * ratio + 1] = z[:, :, i]
            return out

        vae.encoder = encoder
        vae.decoder = decoder
        vae.quant_conv = lambda x: x
        vae.post_quant_conv = lambda z: z
        vae.blend_t = MethodType(AutoencoderKLCausal3D.blend_t, vae)
        vae.temporal_tiled_encode = MethodType(AutoencoderKLCausal3D.temporal_tiled_encode, vae)
        vae.temporal_tiled_decode = MethodType(AutoencoderKLCausal3D.temporal_tiled_decode, vae)
        return vae

    def test_decode_length_and_no_loop_past_192(self):
        vae = self._make_vae(sample_tsize=16)
        # 65 latents -> 257 sample frames, well past the 192/193 boundary.
        t_lat = 65
        z = torch.arange(t_lat, dtype=torch.float32).view(1, 1, t_lat, 1, 1).expand(1, 4, t_lat, 2, 2).contiguous()
        out = vae.temporal_tiled_decode(z, return_dict=False)[0]
        expected = (t_lat - 1) * vae.time_compression_ratio + 1
        self.assertEqual(int(out.shape[2]), expected)
        self.assertGreater(expected, 192)
        ids = out[0, 0, :, 0, 0]
        self.assertGreater(int(ids[192]), int(ids[0]))
        self.assertFalse(torch.equal(ids[193:193 + 8], ids[:8]))
        self.assertEqual(int(ids[-1]), t_lat - 1)

    def test_encode_length_matches_causal_formula(self):
        vae = self._make_vae(sample_tsize=16)
        t_sample = 257
        x = torch.arange(t_sample, dtype=torch.float32).view(1, 1, t_sample, 1, 1).expand(1, 3, t_sample, 2, 2).contiguous()
        posterior = vae.temporal_tiled_encode(x, return_dict=False)[0]
        moments = posterior.parameters
        expected = (t_sample - 1) // vae.time_compression_ratio + 1
        self.assertEqual(int(moments.shape[2]), expected)


if __name__ == "__main__":
    unittest.main()
