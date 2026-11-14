#!/usr/bin/env python3
"""Minimal tiled SR inference script for nadeshiko-sr safetensors models."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from spandrel import ModelLoader


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.inference_mode()
def upscale_rgb(
    model: torch.nn.Module,
    scale: int,
    rgb: Image.Image,
    device: torch.device,
    tile: int,
    overlap: int,
) -> Image.Image:
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    _, _, h, w = t.shape

    if h <= tile and w <= tile:
        out = model(t).clamp(0.0, 1.0)
    else:
        out = torch.zeros((1, 3, h * scale, w * scale), dtype=torch.float32, device=device)
        weight = torch.zeros((1, 1, h * scale, w * scale), dtype=torch.float32, device=device)
        step = max(1, tile - overlap)
        ys = list(range(0, max(1, h - tile + 1), step))
        xs = list(range(0, max(1, w - tile + 1), step))
        if ys[-1] + tile < h:
            ys.append(max(0, h - tile))
        if xs[-1] + tile < w:
            xs.append(max(0, w - tile))

        for y0 in ys:
            y1 = min(h, y0 + tile)
            for x0 in xs:
                x1 = min(w, x0 + tile)
                patch_out = model(t[:, :, y0:y1, x0:x1]).clamp(0.0, 1.0)
                out[:, :, y0 * scale : y1 * scale, x0 * scale : x1 * scale] += patch_out
                weight[:, :, y0 * scale : y1 * scale, x0 * scale : x1 * scale] += 1.0
        out = out / weight.clamp_min(1.0)

    out_np = (out.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(out_np, mode="RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run nadeshiko-sr safetensors inference.")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=24)
    args = parser.parse_args()

    device = pick_device(args.device)
    desc = ModelLoader().load_from_file(str(args.model))
    scale = desc.scale
    model = desc.model.eval().float().to(device)

    with Image.open(args.input) as im:
        im.load()
        has_alpha = False
        alpha_img: Image.Image | None = None
        if im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info:
            rgba = im.convert("RGBA")
            alpha = rgba.getchannel("A")
            if alpha.getextrema()[0] < 254:
                has_alpha = True
                alpha_img = alpha
            rgb = rgba.convert("RGB")
        else:
            rgb = im.convert("RGB")

    sr_rgb = upscale_rgb(model, scale, rgb, device, args.tile, args.overlap)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if has_alpha and alpha_img is not None:
        sr_alpha = alpha_img.resize(sr_rgb.size, Image.Resampling.NEAREST)
        r, g, b = sr_rgb.split()
        Image.merge("RGBA", (r, g, b, sr_alpha)).save(args.output, format="PNG")
    else:
        sr_rgb.save(args.output, format="PNG")


if __name__ == "__main__":
    main()
