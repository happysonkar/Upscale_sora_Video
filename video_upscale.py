#!/usr/bin/env python3
"""
Upscale 9:16 (portrait) Sora videos to 4K using Real-ESRGAN.
Improves perceived quality via AI super-resolution, not just resize.
Output: 2160x3840 (4K UHD portrait).
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

# Optional: use ffmpeg-python if available for cleaner probe
try:
    import ffmpeg
except ImportError:
    ffmpeg = None

# Target 4K resolution for 9:16 portrait
TARGET_W = 2160
TARGET_H = 3840
ASPECT_NUM = 9
ASPECT_DEN = 16
ASPECT_TOLERANCE = 0.05  # allow small deviation from 9:16

# Video extensions to process when input is a folder
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".flv", ".wmv"}


def get_videos_from_folder(folder_path):
    """Return sorted list of video file paths in folder (recursive=False)."""
    folder = Path(folder_path)
    if not folder.is_dir():
        return []
    paths = []
    for f in sorted(folder.iterdir()):
        if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS:
            paths.append(str(f.resolve()))
    return paths


def get_video_info(video_path, ffmpeg_bin="ffmpeg"):
    """Get width, height, fps, frame count, and whether audio exists via ffprobe (or ffmpeg)."""
    import subprocess
    import json
    # Prefer ffprobe (standard for probing); fall back to ffmpeg
    ffprobe_bin = "ffprobe" if ffmpeg_bin == "ffmpeg" else ffmpeg_bin.replace("ffmpeg", "ffprobe", 1)
    try:
        for probe_bin in (ffprobe_bin, ffmpeg_bin):
            cmd = [
                probe_bin, "-v", "error", "-print_format", "json", "-show_streams",
                "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
                "-i", video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                break
            # Try without nb_frames (some formats don't have it)
            cmd = [
                probe_bin, "-v", "error", "-print_format", "json", "-show_streams",
                "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate",
                "-i", video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                break
        else:
            err = (result.stderr or "").strip() or (result.stdout or "").strip()
            if not err:
                err = f"Probe exited with code {result.returncode}. Check that the file exists and ffmpeg/ffprobe can read it."
            raise RuntimeError(f"Could not probe video: {err}")
    except FileNotFoundError as e:
        raise RuntimeError(
            f"ffmpeg/ffprobe not found. Install ffmpeg and ensure it is on your PATH (e.g. apt install ffmpeg). {e}"
        ) from e

    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    if not streams:
        raise RuntimeError("No video stream found")
    s = streams[0]
    w = int(s["width"])
    h = int(s["height"])
    r = s.get("r_frame_rate", "24/1")
    if "/" in r:
        num, den = r.split("/")
        fps = float(num) / float(den)
    else:
        fps = float(r)
    nb_frames = int(s.get("nb_frames", 0))

    # Check for audio
    cmd_audio = [ffmpeg_bin, "-v", "quiet", "-print_format", "json", "-show_streams", "-i", video_path]
    ra = subprocess.run(cmd_audio, capture_output=True, text=True)
    has_audio = False
    if ra.returncode == 0:
        da = json.loads(ra.stdout)
        for st in da.get("streams", []):
            if st.get("codec_type") == "audio":
                has_audio = True
                break

    return {
        "width": w,
        "height": h,
        "fps": fps,
        "nb_frames": nb_frames,
        "has_audio": has_audio,
    }


def aspect_ratio_ok(width, height):
    """Check if video is approximately 9:16 (portrait)."""
    # 9:16 portrait => width/height = 9/16
    actual = width / height
    expected = ASPECT_NUM / ASPECT_DEN
    return abs(actual - expected) <= ASPECT_TOLERANCE


def _patch_torchvision_for_basicsr():
    """Compat for basicsr: torchvision dropped functional_tensor; alias it to functional."""
    if "torchvision.transforms.functional_tensor" in sys.modules:
        return
    try:
        import torchvision.transforms.functional as _ft
        sys.modules["torchvision.transforms.functional_tensor"] = _ft
    except Exception:
        pass


def ensure_realesrgan():
    """Lazy import and validate Real-ESRGAN deps."""
    try:
        import torch
    except ImportError as e:
        print(f"Missing 'torch': {e}", file=sys.stderr)
        print("  pip install torch torchvision", file=sys.stderr)
        _hint_same_python()
        raise SystemExit(1) from e
    _patch_torchvision_for_basicsr()
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from basicsr.utils.download_util import load_file_from_url
    except ImportError as e:
        print(f"Missing 'basicsr': {e}", file=sys.stderr)
        print("  pip install basicsr", file=sys.stderr)
        _hint_same_python()
        raise SystemExit(1) from e
    try:
        from realesrgan import RealESRGANer
    except ImportError as e:
        print(f"Missing 'realesrgan': {e}", file=sys.stderr)
        print("  pip install realesrgan   (or install from repo: pip install git+https://github.com/xinntao/Real-ESRGAN.git)", file=sys.stderr)
        _hint_same_python()
        raise SystemExit(1) from e
    return torch, RRDBNet, load_file_from_url, RealESRGANer


def _hint_same_python():
    """Remind to use the same Python that has the packages installed."""
    print(f"  Using Python: {sys.executable}", file=sys.stderr)
    print("  If you installed in a venv, activate it first (e.g. source venv/bin/activate).", file=sys.stderr)


def get_upsampler(scale=4, tile=256, fp32=False, model_name="RealESRGAN_x4plus", weights_dir="weights", device=None):
    """Build Real-ESRGAN upsampler. scale is 2 or 4. Model is loaded on the given device (GPU if available)."""
    torch, RRDBNet, load_file_from_url, RealESRGANer = ensure_realesrgan()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False. Install PyTorch with CUDA.")

    if model_name == "RealESRGAN_x4plus":
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"]
    elif model_name == "RealESRGAN_x2plus":
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
        file_url = ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"]
    elif model_name == "realesr-general-x4v3":
        from realesrgan.archs.srvgg_arch import SRVGGNetCompact
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type="prelu")
        netscale = 4
        file_url = [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        ]
    else:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"]

    os.makedirs(weights_dir, exist_ok=True)
    dni_weight = None
    if model_name == "realesr-general-x4v3":
        path_main = os.path.join(weights_dir, "realesr-general-x4v3.pth")
        path_wdn = os.path.join(weights_dir, "realesr-general-wdn-x4v3.pth")
        for url in file_url:
            fname = os.path.basename(url.split("/")[-1].split("?")[0])
            dest = os.path.join(weights_dir, fname)
            if not os.path.isfile(dest):
                load_file_from_url(url=url, model_dir=weights_dir, progress=True, file_name=fname)
        model_path = [path_main, path_wdn]
        dni_weight = [0.5, 0.5]
    else:
        model_path = os.path.join(weights_dir, model_name + ".pth")
        if not os.path.isfile(model_path):
            fname = os.path.basename(file_url[0].split("/")[-1].split("?")[0])
            load_file_from_url(url=file_url[0], model_dir=weights_dir, progress=True, file_name=fname)
            model_path = os.path.join(weights_dir, fname)

    outscale = min(scale, netscale)
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=tile,
        tile_pad=10,
        pre_pad=0,
        half=not fp32,
        device=device,
    )
    return upsampler, device, netscale, outscale


def upscale_frame(frame_bgr, upsampler, outscale, target_w, target_h):
    """Upscale one frame with Real-ESRGAN then resize to exact target if needed."""
    out, _ = upsampler.enhance(frame_bgr, outscale=outscale)
    out_h, out_w = out.shape[:2]
    if (out_w, out_h) != (target_w, target_h):
        out = cv2.resize(out, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    return out


def run_upscale(
    input_path,
    output_path,
    ffmpeg_bin="ffmpeg",
    model_name="RealESRGAN_x4plus",
    tile=256,
    fp32=False,
    weights_dir="weights",
    device=None,
):
    from tqdm import tqdm

    info = get_video_info(input_path, ffmpeg_bin)
    w, h = info["width"], info["height"]
    fps = info["fps"]
    has_audio = info["has_audio"]

    if not aspect_ratio_ok(w, h):
        print(f"Warning: video aspect ratio {w}:{h} is not 9:16. Proceeding anyway.", file=sys.stderr)

    # Scale factor to reach at least 4K short side (2160)
    scale_w = TARGET_W / w
    scale_h = TARGET_H / h
    scale_needed = max(scale_w, scale_h)
    if scale_needed <= 2:
        use_scale = 2
        model_name = "RealESRGAN_x2plus" if scale_needed <= 2 else model_name
    else:
        use_scale = 4
    # Prefer x4 model for best quality when we need more than 2x
    if scale_needed > 2:
        model_name = "RealESRGAN_x4plus"

    upsampler, upsampler_device, netscale, outscale = get_upsampler(
        scale=use_scale, tile=tile, fp32=fp32, model_name=model_name, weights_dir=weights_dir, device=device
    )
    print(f"Model loaded on: {upsampler_device}", flush=True)
    # Outscale: we may do 2x or 4x from model, then resize to TARGET_W x TARGET_H
    outscale = min(int(np.ceil(scale_needed)), netscale)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total_frames <= 0 and info.get("nb_frames"):
        total_frames = info["nb_frames"]

    tmp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_video.close()

    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    writer = cv2.VideoWriter(tmp_video.name, fourcc, fps, (TARGET_W, TARGET_H))
    if not writer.isOpened():
        raise RuntimeError("Could not create temporary video writer")

    pbar = tqdm(total=total_frames if total_frames > 0 else None, unit="frame", desc="Upscaling")
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out_frame = upscale_frame(frame, upsampler, outscale, TARGET_W, TARGET_H)
            writer.write(out_frame)
            frame_idx += 1
            pbar.update(1)
    finally:
        cap.release()
        writer.release()
        pbar.close()



    # Mux video + audio if present (explicit map so audio is not dropped)
    if True:
        cmd = [
            ffmpeg_bin, "-y",
            "-i", tmp_video.name,
            "-i", input_path,
            "-map", "0:v:0",   # video from upscaled temp
            "-map", "1:a:0",   # first audio from original (explicit, no optional)
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            output_path,
        ]
    # else:
    #     cmd = [
    #         ffmpeg_bin, "-y",
    #         "-i", tmp_video.name,
    #         "-c:v", "libx264",
    #         "-preset", "medium",
    #         "-crf", "18",
    #         output_path,
    #     ]
    import subprocess
    r = subprocess.run(cmd, capture_output=True, text=True)
    os.unlink(tmp_video.name)
    if r.returncode != 0:
        raise RuntimeError(f"FFmpeg mux failed: {r.stderr}")
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Upscale 9:16 Sora videos to 4K (2160x3840) with Real-ESRGAN."
    )
    parser.add_argument("input", nargs="?", help="Input video file or folder path (9:16 portrait)")
    parser.add_argument("-i", "--input", dest="input_file", help="Input video or folder (alternative to positional)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output path: for a single file, output path; for a folder, output directory (default: same dir as each input)")
    parser.add_argument(
        "-n", "--model",
        default="Real-ESRGAN-General-x4v3",
        choices=["RealESRGAN_x4plus", "RealESRGAN_x2plus", "Real-ESRGAN-General-x4v3"],
        help="Real-ESRGAN model (x4plus recommended for quality)",
    )
    parser.add_argument("-t", "--tile", type=int, default=256,
                        help="Tile size for GPU memory (0 = no tiling, reduce if OOM)")
    parser.add_argument("--fp32", action="store_true", help="Use FP32 (slower, more compatible)")
    parser.add_argument("--weights-dir", default="weights", help="Directory for model weights")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="Path to ffmpeg binary")
    parser.add_argument("--gpu", action="store_true", help="Force GPU (cuda:0); exit if no CUDA)")
    parser.add_argument("--device", type=str, default=None, help="Device: cuda, cuda:0, cuda:1, or cpu (default: auto)")
    args = parser.parse_args()
    input_path = args.input_file or args.input
    if not input_path:
        parser.error("Input required: pass a video file or folder path as argument or use -i/--input")
    input_path = os.path.abspath(input_path)
    if not os.path.exists(input_path):
        print(f"Error: path does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)

    device = args.device
    if args.gpu:
        device = "cuda" if device is None else device
        if not device.startswith("cuda"):
            device = "cuda"

    run_kw = dict(
        ffmpeg_bin=args.ffmpeg,
        model_name=args.model,
        tile=args.tile,
        fp32=args.fp32,
        weights_dir=args.weights_dir,
        device=device,
    )

    if os.path.isfile(input_path):
        # Single file: original behavior
        if args.output:
            output_path = os.path.abspath(args.output)
        else:
            base = Path(input_path).stem
            out_dir = Path(input_path).parent
            output_path = str(out_dir / f"{base}_4k.mp4")
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        run_upscale(input_path, output_path, **run_kw)
        return

    # Folder: collect videos and process one by one
    video_paths = get_videos_from_folder(input_path)
    if not video_paths:
        print(f"No video files found in folder (extensions: {', '.join(sorted(VIDEO_EXTENSIONS))}): {input_path}", file=sys.stderr)
        sys.exit(1)
    output_dir = Path(args.output).resolve() if args.output else Path(input_path)
    if args.output:
        os.makedirs(output_dir, exist_ok=True)
    print(f"Found {len(video_paths)} video(s) in {input_path}. Processing one by one.", flush=True)
    for i, inp in enumerate(video_paths, 1):
        base = Path(inp).stem
        ext = Path(inp).suffix
        out_path = str(output_dir / f"{base}_4k{ext}")
        print(f"\n[{i}/{len(video_paths)}] {Path(inp).name} -> {out_path}", flush=True)
        try:
            run_upscale(inp, out_path, **run_kw)
        except Exception as e:
            print(f"Error processing {inp}: {e}", file=sys.stderr)
            raise
    print(f"\nDone. Processed {len(video_paths)} video(s).")


if __name__ == "__main__":
    main()
