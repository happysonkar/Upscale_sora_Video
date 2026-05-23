# Sora 9:16 → 4K Video Upscaler

Upscales **9:16 portrait** videos (e.g. from Sora) to **4K (2160×3840)** using [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) for AI super-resolution. Output is sharper and more detailed than a simple resize.

## Requirements

- **Python 3.9+**
- **FFmpeg** and **ffprobe** on your `PATH` (probe metadata, encode video, mux audio)
- **NVIDIA GPU with CUDA** recommended; CPU works but is much slower

Install system FFmpeg (example on Debian/Ubuntu):

```bash
sudo apt install ffmpeg
```

## Install

```bash
pip install -r requirements.txt
```

On first run, model weights are downloaded automatically into `weights/` (hundreds of MB per model).

Use the same Python environment for install and run (activate your venv if you use one).

## Usage

### Single video

```bash
# Output: same folder as input, <name>_4k.mp4
python video_upscale.py path/to/video.mp4

# Custom output path
python video_upscale.py input.mp4 -o output_4k.mp4

# Explicit input flag (same as positional)
python video_upscale.py -i input.mp4 -o output_4k.mp4
```

### Folder (batch)

Process every video in a directory (non-recursive). Supported extensions: `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`, `.m4v`, `.flv`, `.wmv`.

```bash
# Outputs <name>_4k.<ext> next to each input file
python video_upscale.py path/to/folder/

# Write all outputs to a separate directory
python video_upscale.py path/to/folder/ -o path/to/output/
```

If one file fails, processing stops and the error is reported.

## Options

| Option | Description |
|--------|-------------|
| `input` | Video file or folder (positional), or use `-i` / `--input` |
| `-o`, `--output` | Output file (single input) or output directory (folder input). Default: same location as input with `_4k` suffix |
| `-n`, `--model` | Real-ESRGAN model (see table below). Default: `Real-ESRGAN-General-x4v3` |
| `-t`, `--tile` | Tile size for GPU memory (default: `256`). Lower (e.g. `128`) if you hit OOM; `0` disables tiling |
| `--fp32` | Use FP32 instead of FP16 (slower, more compatible) |
| `--weights-dir` | Directory for downloaded weights (default: `weights`) |
| `--ffmpeg` | Path to `ffmpeg` binary (default: `ffmpeg`) |
| `--gpu` | Force CUDA; exit if CUDA is unavailable |
| `--device` | Device string: `cuda`, `cuda:0`, `cuda:1`, or `cpu` (default: auto — GPU if available) |

### Models (`-n` / `--model`)

| Name | Notes |
|------|--------|
| `RealESRGAN_x4plus` | 4× upscale; strong quality (used automatically when input needs more than 2×) |
| `RealESRGAN_x2plus` | 2× upscale; faster, good when source is already close to 4K |
| `Real-ESRGAN-General-x4v3` | General-purpose 4× model (CLI default) |

The script may still switch between 2× and 4× models based on how much scaling is needed to reach 2160×3840.

### Examples

```bash
# General model (default)
python video_upscale.py input.mp4

# Explicit x4plus for maximum quality
python video_upscale.py input.mp4 -n RealESRGAN_x4plus

# 2× model (faster for sources already near 1080p width)
python video_upscale.py input.mp4 -n RealESRGAN_x2plus

# Less VRAM
python video_upscale.py input.mp4 -t 128

# CPU or compatibility mode
python video_upscale.py input.mp4 --fp32 --device cpu

# Specific GPU
python video_upscale.py input.mp4 --device cuda:0
```

## Aspect ratio

Input should be **9:16 portrait** (width:height ≈ 9:16). A small deviation is allowed; otherwise a warning is printed and processing continues. Output is always **2160×3840**.

## How it works

1. **Probe** — Read width, height, FPS, frame count, and audio presence via ffprobe/ffmpeg.
2. **Scale** — Choose 2× or 4× Real-ESRGAN so the upscaled frame reaches at least 4K, then resize to exactly 2160×3840 (LANCZOS) if needed.
3. **Enhance** — Run each frame through Real-ESRGAN on GPU (or CPU).
4. **Encode** — Write frames to a temporary file, then mux with **libx264** (CRF 18, medium preset) and **AAC** audio (192k) from the original when audio exists.

Progress is shown per frame with `tqdm`.

## Troubleshooting

| Issue | What to try |
|-------|-------------|
| `ffmpeg/ffprobe not found` | Install FFmpeg and ensure it is on `PATH` |
| CUDA OOM | Lower `-t` (e.g. `128` or `64`) or use `--fp32` |
| Missing `torch` / `basicsr` / `realesrgan` | `pip install -r requirements.txt` in the same environment you use to run the script |
| `torchvision` / `basicsr` import errors | Upgrade `torch` and `torchvision`; the script patches a common `functional_tensor` compatibility issue |

## Project layout

```
video_upscale.py   # Main script
requirements.txt   # Python dependencies
weights/           # Downloaded models (created on first run)
```

## License

Real-ESRGAN and its weights are subject to their upstream licenses. See the [Real-ESRGAN repository](https://github.com/xinntao/Real-ESRGAN) for details.
