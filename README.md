# Xbox Bootanim Decoder

Convert Xbox `bootanim.dat` files into normal MP4 files with audio.

This tool targets the currently researched `FS01`/`FSEG` Xbox boot animation format. It parses the DAT container, reconstructs a CPU-decoder-friendly H.264 Annex-B stream from the embedded AMD UVD messages and type-256 slice payloads, extracts the type-0 audio segment, and muxes the result into MP4.

## Features

- Parses `FS01` / `FSEG` sections.
- Lists animation segments, decode sessions, IDRs, and audio metadata.
- Reconstructs H.264 video from UVD message fields and embedded slice NALs.
- Extracts audio from both observed audio layouts:
  - raw f32le 8-channel 48 kHz audio used by Xbox One samples;
  - Series X/S `PEAK` + `data` chunk wrapper.
- Outputs MP4 with H.264 video and AAC 8-channel audio.

## Requirements

Python 3.10+ and FFmpeg must be installed.

Python dependency:

```bash
pip install -r requirements.txt
```

System dependency:

```bash
ffmpeg -version
```

`ffmpeg-python` is a Python wrapper around the FFmpeg executable, so the `ffmpeg` binary still needs to be available in `PATH`.

## Install

For development/use from a checkout:

```bash
git clone <your-repo-url> xbox-bootanim-decoder
cd xbox-bootanim-decoder
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

After `pip install -e .`, the command is:

```bash
xbox-bootanim-to-mp4 --help
```

You can also run the module directly from the checkout:

```bash
python -m xbox_bootanim.cli --help
```

## Basic usage

Convert a full DAT:

```bash
xbox-bootanim-to-mp4 /path/to/bootanim.dat -o output.mp4
```

List detected structure without writing media:

```bash
xbox-bootanim-to-mp4 /path/to/bootanim.dat --list
```

Export a specific decode range:

```bash
xbox-bootanim-to-mp4 /path/to/bootanim.dat -o visible_plus_loop.mp4 --start 202 --count 296
```

Export a detected session:

```bash
xbox-bootanim-to-mp4 /path/to/bootanim.dat -o loop.mp4 --session 3
```

## Options

```text
--list                  Print segments/sessions/audio info and exit
--fps FPS               Output frame rate, default 30
--start N               Decode picture index to start from
--count N               Number of decode pictures to export
--session N             Export detected decode session N
--no-audio              Write video-only MP4
--pad-audio             Pad short audio with silence to selected video duration
--audio-advance SEC     Trim SEC from audio start, making audio earlier
--audio-delay SEC       Delay audio by SEC
--crf N                 libx264 CRF, default 18
--preset NAME           libx264 preset, default medium
--audio-bitrate RATE    AAC bitrate, default 512k
--workdir DIR           Temporary/work directory
--keep-temp             Keep intermediate .h264/.raw/.mp4 files
```

## Notes and limitations

The DAT contains AMD UVD decode messages and per-picture H.264 reference state. This converter synthesizes SPS/PPS and creates a normal CPU-decodable H.264 stream. That is good enough for viewable MP4 output, but it is not a perfect hardware replay of the original UVD command stream.
