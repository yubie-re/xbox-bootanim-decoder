#!/usr/bin/env python3
"""
Convert an Xbox bootanim.dat to a normal MP4 with audio.

This is a standalone convenience tool for the currently understood FS01/FSEG
bootanim.dat format. It reconstructs a CPU-decoder-friendly H.264 Annex-B
stream from the embedded AMD UVD messages + type-256 slice payloads, extracts
the type-0 audio segment, and muxes both into an MP4 using ffmpeg.

Requirements:
  - Python 3.10+
  - ffmpeg in PATH
  - ffmpeg-python (`pip install ffmpeg-python`)

Examples:
  xbox-bootanim-to-mp4 "../source/Xbox One X/bootanim.dat" \
      -o xbox_one_x_full.mp4

  xbox-bootanim-to-mp4 "../source/Xbox One X/bootanim.dat" \
      -o xbox_one_x_main_plus_loop.mp4 --start 202 --count 296

  xbox-bootanim-to-mp4 "../source/Xbox One S OG/bootanim.dat" \
      -o xbox_one_s.mp4 --session 1
"""

from __future__ import annotations

import argparse
import ffmpeg  # type: ignore
import json
import shutil
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

def u32(data: bytes, off: int) -> int:
    return struct.unpack_from("<I", data, off)[0]


def u64(data: bytes, off: int) -> int:
    return struct.unpack_from("<Q", data, off)[0]


def s8(value: int) -> int:
    return value - 256 if value >= 128 else value

class BitWriter:
    def __init__(self) -> None:
        self.bits: list[int] = []

    def bit(self, value: int, count: int = 1) -> None:
        for i in range(count - 1, -1, -1):
            self.bits.append((int(value) >> i) & 1)

    def ue(self, value: int) -> None:
        if value < 0:
            raise ValueError(f"ue() cannot encode negative value {value}")
        code_num = value + 1
        leading_zero_bits = code_num.bit_length() - 1
        self.bits.extend([0] * leading_zero_bits)
        self.bit(code_num, leading_zero_bits + 1)

    def se(self, value: int) -> None:
        self.ue(-2 * value if value <= 0 else 2 * value - 1)

    def rbsp(self) -> bytes:
        # rbsp_stop_one_bit and byte alignment.
        self.bit(1)
        while len(self.bits) % 8:
            self.bit(0)

        raw = bytearray()
        for i in range(0, len(self.bits), 8):
            byte = 0
            for bit in self.bits[i : i + 8]:
                byte = (byte << 1) | bit
            raw.append(byte)

        # Emulation-prevention bytes.
        out = bytearray()
        zero_count = 0
        for byte in raw:
            if zero_count >= 2 and byte <= 3:
                out.append(3)
                zero_count = 0
            out.append(byte)
            zero_count = zero_count + 1 if byte == 0 else 0
        return bytes(out)


def write_flat_scaling_list(writer: BitWriter, size: int) -> None:
    # Flat scaling list: first delta sets 8, rest stay unchanged.
    writer.se(8)
    for _ in range(size - 1):
        writer.se(0)


def build_sps_pps(params: dict[str, int | bool]) -> bytes:
    """Build synthetic SPS/PPS from the UVD H.264 message fields."""

    profile = int(params.get("profile", 100))
    level = int(params.get("level", 41))
    high_profile = profile in (100, 110, 122, 244, 44, 83, 86, 118, 128, 138, 139, 134, 135)

    writer = BitWriter()
    writer.bit(profile, 8)
    writer.bit(int(params.get("constraints", 0)), 8)
    writer.bit(level, 8)
    writer.ue(0)  # seq_parameter_set_id

    if high_profile:
        writer.ue(int(params.get("chroma", 1)))
        writer.ue(int(params.get("bit_depth_luma_minus8", 0)))
        writer.ue(int(params.get("bit_depth_chroma_minus8", 0)))
        writer.bit(0)  # qpprime_y_zero_transform_bypass_flag
        writer.bit(1 if params.get("sps_flat_scaling", False) else 0)
        if params.get("sps_flat_scaling", False):
            for i in range(8):
                writer.bit(1)
                write_flat_scaling_list(writer, 16 if i < 6 else 64)

    writer.ue(int(params["log2fn"]) - 4)
    poc_type = int(params.get("poc_type", 0))
    writer.ue(poc_type)
    if poc_type == 0:
        writer.ue(int(params.get("log2poc", 4)) - 4)
    elif poc_type == 1:
        writer.bit(int(params.get("delta_pic_order_always_zero_flag", 0)))
        writer.se(0)
        writer.se(0)
        writer.ue(0)

    writer.ue(int(params.get("refs", 1)))
    writer.bit(0)  # gaps_in_frame_num_value_allowed_flag
    writer.ue(int(params.get("pic_width_mbs_minus1", 119)))
    writer.ue(int(params.get("pic_height_map_units_minus1", 67)))
    frame_mbs_only = int(params.get("frame_mbs_only", 1))
    writer.bit(frame_mbs_only)
    if not frame_mbs_only:
        writer.bit(0)  # mb_adaptive_frame_field_flag
    writer.bit(int(params.get("direct8x8", 1)))

    # Crop coded 1920x1088 to visible 1920x1080. For 4:2:0, bottom offset 4 => 8 luma pixels.
    writer.bit(1)  # frame_cropping_flag
    writer.ue(0)
    writer.ue(0)
    writer.ue(0)
    writer.ue(int(params.get("crop_bottom", 4)))
    writer.bit(0)  # vui_parameters_present_flag
    sps = b"\x00\x00\x01\x67" + writer.rbsp()

    writer = BitWriter()
    writer.ue(0)  # pic_parameter_set_id
    writer.ue(0)  # seq_parameter_set_id
    writer.bit(int(params.get("cabac", 1)))
    writer.bit(int(params.get("pic_order_present", 0)))
    writer.ue(int(params.get("num_slice_groups_minus1", 0)))
    writer.ue(int(params.get("num_ref_idx_l0_default_active_minus1", 0)))
    writer.ue(int(params.get("num_ref_idx_l1_default_active_minus1", 0)))
    writer.bit(int(params.get("weighted_pred", 0)))
    writer.bit(int(params.get("weighted_bipred_idc", 0)), 2)
    writer.se(int(params.get("pic_init_qp_minus26", 0)))
    writer.se(int(params.get("pic_init_qs_minus26", 0)))
    writer.se(int(params.get("chroma_qp_index_offset", 0)))
    writer.bit(int(params.get("deblocking_filter_control_present", 1)))
    writer.bit(int(params.get("constrained_intra_pred", 0)))
    writer.bit(int(params.get("redundant_pic_cnt_present", 0)))

    if int(params.get("pps_extra", 1)):
        transform8 = int(params.get("transform8", 1))
        writer.bit(transform8)
        writer.bit(1 if params.get("pps_flat_scaling", False) else 0)
        if params.get("pps_flat_scaling", False):
            for i in range(6 + (2 if transform8 else 0)):
                writer.bit(1)
                write_flat_scaling_list(writer, 16 if i < 6 else 64)
        writer.se(int(params.get("second_chroma_qp_index_offset", params.get("chroma_qp_index_offset", 0))))

    pps = b"\x00\x00\x01\x68" + writer.rbsp()
    return sps + pps


@dataclass
class Segment:
    index: int
    offset: int
    kind: int
    payload: int
    size: int
    end: int


@dataclass
class Resource:
    kind: str
    rtype: int
    size: int
    rid: int
    payload: int | None


@dataclass
class Frame:
    segment_index: int
    segment_type: int
    frame_index: int
    marker: int
    resources: list[Resource]


@dataclass
class DecodePicture:
    index: int
    segment_index: int
    segment_type: int
    frame_index: int
    marker: int
    message: tuple[int, ...]
    nal: bytes
    nal_type: int


@dataclass
class Session:
    index: int
    segment_index: int
    segment_type: int
    create_frame: int | None
    destroy_frame: int | None
    decode_start: int | None
    decode_count: int


def parse_segments(data: bytes) -> list[Segment]:
    out: list[Segment] = []
    off = 0
    index = 0
    while off + 12 <= len(data) and data[off : off + 4] in (b"FS01", b"FSEG"):
        kind = u32(data, off + 4)
        size = u32(data, off + 8)
        payload = off + 12
        end = payload + size
        if end > len(data):
            raise ValueError(f"segment {index} extends past EOF")
        out.append(Segment(index, off, kind, payload, size, end))
        off = end
        index += 1
    if not out:
        raise ValueError("input does not start with FS01/FSEG")
    return out


def parse_frames(data: bytes, segments: Iterable[Segment]) -> list[Frame]:
    frames: list[Frame] = []
    for segment in segments:
        if segment.kind not in (1, 2):
            continue
        pos = segment.payload
        frame_index = -1
        while pos < segment.end:
            if data[pos : pos + 4] != b"FRMB":
                raise ValueError(f"expected FRMB at 0x{pos:x}, got {data[pos:pos+4]!r}")
            pos += 4
            frame_index += 1
            marker = u32(data, pos)
            pos += 4

            token = data[pos : pos + 4]
            pos += 4
            if token == b"MDTB":
                pos += 16
                token = data[pos : pos + 4]
                pos += 4

            resources: list[Resource] = []
            if token == b"RSSB":
                while True:
                    record = data[pos : pos + 4]
                    pos += 4
                    if record == b"RSSE":
                        token = data[pos : pos + 4]
                        pos += 4
                        break
                    if record != b"RSCB":
                        raise ValueError(f"expected RSCB/RSSE at 0x{pos-4:x}, got {record!r}")
                    kind_b = data[pos : pos + 4]
                    kind = kind_b.decode("ascii", "replace")
                    rtype = u32(data, pos + 4)
                    rsize = u32(data, pos + 8)
                    rid = u64(data, pos + 12)
                    pos += 20
                    payload = None
                    if kind_b == b"REAR":
                        payload = pos
                        pos += rsize
                    if data[pos : pos + 4] != b"RSCE":
                        raise ValueError(f"expected RSCE at 0x{pos:x}, got {data[pos:pos+4]!r}")
                    pos += 4
                    resources.append(Resource(kind, rtype, rsize, rid, payload))

            if token != b"IBSB":
                raise ValueError(f"expected IBSB at 0x{pos-4:x}, got {token!r}")
            ib_size = u32(data, pos)
            pos += 4 + ib_size
            if data[pos : pos + 4] != b"IBSE":
                raise ValueError(f"expected IBSE at 0x{pos:x}, got {data[pos:pos+4]!r}")
            pos += 4
            if data[pos : pos + 4] != b"FRME":
                raise ValueError(f"expected FRME at 0x{pos:x}, got {data[pos:pos+4]!r}")
            pos += 4

            frames.append(Frame(segment.index, segment.kind, frame_index, marker, resources))
    return frames


def first_nal_type(nal: bytes) -> int:
    start = nal.find(b"\x00\x00\x01")
    if start < 0 or start + 3 >= len(nal):
        return -1
    return nal[start + 3] & 0x1F


def parse_decode_pictures(data: bytes, frames: Iterable[Frame]) -> list[DecodePicture]:
    pictures: list[DecodePicture] = []
    for frame in frames:
        msg_res = next((r for r in frame.resources if r.rtype == 0 and r.payload is not None), None)
        bit_res = next((r for r in frame.resources if r.rtype == 256 and r.payload is not None), None)
        if msg_res is None or bit_res is None:
            continue
        if msg_res.size < 256 * 4:
            continue
        message = struct.unpack_from("<256I", data, msg_res.payload)
        if message[1] != 1:  # UVD decode message
            continue
        nal = data[bit_res.payload : bit_res.payload + bit_res.size]
        pictures.append(
            DecodePicture(
                index=len(pictures),
                segment_index=frame.segment_index,
                segment_type=frame.segment_type,
                frame_index=frame.frame_index,
                marker=frame.marker,
                message=message,
                nal=nal,
                nal_type=first_nal_type(nal),
            )
        )
    return pictures


def params_from_uvd_message(v: tuple[int, ...]) -> dict[str, int | bool]:
    sps_flags = v[58]
    pps_flags = v[59]
    b60 = v[60].to_bytes(4, "little")
    b61 = v[61].to_bytes(4, "little")
    b62 = v[62].to_bytes(4, "little")
    b63 = v[63].to_bytes(4, "little")

    # UVD profile enum observed as 0 in some streams that still need High-profile
    # syntax for CPU decode because CABAC/8x8 transform fields are enabled.
    profile_map = {0: 66, 1: 77, 2: 100, 3: 100, 4: 100}
    profile = profile_map.get(v[56], 100)
    if ((pps_flags >> 8) & 1) or (pps_flags & 1):
        profile = 100

    return {
        "profile": profile,
        "level": v[57],
        "chroma": b60[0],
        "bit_depth_luma_minus8": b60[1],
        "bit_depth_chroma_minus8": b60[2],
        "log2fn": b60[3] + 4,
        "poc_type": b61[0],
        "log2poc": b61[1] + 4 if b61[0] == 0 else 4,
        "refs": b61[2],
        "pic_width_mbs_minus1": 119,
        "pic_height_map_units_minus1": 67,
        "frame_mbs_only": (sps_flags >> 2) & 1,
        "direct8x8": sps_flags & 1,
        "delta_pic_order_always_zero_flag": (sps_flags >> 3) & 1,
        "transform8": pps_flags & 1,
        "redundant_pic_cnt_present": (pps_flags >> 1) & 1,
        "constrained_intra_pred": (pps_flags >> 2) & 1,
        "deblocking_filter_control_present": (pps_flags >> 3) & 1,
        "weighted_bipred_idc": (pps_flags >> 4) & 3,
        "weighted_pred": (pps_flags >> 6) & 1,
        "pic_order_present": (pps_flags >> 7) & 1,
        "cabac": (pps_flags >> 8) & 1,
        "num_slice_groups_minus1": b63[0],
        "slice_group_map_type": b63[1],
        "num_ref_idx_l0_default_active_minus1": b63[2],
        "num_ref_idx_l1_default_active_minus1": b63[3],
        "pic_init_qp_minus26": s8(b62[0]),
        "pic_init_qs_minus26": s8(b62[1]),
        "chroma_qp_index_offset": s8(b62[2]),
        "second_chroma_qp_index_offset": s8(b62[3]),
        "crop_bottom": 4,
        "pps_extra": 1 if profile == 100 else 0,
    }


def params_key(params: dict[str, int | bool]) -> tuple[tuple[str, int | bool], ...]:
    return tuple(sorted(params.items()))


def build_annexb(pictures: list[DecodePicture], start: int, count: int) -> bytes:
    selected = pictures[start : start + count]
    if not selected:
        raise ValueError("selected decode range is empty")

    out = bytearray()
    last_key: tuple[tuple[str, int | bool], ...] | None = None
    aud = b"\x00\x00\x01\x09\xf0"

    for i, picture in enumerate(selected):
        params = params_from_uvd_message(picture.message)
        key = params_key(params)
        if i == 0 or picture.nal_type == 5 or key != last_key:
            out += build_sps_pps(params)
            last_key = key
        out += aud
        out += picture.nal
    return bytes(out)


def extract_audio(data: bytes, segments: Iterable[Segment]) -> tuple[bytes, dict[str, object]] | tuple[None, dict[str, object]]:
    audio_segment = next((s for s in segments if s.kind == 0), None)
    if audio_segment is None:
        return None, {"present": False}

    segment = data[audio_segment.payload : audio_segment.end]
    data_tag = segment.find(b"data")
    peak_tag = segment.find(b"PEAK")

    if data_tag >= 0 and data_tag + 8 <= len(segment):
        data_size = u32(segment, data_tag + 4)
        start = audio_segment.payload + data_tag + 8
        end = min(start + data_size, audio_segment.end)
        mode = "chunked_data"
    else:
        start = audio_segment.payload
        end = audio_segment.end
        data_size = audio_segment.size
        mode = "raw_segment"

    frame_bytes = 8 * 4
    end = start + ((end - start) // frame_bytes) * frame_bytes
    audio = data[start:end]
    meta = {
        "present": True,
        "mode": mode,
        "segment_size": audio_segment.size,
        "peak_tag_offset_in_segment": None if peak_tag < 0 else peak_tag,
        "data_tag_offset_in_segment": None if data_tag < 0 else data_tag,
        "declared_data_size": data_size,
        "audio_file_offset": start,
        "audio_size": len(audio),
        "sample_format": "f32le",
        "sample_rate": 48000,
        "channels": 8,
        "duration_seconds": len(audio) / frame_bytes / 48000,
    }
    return audio, meta


def detect_sessions(frames: list[Frame], pictures: list[DecodePicture]) -> list[Session]:
    frame_to_decode: dict[tuple[int, int], int] = {
        (p.segment_index, p.frame_index): p.index for p in pictures
    }
    sessions: list[Session] = []
    current_create: Frame | None = None
    current_decodes: list[int] = []

    for frame in frames:
        msg_res = next((r for r in frame.resources if r.rtype == 0 and r.payload is not None), None)
        op = None
        if msg_res is not None and msg_res.size >= 8:
            # msg_type is dword 1.
            # Safe because payload points into the original file, unavailable here, so infer op via decode map where possible.
            pass
        decode_index = frame_to_decode.get((frame.segment_index, frame.frame_index))
        if decode_index is not None:
            if current_create is None:
                current_create = frame
            current_decodes.append(decode_index)
            continue

        # Lifecycle inference by absence/presence of decode resources is enough for listing sessions in observed files.
        has_msg = any(r.rtype == 0 and r.payload is not None for r in frame.resources)
        has_bitstream = any(r.rtype == 256 for r in frame.resources)
        if has_msg and not has_bitstream:
            if current_create is None:
                current_create = frame
            else:
                sessions.append(
                    Session(
                        index=len(sessions),
                        segment_index=current_create.segment_index,
                        segment_type=current_create.segment_type,
                        create_frame=current_create.frame_index,
                        destroy_frame=frame.frame_index,
                        decode_start=current_decodes[0] if current_decodes else None,
                        decode_count=len(current_decodes),
                    )
                )
                current_create = None
                current_decodes = []

    if current_create is not None:
        sessions.append(
            Session(
                index=len(sessions),
                segment_index=current_create.segment_index,
                segment_type=current_create.segment_type,
                create_frame=current_create.frame_index,
                destroy_frame=None,
                decode_start=current_decodes[0] if current_decodes else None,
                decode_count=len(current_decodes),
            )
        )
    return sessions


def ffprobe_summary(path: Path) -> str:
    try:
        probe = ffmpeg.probe(str(path))
        lines: list[str] = []
        for stream in probe.get("streams", []):
            parts = [
                f"index={stream.get('index')}",
                f"codec_name={stream.get('codec_name')}",
                f"codec_type={stream.get('codec_type')}",
            ]
            if "channels" in stream:
                parts.append(f"channels={stream.get('channels')}")
            if "duration" in stream:
                parts.append(f"duration={stream.get('duration')}")
            lines.append("|".join(parts))
        fmt = probe.get("format", {})
        if "duration" in fmt:
            lines.append(f"duration={fmt.get('duration')}")
        return "\n".join(lines)
    except Exception as exc:
        return f"ffmpeg.probe failed: {exc}"


def encode_reconstructed_h264(
    h264_path: Path,
    video_mp4: Path,
    *,
    fps: float,
    crf: int,
    preset: str,
) -> None:
    """Encode reconstructed Annex-B H.264 into a normal yuv420p MP4."""

    print(f"ffmpeg-python: encode {h264_path} -> {video_mp4}")
    stream = ffmpeg.input(
        str(h264_path),
        format="h264",
        framerate=fps,
        err_detect="ignore_err",
        fflags="+genpts",
        flags2="+showall",
        max_error_rate="1.0",
    )
    stream = ffmpeg.output(
        stream,
        str(video_mp4),
        vcodec="libx264",
        preset=preset,
        crf=crf,
        pix_fmt="yuv420p",
    )
    ffmpeg.run(stream, overwrite_output=True, quiet=False)


def mux_video_and_audio(
    video_mp4: Path,
    audio_raw: Path,
    output: Path,
    *,
    audio_bitrate: str,
    video_duration: float,
    pad_audio: bool,
    audio_advance: float,
    audio_delay: float,
) -> None:
    """Mux the video MP4 with raw f32le 8ch/48k audio into final MP4."""

    print(f"ffmpeg-python: mux {video_mp4} + {audio_raw} -> {output}")
    video_in = ffmpeg.input(str(video_mp4)).video
    audio_in = ffmpeg.input(str(audio_raw), format="f32le", ar=48000, ac=8).audio

    if audio_advance:
        audio_in = audio_in.filter("atrim", start=audio_advance).filter("asetpts", "PTS-STARTPTS")
    if audio_delay:
        audio_in = audio_in.filter("adelay", f"{int(round(audio_delay * 1000))}:all=1")
    if pad_audio:
        audio_in = audio_in.filter("apad", whole_dur=f"{video_duration:.6f}")

    kwargs = {"vcodec": "copy", "acodec": "aac", "audio_bitrate": audio_bitrate}
    if pad_audio:
        kwargs["shortest"] = None
    stream = ffmpeg.output(video_in, audio_in, str(output), **kwargs)
    ffmpeg.run(stream, overwrite_output=True, quiet=False)


def choose_range(args: argparse.Namespace, pictures: list[DecodePicture], sessions: list[Session]) -> tuple[int, int]:
    if args.session is not None:
        if args.session < 0 or args.session >= len(sessions):
            raise ValueError(f"session index {args.session} out of range")
        session = sessions[args.session]
        if session.decode_start is None or session.decode_count == 0:
            raise ValueError(f"session {args.session} has no decode pictures")
        return session.decode_start, session.decode_count

    start = args.start
    if start < 0 or start >= len(pictures):
        raise ValueError(f"start {start} out of range for {len(pictures)} pictures")
    count = len(pictures) - start if args.count is None else args.count
    if count <= 0:
        raise ValueError("count must be positive")
    count = min(count, len(pictures) - start)
    return start, count


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert Xbox bootanim.dat to MP4 with audio")
    parser.add_argument("dat", type=Path, help="input bootanim.dat")
    parser.add_argument("-o", "--output", type=Path, help="output MP4 path; not required with --list")
    parser.add_argument("--fps", type=float, default=30.0, help="video frame rate/render rate, default 30")
    parser.add_argument("--start", type=int, default=0, help="decode picture index to start from")
    parser.add_argument("--count", type=int, default=None, help="number of decode pictures; default to end")
    parser.add_argument("--session", type=int, default=None, help="use a detected decode session instead of --start/--count")
    parser.add_argument("--list", action="store_true", help="list detected segments/sessions and exit")
    parser.add_argument("--no-audio", action="store_true", help="write video-only MP4")
    parser.add_argument("--pad-audio", action="store_true", help="pad short audio with silence to the selected video duration")
    parser.add_argument("--audio-advance", type=float, default=0.0, help="trim this many seconds from start of audio, making audio earlier")
    parser.add_argument("--audio-delay", type=float, default=0.0, help="delay audio by this many seconds")
    parser.add_argument("--crf", type=int, default=18, help="libx264 CRF, default 18")
    parser.add_argument("--preset", default="medium", help="libx264 preset, default medium")
    parser.add_argument("--audio-bitrate", default="512k", help="AAC bitrate, default 512k")
    parser.add_argument("--workdir", type=Path, default=None, help="temporary/work directory")
    parser.add_argument("--keep-temp", action="store_true", help="keep intermediate .h264/.raw/.mp4 files")
    args = parser.parse_args()

    if not args.dat.is_file():
        raise FileNotFoundError(args.dat)
    if not args.list and args.output is None:
        parser.error("-o/--output is required unless --list is used")
    if args.session is not None and (args.start != 0 or args.count is not None):
        raise ValueError("use either --session or --start/--count, not both")
    if args.audio_advance < 0 or args.audio_delay < 0:
        raise ValueError("audio advance/delay values must be non-negative")

    data = args.dat.read_bytes()
    segments = parse_segments(data)
    frames = parse_frames(data, segments)
    pictures = parse_decode_pictures(data, frames)
    if not pictures:
        raise ValueError("no decode pictures found")
    sessions = detect_sessions(frames, pictures)

    if args.list:
        print("Segments:")
        for segment in segments:
            print(
                f"  #{segment.index}: type={segment.kind} offset=0x{segment.offset:x} "
                f"payload=0x{segment.payload:x} size=0x{segment.size:x}"
            )
        print("Decode pictures:", len(pictures))
        print("IDRs:", [p.index for p in pictures if p.nal_type == 5])
        print("Sessions:")
        for session in sessions:
            print(
                f"  #{session.index}: segment={session.segment_index}/type{session.segment_type} "
                f"create_frame={session.create_frame} destroy_frame={session.destroy_frame} "
                f"decode_start={session.decode_start} decode_count={session.decode_count}"
            )
        audio, audio_meta = extract_audio(data, segments)
        print("Audio:", json.dumps(audio_meta, indent=2))
        return 0

    start, count = choose_range(args, pictures, sessions)
    video_duration = count / args.fps

    assert args.output is not None
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    workdir = args.workdir or output.with_suffix("").with_name(output.stem + "_work")
    if workdir.exists() and not args.keep_temp:
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    h264_path = workdir / "video_reconstructed.h264"
    video_mp4 = workdir / "video_only.mp4"
    audio_raw = workdir / "audio_f32le_8ch_48k.raw"
    meta_path = workdir / "conversion_meta.json"

    h264_path.write_bytes(build_annexb(pictures, start, count))

    encode_reconstructed_h264(
        h264_path,
        video_mp4,
        fps=args.fps,
        crf=args.crf,
        preset=args.preset,
    )

    audio, audio_meta = extract_audio(data, segments)
    if args.no_audio or audio is None:
        shutil.copy2(video_mp4, output)
    else:
        audio_raw.write_bytes(audio)
        mux_video_and_audio(
            video_mp4,
            audio_raw,
            output,
            audio_bitrate=args.audio_bitrate,
            video_duration=video_duration,
            pad_audio=args.pad_audio,
            audio_advance=args.audio_advance,
            audio_delay=args.audio_delay,
        )

    meta = {
        "input": str(args.dat.resolve()),
        "output": str(output),
        "fps": args.fps,
        "decode_picture_count_total": len(pictures),
        "selected_start": start,
        "selected_count": count,
        "selected_duration_seconds": video_duration,
        "idrs": [p.index for p in pictures if p.nal_type == 5],
        "segments": [segment.__dict__ for segment in segments],
        "sessions": [session.__dict__ for session in sessions],
        "audio": audio_meta,
        "ffmpeg_backend": "ffmpeg-python",
        "ffprobe": ffprobe_summary(output),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    print("\nWrote:", output)
    print("Workdir:", workdir)
    print("ffprobe:")
    print(meta["ffprobe"])

    if not args.keep_temp:
        # Keep metadata next to the output, remove bulky intermediates.
        sidecar = output.with_suffix(output.suffix + ".json")
        sidecar.write_text(meta_path.read_text())
        shutil.rmtree(workdir)
        print("Metadata:", sidecar)
    else:
        print("Metadata:", meta_path)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        if exc.__class__.__name__ == "Error" and hasattr(exc, "stderr"):
            stderr = exc.stderr.decode("utf-8", "replace") if exc.stderr else str(exc)
            print(stderr, file=sys.stderr)
            raise SystemExit(1)
        raise
