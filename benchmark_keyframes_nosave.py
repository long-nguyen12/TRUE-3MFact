import argparse
import logging
import os
import time
from pathlib import Path

from Config import DATASET_CONFIG, VIDEO_DESCRIPTOR_CONFIG


class _MemoryWriter:
    def __init__(self):
        self.frame_count = 0

    def write(self, filepath, data):
        self.frame_count = len(data or [])


def _resolve_video_file(video_dir, video_id):
    for ext in (".mp4", ".mkv", ".avi", ".mov"):
        candidate = os.path.join(video_dir, f"{video_id}{ext}")
        if os.path.exists(candidate):
            return candidate
    return None


def _load_video_ids(annotation_path, video_dir):
    if annotation_path and os.path.exists(annotation_path):
        with open(annotation_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    ids = []
    for p in sorted(Path(video_dir).glob("*")):
        if p.suffix.lower() in {".mp4", ".mkv", ".avi", ".mov"}:
            ids.append(p.stem)
    return ids


def katna_keyframes_extraction_nosave(video_file_path, no_of_frames_to_returned):
    try:
        from Katna.video import Video
    except Exception as e:
        raise ImportError(
            "Katna is required for katna_keyframes_extraction_nosave."
        ) from e

    vd = Video()
    memory_writer = _MemoryWriter()
    vd.extract_video_keyframes(
        no_of_frames=no_of_frames_to_returned,
        file_path=video_file_path,
        writer=memory_writer,
    )
    return memory_writer.frame_count


def clip_chunk_keyframes_extraction_nosave(
    video_file_path,
    chunk_count=10,
    samples_per_chunk=8,
    model_name="openai/clip-vit-base-patch32",
    spectral_clusters=2,
):
    try:
        import cv2
        import torch
        from ClusterFrame.video import (
            _load_clip_vision,
            _read_frames_at_indices,
            _select_representative_frame,
            _select_representative_frame_spectral,
        )
    except Exception as e:
        raise ImportError(
            "opencv-python, torch, transformers, and ClusterFrame.video dependencies "
            "are required for clip_chunk_keyframes_extraction_nosave."
        ) from e

    cap = cv2.VideoCapture(str(video_file_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_file_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"Video has no frames: {video_file_path}")

    chunk_count = min(chunk_count, total_frames)
    chunk_size = total_frames / chunk_count

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor, model = _load_clip_vision(model_name)
    model.to(device)

    selected = 0
    for chunk_idx in range(chunk_count):
        start = int(chunk_idx * chunk_size)
        end = int(min(total_frames, (chunk_idx + 1) * chunk_size))
        if end <= start:
            continue

        if samples_per_chunk <= 1 or (end - start) <= 1:
            indices = [start]
        else:
            step = max(1, (end - start) // samples_per_chunk)
            indices = list(range(start, end, step))[:samples_per_chunk]

        frames = _read_frames_at_indices(cap, indices)
        embeddings = _select_representative_frame(frames, processor, model, device)
        if embeddings is None:
            continue

        best_frame = _select_representative_frame_spectral(
            frames, embeddings, spectral_clusters
        )
        if best_frame is None:
            continue

        selected += 1

    cap.release()
    return selected


def benchmark_keyframe_extraction_times_nosave(
    split="test",
    video_dir=None,
    annotation_path=None,
    max_videos=None,
    keyframes_per_video=None,
):
    root_dir = DATASET_CONFIG["root_dir"]
    if video_dir is None:
        video_dir = os.path.join(root_dir, DATASET_CONFIG["video_dir"][split])
    if annotation_path is None:
        annotation_path = os.path.join(root_dir, DATASET_CONFIG["annotation"][split])
    if keyframes_per_video is None:
        keyframes_per_video = VIDEO_DESCRIPTOR_CONFIG.get("keyframes_per_video", 7)

    if not os.path.isdir(video_dir):
        raise FileNotFoundError(f"Video directory does not exist: {video_dir}")

    video_ids = _load_video_ids(annotation_path, video_dir)
    if max_videos is not None:
        video_ids = video_ids[: int(max_videos)]

    rows = []
    extractors = ("katna", "clip_chunk")

    for idx, video_id in enumerate(video_ids, start=1):
        logging.info(
            "Benchmarking keyframe extractors for video %s (%d/%d)",
            video_id,
            idx,
            len(video_ids),
        )
        video_file = _resolve_video_file(video_dir, video_id)
        if not video_file:
            rows.append(
                {
                    "video_id": video_id,
                    "video_file": "",
                    "extractor": "",
                    "elapsed_seconds": 0.0,
                    "keyframes_requested": keyframes_per_video,
                    "keyframes_generated": 0,
                    "status": "missing_video",
                    "error": "Video file not found",
                }
            )
            continue

        for extractor in extractors:
            generated = 0
            status = "ok"
            error = ""
            start = time.perf_counter()
            try:
                if extractor == "katna":
                    generated = katna_keyframes_extraction_nosave(
                        video_file, keyframes_per_video
                    )
                else:
                    generated = clip_chunk_keyframes_extraction_nosave(
                        video_file_path=video_file,
                        chunk_count=keyframes_per_video,
                    )
            except Exception as e:
                status = "error"
                error = str(e)
                logging.error(
                    "Benchmark failed for %s (%s): %s", video_id, extractor, e
                )
            elapsed = time.perf_counter() - start
            rows.append(
                {
                    "video_id": video_id,
                    "video_file": video_file,
                    "extractor": extractor,
                    "elapsed_seconds": round(elapsed, 6),
                    "keyframes_requested": keyframes_per_video,
                    "keyframes_generated": generated,
                    "status": status,
                    "error": error,
                }
            )

    return rows


def print_rows(rows):
    fieldnames = [
        "video_id",
        "video_file",
        "extractor",
        "elapsed_seconds",
        "keyframes_requested",
        "keyframes_generated",
        "status",
        "error",
    ]
    print("\t".join(fieldnames))
    for row in rows:
        values = []
        for field in fieldnames:
            value = str(row.get(field, ""))
            values.append(value.replace("\t", " ").replace("\n", " "))
        print("\t".join(values))


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark keyframe extraction times without saving outputs."
    )
    parser.add_argument("--split", default="test", help="Dataset split key, e.g. test")
    parser.add_argument(
        "--video-dir",
        default=None,
        help="Optional absolute/relative path to a video folder",
    )
    parser.add_argument(
        "--annotation-path",
        default=None,
        help="Optional path to annotation file with video IDs",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Optional cap on number of videos to benchmark",
    )
    parser.add_argument(
        "--keyframes-per-video",
        type=int,
        default=None,
        help="Number of keyframes requested from each extractor",
    )
    args = parser.parse_args()

    rows = benchmark_keyframe_extraction_times_nosave(
        split=args.split,
        video_dir=args.video_dir,
        annotation_path=args.annotation_path,
        max_videos=args.max_videos,
        keyframes_per_video=args.keyframes_per_video,
    )
    print_rows(rows)
    print(f"Total benchmark rows: {len(rows)}")


if __name__ == "__main__":
    main()
