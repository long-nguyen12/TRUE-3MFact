import argparse

from VideoDescriptor import benchmark_keyframe_extraction_times


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark katna and clip-chunk keyframe extraction times."
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

    rows = benchmark_keyframe_extraction_times(
        split=args.split,
        video_dir=args.video_dir,
        annotation_path=args.annotation_path,
        max_videos=args.max_videos,
        keyframes_per_video=args.keyframes_per_video,
    )
    print(f"Total benchmark rows: {len(rows)}")


if __name__ == "__main__":
    main()
