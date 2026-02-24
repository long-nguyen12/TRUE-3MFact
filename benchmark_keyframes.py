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
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Output CSV path; default uses DATASET_CONFIG root",
    )
    parser.add_argument(
        "--keep-outputs",
        action="store_true",
        help="Keep extracted benchmark keyframes on disk",
    )
    args = parser.parse_args()

    csv_path = benchmark_keyframe_extraction_times(
        output_csv_path=args.output_csv,
        split=args.split,
        video_dir=args.video_dir,
        annotation_path=args.annotation_path,
        max_videos=args.max_videos,
        keyframes_per_video=args.keyframes_per_video,
        cleanup_outputs=not args.keep_outputs,
    )
    print(f"Benchmark CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
