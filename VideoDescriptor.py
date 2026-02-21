import glob
import json
import logging
import os
import time
import traceback
from datetime import datetime

import pytz
from PIL import Image
from decord import VideoReader, cpu

from Config import DATASET_CONFIG, VIDEO_DESCRIPTOR_CONFIG

MAX_NUM_FRAMES = 64  # if cuda OOM set a smaller number

qwen_model = None
qwen_processor = None


def set_qwen_model(processor, model):
    """Inject preloaded Qwen processor/model for video-image generation."""
    global qwen_processor, qwen_model
    qwen_processor = processor
    qwen_model = model


def uniform_sample(l, n):
    gap = len(l) / n
    idxs = [int(i * gap + gap / 2) for i in range(n)]
    return [l[i] for i in idxs]


def encode_video(video_path):
    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = max(1, round(vr.get_avg_fps() / 1))  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx) > MAX_NUM_FRAMES:
        frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype("uint8")) for v in frames]
    return frames


def _ensure_qwen_loaded():
    global qwen_model, qwen_processor

    if qwen_model is None or qwen_processor is None:
        raise RuntimeError(
            "Qwen model is not set. Call set_qwen_model(...) from main.py "
            "before process_folder_videos_with_logging()."
        )

    return qwen_processor, qwen_model


def process_image(image_path, question):
    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]
    return _generate_qwen_response(messages)


def process_video(video_path, question):
    frames = encode_video(video_path)
    messages = [
        {
            "role": "user",
            "content": (
                [{"type": "image", "image": frame} for frame in frames]
                + [{"type": "text", "text": question}]
            ),
        }
    ]
    return _generate_qwen_response(messages)


def _generate_qwen_response(messages):
    try:
        from qwen_vl_utils import process_vision_info
    except Exception as e:
        raise ImportError(
            "qwen-vl-utils is required for Qwen2.5-VL inference. "
            "Install with: pip install qwen-vl-utils[decord]==0.0.8"
        ) from e

    processor, model = _ensure_qwen_loaded()

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    try:
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )
    except TypeError:
        image_inputs, video_inputs = process_vision_info(messages)
        video_kwargs = {}

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.2,
        do_sample=True,
    )
    outputs_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, outputs)
    ]
    return processor.batch_decode(
        outputs_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

from Katna.video import Video
from Katna.writer import KeyFrameDiskWriter
from ClusterFrame.video import clip_chunk_keyframes_extraction


def katna_keyframes_extraction(video_file_path, no_of_frames_to_returned):
    vd = Video()

    video_dir, video_name = os.path.split(video_file_path)
    video_base_name = os.path.splitext(video_name)[0]
    target_path = os.path.join(video_dir, video_base_name)

    if (
        os.path.exists(target_path)
        and len([f for f in os.listdir(target_path) if f.endswith(".jpeg")])
        >= no_of_frames_to_returned
    ):
        logging.info(f"Keyframes already extracted and present in {target_path}")
        return target_path

    disk_writer = KeyFrameDiskWriter(location=target_path)

    logging.info(f"Input video file path = {video_file_path}")

    vd.extract_video_keyframes(
        no_of_frames=no_of_frames_to_returned,
        file_path=video_file_path,
        writer=disk_writer,
    )
    logging.info(f"video {video_base_name}：Keyframes extracted successfully")

    reorder_and_rename_images(target_path)

    return target_path


def reorder_and_rename_images(directory_path):
    images = sorted(
        glob.glob(os.path.join(directory_path, "*.jpeg")), key=os.path.getmtime
    )

    for i, image_path in enumerate(images, start=1):
        new_name = os.path.join(directory_path, f"{i}.jpeg")
        os.rename(image_path, new_name)

    logging.info("Images have been renamed successfully.")


def gpt_summary(video_llava_answer, key_frame_highlights):
    prompt = f"""
Your task is to generate a coherent, logically structured, and accurate video description.
Requirements:
1. Strictly use the provided information only. Do not speculate or add facts.
2. Integrate the overall video analysis with details from key frames.
3. Keep accuracy as the highest priority.
4. Keep the narrative fluent and logically clear.
5. Target 100-500 words.

Input:
- Overall video content: {video_llava_answer}
- Key frame highlights: {key_frame_highlights}

Output:
Produce one cohesive, objective description that someone can understand without seeing the video.
"""
    return _generate_qwen_text(prompt)


def _generate_qwen_text(prompt):
    processor, model = _ensure_qwen_loaded()
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.2,
        do_sample=True,
    )
    outputs_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, outputs)
    ]
    return processor.batch_decode(
        outputs_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


def pipe_prompt_2_only_accuracy(video_file_path, image_folder_path):

    video_prompt = "Describe the key events in the video chronologically, including time, location, and participants. Focus on observable actions and processes, avoiding speculation. Provide a brief and accurate summary of the video content."

    image_prompt = "Accurately describe this image, including visible subjects, their actions, and the main scene or background information. Extract and describe any visible text. Only describe what can be directly observed in the image. Do not speculate on uncertain details, only describe the most certain elements."

    all_answers = {"Video": {}, "Image": {}, "LLM": {}}

    video_summary_answer = process_video(video_file_path, video_prompt)
    all_answers["Video"]["question"] = video_prompt
    all_answers["Video"]["answer"] = video_summary_answer

    all_answers["Image"]["question"] = image_prompt

    image_files = [
        f
        for f in os.listdir(image_folder_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    sorted_image_files = sorted(image_files)

    for image_path in sorted_image_files:
        image_name = os.path.join(image_folder_path, image_path)
        answer = process_image(image_name, image_prompt)
        all_answers["Image"][image_path] = {"answer": answer}

    key_frame_summary = "; ".join(
        [
            f"{image_file}: {details['answer']}"
            for image_file, details in all_answers["Image"].items()
            if "answer" in details and details["answer"]
        ]
    )

    gpt_summary_answer = gpt_summary(video_summary_answer, key_frame_summary)
    all_answers["LLM"]["answer"] = gpt_summary_answer
    
    return all_answers


def process_folder_videos():
    root_dir = DATASET_CONFIG["root_dir"]
    test_annotation = os.path.join(root_dir, DATASET_CONFIG["annotation"]["test"])
    test_data_dir = os.path.join(root_dir, DATASET_CONFIG["data_dir"]["test"])
    test_video_dir = os.path.join(root_dir, DATASET_CONFIG["video_dir"]["test"])

    test_output_dir = os.path.join(root_dir, DATASET_CONFIG["output_dir"]["test"])

    test_VD_result_dir = test_output_dir
    if not os.path.exists(test_VD_result_dir):
        os.makedirs(test_VD_result_dir)

    with open(test_annotation, "r") as f:
        video_ids = [line.strip() for line in f if line.strip()]

    for i, video_id in enumerate(video_ids):
        try:
            logging.info(f"Processing video {i+1}/{len(video_ids)}: {video_id}")
            start_time = time.time()

            video_file = None
            for ext in [".mp4", ".mkv"]:
                potential_file = os.path.join(test_video_dir, f"{video_id}{ext}")
                if os.path.exists(potential_file):
                    video_file = potential_file
                    break

            if not video_file:
                logging.warning(f"No video file found for ID: {video_id}")
                continue

            data_folder = os.path.join(test_video_dir, str(video_id))

            if not os.path.exists(data_folder):
                os.makedirs(data_folder)
            original_output_path = os.path.join(data_folder, f"{video_id}.json")

            logging.info(f"Extracting keyframes for video: {video_id}")
            try:
                print(f"Extracting keyframes for video: {video_id}")
                no_of_frames_to_returned = VIDEO_DESCRIPTOR_CONFIG.get(
                    "keyframes_per_video", 7
                )
                extractor = VIDEO_DESCRIPTOR_CONFIG.get(
                    "keyframe_extractor", "clip_chunk"
                ).lower()

                if extractor == "katna":
                    keyframes_folder = katna_keyframes_extraction(
                        video_file, no_of_frames_to_returned
                    )
                else:
                    keyframes_folder = clip_chunk_keyframes_extraction(
                        video_file_path=video_file, chunk_count=no_of_frames_to_returned
                    )
                logging.info(f"Keyframes extracted successfully for video: {video_id}")
            except Exception as e:
                logging.error(
                    f"Failed to extract keyframes for video {video_id}: {str(e)}"
                )
                continue

            descriptor_result = pipe_prompt_2_only_accuracy(
                video_file, keyframes_folder
            )

            with open(original_output_path, "w", encoding="utf-8-sig") as f:
                json.dump(
                    {"Zero-Shot Detailed Inquiry Prompt": descriptor_result},
                    f,
                    indent=4,
                )

            original_json_path = os.path.join(test_data_dir, f"{video_id}.json")
            try:
                with open(original_json_path, "r", encoding="utf-8-sig") as f:
                    original_data = json.load(f)

                video_info = original_data.get("video_information", {})

                video_info["Video_descriptor"] = descriptor_result["LLM"]["answer"]

                final_result = {
                    "claim": original_data.get("claim", ""),
                    "Video_information": video_info,
                }

                result_json_path = os.path.join(test_VD_result_dir, f"{video_id}.json")
                with open(result_json_path, "w", encoding="utf-8") as f:
                    json.dump(final_result, f, indent=4, ensure_ascii=False)

                end_time = time.time()
                logging.info(
                    f"Time taken for video {video_id}: {(end_time - start_time)/60:.2f} minutes"
                )
                logging.info(
                    f"----------------------------- Analysis for {video_id} completed successfully -----------------------------"
                )

            except Exception as e:
                logging.error(f"Error processing JSON for video {video_id}: {str(e)}")
                continue

        except Exception as e:
            logging.error(f"Error processing video {video_id}: {str(e)}")
            continue


class TimeZoneFormatter(logging.Formatter):
    def converter(self, timestamp):
        dt = datetime.fromtimestamp(timestamp, tz=pytz.UTC)
        return dt.astimezone(pytz.timezone("Europe/Paris"))

    def formatTime(self, record, datefmt=None):
        dt = self.converter(record.created)
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    def format(self, record):
        record.asctime = self.formatTime(record)
        return super().format(record)


logger = logging.getLogger()
logger.setLevel(logging.INFO)


def process_folder_videos_with_logging():
    """
    Wrapper function for process_folder_videos that sets up logging
    and generates a separate log file for this function's operations.
    """
    root_dir = DATASET_CONFIG["root_dir"]
    test_video_dir = os.path.join(root_dir, DATASET_CONFIG["video_dir"]["test"])

    log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_name = f"process_folder_videos_{log_timestamp}.log"
    log_file_path = os.path.join(test_video_dir, log_file_name)

    os.makedirs(test_video_dir, exist_ok=True)

    file_handler = logging.FileHandler(log_file_path)
    formatter = TimeZoneFormatter(
        "%(asctime)s - %(threadName)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    try:
        logger.info(f"Starting process_folder_videos at {log_timestamp}")
        logger.info(f"Log file: {log_file_path}")

        process_folder_videos()

        logger.info("process_folder_videos completed successfully")

    except Exception as e:
        logger.error(f"Error in process_folder_videos: {e}")
        logger.error(traceback.format_exc())
        raise

    finally:
        logger.removeHandler(file_handler)

        file_handler.close()
