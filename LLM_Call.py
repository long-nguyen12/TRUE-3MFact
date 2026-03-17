import json
from platform import processor
import regex
import requests
from local_llm.llms import initialize, run_llama, run_deepseek, run_qwen, run_qwen_text
from Config import MODEL_CONFIG, DATASET_CONFIG
from qwen_vl_utils import process_vision_info
from pathlib import Path


def gpt_mini_analysis(prompt):
    url = "https://cn2us02.opapi.win/v1/chat/completions"

    api_key = MODEL_CONFIG["llm"]["model_key"]

    payload = json.dumps(
        {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        }
    )
    headers = {
        "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
        "Content-Type": "application/json",
        "Authorization": "Bearer " + api_key,
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    res = response.json()
    res_content = res["choices"][0]["message"]["content"]
    return res_content


def local_llm_analysis(model, tokenizer, prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    response = run_qwen_text(
        model, tokenizer, prompt, system_prompt="You are a helpful assistant."
    )

    return response


# 提取完整的JSON数据
def extract_complete_json(response_text):
    json_pattern = r"(\{(?:[^{}]|(?1))*\})"
    matches = regex.findall(json_pattern, response_text)
    if matches:
        try:
            for match in matches:
                json_data = json.loads(match)
                return json_data
        except json.JSONDecodeError as e:
            print(f"JSON Parsing Error: {e}")
    return None


import os
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from decord import VideoReader, cpu


def analysis_video_minicpm(video_path, question):

    model_path = MODEL_CONFIG["video_lmm"]["local_model_path"]

    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    def encode_video(video_path):
        MAX_NUM_FRAMES = 64  # if cuda OOM set a smaller number
        vr = VideoReader(video_path, ctx=cpu(0))
        sample_fps = round(vr.get_avg_fps() / 1)  # FPS
        frame_idx = [i for i in range(0, len(vr), sample_fps)]
        if len(frame_idx) > MAX_NUM_FRAMES:
            frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
        frames = vr.get_batch(frame_idx).asnumpy()
        frames = [Image.fromarray(v.astype("uint8")) for v in frames]
        print(f"Video: {video_path}, num frames: {len(frames)}")
        return frames

    try:
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
        )
        model = model.eval().cuda()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        frames = encode_video(video_path)
        msgs = [{"role": "user", "content": frames + [question]}]

        params = {
            "use_image_id": False,
            "max_slice_nums": 1,  # use 1 if cuda OOM and video resolution > 448*448
        }

        answer = model.chat(image=None, msgs=msgs, tokenizer=tokenizer, **params)

        return answer

    except Exception as e:
        print(f"An error occurred while processing the video: {str(e)}")
        return None
    finally:
        if "model" in locals():
            del model
            torch.cuda.empty_cache()


def analysis_video_qwenvl(video_path, question):

    # Allow callers to pass an absolute path directly.
    try_path = Path(str(video_path))
    if try_path.is_absolute() and try_path.exists():
        video_path = str(try_path)
    else:
        root_dir = Path(DATASET_CONFIG.get("root_dir", ""))
        if not root_dir.is_absolute():
            root_dir = (Path(__file__).resolve().parent / root_dir).resolve()

        video_dirs_cfg = DATASET_CONFIG.get("video_dir", {}) or {}
        candidate_dirs = []
        # Prefer train/val first, then test, but also accept any other configured splits.
        for split in ("train_val", "test"):
            v = video_dirs_cfg.get(split)
            if v:
                candidate_dirs.append(root_dir / v)
        for split, v in video_dirs_cfg.items():
            if split in ("train_val", "test"):
                continue
            if v:
                candidate_dirs.append(root_dir / v)

        rel_video = str(video_path).lstrip("/\\")
        tried = []
        found_path = None
        for vdir in candidate_dirs:
            candidate = (Path(vdir) / rel_video)
            tried.append(str(candidate))
            if candidate.exists():
                found_path = str(candidate)
                break

        if found_path is None:
            print(
                f"Error: Video '{video_path}' not found under dataset root '{root_dir}'. "
                f"Tried: {tried}"
            )
            return None

        video_path = found_path

    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    def encode_video(video_path):
        MAX_NUM_FRAMES = 64  # if cuda OOM set a smaller number
        vr = VideoReader(video_path, ctx=cpu(0))
        sample_fps = round(vr.get_avg_fps() / 1)  # FPS
        frame_idx = [i for i in range(0, len(vr), sample_fps)]
        if len(frame_idx) > MAX_NUM_FRAMES:
            frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
        frames = vr.get_batch(frame_idx).asnumpy()
        frames = [Image.fromarray(v.astype("uint8")) for v in frames]
        print(f"Video: {video_path}, num frames: {len(frames)}")
        return frames

    try:
        processor, model = initialize("Qwen", "2.5B")

        frames = encode_video(video_path)

        msgs = [
            {
                "role": "user",
                "content": (
                    [{"type": "image", "image": f} for f in frames]
                    + [{"type": "text", "text": question}]
                ),
            }
        ]

        text = processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        try:
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                msgs, return_video_kwargs=True
            )
        except TypeError:
            image_inputs, video_inputs = process_vision_info(msgs)
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
            **inputs, max_new_tokens=512, temperature=0.7, do_sample=True
        )
        outputs_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, outputs)
        ]
        response = processor.batch_decode(
            outputs_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return response

    except Exception as e:
        print(f"An error occurred while processing the video: {str(e)}")
        return None
    finally:
        if "model" in locals():
            del model
            torch.cuda.empty_cache()
