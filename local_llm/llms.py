# Set GPU device
import os
import time
import warnings
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from PIL import Image
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

project_root = Path(__file__).resolve().parents[1]
local_llm_root = Path(__file__).resolve().parent

if load_dotenv is not None:
    # Load .env from project root or local_llm folder if present.
    for dotenv_path in (project_root / ".env", local_llm_root / ".env"):
        if dotenv_path.exists():
            load_dotenv(dotenv_path, override=False)
else:
    print(
        "python-dotenv is not installed; .env files will not be auto-loaded. "
        "Set HF_TOKEN as an environment variable instead."
    )

hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    print(
        "No Hugging Face token found. Set HF_TOKEN in your environment or .env "
        "to access gated models."
    )

warnings.filterwarnings("ignore")


def initialize(model_name, model_size):
    if model_name == "LLaMA":
        print(f"Selected model: LLaMA {model_size}")
        model_map = {
            "1B": "meta-llama/Llama-3.2-1B-Instruct",
            "3B": "meta-llama/Llama-3.2-3B-Instruct",
            "8B": "meta-llama/Llama-3.1-8B-Instruct",
        }
        name = model_map.get(model_size, model_map["1B"])

        print(f"Loading LLaMA {model_size}...")
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model = model.eval()
        return tokenizer, model
    elif model_name == "DeepSeek":
        print(f"Selected model: DeepSeek R1 {model_size}")

        model_map = {
            "7B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "14B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        }

        name = model_map.get(model_size, model_map["7B"])

        print(f"Loading DeepSeek {model_size}...")
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = model.eval()
        return tokenizer, model
    elif model_name == "Qwen":
        print(f"Selected model: Qwen {model_size}")
        model_map = {
            "3B": "Qwen/Qwen2.5-VL-3B-Instruct",
            "7B": "Qwen/Qwen2.5-VL-7B-Instruct",
        }
        name = model_map.get(model_size, model_map["3B"])

        print(f"Loading Qwen {model_size}...")
        from transformers import Qwen2_5_VLForConditionalGeneration

        processor = AutoProcessor.from_pretrained(name)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            name,
            torch_dtype="auto",
            device_map="auto",
        )
        model = model.eval()
        return processor, model


def run_llama(model, tokenizer, prompt, model_size="1B"):
    messages = [{"role": "user", "content": prompt}]

    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    print("Generating response...")
    start_time = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
    )
    end_time = time.time() - start_time
    print(f"Generation completed in {end_time:.2f} seconds.")

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract just the assistant's response
    if "assistant" in response:
        parts = response.split("assistant")
        if len(parts) > 1:
            response = parts[-1].strip()
            # Remove leading colons or newlines
            response = response.lstrip(":\n").strip()

    return response


def run_deepseek(model, tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]

    print("Generating response...")

    try:
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        outputs = model.generate(
            **inputs, max_new_tokens=512, temperature=0.7, do_sample=True
        )
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
        )
        return response
    except Exception as e:
        print(f"Chat template failed, falling back to manual formatting: {e}")
        return None


def run_llama_quantized(prompt):
    try:
        from transformers import BitsAndBytesConfig
    except ImportError:
        print(
            "Error: bitsandbytes not installed. Install with: pip install bitsandbytes"
        )
        return None

    print("Loading LLaMA 8B (4-bit quantized)...")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
    )

    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    print("Generating response...")
    outputs = model.generate(**inputs, max_new_tokens=512)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract assistant's response
    if "assistant" in response:
        parts = response.split("assistant")
        if len(parts) > 1:
            response = parts[-1].strip().lstrip(":\n").strip()

    return response


def run_qwen(model, processor, prompt, video_path=None, frames=None):
    print("Generating response...")

    if video_path and frames:
        raise ValueError("Provide either video_path or frames, not both.")

    try:
        from qwen_vl_utils import process_vision_info
    except Exception as e:
        raise ImportError(
            "qwen-vl-utils is required for video inputs. Install with: "
            "pip install qwen-vl-utils[decord]==0.0.8"
        ) from e

    if frames is not None:
        if not isinstance(frames, (list, tuple)) or len(frames) == 0:
            raise ValueError("frames must be a non-empty list of images.")
        normalized_frames = []
        for frame in frames:
            if isinstance(frame, Image.Image):
                normalized_frames.append(frame)
            else:
                normalized_frames.append(Image.fromarray(frame).convert("RGB"))

        messages = [
            {
                "role": "user",
                "content": (
                    [{"type": "image", "image": f} for f in normalized_frames]
                    + [{"type": "text", "text": prompt}]
                ),
            }
        ]
    else:
        if not video_path:
            raise ValueError("video_path is required when frames are not provided.")

        video_ref = video_path
        if os.path.exists(video_path):
            video_ref = Path(video_path).resolve().as_uri()
        elif not video_path.startswith(("http://", "https://", "file://")):
            raise FileNotFoundError(f"Video not found: {video_path}")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_ref},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

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
        **inputs, max_new_tokens=512, temperature=0.7, do_sample=True
    )
    outputs_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, outputs)
    ]
    response = processor.batch_decode(
        outputs_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return response


def check_system():
    """Check system capabilities."""
    print("=" * 70)
    print("SYSTEM CHECK")
    print("=" * 70)

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}")
            print(f"    Total memory: {props.total_memory / 1024**3:.2f} GB")
    else:
        print("Running on CPU (will be slower)")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    check_system()

    print("\n" + "=" * 70)
    print("EXAMPLE 1: Qwen2.5-VL-3B-Instruct (Video)")
    print("=" * 70)
    processor, model = initialize("Qwen", "3B")

    try:
        video_path = (
            "C:/Users/hnguyen/Documents/PhD/Code/TRUE-3MFact/local_llm/1942500.mp4"
        )
        prompt = "Describe the key events in this video."
        response = run_qwen(model, processor, prompt, video_path)
        print(f"\nResponse:\n{response}\n")
    except Exception as e:
        print(f"Error running Qwen2.5-VL: {e}")

    # print("\n" + "=" * 70)
    # print("EXAMPLE 2: DeepSeek R1 7B")
    # print("=" * 70)

    # try:
    #     response = run_deepseek(
    #         "Write a paragraph about the benefits of using local LLMs.",
    #         model_variant="7B",
    #     )
    #     print(f"\nResponse:\n{response}\n")
    # except Exception as e:
    #     print(f"Error running DeepSeek: {e}")
