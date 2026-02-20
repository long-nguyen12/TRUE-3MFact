import logging
import os
from pathlib import Path

import cv2
import torch
from transformers import CLIPImageProcessor, CLIPVisionModel
from sklearn.cluster import SpectralClustering


logger = logging.getLogger()
logger.setLevel(logging.INFO)


def reorder_and_rename_images(directory_path):
    directory = Path(directory_path)
    images = sorted(directory.glob("*.jpeg"), key=lambda p: p.stat().st_mtime)
    if not images:
        logging.info("No JPEG images found to rename in %s", directory_path)
        return

    # Use temporary names to avoid collisions when target names already exist.
    tmp_paths = []
    for idx, image_path in enumerate(images, start=1):
        tmp_path = image_path.with_name(f"__tmp_{idx:06d}.jpeg")
        image_path.rename(tmp_path)
        tmp_paths.append(tmp_path)

    for idx, tmp_path in enumerate(tmp_paths, start=1):
        tmp_path.rename(directory / f"{idx}.jpeg")

    logging.info("Images have been renamed successfully.")


def _load_clip_vision(model_name):
    processor = CLIPImageProcessor.from_pretrained(model_name)
    model = CLIPVisionModel.from_pretrained(model_name)
    model.eval()
    return processor, model


def _read_frames_at_indices(cap, indices):
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    return frames


def _select_representative_frame(frames, processor, model, device):
    if not frames:
        return None
    inputs = processor(images=frames, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
    return embeddings


def _select_representative_frame_spectral(frames, embeddings, n_clusters):
    if not frames:
        return None
    if embeddings.shape[0] == 1:
        return frames[0]

    n_clusters = max(2, min(int(n_clusters), int(embeddings.shape[0])))
    n_samples = int(embeddings.shape[0])
    n_neighbors = max(1, min(10, n_samples - 1))
    labels = SpectralClustering(
        n_clusters=n_clusters,
        affinity="nearest_neighbors",
        n_neighbors=n_neighbors,
        assign_labels="kmeans",
        random_state=0,
    ).fit_predict(embeddings.cpu().numpy())

    largest_label = max(set(labels), key=lambda l: (labels == l).sum())
    idxs = [i for i, l in enumerate(labels) if l == largest_label]
    cluster_emb = embeddings[idxs]
    centroid = cluster_emb.mean(dim=0, keepdim=True)
    distances = torch.cdist(cluster_emb, centroid).squeeze(1)
    best_local = int(torch.argmin(distances).item())
    return frames[idxs[best_local]]


def clip_chunk_keyframes_extraction(
    video_file_path,
    chunk_count=10,
    samples_per_chunk=8,
    model_name="openai/clip-vit-base-patch32",
    spectral_clusters=2,
    output_dir=None,
):
    logging.info(f"Extract keyframes using CLIP for video: {video_file_path}")
    video_path = Path(video_file_path)
    if output_dir is None:
        target_path = video_path.parent / video_path.stem
    else:
        target_path = Path(output_dir) / video_path.stem

    if target_path.exists() and len(list(target_path.glob("*.jpeg"))) >= chunk_count:
        logging.info("Keyframes already extracted and present in %s", target_path)
        return str(target_path)

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

    target_path.mkdir(parents=True, exist_ok=True)
    saved = 0

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

        saved += 1
        out_path = target_path / f"{saved}.jpeg"
        out_bgr = cv2.cvtColor(best_frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_path), out_bgr)

    cap.release()

    reorder_and_rename_images(str(target_path))
    logging.info(
        "video %s: CLIP chunk keyframes extracted successfully", video_path.stem
    )
    return str(target_path)
