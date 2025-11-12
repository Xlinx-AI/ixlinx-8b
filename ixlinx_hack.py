


from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import torch
except ModuleNotFoundError as exc:  
    raise RuntimeError(
        "PyTorch is required to run ixlinx_hack.py. Install it via `pip install torch`."
    ) from exc

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.func import functional_call

try:  
    from datasets import load_dataset
except Exception:  
    load_dataset = None  

try:  
    from transformers import AutoTokenizer
except Exception:  
    AutoTokenizer = None  

try:  
    from PIL import Image
except Exception:  
    Image = None  

try:  
    import librosa  
except Exception:  
    librosa = None  

try:  
    from torchvision.io import read_video  
except Exception:  
    read_video = None  

try:  
    import cv2  
except Exception:  
    cv2 = None  






def configure_logging(verbosity: str = "info") -> None:
    

    level = getattr(logging, verbosity.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def detect_device(preferred: Optional[str] = None) -> torch.device:
    

    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pil_image_to_tensor(image: "Image.Image", normalize: bool = True) -> torch.Tensor:
    if Image is None:
        raise RuntimeError("Pillow is required for image processing but is not available")
    data = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
    channels = len(image.getbands())
    height, width = image.size[1], image.size[0]
    tensor = data.view(height, width, channels).permute(2, 0, 1).float()
    if channels == 1:
        tensor = tensor.repeat(3, 1, 1)
    if normalize:
        tensor = tensor.div(255.0)
    return tensor


def extract_video_frames(
    video_path: str,
    num_frames: int = 16,
    size: Tuple[int, int] = (224, 224),
    sampling: str = "uniform",
) -> torch.Tensor:
    
    if cv2 is not None:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.warning("Failed to open video %s, returning zeros", video_path)
            return torch.zeros(num_frames, 3, size[0], size[1])
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return torch.zeros(num_frames, 3, size[0], size[1])
        
        if sampling == "uniform":
            frame_indices = torch.linspace(0, total_frames - 1, num_frames).long().tolist()
        elif sampling == "random":
            if total_frames >= num_frames:
                frame_indices = sorted(random.sample(range(total_frames), num_frames))
            else:
                frame_indices = list(range(total_frames))
        else:
            frame_indices = torch.linspace(0, total_frames - 1, num_frames).long().tolist()
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (size[1], size[0]))
                frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                frames.append(frame_tensor)
        
        cap.release()
        
        if len(frames) < num_frames:
            padding = [torch.zeros(3, size[0], size[1]) for _ in range(num_frames - len(frames))]
            frames.extend(padding)
        
        return torch.stack(frames[:num_frames])
    
    elif read_video is not None:
        try:
            video_tensor, _, _ = read_video(video_path, pts_unit="sec")
            total_frames = video_tensor.shape[0]
            
            if total_frames == 0:
                return torch.zeros(num_frames, 3, size[0], size[1])
            
            if sampling == "uniform":
                frame_indices = torch.linspace(0, total_frames - 1, num_frames).long()
            elif sampling == "random":
                if total_frames >= num_frames:
                    frame_indices = torch.tensor(sorted(random.sample(range(total_frames), num_frames)))
                else:
                    frame_indices = torch.arange(total_frames)
            else:
                frame_indices = torch.linspace(0, total_frames - 1, num_frames).long()
            
            frames = video_tensor[frame_indices]
            frames = F.interpolate(
                frames.permute(0, 3, 1, 2).float(),
                size=size,
                mode="bilinear",
                align_corners=False,
            )
            frames = frames / 255.0
            
            if frames.shape[0] < num_frames:
                padding = torch.zeros(num_frames - frames.shape[0], 3, size[0], size[1])
                frames = torch.cat([frames, padding], dim=0)
            
            return frames[:num_frames]
        except Exception as exc:
            logging.warning("Failed to read video %s: %s", video_path, exc)
            return torch.zeros(num_frames, 3, size[0], size[1])
    
    else:
        logging.warning("No video backend available (cv2 or torchvision), returning zeros")
        return torch.zeros(num_frames, 3, size[0], size[1])







@dataclass
class ModelConfig:
    

    vocab_size: int = 32_000
    dim: int = 4096
    layers: int = 32
    n_heads: int = 32
    n_kv_heads: int = 8
    rmc_hidden: int = 8192
    low_rank: int = 512
    ff_mult: int = 4
    dropout: float = 0.1
    image_patch: int = 16
    audio_window: int = 400
    video_frames: int = 16
    video_size: int = 224
    video_patch: int = 16
    video_temporal_patch: int = 2
    entropy_beta: float = 0.05
    max_seq_len: int = 4096
    rope_theta: float = 10000.0
    use_flash_attn: bool = False
    use_gradient_checkpointing: bool = True
    quantize_linear: bool = False
    mixed_precision: bool = True


@dataclass
class TrainingConfig:
    

    epochs: int = 1
    meta_batch_size: int = 2
    inner_steps_min: int = 1
    inner_steps_max: int = 4
    inner_lr: float = 1e-3
    outer_lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    support_size: int = 4
    query_size: int = 4
    epsilon: float = 0.1
    gamma: float = 0.9
    q_alpha: float = 0.1
    reward_stability_weight: float = 0.5
    reward_step_penalty: float = 0.05
    max_meta_tasks: int = 100
    log_interval: int = 5
    synthetic: bool = True
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    use_amp: bool = True
    ema_decay: float = 0.999
    ema_start_step: int = 100
    scheduler: str = "cosine"
    warmup_steps: int = 2000
    resume_from: Optional[str] = None


@dataclass
class EvalConfig:
    batch_size: int = 4
    max_batches: int = 25


@dataclass
class QuantConfig:
    enabled: bool = True
    dtype: str = "int8"







class SyntheticMultimodalDataset:
    

    def __init__(
        self,
        length: int = 2048,
        vocab_size: int = 32_000,
        seq_len: int = 128,
        image_size: int = 64,
        audio_len: int = 16000,
        video_frames: int = 16,
        video_size: int = 224,
    ) -> None:
        self.length = length
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.image_size = image_size
        self.audio_len = audio_len
        self.video_frames = video_frames
        self.video_size = video_size

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        generator = torch.Generator().manual_seed(idx)
        text = torch.randint(
            low=0,
            high=self.vocab_size,
            size=(self.seq_len + 1,),
            generator=generator,
        )
        image = torch.randn(3, self.image_size, self.image_size, generator=generator)
        audio = torch.randn(self.audio_len, generator=generator)
        video = torch.randn(
            self.video_frames,
            3,
            self.video_size,
            self.video_size,
            generator=generator,
        )
        modalities = torch.tensor([
            random.random(),  
            random.random(),  
            random.random(),  
            random.random(),  
        ])
        return {
            "text": text[:-1],
            "targets": text[1:],
            "image": image,
            "audio": audio,
            "video": video,
            "modalities": modalities,
            "length": self.seq_len,
        }


def load_multimodal_dataset(
    config: TrainingConfig,
    model_config: ModelConfig,
    split: str = "train",
) -> SyntheticMultimodalDataset:
    

    if config.synthetic or load_dataset is None:
        logging.info("Using synthetic multimodal dataset (%s split).", split)
        return SyntheticMultimodalDataset(
            length=2048 if split == "train" else 512,
            vocab_size=model_config.vocab_size,
            seq_len=min(128, model_config.max_seq_len - 1),
            video_frames=model_config.video_frames,
            video_size=model_config.video_size,
        )

    try:
        text_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        vision_ds = load_dataset("laion/laion-aesthetics_v2", split=f"{split}[:1%]")
        audio_ds = load_dataset("ashraq/esc50", split=split)
    except Exception as exc:  
        logging.warning("Falling back to synthetic dataset: %s", exc)
        return SyntheticMultimodalDataset(
            length=1024 if split == "train" else 256,
            vocab_size=model_config.vocab_size,
            seq_len=min(128, model_config.max_seq_len - 1),
        )

    
    
    min_len = min(len(text_ds), len(vision_ds), len(audio_ds))

    class HuggingFaceMultimodalDataset(SyntheticMultimodalDataset):  
        def __init__(self) -> None:
            super().__init__(
                length=min_len,
                vocab_size=model_config.vocab_size,
                video_frames=model_config.video_frames,
                video_size=model_config.video_size,
            )

        def __getitem__(self, idx: int) -> Dict[str, Any]:
            text_item = text_ds[int(idx % len(text_ds))]
            vision_item = vision_ds[int(idx % len(vision_ds))]
            audio_item = audio_ds[int(idx % len(audio_ds))]

            text_tokens = text_item.get("text", "").split()
            if len(text_tokens) < 2:
                text_tokens = ["ixlinx", "hackathon", "sample"]
            token_ids = torch.tensor([
                hash(tok) % self.vocab_size for tok in text_tokens
            ])
            token_ids = token_ids[: self.seq_len + 1]
            if token_ids.numel() < self.seq_len + 1:
                pad = torch.full(
                    (self.seq_len + 1 - token_ids.numel(),),
                    fill_value=0,
                    dtype=torch.long,
                )
                token_ids = torch.cat([token_ids, pad], dim=0)

            
            if Image is not None and isinstance(vision_item.get("image"), Image.Image):
                try:
                    image = pil_image_to_tensor(vision_item["image"])
                except Exception:
                    image = torch.randn(3, 64, 64)
            else:
                image = torch.randn(3, 64, 64)

            
            if librosa is not None and "file" in audio_item:
                file_path = audio_item["file"]
                if os.path.exists(file_path):
                    try:
                        audio_array, _ = librosa.load(file_path, sr=16_000)
                        audio = torch.tensor(audio_array, dtype=torch.float32)
                    except Exception:
                        audio = torch.randn(self.audio_len)
                else:
                    audio = torch.randn(self.audio_len)
            else:
                audio = torch.randn(self.audio_len)
            if audio.numel() < self.audio_len:
                audio = F.pad(audio, (0, self.audio_len - audio.numel()))
            audio = audio[: self.audio_len]

            
            video = torch.randn(self.video_frames, 3, self.video_size, self.video_size)
            candidate_path = vision_item.get("path") if isinstance(vision_item, dict) else None
            if isinstance(candidate_path, str) and os.path.exists(candidate_path):
                try:
                    video = extract_video_frames(
                        candidate_path,
                        num_frames=self.video_frames,
                        size=(self.video_size, self.video_size),
                        sampling="uniform",
                    )
                except Exception:
                    video = torch.randn(self.video_frames, 3, self.video_size, self.video_size)

            modalities = torch.tensor([
                random.random(),
                random.random(),
                random.random(),
                random.random(),
            ])

            return {
                "text": token_ids[:-1],
                "targets": token_ids[1:],
                "image": image,
                "audio": audio,
                "video": video,
                "modalities": modalities,
                "length": int(token_ids.numel() - 1),
            }

    logging.info(
        "Loaded hybrid multimodal dataset with %d samples from Hugging Face", min_len
    )
    return HuggingFaceMultimodalDataset()







@dataclass
class MetaTask:
    support: Dict[str, torch.Tensor]
    query: Dict[str, torch.Tensor]
    metadata: Dict[str, Any]


def collate_batch(samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    text = torch.stack([s["text"] for s in samples], dim=0)
    targets = torch.stack([s["targets"] for s in samples], dim=0)
    images = torch.stack([s["image"] for s in samples], dim=0)
    audios = torch.stack([s["audio"] for s in samples], dim=0)
    videos = torch.stack([s["video"] for s in samples], dim=0)
    modalities = torch.stack([s["modalities"] for s in samples], dim=0)
    lengths = torch.tensor([s["length"] for s in samples])
    return {
        "text": text,
        "targets": targets,
        "image": images,
        "audio": audios,
        "video": videos,
        "modalities": modalities,
        "lengths": lengths,
    }


class MetaTaskSampler:
    

    def __init__(
        self,
        dataset: SyntheticMultimodalDataset,
        config: TrainingConfig,
        split: str = "train",
    ) -> None:
        self.dataset = dataset
        self.config = config
        self.split = split
        self.indices = list(range(len(dataset)))

    def __iter__(self) -> Iterable[MetaTask]:
        random.shuffle(self.indices)
        max_tasks = self.config.max_meta_tasks
        task_count = 0

        while task_count < max_tasks:
            support_samples = []
            query_samples = []
            for _ in range(self.config.support_size):
                idx = random.choice(self.indices)
                support_samples.append(self.dataset[idx])
            for _ in range(self.config.query_size):
                idx = random.choice(self.indices)
                query_samples.append(self.dataset[idx])

            metadata = {
                "modality_ratio": float(torch.stack([
                    s["modalities"] for s in support_samples + query_samples
                ]).mean().item()),
                "context_mean": float(mean([
                    s["length"] for s in support_samples + query_samples
                ])),
                "task_id": task_count,
                "split": self.split,
            }

            yield MetaTask(
                support=collate_batch(support_samples),
                query=collate_batch(query_samples),
                metadata=metadata,
            )
            task_count += 1

    def __len__(self) -> int:
        return self.config.max_meta_tasks







class QAgent:
    

    def __init__(
        self,
        modality_bins: int = 10,
        context_bins: int = 10,
        action_dim: int = 6,
        epsilon: float = 0.1,
        alpha: float = 0.1,
        gamma: float = 0.9,
    ) -> None:
        self.modality_bins = modality_bins
        self.context_bins = context_bins
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = torch.zeros(modality_bins, context_bins, action_dim)

    def encode_state(self, metadata: Dict[str, Any]) -> Tuple[int, int]:
        modality_ratio = metadata.get("modality_ratio", 0.5)
        context_mean = metadata.get("context_mean", 128)
        mod_idx = int(max(0, min(self.modality_bins - 1, modality_ratio * self.modality_bins)))
        ctx_idx = int(
            max(0, min(self.context_bins - 1, context_mean / 1024 * self.context_bins))
        )
        return mod_idx, ctx_idx

    def select_action(self, state: Tuple[int, int]) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        mod_idx, ctx_idx = state
        q_values = self.q_table[mod_idx, ctx_idx]
        return int(torch.argmax(q_values).item())

    def update(
        self,
        state: Tuple[int, int],
        action: int,
        reward: float,
        next_state: Tuple[int, int],
    ) -> None:
        mod_idx, ctx_idx = state
        next_mod, next_ctx = next_state
        best_next = torch.max(self.q_table[next_mod, next_ctx]).item()
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[mod_idx, ctx_idx, action].item()
        self.q_table[mod_idx, ctx_idx, action] += self.alpha * td_error

    def action_to_specs(
        self,
        action: int,
        inner_steps_bounds: Tuple[int, int],
    ) -> Dict[str, Any]:
        

        steps_min, steps_max = inner_steps_bounds
        step_choices = list(range(steps_min, steps_max + 1))
        step_idx = action % len(step_choices)
        inner_steps = step_choices[step_idx]
        augmentation_strength = [0.1, 0.25, 0.5, 0.75, 1.0]
        strength = augmentation_strength[action % len(augmentation_strength)]
        task_mix = [
            "text-heavy",
            "vision-heavy",
            "audio-heavy",
            "video-heavy",
            "balanced",
            "long-context",
        ][action % 6]
        return {
            "inner_steps": inner_steps,
            "augmentation_strength": strength,
            "task_mix": task_mix,
        }







def precompute_rope_freqs(
    dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(max_seq_len, device=device, dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return cos, sin


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    
    seq_len = x.shape[1]
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(2)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(2)
    
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    
    x_rope = torch.stack([
        x_even * cos - x_odd * sin,
        x_even * sin + x_odd * cos,
    ], dim=-1)
    
    return x_rope.flatten(-2)


class RMSNorm(nn.Module):
    

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class SwiGLU(nn.Module):
    

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class EntropyRegGELU(nn.Module):
    

    def __init__(self, beta: float = 0.05) -> None:
        super().__init__()
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gelu = F.gelu(x)
        probs = torch.sigmoid(x)
        entropy = -(probs * torch.log(probs + 1e-9) + (1 - probs) * torch.log(1 - probs + 1e-9))
        return gelu - self.beta * entropy


class LowRankLinear(nn.Module):
    

    def __init__(self, in_features: int, out_features: int, rank: int) -> None:
        super().__init__()
        self.u = nn.Linear(in_features, rank, bias=False)
        self.v = nn.Linear(rank, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.v(self.u(x)) + self.bias


class LowRankFFN(nn.Module):
    def __init__(self, dim: int, rank: int, mult: int, dropout: float, beta: float) -> None:
        super().__init__()
        inner_dim = dim * mult
        self.fc1 = LowRankLinear(dim, inner_dim, rank)
        self.act = EntropyRegGELU(beta=beta)
        self.fc2 = LowRankLinear(inner_dim, dim, rank)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.act(self.fc1(x))))


class VideoPatchEmbed(nn.Module):
    

    def __init__(self, dim: int, patch_size: int, temporal_patch: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch = temporal_patch
        self.proj = nn.Linear(3 * patch_size * patch_size * temporal_patch, dim)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        
        batch, frames, channels, height, width = video.shape
        pad_t = (self.temporal_patch - frames % self.temporal_patch) % self.temporal_patch
        if pad_t:
            pad = torch.zeros(batch, pad_t, channels, height, width, device=video.device, dtype=video.dtype)
            video = torch.cat([video, pad], dim=1)
            frames += pad_t
        pad_h = (self.patch_size - height % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - width % self.patch_size) % self.patch_size
        if pad_h or pad_w:
            video = F.pad(video, (0, pad_w, 0, pad_h))
            height += pad_h
            width += pad_w
        video = video.view(batch, frames // self.temporal_patch, self.temporal_patch, channels, height, width)
        video = video.permute(0, 1, 3, 2, 4, 5).contiguous()
        b, windows, ch, tp, h, w = video.shape
        video = video.view(b * windows, ch * tp, h, w)
        patches = F.unfold(video, kernel_size=self.patch_size, stride=self.patch_size)
        patches = patches.transpose(1, 2)
        patches = patches.reshape(b, windows, patches.size(1), patches.size(2))
        patches = patches.reshape(b, -1, patches.size(-1))
        return self.proj(patches)


class GroupedQueryAttention(nn.Module):
    

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        dropout: float = 0.0,
        use_flash: bool = False,
    ) -> None:
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = dim // n_heads
        self.use_flash = use_flash
        
        self.q_proj = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)
        
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=2)
            v = v.repeat_interleave(self.n_rep, dim=2)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        if self.use_flash:
            try:
                attn_out = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=mask,
                    dropout_p=self.dropout.p if self.training else 0.0,
                )
            except Exception:
                attn_out = self._standard_attention(q, k, v, mask)
        else:
            attn_out = self._standard_attention(q, k, v, mask)
        
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(attn_out)

    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        scale = self.head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        if mask is not None:
            scores = scores + mask
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        return torch.matmul(attn, v)


class SSMSequenceMixer(nn.Module):
    

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.state_proj = nn.Linear(dim, hidden)
        self.input_proj = nn.Linear(dim, hidden)
        self.out_proj = nn.Linear(hidden, dim)
        self.gamma = nn.Parameter(torch.ones(hidden))
        self.beta = nn.Parameter(torch.zeros(hidden))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        state = torch.zeros(batch, self.state_proj.out_features, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(seq_len):
            inp = self.input_proj(x[:, t, :])
            state = torch.tanh(inp + state * self.gamma + self.beta)
            outputs.append(self.out_proj(state))
        return torch.stack(outputs, dim=1)


class RMCBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden: int,
        low_rank: int,
        ff_mult: int,
        dropout: float,
        beta: float,
        n_heads: int,
        n_kv_heads: int,
        use_flash: bool,
    ) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.ssm_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
        self.attn = GroupedQueryAttention(dim, n_heads, n_kv_heads, dropout, use_flash)
        self.mixer = SSMSequenceMixer(dim, hidden)
        self.ffn = LowRankFFN(dim, low_rank, ff_mult, dropout, beta)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_out = self.attn(self.attn_norm(x), cos, sin, mask)
        x = x + self.dropout(attn_out)
        ssm_out = self.mixer(self.ssm_norm(x))
        x = x + self.dropout(ssm_out)
        ffn_out = self.ffn(self.ffn_norm(x))
        return x + self.dropout(ffn_out)


class UnifiedProjectionModule(nn.Module):
    def __init__(
        self,
        dim: int,
        vocab_size: int,
        image_patch: int,
        audio_window: int,
        video_patch: int,
        video_temporal_patch: int,
    ) -> None:
        super().__init__()
        self.text_embed = nn.Embedding(vocab_size, dim)
        self.image_patch = image_patch
        self.audio_window = audio_window
        image_dim = (3 * image_patch * image_patch)
        audio_dim = audio_window
        self.image_proj = nn.Linear(image_dim, dim)
        self.audio_proj = nn.Linear(audio_dim, dim)
        self.video_embed = VideoPatchEmbed(dim, video_patch, video_temporal_patch)
        self.text_pos_embed = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.image_pos_embed = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.audio_pos_embed = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.video_pos_embed = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

    def forward(
        self,
        text: torch.Tensor,
        image: torch.Tensor,
        audio: torch.Tensor,
        video: torch.Tensor,
        modalities: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        text_tokens = self.text_embed(text)
        text_tokens = text_tokens + self.text_pos_embed

        batch, _, height, width = image.shape
        patch = self.image_patch
        if height % patch != 0 or width % patch != 0:
            pad_h = (patch - height % patch) % patch
            pad_w = (patch - width % patch) % patch
            image = F.pad(image, (0, pad_w, 0, pad_h))
        unfold = F.unfold(image, kernel_size=patch, stride=patch)
        patches = unfold.transpose(1, 2)
        image_features = self.image_proj(patches) + self.image_pos_embed

        audio = audio.unsqueeze(1)
        audio = F.unfold(audio.unsqueeze(-1), kernel_size=(self.audio_window, 1), stride=(self.audio_window // 2, 1))
        audio = audio.transpose(1, 2)
        audio_features = self.audio_proj(audio.squeeze(-1)) + self.audio_pos_embed

        video_features = self.video_embed(video) + self.video_pos_embed

        if modalities is not None:
            modalities = modalities.to(text_tokens.device)
            text_scale = modalities[:, 0].view(batch, 1, 1)
            vision_scale = modalities[:, 1].view(batch, 1, 1)
            audio_scale = modalities[:, 2].view(batch, 1, 1)
            video_scale = modalities[:, 3].view(batch, 1, 1)
            text_tokens = text_tokens * (1.0 + text_scale)
            image_features = image_features * (1.0 + vision_scale)
            audio_features = audio_features * (1.0 + audio_scale)
            video_features = video_features * (1.0 + video_scale)

        return torch.cat([text_tokens, image_features, audio_features, video_features], dim=1)


class IXLinx8B(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.upm = UnifiedProjectionModule(
            dim=config.dim,
            vocab_size=config.vocab_size,
            image_patch=config.image_patch,
            audio_window=config.audio_window,
            video_patch=config.video_patch,
            video_temporal_patch=config.video_temporal_patch,
        )
        self.blocks = nn.ModuleList([
            RMCBlock(
                dim=config.dim,
                hidden=config.rmc_hidden,
                low_rank=config.low_rank,
                ff_mult=config.ff_mult,
                dropout=config.dropout,
                beta=config.entropy_beta,
                n_heads=config.n_heads,
                n_kv_heads=config.n_kv_heads,
                use_flash=config.use_flash_attn,
            )
            for _ in range(config.layers)
        ])
        self.norm = RMSNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        
        self.rope_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        if config.use_gradient_checkpointing:
            self._enable_gradient_checkpointing()

    def _enable_gradient_checkpointing(self) -> None:
        for block in self.blocks:
            block.requires_grad_(True)

    def _get_rope_cache(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.rope_cache is None or self.rope_cache[0].device != device:
            self.rope_cache = precompute_rope_freqs(
                self.config.dim // self.config.n_heads,
                self.config.max_seq_len * 4,
                self.config.rope_theta,
                device,
            )
        return self.rope_cache

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        device = batch["text"].device
        cos, sin = self._get_rope_cache(device)
        hidden = self.upm(
            batch["text"],
            batch["image"],
            batch["audio"],
            batch["video"],
            batch.get("modalities"),
        )
        cos = cos.to(dtype=hidden.dtype)
        sin = sin.to(dtype=hidden.dtype)
        
        for block in self.blocks:
            if self.config.use_gradient_checkpointing and self.training:
                try:
                    hidden = torch.utils.checkpoint.checkpoint(
                        block,
                        hidden,
                        cos,
                        sin,
                        None,
                        use_reentrant=False,
                    )
                except TypeError:
                    hidden = torch.utils.checkpoint.checkpoint(block, hidden, cos, sin, None)
            else:
                hidden = block(hidden, cos, sin, None)
        
        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)
        text_len = batch["text"].shape[1]
        return logits[:, :text_len, :]

    def generate(
        self,
        prompt_ids: torch.Tensor,
        steps: int = 32,
        temperature: float = 0.9,
        top_k: Optional[int] = 50,
        image: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self.eval()
        device = next(self.parameters()).device
        generated = prompt_ids.clone().to(device)
        
        if image is None:
            image = torch.zeros(1, 3, 64, 64, device=device)
        if audio is None:
            audio = torch.zeros(1, 16000, device=device)
        if video is None:
            video = torch.zeros(1, self.config.video_frames, 3, self.config.video_size, self.config.video_size, device=device)
        
        for _ in range(steps):
            batch = {
                "text": generated.unsqueeze(0),
                "image": image,
                "audio": audio,
                "video": video,
            }
            with torch.no_grad():
                logits = self.forward(batch)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                top_values, top_indices = torch.topk(logits, top_k, dim=-1)
                probs = torch.softmax(top_values, dim=-1)
                next_token = top_indices[0, torch.multinomial(probs[0], num_samples=1)]
            else:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)[0]
            generated = torch.cat([generated, next_token], dim=0)
        return generated

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def quantize_dynamic(self, dtype: torch.dtype = torch.qint8) -> "IXLinx8B":
        try:
            quantized = torch.quantization.quantize_dynamic(
                self,
                {nn.Linear},
                dtype=dtype,
            )
            logging.info("Applied dynamic quantisation (%s)", dtype)
            return quantized  
        except Exception as exc:
            logging.warning("Dynamic quantisation failed: %s", exc)
            return self







def compute_loss(
    model: IXLinx8B,
    batch: Dict[str, torch.Tensor],
    params: Optional[OrderedDict[str, torch.Tensor]] = None,
    buffers: Optional[OrderedDict[str, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if params is not None:
        params_and_buffers = OrderedDict(params)
        if buffers is not None:
            params_and_buffers.update(buffers)
        logits = functional_call(model, params_and_buffers, (batch,))
    else:
        logits = model(batch)
    targets = batch["targets"].to(logits.device)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    with torch.no_grad():
        pred = torch.argmax(logits, dim=-1)
        accuracy = (pred == targets).float().mean().item()
        perplexity = float(torch.exp(loss.detach()))
    return loss, {"loss": float(loss.item()), "accuracy": accuracy, "perplexity": perplexity}


def compute_reward(
    metrics: Dict[str, float],
    inner_steps: int,
    config: TrainingConfig,
    grad_norm: Optional[float] = None,
) -> float:
    perplexity = metrics.get("perplexity", 10.0)
    stability = 0.0
    if grad_norm is not None:
        stability = 1.0 / (1.0 + grad_norm)
    base_reward = math.exp(-perplexity / 10.0)
    stability_bonus = config.reward_stability_weight * stability
    step_penalty = config.reward_step_penalty * (inner_steps - config.inner_steps_min)
    reward = base_reward + stability_bonus - step_penalty
    return reward







def apply_task_augmentation(
    task: MetaTask,
    strength: float,
    mix: str,
) -> MetaTask:
    

    def augment(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        augmented = {k: v.clone() for k, v in batch.items()}
        noise_scale = strength * 0.05
        if mix in {"vision-heavy", "balanced", "video-heavy"}:
            augmented["image"] = augmented["image"] + noise_scale * torch.randn_like(augmented["image"])
        if mix in {"audio-heavy", "balanced"}:
            augmented["audio"] = augmented["audio"] + noise_scale * torch.randn_like(augmented["audio"])
        if mix in {"video-heavy", "balanced"} and "video" in augmented:
            video_noise = noise_scale * torch.randn_like(augmented["video"])
            augmented["video"] = augmented["video"] + video_noise
            if strength > 0.5:
                drop_rate = min(0.3, strength * 0.2)
                mask = (torch.rand(augmented["video"].shape[:2], device=augmented["video"].device) < drop_rate)
                augmented["video"] = augmented["video"] * (~mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
        if mix in {"text-heavy", "long-context"}:
            probability = min(0.15, strength * 0.1)
            mask = torch.rand_like(augmented["text"].float()) < probability
            noise_tokens = torch.randint(0, max(augmented["text"].max().item(), 1), augmented["text"].shape, device=augmented["text"].device)
            augmented["text"] = torch.where(mask.bool(), noise_tokens, augmented["text"])
            augmented["targets"] = torch.where(mask.bool(), noise_tokens, augmented["targets"])
        if mix == "long-context":
            pad_len = max(1, int(augmented["text"].shape[1] * 0.1))
            pad_tokens = torch.full((augmented["text"].shape[0], pad_len), fill_value=0, dtype=torch.long, device=augmented["text"].device)
            augmented["text"] = torch.cat([augmented["text"], pad_tokens], dim=1)
            augmented["targets"] = torch.cat([augmented["targets"], pad_tokens], dim=1)
        return augmented

    return MetaTask(
        support=augment(task.support),
        query=augment(task.query),
        metadata=task.metadata,
    )


class ModelEMA:
    
    
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def q_amaml_train(
    model: IXLinx8B,
    train_sampler: MetaTaskSampler,
    val_sampler: Optional[MetaTaskSampler],
    config: TrainingConfig,
    model_config: ModelConfig,
    device: torch.device,
    resume_state: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Path] = None,
    save_every: int = 0,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler, Optional[ModelEMA], int, int]:
    logger = logging.getLogger("train")
    buffers = OrderedDict(model.named_buffers())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.outer_lr,
        weight_decay=config.weight_decay,
    )
    
    total_steps = config.epochs * config.max_meta_tasks
    scheduler = get_cosine_schedule_with_warmup(optimizer, config.warmup_steps, total_steps)
    
    scaler = torch.cuda.amp.GradScaler() if config.use_amp and device.type == "cuda" else None
    
    ema = ModelEMA(model, decay=config.ema_decay) if config.ema_decay > 0 else None
    
    global_step = 0
    start_epoch = 0

    if resume_state is not None:
        if "optimizer_state" in resume_state:
            optimizer.load_state_dict(resume_state["optimizer_state"])
        if "scheduler_state" in resume_state:
            scheduler.load_state_dict(resume_state["scheduler_state"])
        if "ema_shadow" in resume_state and ema is not None:
            ema.shadow = {k: v.to(device) for k, v in resume_state["ema_shadow"].items()}
        global_step = resume_state.get("step", 0)
        start_epoch = resume_state.get("epoch", 0)
        logger.info("Resumed from step %d, epoch %d", global_step, start_epoch)
    
    q_agent = QAgent(
        epsilon=config.epsilon,
        alpha=config.q_alpha,
        gamma=config.gamma,
    )

    for epoch in range(start_epoch, config.epochs):
        logger.info("Starting epoch %d/%d", epoch + 1, config.epochs)
        for meta_task in train_sampler:
            model.train()
            task = apply_task_augmentation(
                meta_task,
                strength=q_agent.action_to_specs(0, (config.inner_steps_min, config.inner_steps_max))["augmentation_strength"],
                mix="balanced",
            )  

            state = q_agent.encode_state(task.metadata)
            action = q_agent.select_action(state)
            specs = q_agent.action_to_specs(action, (config.inner_steps_min, config.inner_steps_max))
            task = apply_task_augmentation(task, specs["augmentation_strength"], specs["task_mix"])
            inner_steps = specs["inner_steps"]

            params = OrderedDict(model.named_parameters())
            grads_for_reward: List[torch.Tensor] = []

            
            for _ in range(inner_steps):
                optimizer.zero_grad()
                support_batch = {k: v.to(device) for k, v in task.support.items()}
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        loss, metrics = compute_loss(model, support_batch, params=params, buffers=buffers)
                else:
                    loss, metrics = compute_loss(model, support_batch, params=params, buffers=buffers)
                grads = torch.autograd.grad(
                    loss,
                    params.values(),
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )
                params = OrderedDict(
                    (name, param - config.inner_lr * grad if grad is not None else param)
                    for (name, param), grad in zip(params.items(), grads)
                )
                grads_for_reward.extend([g.detach() for g in grads if g is not None])

            query_batch = {k: v.to(device) for k, v in task.query.items()}
            
            optimizer.zero_grad()
            
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    query_loss, query_metrics = compute_loss(model, query_batch, params=params, buffers=buffers)
                scaler.scale(query_loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = float(clip_grad_norm_(model.parameters(), config.grad_clip))
                scaler.step(optimizer)
                scaler.update()
            else:
                query_loss, query_metrics = compute_loss(model, query_batch, params=params, buffers=buffers)
                query_loss.backward()
                grad_norm = float(clip_grad_norm_(model.parameters(), config.grad_clip))
                optimizer.step()
            
            scheduler.step()
            
            if ema is not None and global_step >= config.ema_start_step:
                ema.update()

            reward = compute_reward(query_metrics, inner_steps, config, grad_norm)
            next_state = q_agent.encode_state(task.metadata)
            q_agent.update(state, action, reward, next_state)

            if global_step % config.log_interval == 0:
                logger.info(
                    "step=%d epoch=%d task=%d loss=%.4f ppl=%.2f acc=%.3f reward=%.4f inner_steps=%d lr=%.6f",
                    global_step,
                    epoch + 1,
                    task.metadata["task_id"],
                    query_metrics["loss"],
                    query_metrics["perplexity"],
                    query_metrics["accuracy"],
                    reward,
                    inner_steps,
                    optimizer.param_groups[0]["lr"],
                )
            global_step += 1

            if save_every > 0 and global_step % save_every == 0 and output_dir is not None:
                checkpoint_path = output_dir / f"checkpoint_step_{global_step}.ckpt"
                logger.info("Saving intermediate checkpoint to %s", checkpoint_path)
                save_checkpoint(
                    model,
                    checkpoint_path,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    ema=ema,
                    step=global_step,
                    epoch=epoch + 1,
                )

        if val_sampler is not None:
            if ema is not None:
                ema.apply_shadow()
            evaluate(model, val_sampler, device)
            if ema is not None:
                ema.restore()

    logger.info("Training complete")
    return optimizer, scheduler, ema, global_step, config.epochs







def evaluate(model: IXLinx8B, sampler: MetaTaskSampler, device: torch.device) -> Dict[str, float]:
    logger = logging.getLogger("eval")
    model.eval()
    metrics_accum: Dict[str, float] = {"loss": 0.0, "accuracy": 0.0, "perplexity": 0.0}
    count = 0
    with torch.no_grad():
        for task in sampler:
            batch = {k: v.to(device) for k, v in task.query.items()}
            loss, metrics = compute_loss(model, batch)
            metrics_accum["loss"] += metrics["loss"]
            metrics_accum["accuracy"] += metrics["accuracy"]
            metrics_accum["perplexity"] += metrics["perplexity"]
            count += 1
    if count > 0:
        for k in metrics_accum:
            metrics_accum[k] /= count
    logger.info(
        "Eval metrics: loss=%.4f ppl=%.2f acc=%.3f",
        metrics_accum["loss"],
        metrics_accum["perplexity"],
        metrics_accum["accuracy"],
    )
    return metrics_accum







def save_checkpoint(
    model: IXLinx8B,
    path: Path,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    ema: Optional[ModelEMA] = None,
    step: int = 0,
    epoch: int = 0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": model.state_dict(),
        "config": model.config.__dict__,
        "step": step,
        "epoch": epoch,
    }
    if optimizer is not None:
        checkpoint["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state"] = scheduler.state_dict()
    if ema is not None:
        checkpoint["ema_shadow"] = ema.shadow
    torch.save(checkpoint, path)
    logging.info("Saved checkpoint to %s", path)


def load_checkpoint(
    path: Path,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
) -> Tuple[IXLinx8B, Dict[str, Any]]:
    payload = torch.load(path, map_location=device)
    config = ModelConfig(**payload["config"])
    model = IXLinx8B(config)
    model.load_state_dict(payload["model_state"])
    model.to(device)
    
    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and "scheduler_state" in payload:
        scheduler.load_state_dict(payload["scheduler_state"])
    
    metadata = {
        "step": payload.get("step", 0),
        "epoch": payload.get("epoch", 0),
        "ema_shadow": payload.get("ema_shadow"),
    }
    
    logging.info("Loaded checkpoint from %s (step=%d, epoch=%d)", path, metadata["step"], metadata["epoch"])
    return model, metadata


def load_checkpoint_simple(path: Path, device: torch.device) -> IXLinx8B:
    
    model, _ = load_checkpoint(path, device)
    return model







def ensure_tokenizer(model_config: ModelConfig) -> Any:
    if AutoTokenizer is not None:
        try:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token
            return tokenizer
        except Exception as exc:  
            logging.warning("Falling back to naive tokenizer: %s", exc)
    logging.warning("Using whitespace tokenizer fallback")

    class NaiveTokenizer:
        def __init__(self) -> None:
            self.vocab: Dict[str, int] = {"<pad>": 0, "<unk>": 1}
            self.inverse_vocab: Dict[int, str] = {0: "<pad>", 1: "<unk>"}

        def encode(self, text: str) -> List[int]:
            tokens = text.strip().split()
            ids = []
            for token in tokens:
                if token not in self.vocab:
                    idx = len(self.vocab)
                    if idx >= model_config.vocab_size:
                        token = "<unk>"
                        idx = self.vocab[token]
                    else:
                        self.vocab[token] = idx
                        self.inverse_vocab[idx] = token
                ids.append(self.vocab[token])
            if not ids:
                ids = [0]
            return ids

        def decode(self, ids: Iterable[int]) -> str:
            return " ".join(self.inverse_vocab.get(i, "<unk>") for i in ids)

        @property
        def eos_token_id(self) -> int:
            return 0

    return NaiveTokenizer()


def load_image_tensor(path: Optional[str], device: torch.device, size: int = 64) -> torch.Tensor:
    if path is None or Image is None:
        return torch.zeros(1, 3, size, size, device=device)
    image = Image.open(path).convert("RGB")
    image = image.resize((size, size))
    tensor = pil_image_to_tensor(image)
    return tensor.unsqueeze(0).to(device)


def load_audio_tensor(path: Optional[str], device: torch.device, length: int = 16000) -> torch.Tensor:
    if path is None or librosa is None:
        return torch.zeros(1, length, device=device)
    audio, _ = librosa.load(path, sr=16_000)
    tensor = torch.tensor(audio, dtype=torch.float32)
    if tensor.numel() < length:
        tensor = F.pad(tensor, (0, length - tensor.numel()))
    tensor = tensor[:length]
    return tensor.unsqueeze(0).to(device)


def load_video_tensor(
    path: Optional[str],
    device: torch.device,
    num_frames: int = 16,
    size: int = 224,
) -> torch.Tensor:
    if path is None:
        return torch.zeros(1, num_frames, 3, size, size, device=device)
    try:
        video = extract_video_frames(path, num_frames=num_frames, size=(size, size))
        return video.unsqueeze(0).to(device)
    except Exception as exc:
        logging.warning("Failed to load video %s: %s", path, exc)
        return torch.zeros(1, num_frames, 3, size, size, device=device)


def cli_train(args: argparse.Namespace) -> None:
    configure_logging(args.verbosity)
    set_seed(args.seed)
    device = detect_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resume_state: Optional[Dict[str, Any]] = None
    if args.resume_from:
        resume_path = Path(args.resume_from)
        logging.info("Resuming training from checkpoint: %s", resume_path)
        resume_state = torch.load(resume_path, map_location=device)
        model_config = ModelConfig(**resume_state["config"])
    else:
        presets = {
            "prototype": {
                "dim": 768, "layers": 12, "n_heads": 12, "n_kv_heads": 4,
                "rmc_hidden": 1536, "low_rank": 192, "ff_mult": 2, "max_seq_len": 512,
            },
            "8b-lite": {
                "dim": 2048, "layers": 24, "n_heads": 16, "n_kv_heads": 4,
                "rmc_hidden": 4096, "low_rank": 256, "ff_mult": 3, "max_seq_len": 2048,
            },
            "8b": {
                "dim": 4096, "layers": 32, "n_heads": 32, "n_kv_heads": 8,
                "rmc_hidden": 8192, "low_rank": 512, "ff_mult": 4, "max_seq_len": 4096,
            },
        }
        preset = presets[args.preset]
        model_config = ModelConfig(
            vocab_size=args.vocab_size or 32000,
            dim=args.dim or preset["dim"],
            layers=args.layers or preset["layers"],
            n_heads=args.n_heads or preset["n_heads"],
            n_kv_heads=args.n_kv_heads or preset["n_kv_heads"],
            rmc_hidden=args.rmc_hidden or preset["rmc_hidden"],
            low_rank=args.low_rank or preset["low_rank"],
            ff_mult=args.ff_mult or preset["ff_mult"],
            dropout=args.dropout or 0.1,
            image_patch=args.image_patch or 16,
            audio_window=args.audio_window or 400,
            video_frames=args.video_frames or 16,
            video_size=args.video_size or 224,
            video_patch=args.video_patch or 16,
            video_temporal_patch=args.video_temporal_patch or 2,
            entropy_beta=args.entropy_beta or 0.05,
            max_seq_len=args.max_seq_len or preset["max_seq_len"],
            use_flash_attn=args.use_flash_attn,
            use_gradient_checkpointing=not args.no_gradient_checkpointing,
            mixed_precision=not args.no_mixed_precision,
        )

    if args.use_flash_attn:
        model_config.use_flash_attn = True
    if args.no_gradient_checkpointing:
        model_config.use_gradient_checkpointing = False
    if args.no_mixed_precision:
        model_config.mixed_precision = False

    use_amp = (not args.no_amp) and model_config.mixed_precision and device.type == "cuda"

    training_config = TrainingConfig(
        epochs=args.epochs,
        meta_batch_size=args.meta_batch_size,
        inner_steps_min=args.inner_steps_min,
        inner_steps_max=args.inner_steps_max,
        inner_lr=args.inner_lr,
        outer_lr=args.outer_lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        support_size=args.support_size,
        query_size=args.query_size,
        epsilon=args.epsilon,
        gamma=args.gamma,
        q_alpha=args.q_alpha,
        reward_stability_weight=args.reward_stability_weight,
        reward_step_penalty=args.reward_step_penalty,
        max_meta_tasks=args.max_meta_tasks,
        log_interval=args.log_interval,
        synthetic=args.synthetic,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        use_amp=use_amp,
        ema_decay=args.ema_decay,
        ema_start_step=args.ema_start_step,
        scheduler=args.scheduler,
        warmup_steps=args.warmup_steps,
        resume_from=args.resume_from,
    )

    logging.info("Device: %s", device)
    logging.info("Model config: dim=%d, layers=%d, n_heads=%d, n_kv_heads=%d",
                 model_config.dim, model_config.layers, model_config.n_heads, model_config.n_kv_heads)

    model = IXLinx8B(model_config).to(device)
    if resume_state is not None:
        model.load_state_dict(resume_state["model_state"])
        logging.info("Checkpoint weights loaded")

    actual_params = model.parameter_count()
    logging.info("Actual model parameters: %.2fB", actual_params / 1e9)

    train_dataset = load_multimodal_dataset(training_config, model_config, split="train")
    val_dataset = load_multimodal_dataset(training_config, model_config, split="validation")
    train_sampler = MetaTaskSampler(train_dataset, training_config, split="train")
    val_sampler = MetaTaskSampler(val_dataset, training_config, split="validation")

    optimizer, scheduler, ema, final_step, final_epoch = q_amaml_train(
        model,
        train_sampler,
        val_sampler,
        training_config,
        model_config,
        device,
        resume_state=resume_state,
        output_dir=output_dir,
        save_every=args.save_every,
    )

    checkpoint_path = output_dir / "ixlinx_hack.ckpt"
    ema_applied = False
    if ema is not None:
        ema.apply_shadow()
        ema_applied = True
    save_checkpoint(
        model,
        checkpoint_path,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        step=final_step,
        epoch=final_epoch,
    )
    if ema_applied:
        ema.restore()
    logging.info("Saved final checkpoint to %s", checkpoint_path)

    if args.quantize:
        dtype = torch.qint8 if args.quant_dtype == "int8" else torch.qint8
        quantized = model.quantize_dynamic(dtype=dtype)
        quant_path = output_dir / "ixlinx_hack_quantized.ckpt"
        torch.save({"model_state": quantized.state_dict(), "config": model_config.__dict__}, quant_path)
        logging.info("Saved quantised checkpoint to %s", quant_path)


def cli_eval(args: argparse.Namespace) -> None:
    configure_logging(args.verbosity)
    set_seed(args.seed)
    device = detect_device(args.device)
    model = load_checkpoint_simple(Path(args.checkpoint), device)
    config = TrainingConfig(max_meta_tasks=args.max_meta_tasks, synthetic=args.synthetic)
    dataset = load_multimodal_dataset(config, model.config, split="validation")
    sampler = MetaTaskSampler(dataset, config, split="validation")
    evaluate(model, sampler, device)


def cli_chat(args: argparse.Namespace) -> None:
    configure_logging(args.verbosity)
    device = detect_device(args.device)
    model = load_checkpoint_simple(Path(args.checkpoint), device)
    tokenizer = ensure_tokenizer(model.config)

    if args.prompt is None:
        prompt = input("Prompt: ")
    else:
        prompt = args.prompt

    prompt_ids = tokenizer.encode(prompt)
    prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    image = load_image_tensor(args.image, device, size=model.config.video_size)
    audio = load_audio_tensor(args.audio, device, length=model.config.audio_window * 40)
    video = load_video_tensor(args.video, device, num_frames=model.config.video_frames, size=model.config.video_size)

    generated_ids = model.generate(
        prompt_tensor,
        steps=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        image=image,
        audio=audio,
        video=video,
    )
    response = tokenizer.decode(generated_ids.tolist())
    print(response)


def cli_video_test(args: argparse.Namespace) -> None:
    
    configure_logging(args.verbosity)
    device = detect_device(args.device)
    
    logging.info("Testing video extraction from: %s", args.video_path)
    video = extract_video_frames(
        args.video_path,
        num_frames=args.num_frames,
        size=(args.frame_size, args.frame_size),
        sampling=args.sampling,
    )
    logging.info("Extracted video tensor shape: %s", video.shape)
    logging.info("Video dtype: %s, min: %.3f, max: %.3f", video.dtype, video.min(), video.max())
    
    if args.checkpoint:
        logging.info("Testing with model from checkpoint: %s", args.checkpoint)
        model = load_checkpoint_simple(Path(args.checkpoint), device)
        model.eval()
        
        with torch.no_grad():
            batch = {
                "text": torch.zeros(1, 10, dtype=torch.long, device=device),
                "image": torch.zeros(1, 3, 64, 64, device=device),
                "audio": torch.zeros(1, 16000, device=device),
                "video": video.unsqueeze(0).to(device),
            }
            logits = model(batch)
            logging.info("Model output logits shape: %s", logits.shape)
    
    logging.info("Video test completed successfully")


def cli_export_config(args: argparse.Namespace) -> None:
    configure_logging(args.verbosity)
    model_config = ModelConfig()
    training_config = TrainingConfig()
    payload = {
        "model": model_config.__dict__,
        "training": training_config.__dict__,
    }
    path = Path(args.output)
    path.write_text(json.dumps(payload, indent=2))
    logging.info("Exported default configuration to %s", path)







def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="iXlinx-8B hackathon prototype")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    def add_common_arguments(p: argparse.ArgumentParser) -> None:
        p.add_argument("--verbosity", default="info", choices=["debug", "info", "warning", "error"])
        p.add_argument("--device", default=None, help="Force device: cpu, cuda, mps")
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--synthetic", action="store_true", help="Force synthetic dataset")

    train_parser = subparsers.add_parser("train", help="Train the model with Q-AMAML")
    add_common_arguments(train_parser)
    train_parser.add_argument("--epochs", type=int, default=1)
    train_parser.add_argument("--meta-batch-size", dest="meta_batch_size", type=int, default=2)
    train_parser.add_argument("--inner-lr", dest="inner_lr", type=float, default=1e-3)
    train_parser.add_argument("--outer-lr", dest="outer_lr", type=float, default=3e-4)
    train_parser.add_argument("--weight-decay", dest="weight_decay", type=float, default=0.01)
    train_parser.add_argument("--grad-clip", dest="grad_clip", type=float, default=1.0)
    train_parser.add_argument("--support-size", dest="support_size", type=int, default=4)
    train_parser.add_argument("--query-size", dest="query_size", type=int, default=4)
    train_parser.add_argument("--inner-steps-min", dest="inner_steps_min", type=int, default=1)
    train_parser.add_argument("--inner-steps-max", dest="inner_steps_max", type=int, default=4)
    train_parser.add_argument("--epsilon", type=float, default=0.1)
    train_parser.add_argument("--gamma", type=float, default=0.9)
    train_parser.add_argument("--q-alpha", type=float, default=0.1)
    train_parser.add_argument("--reward-stability-weight", type=float, default=0.5)
    train_parser.add_argument("--reward-step-penalty", type=float, default=0.05)
    train_parser.add_argument("--max-meta-tasks", type=int, default=100)
    train_parser.add_argument("--log-interval", type=int, default=5)
    train_parser.add_argument("--wandb-project", default=None)
    train_parser.add_argument("--wandb-run-name", default=None)
    train_parser.add_argument("--output-dir", default="./outputs")
    train_parser.add_argument("--quantize", action="store_true")
    train_parser.add_argument("--quant-dtype", choices=["int8", "int4"], default="int8")
    train_parser.add_argument("--preset", choices=["prototype", "8b-lite", "8b"], default="8b-lite")
    train_parser.add_argument("--vocab-size", type=int, default=None)
    train_parser.add_argument("--dim", type=int, default=None)
    train_parser.add_argument("--layers", type=int, default=None)
    train_parser.add_argument("--n-heads", type=int, default=None)
    train_parser.add_argument("--n-kv-heads", type=int, default=None)
    train_parser.add_argument("--rmc-hidden", dest="rmc_hidden", type=int, default=None)
    train_parser.add_argument("--low-rank", dest="low_rank", type=int, default=None)
    train_parser.add_argument("--ff-mult", dest="ff_mult", type=int, default=None)
    train_parser.add_argument("--dropout", type=float, default=None)
    train_parser.add_argument("--image-patch", type=int, default=None)
    train_parser.add_argument("--audio-window", type=int, default=None)
    train_parser.add_argument("--video-frames", type=int, default=None)
    train_parser.add_argument("--video-size", type=int, default=None)
    train_parser.add_argument("--video-patch", type=int, default=None)
    train_parser.add_argument("--video-temporal-patch", type=int, default=None)
    train_parser.add_argument("--entropy-beta", type=float, default=None)
    train_parser.add_argument("--max-seq-len", type=int, default=None)
    train_parser.add_argument("--use-flash-attn", action="store_true")
    train_parser.add_argument("--no-gradient-checkpointing", action="store_true")
    train_parser.add_argument("--no-mixed-precision", action="store_true")
    train_parser.add_argument("--no-amp", action="store_true")
    train_parser.add_argument("--ema-decay", type=float, default=0.999)
    train_parser.add_argument("--ema-start-step", type=int, default=100)
    train_parser.add_argument("--scheduler", choices=["cosine"], default="cosine")
    train_parser.add_argument("--warmup-steps", type=int, default=2000)
    train_parser.add_argument("--resume-from", default=None)
    train_parser.add_argument("--save-every", type=int, default=0, help="Save checkpoint every N steps (0 disables)")

    eval_parser = subparsers.add_parser("eval", help="Evaluate a checkpoint")
    add_common_arguments(eval_parser)
    eval_parser.add_argument("--checkpoint", required=True)
    eval_parser.add_argument("--max-meta-tasks", type=int, default=25)

    chat_parser = subparsers.add_parser("chat", help="Interactive chat demo")
    add_common_arguments(chat_parser)
    chat_parser.add_argument("--checkpoint", required=True)
    chat_parser.add_argument("--prompt", default=None)
    chat_parser.add_argument("--image", default=None, help="Path to image file")
    chat_parser.add_argument("--audio", default=None, help="Path to audio file")
    chat_parser.add_argument("--video", default=None, help="Path to video file")
    chat_parser.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=50)
    chat_parser.add_argument("--temperature", type=float, default=0.9)
    chat_parser.add_argument("--top-k", dest="top_k", type=int, default=50)

    video_parser = subparsers.add_parser("video-test", help="Test video preprocessing pipeline")
    add_common_arguments(video_parser)
    video_parser.add_argument("--video-path", required=True)
    video_parser.add_argument("--num-frames", type=int, default=16)
    video_parser.add_argument("--frame-size", type=int, default=224)
    video_parser.add_argument("--sampling", choices=["uniform", "random", "adaptive"], default="uniform")
    video_parser.add_argument("--checkpoint", default=None)

    export_parser = subparsers.add_parser("export-config", help="Write default config to JSON")
    add_common_arguments(export_parser)
    export_parser.add_argument("--output", default="ixlinx_default_config.json")

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.mode == "train":
        cli_train(args)
    elif args.mode == "eval":
        cli_eval(args)
    elif args.mode == "chat":
        cli_chat(args)
    elif args.mode == "video-test":
        cli_video_test(args)
    elif args.mode == "export-config":
        cli_export_config(args)
    else:  
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
