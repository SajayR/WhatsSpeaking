import torch
from torch.utils.data import Dataset, Sampler
from pathlib import Path
import numpy as np
import random
import av
from typing import Dict, List
import torch.nn as nn
import torchaudio.transforms as T
import warnings
warnings.filterwarnings("ignore")
import multiprocessing
#import dataloader
from torchcodec.decoders import VideoDecoder
from torch.utils.data import DataLoader
# Attempt to use fork for potentially faster dataloader start
#try:
    #multiprocessing.set_start_method('fork', force=True)
#except:
multiprocessing.set_start_method('spawn', force=True)
import gc
import sys
sys.path.append("/home/cis/heyo/AudTok/WavTokenizer")
from decoder.pretrained import WavTokenizer
from encoder.utils import convert_audio

# Global normalization constants (ImageNet)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)


def extract_audio_from_video(video_path: Path) -> torch.Tensor:
    """Extract entire 1s audio from video."""
    #print("extracting audio from video")
    device = torch.device("cuda")
    config_path = "/home/cis/heyo/AudTok/WavTokenizer/checkpoints/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
    model_path = "/home/cis/heyo/AudTok/WavTokenizer/checkpoints/WavTokenizer_small_600_24k_4096.ckpt"
    wavtokenizer = WavTokenizer.from_pretrained0802(config_path, model_path)
    wavtokenizer = wavtokenizer.to(device)
    bandwidth_id = torch.tensor([0])
    try:
        container = av.open(str(video_path))
        audio = container.streams.audio[0]
        resampler = av.audio.resampler.AudioResampler(format='s16', layout='mono', rate=24000)
        
        samples = []
        for frame in container.decode(audio):
            frame.pts = None
            frame = resampler.resample(frame)[0]
            samples.append(frame.to_ndarray().reshape(-1))
        container.close()

        samples = torch.tensor(np.concatenate(samples))#[:24000*1] #only take first second
        samples = samples.float() / 32768.0  # Convert to float and normalize
        samples = samples.unsqueeze(0).to(device)
        print(samples.shape)
        _,discrete_code= wavtokenizer.encode_infer(samples, bandwidth_id=bandwidth_id)
        return discrete_code.squeeze(0) # (1, 40) 
        
    except:
        print(f"Failed to load audio from {video_path}")
        return torch.zeros(16331)
    finally:
        if container:
            container.close()


def load_and_preprocess_video(video_path: str) -> torch.Tensor:
    decoder = VideoDecoder(video_path, device="cpu")
    #picking random time between 0 and 1 second
    time = random.uniform(0, 1)
    frame = decoder.get_frames_played_at(seconds=[time]).data # [B, C, H, W]
    transforms = T.Compose([
        T.Resize((224, 224), antialias=True),
        T.Lambda(lambda x: x / 255.0),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    frame_tensor = transforms(frame)
    frame_tensor = frame_tensor.to(torch.device("cuda"))
    return frame_tensor


if __name__ == "__main__":
    video_path = "/home/cis/VGGSound_Splits/0_0.mp4"
    audio = extract_audio_from_video(video_path)
    print(audio.shape)