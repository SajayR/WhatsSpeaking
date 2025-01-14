import sys
from pathlib import Path
import io
import warnings
import multiprocessing
import subprocess
import random
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
import torchaudio
import torchaudio.transforms as T
from torchcodec.decoders import VideoDecoder
import torchvision.transforms as transforms
# Local imports
sys.path.append("/home/cis/heyo/AudTok/WavTokenizer")
from decoder.pretrained import WavTokenizer
warnings.filterwarnings("ignore")
multiprocessing.set_start_method('spawn', force=True)
# Constants
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)

def quick_audio_load(video_path):
    # Extract mono audio at 24kHz
    cmd = ['ffmpeg','-hide_banner', '-i', video_path, '-ac', '1', '-ar', '24000', '-f', 'wav', '-', '-loglevel', 'error']  # Added -loglevel error
    #audio_buffer = io.BytesIO(subprocess.check_output(cmd))
    audio_buffer = io.BytesIO(subprocess.check_output(cmd, stderr=subprocess.DEVNULL))

    waveform, sample_rate = torchaudio.load(audio_buffer)
    assert sample_rate == 24000
    assert waveform.size(0) == 1  # mono
    
    return waveform, sample_rate

def extract_audio_from_video(video_path: Path) -> torch.Tensor:
    """Extract entire 1s audio from video."""
    device = torch.device("cpu")
    config_path = "/home/cis/heyo/AudTok/WavTokenizer/checkpoints/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
    model_path = "/home/cis/heyo/AudTok/WavTokenizer/checkpoints/WavTokenizer_small_600_24k_4096.ckpt"
    wavtokenizer = WavTokenizer.from_pretrained0802(config_path, model_path)
    wavtokenizer = wavtokenizer.to(device)
    bandwidth_id = torch.tensor([0])
    try:
        waveform, sample_rate = quick_audio_load(video_path)
        waveform = waveform[:, :24000*1]
        waveform = waveform.to(device)
        _,discrete_code= wavtokenizer.encode_infer(waveform, bandwidth_id=bandwidth_id)
        return discrete_code.squeeze(0) # (1, 40) 
    except:
        print(f"Failed to load audio from {video_path}")
        return torch.zeros(1, 40)

def load_and_preprocess_video(video_path: str) -> torch.Tensor:
    decoder = VideoDecoder(video_path, device="cpu")
    time = random.uniform(0, 1)
    frame = decoder.get_frames_played_at(seconds=[time]).data # [B, C, H, W]
    transform = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.Lambda(lambda x: x / 255.0),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    frame_tensor = transform(frame)
    frame_tensor = frame_tensor.to(torch.device("cuda"))
    del decoder
    return frame_tensor

class VideoAudioDataset(Dataset):
    def __init__(self, video_dir):
        self.video_paths = list(Path(video_dir).glob('*.mp4'))  # or whatever pattern matches your files
        self.audio_tokens_path = Path("/home/cis/VGGSound_sound_tokens/")
        
    def __len__(self):
        return len(self.video_paths)
        
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        audio = torch.load(self.audio_tokens_path / f"{video_path.stem}.pt").squeeze()
        frames = load_and_preprocess_video(str(video_path)).squeeze()  # This needs updating
        #print(audio)
        return {
            'path': str(video_path),
            'frames': frames,
            'audio': audio
        }

if __name__ == "__main__":
    # Test with a small directory first
    video_dir = "/home/cis/VGGSound_Splits"
    dataset = VideoAudioDataset(video_dir)
    
    # Let's look at one sample
    sample = dataset[0]
    print(f"Video path: {sample['path']}")
    print(f"Frames shape: {sample['frames'].shape}")  # Should be like [C, H, W]
    print(f"Audio shape: {sample['audio'].shape}")    # Your discrete codes [40]
    
    # Test with DataLoader to make sure batching works
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2, persistent_workers=True)
    import time
    start = time.time()
    for batch in loader:
        print("\nBatch info:")
        #print(f"Batch paths: {batch['path']}")
        print(f"Batch frames: {batch['frames'].shape}") #
        print(f"Batch audio: {batch['audio'].shape}")
        end = time.time()
        print(f"Time taken: {end - start}")
        start = end
          # Just test one batch
