import sys
from pathlib import Path
import io
import warnings
import multiprocessing
import subprocess
import torch
import torchaudio
from tqdm import tqdm

# Make sure you have access to your WavTokenizer code
sys.path.append("/home/cis/heyo/AudTok/WavTokenizer")
from decoder.pretrained import WavTokenizer

warnings.filterwarnings("ignore")

# -------------------------------------------------------------------
# 1) Adapted from your existing code
# -------------------------------------------------------------------
def quick_audio_load(video_path):
    """
    Extract mono audio at 24kHz from the given `video_path`.
    Returns:
        waveform (Tensor): shape [1, N]
        sample_rate (int): should be 24000
    """
    cmd = [
        'ffmpeg', '-hide_banner',
        '-i', str(video_path),
        '-ac', '1',
        '-ar', '24000',
        '-f', 'wav', '-',
        '-loglevel', 'error'
    ]
    audio_buffer = io.BytesIO(subprocess.check_output(cmd, stderr=subprocess.DEVNULL))

    waveform, sample_rate = torchaudio.load(audio_buffer)
    assert sample_rate == 24000, f"Expected 24000Hz, got {sample_rate}"
    assert waveform.size(0) == 1, f"Expected mono audio, got shape {waveform.shape}"
    return waveform, sample_rate


def tokenize_audio_from_video(video_path: Path, wavtokenizer, max_seconds=1):
    """
    Loads 1 second of audio from `video_path`,
    then uses `wavtokenizer` to extract discrete tokens.
    """
    try:
        waveform, sample_rate = quick_audio_load(video_path)
        # Trim or pad wave to exactly 1 second
        max_length = 24000 * max_seconds  # 24000 = sample_rate
        # Just slice to 1s (your original code did waveform[:, :24000*1])
        waveform = waveform[:, :max_length]

        # Send to CPU or GPU as you prefer -- here we keep it on CPU
        waveform = waveform.to("cuda")

        # Example bandwidth_id = 0. Feel free to make this variable if needed
        bandwidth_id = torch.tensor([0], device="cuda")

        # WavTokenizer encode_infer
        # Returns (embedding, discrete_code)
        _, discrete_code = wavtokenizer.encode_infer(waveform, bandwidth_id=bandwidth_id)
        # discrete_code is shape [1, 40], so squeeze batch dim to get [40]
        return discrete_code.squeeze(0)  
    except Exception as e:
        print(f"Failed to load/tokenize audio from {video_path}: {e}")
        # Return a dummy tensor of shape [40] if you want to keep everything consistent
        return torch.zeros(40, dtype=torch.long)

# -------------------------------------------------------------------
# 2) Multiprocessing initialization
# -------------------------------------------------------------------
# We'll load the tokenizer once per worker via `init_tokenizer`.
def init_tokenizer(config_path, model_path):
    """
    This function is called ONCE in each worker, creating a global `wavtokenizer`.
    """
    global wavtokenizer
    wavtokenizer = WavTokenizer.from_pretrained0802(config_path, model_path).to("cuda")

def process_video(video_path):
    """
    The worker job: tokenize the audio from `video_path` and save to disk.
    Note that `wavtokenizer` is taken from the global scope in each worker.
    """
    global wavtokenizer
    tokens_cache_dir = Path("/home/cis/VGGSound_sound_tokens")
    try:
        # Where you want to save the tokens, e.g., tokens_cache/<video_name>.pt
        out_name = video_path.stem + ".pt"
        out_path = tokens_cache_dir / out_name

        # To avoid re-processing if you've already got the file
        if out_path.exists():
            return

        # Actually do the tokenization
        discrete_code = tokenize_audio_from_video(video_path, wavtokenizer)
        
        # Save the discrete_code to out_path
        # (You could also store a dict with multiple keys if desired)
        torch.save(discrete_code, out_path)
    except Exception as e:
        print(f"[process_video] Error processing {video_path} : {e}")

# -------------------------------------------------------------------
# 3) Main
# -------------------------------------------------------------------
if __name__ == "__main__":
    # If you need 'spawn' (sometimes necessary on certain platforms)
    multiprocessing.set_start_method('spawn', force=True)

    # Modify these variables as desired (instead of using argparse)
    video_dir = Path("/home/cis/VGGSound_Splits")
    tokens_cache_dir = Path("/home/cis/VGGSound_sound_tokens")
    tokens_cache_dir.mkdir(exist_ok=True, parents=True)  # Ensure dir exists

    # Path to your WavTokenizer config & model
    config_path = "/home/cis/heyo/AudTok/WavTokenizer/checkpoints/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
    model_path = "/home/cis/heyo/AudTok/WavTokenizer/checkpoints/wavtokenizer_large_unify_600_24k.ckpt"

    # Number of parallel workers
    num_workers = 16

    # Gather all .mp4 files
    video_paths = list(video_dir.glob("*.mp4"))
    print(f"Found {len(video_paths)} .mp4 files in {video_dir}")

    # Create a Pool that initializes one tokenizer per process
    with multiprocessing.Pool(
        processes=num_workers,
        initializer=init_tokenizer,
        initargs=(config_path, model_path)
    ) as pool:

        # Use imap or imap_unordered for a nice TQDM progress bar
        # (In Python 3.9+, you can specify total in tqdm)
        for _ in tqdm(
            pool.imap_unordered(process_video, video_paths),
            total=len(video_paths),
            desc="Tokenizing Audio"
        ):
            pass

    print("All audio tokens have been saved in:", tokens_cache_dir)
