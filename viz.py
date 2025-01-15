import torch
import cv2
import numpy as np
from pathlib import Path
from torchcodec.decoders import VideoDecoder
import torchvision.transforms as transforms

def load_test_frame(video_path):
    """
    Load a single frame from a real video for testing
    """
    decoder = VideoDecoder(video_path, device="cpu")
    frame = decoder.get_frames_played_at(seconds=[0.5]).data  # mid-point frame
    
    # Apply same transforms as dataset.py
    transform = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.Lambda(lambda x: x / 255.0),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    frame_tensor = transform(frame)
    return frame_tensor.squeeze(0)  # [3, 224, 224]

def unnormalize_frame(frame):
    """
    Same as your existing code
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(frame.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(frame.device)
    frame = frame * std + mean
    frame = (frame * 255).clamp(0, 255).to(torch.uint8)
    return frame

def get_token_heatmap(patch_probs, token_idx):
    """
    Args:
        patch_probs: [256, 4096] probabilities for each patch
        token_idx: scalar index of the token
    Returns:
        heatmap: [16, 16] probability map for this token
    """
    token_probs = patch_probs[:, token_idx]  # [256]
    heatmap = token_probs.view(16, 16)  # Reshape to spatial grid
    return heatmap

def create_sequential_heatmaps(cross_attn, num_steps=40):
    """
    Instead of patch_probs [256, 4096], we have cross_attn [T, 256].
    We assume T==num_steps. We'll reshape each step from [256] -> [16, 16].
    """
    # cross_attn is [T, 256]
    # We'll just gather each slice
    heatmap_sequence = []
    for t in range(num_steps):
        attn_t = cross_attn[t]  # [256]
        # Reshape to [16, 16]
        attn_grid = attn_t.view(16, 16)
        # Optionally normalize to 0..1
        attn_grid = (attn_grid - attn_grid.min()) / (attn_grid.max() - attn_grid.min() + 1e-6)
        heatmap_sequence.append(attn_grid)
    return heatmap_sequence  # list of length T, each is [16, 16]

def overlay_heatmap_on_frame(frame, heatmap, alpha=0.6):
    """
    No real change except we pass in [16,16] for heatmap
    """
    frame_np = frame.permute(1, 2, 0).cpu().numpy()  # [224,224,3]
    
    heatmap_np = heatmap.cpu().numpy()
    heatmap_np = cv2.resize(heatmap_np, (224, 224))
    
    heatmap_color = cv2.applyColorMap(
        (heatmap_np * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )
    overlaid = cv2.addWeighted(frame_np, 1-alpha, heatmap_color, alpha, 0)
    return overlaid


def create_visualization_video(frame, cross_attn, output_path, fps=40):
    """
    - 'frame' is [3,224,224]
    - 'cross_attn' is [T, 256] => T steps
    """
    # 1) Unnormalize
    frame_unnorm = unnormalize_frame(frame)
    
    # 2) Build heatmap sequence
    T = cross_attn.size(0)
    heatmaps = create_sequential_heatmaps(cross_attn, num_steps=T)

    # 3) Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (224, 224))

    # 4) Write each step's overlaid frame
    for t in range(T):
        overlaid = overlay_heatmap_on_frame(frame_unnorm, heatmaps[t])
        video_writer.write(overlaid)
    video_writer.release()


def process_batch_visualizations(frames, audio_tokens, patch_probs, file_paths, output_dir, step=None):
    """
    Added step parameter for unique file naming
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    for i in range(len(frames)):
        # Add step number to filename if provided
        if step is not None:
            output_path = output_dir / f"{Path(file_paths[i]).stem}_step{step}_viz.mp4"
        else:
            output_path = output_dir / f"{Path(file_paths[i]).stem}_viz.mp4"
            
        create_visualization_video(
            frames[i],
            audio_tokens[i],
            patch_probs[i],
            output_path
        )

def create_snapshot_grid(frames, audio_tokens, patch_probs, n_samples=6):
    """
    Creates grids of n_samples evenly spaced frames for each sample in batch,
    stacked vertically
    
    Args:
        frames: [B, 3, 224, 224] normalized tensor
        audio_tokens: [B, 40] sequence of tokens
        patch_probs: [B, 256, 4096] patch probabilities
        n_samples: number of snapshots to include (including original frame)
    
    Returns:
        grid: [3, H*B, W] tensor where H is computed to fit n_samples in a grid
    """
    import math
    
    batch_size = frames.size(0)
    sample_grids = []
    
    for b in range(batch_size):
        # Unnormalize frame
        frame = unnormalize_frame(frames[b])
        
        # Get full sequence of heatmaps for this sample
        heatmap_sequence = create_sequential_heatmaps(patch_probs[b], audio_tokens[b])
        
        # Sample indices evenly from sequence
        n_samples = min(n_samples, len(heatmap_sequence) + 1)
        indices = torch.linspace(0, len(heatmap_sequence)-1, n_samples-1).long()
        
        # Create frames for each sampled heatmap
        sample_frames = [frame.clone()]
        for idx in indices:
            overlaid = overlay_heatmap_on_frame(frame, heatmap_sequence[idx])
            overlaid = torch.from_numpy(overlaid).permute(2, 0, 1)
            sample_frames.append(overlaid)
            
        # Compute grid dimensions for one sample
        grid_size = math.ceil(math.sqrt(n_samples))
        
        # Create grid for this sample
        h, w = 224, 224
        sample_grid = torch.zeros(3, grid_size * h, grid_size * w)
        
        for idx, f in enumerate(sample_frames):
            i = idx // grid_size
            j = idx % grid_size
            sample_grid[:, i*h:(i+1)*h, j*w:(j+1)*w] = f
            
        sample_grids.append(sample_grid)
    
    # Stack all sample grids vertically
    final_grid = torch.cat(sample_grids, dim=1)
    return final_grid

def save_snapshot_grid(frames, audio_tokens, patch_probs, output_dir, step=None, n_samples=6):
    """
    Creates and saves snapshot grid with step number in filename
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    grid = create_snapshot_grid(frames, audio_tokens, patch_probs, n_samples)
    grid_np = grid.permute(1, 2, 0).numpy()
    
    # Add step number to filename if provided
    if step is not None:
        output_path = output_dir / f"snapshot_grid_step{step}.png"
    else:
        output_path = output_dir / "snapshot_grid.png"
        
    cv2.imwrite(str(output_path), cv2.cvtColor(grid_np, cv2.COLOR_RGB2BGR))
    return grid

def test_visualization():
    # Load a real frame from one of your videos
    video_path = "0_2.mp4"  # Use any video path
    real_frame = load_test_frame(video_path)
    
    # Create batch with the real frame repeated
    batch_size = 2
    frames = real_frame.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    
    # Random audio tokens (integers between 0-4095)
    audio_tokens = torch.randint(0, 4096, (batch_size, 40))
    
    # Random patch probabilities (after sigmoid, so between 0 and 1)
    patch_probs = torch.rand(batch_size, 256, 4096)
    
    print("Input shapes:")
    print(f"Frames: {frames.shape}")
    print(f"Audio tokens: {audio_tokens.shape}")
    print(f"Patch probs: {patch_probs.shape}")

    file_paths = [
        "0_2.mp4",
        "0_2.mp4"
    ]
    # Use the batched frames instead of real_frame
    steps = [0, 100, 500]
    for step in steps:
        # Save snapshot grid
        save_snapshot_grid(frames, audio_tokens, patch_probs, 
                         '/home/cis/heyo/AudTok/WhosSpeaking/test_visualizations', 
                         step=step)
        print(f"\nSnapshot grid saved for step {step}")
        
        # Save videos
        process_batch_visualizations(
            frames, 
            audio_tokens, 
            patch_probs, 
            file_paths, 
            Path("/home/cis/heyo/AudTok/WhosSpeaking/test_visualizations"),
            step=step
        )
        print(f"Videos saved for step {step}")

if __name__ == "__main__":
    test_visualization()