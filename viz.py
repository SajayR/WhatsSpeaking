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
    Reverses ImageNet normalization on a frame while keeping size at 224x224
    Args:
        frame: [3, 224, 224] normalized tensor
    Returns:
        frame: [3, 224, 224] unnormalized tensor (0-255 range)
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(frame.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(frame.device)
    frame = frame * std + mean
    frame = (frame * 255).clamp(0, 255).to(torch.uint8)
    return frame

def get_token_heatmap(patch_probs, timestep):
    """
    Args:
        patch_probs: [256, 40] probabilities for each patch over time
        timestep: which time step to visualize
    Returns:
        heatmap: [16, 16] probability map for this timestep
    """
    token_probs = patch_probs[:, timestep]  # [256]
    heatmap = token_probs.view(16, 16)  # Reshape to spatial grid
    return heatmap

def create_sequential_heatmaps(patch_probs):
    """
    Args:
        patch_probs: [256, 40] probabilities for all patches over time
    Returns:
        heatmap_sequence: [40, 16, 16] sequence of spatial heatmaps
    """
    heatmap_sequence = []
    
    for t in range(patch_probs.shape[1]):  # Loop through timesteps
        heatmap = get_token_heatmap(patch_probs, t)
        heatmap_sequence.append(heatmap)
        
    return torch.stack(heatmap_sequence)  # [40, 16, 16]

def overlay_heatmap_on_frame(frame, heatmap, alpha=0.6):
    """
    Args:
        frame: [3, 224, 224] uint8 tensor
        heatmap: [16, 16] probability tensor 
        alpha: transparency of overlay
    Returns:
        overlaid_frame: [224, 224, 3] numpy array (BGR for cv2)
    """
    # Convert frame to numpy and correct channel order
    frame = frame.permute(1, 2, 0).cpu().numpy()  # [224, 224, 3]
    
    # Resize heatmap to frame size
    heatmap = heatmap.cpu().numpy()
    heatmap = cv2.resize(heatmap, (224, 224))
    
    # Apply colormap (let's use COLORMAP_JET for now)
    heatmap = cv2.applyColorMap(
        (heatmap * 255).astype(np.uint8), 
        cv2.COLORMAP_JET
    )
    
    # Overlay
    overlaid = cv2.addWeighted(frame, 1-alpha, heatmap, alpha, 0)
    return overlaid


def create_visualization_video(frame, patch_probs, output_path, fps=40):
    """
    Creates and saves a visualization video showing audio localizations over time
    
    Args:
        frame: [3, 224, 224] normalized tensor
        patch_probs: [256, 40] patch probabilities over time
        output_path: path to save the video
        fps: frames per second (40 for 1:1 with audio tokens)
    """
    # Unnormalize the input frame
    frame = unnormalize_frame(frame)
    
    # Get sequence of heatmaps
    heatmap_sequence = create_sequential_heatmaps(patch_probs)
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        str(output_path), 
        fourcc, 
        fps, 
        (224, 224)  # frame size
    )
    
    # Create and write each frame
    for heatmap in heatmap_sequence:
        overlaid_frame = overlay_heatmap_on_frame(frame, heatmap)
        video_writer.write(overlaid_frame)
    
    video_writer.release()


def process_batch_visualizations(frames, patch_probs, file_paths, output_dir, step=None):
    """
    Process visualization for a batch of samples
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    for i in range(len(frames)):
        if step is not None:
            output_path = output_dir / f"{Path(file_paths[i]).stem}_step{step}_viz.mp4"
        else:
            output_path = output_dir / f"{Path(file_paths[i]).stem}_viz.mp4"
            
        create_visualization_video(
            frames[i],
            patch_probs[i],
            output_path
        )

def create_snapshot_grid(frames, patch_probs, n_samples=6):
    """
    Creates visualization grid with evenly spaced temporal samples
    
    Args:
        frames: [B, 3, 224, 224] normalized tensor
        patch_probs: [B, 256, 40] patch probabilities over time
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
        
        # Get sequence of heatmaps
        heatmap_sequence = create_sequential_heatmaps(patch_probs[b])
        
        # Sample indices evenly from sequence
        n_samples = min(n_samples, len(heatmap_sequence) + 1)
        indices = torch.linspace(0, len(heatmap_sequence)-1, n_samples-1).long()
        
        # Create frames for each sampled heatmap
        sample_frames = [frame.clone()]
        for idx in indices:
            overlaid = overlay_heatmap_on_frame(frame, heatmap_sequence[idx])
            overlaid = torch.from_numpy(overlaid).permute(2, 0, 1)
            sample_frames.append(overlaid)
            
        # Create grid layout
        grid_size = math.ceil(math.sqrt(n_samples))
        h, w = 224, 224
        sample_grid = torch.zeros(3, grid_size * h, grid_size * w)
        
        for idx, f in enumerate(sample_frames):
            i = idx // grid_size
            j = idx % grid_size
            sample_grid[:, i*h:(i+1)*h, j*w:(j+1)*w] = f
            
        sample_grids.append(sample_grid)
    
    final_grid = torch.cat(sample_grids, dim=1)
    return final_grid

def save_snapshot_grid(frames, patch_probs, output_dir, step=None, n_samples=6):
    """
    Creates and saves snapshot grid
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    grid = create_snapshot_grid(frames, patch_probs, n_samples)
    grid_np = grid.permute(1, 2, 0).numpy()
    
    if step is not None:
        output_path = output_dir / f"snapshot_grid_step{step}.png"
    else:
        output_path = output_dir / "snapshot_grid.png"
        
    cv2.imwrite(str(output_path), cv2.cvtColor(grid_np, cv2.COLOR_RGB2BGR))
    return grid

if __name__ == "__main__":
    # Test setup
    video_path = "0_2.mp4"
    real_frame = load_test_frame(video_path)
    
    # Create test batch
    batch_size = 2
    frames = real_frame.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    
    # Create random patch probabilities [B, 256, 40]
    patch_probs = torch.rand(batch_size, 256, 40)
    
    print("\nInput shapes:")
    print(f"Frames: {frames.shape}")
    print(f"Patch probs: {patch_probs.shape}")
    
    file_paths = ["0_2.mp4", "0_2.mp4"]
    test_steps = [0, 100, 500]
    
    for step in test_steps:
        # Test grid visualization
        save_snapshot_grid(frames, patch_probs, 
                         './test_visualizations', 
                         step=step)
        print(f"Snapshot grid saved for step {step}")
        
        # Test video visualization
        process_batch_visualizations(
            frames, 
            patch_probs, 
            file_paths, 
            Path("./test_visualizations"),
            step=step
        )
        print(f"Videos saved for step {step}")