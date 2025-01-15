import torch
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
import time
import numpy as np
from viz import save_snapshot_grid, create_visualization_video
import gc
from dataset import VideoAudioDataset
from model import SequentialAudioVisualModel
DO_WANDB = False
import resource
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
print(f"File limit increased from {soft} to {hard}")

def get_visualization_samples(dataset, num_samples=16):
    """
    Get fixed samples for visualization throughout training
    Returns dict with original indices and batched data
    """
    vis_indices = torch.randperm(len(dataset))[:num_samples]
    vis_samples = {
        'indices': vis_indices,
        'data': [dataset[i] for i in vis_indices]
    }
    
    # Collate into batched tensors
    frames = torch.stack([s['frames'] for s in vis_samples['data']])
    audio = torch.stack([s['audio'] for s in vis_samples['data']])
    paths = [s['path'] for s in vis_samples['data']]
    
    return {
        'indices': vis_indices,
        'frames': frames,
        'audio': audio,
        'paths': paths
    }

def train(
    model, 
    dataset,
    num_epochs=25,
    batch_size=32,
    learning_rate=1e-4,
    vis_interval=1000,  # steps between visualizations
    checkpoint_interval=5000,
    output_dir="./outputs",
    resume_from=None,
    warmup_steps=5000,
):
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Cosine learning rate scheduler with warmup
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=num_epochs * len(dataset) // batch_size,
        eta_min=learning_rate * 0.01
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        persistent_workers=False,
        prefetch_factor=8
        #pin_memory=True
    )
    
    # Get visualization samples
    vis_samples = get_visualization_samples(dataset)
    for k in ['frames', 'audio']:
        vis_samples[k] = vis_samples[k].to(device)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0
    best_loss = float('inf')
    if resume_from:
        print(f"Loading checkpoint from {resume_from}")
        ckpt = torch.load(resume_from)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch']
        global_step = ckpt['global_step']
        best_loss = ckpt['best_loss']
        print(f"Resumed from epoch {start_epoch}, step {global_step}")
    
    # Initialize wandb
    if DO_WANDB:
        wandb.init(
            project="PoopyPants",
            config={
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "warmup_steps": warmup_steps,
                "model_name": model.__class__.__name__,
                "dataset_size": len(dataset)
            }
        )
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_losses = []
        start_time = time.time()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            # Warmup learning rate
            if global_step < warmup_steps:
                lr = learning_rate * (global_step + 1) / warmup_steps
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            
            # Move batch to device
            frames = batch['frames'].to(device)
            audio = batch['audio'].to(device)
            
            # Forward pass
            #print("Going to forward pass")
            loss = model.compute_loss(frames, audio)
            #print("Forward pass complete")
            epoch_losses.append(loss.item())
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if global_step >= warmup_steps:
                scheduler.step()
            
            # Log metrics
            if DO_WANDB:
                wandb.log({
                    "train_loss": loss.item(),
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "epoch": epoch,
                    "global_step": global_step,
                }, step=global_step)
            
            # Visualization step
            if global_step % vis_interval == 0 and global_step > 0:
                model.eval()
                with torch.no_grad():
                    patch_probs = model(vis_samples['frames'], vis_samples['audio'], return_all_logits=False)
                    # Debug prediction statistics
                    print("\nVisualization Sample Statistics:")
                    print(f"Patch probs shape: {patch_probs.shape}")  # Should be [batch, 256, 4096]
                    probs = patch_probs
                    
                    print(f"Overall stats:")
                    print(f"Mean prob: {probs.mean():.4f}")
                    print(f"Prob >0.5: {(probs > 0.5).float().mean():.4f}")
                    print(f"Prob >0.7: {(probs > 0.7).float().mean():.4f}")
                    print(f"Prob >0.9: {(probs > 0.9).float().mean():.4f}")
            
                    
                print(f"Range of patch probs: {patch_probs.min().item()} to {patch_probs.max().item()}")
                grid = save_snapshot_grid(
                    vis_samples['frames'], 
                    patch_probs, # now is of shape [B, 256, 40]
                    output_dir,
                    step=global_step
                )
                #saving video for the first sample
                #create_visualization_video(vis_samples['frames'][0], vis_samples['audio'][0], patch_probs[0], output_dir / f"visualization_step{global_step}.mp4")
                # Log to wandb
                if DO_WANDB:
                    wandb.log({
                        "visualizations": wandb.Image(grid)
                    }, step=global_step)
                
                model.train()
                torch.cuda.empty_cache()
                gc.collect()
            
            # Save checkpoint
            if global_step % checkpoint_interval == 0 and global_step > 0:
                avg_loss = np.mean(epoch_losses)
                is_best = avg_loss < best_loss
                best_loss = min(avg_loss, best_loss)
                
                ckpt_path = output_dir / f"checkpoint_step{global_step}.pt"
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'global_step': global_step,
                    'best_loss': best_loss
                }, ckpt_path)
                
                if is_best:
                    best_path = output_dir / "best_model.pt"
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch,
                        'global_step': global_step,
                        'best_loss': best_loss
                    }, best_path)
            
            global_step += 1
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
            })
        
        # Epoch summary
        epoch_time = time.time() - start_time
        avg_loss = np.mean(epoch_losses)
        print(f"\nEpoch {epoch} Summary:")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Time: {epoch_time:.2f}s")
        print(f"Steps: {global_step}")
        if DO_WANDB:
            wandb.log({
                "epoch_loss": avg_loss,
                "epoch_time": epoch_time
            }, step=global_step)

if __name__ == "__main__":
    # Training parameters
    NUM_EPOCHS = 50
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    VIS_INTERVAL = 1000
    CHECKPOINT_INTERVAL = 5000
    OUTPUT_DIR = "./outputsAR"
    WARMUP_STEPS = 5000
    
    # Initialize dataset and model
    dataset = VideoAudioDataset("/home/cis/VGGSound_Splits")
    model = SequentialAudioVisualModel()
    
    # Start training
    train(
        model=model,
        dataset=dataset,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        vis_interval=VIS_INTERVAL,
        checkpoint_interval=CHECKPOINT_INTERVAL,
        output_dir=OUTPUT_DIR,
        warmup_steps=WARMUP_STEPS,
        resume_from=None
    )