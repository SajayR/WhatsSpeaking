import torch
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
import time
import numpy as np
import gc
import resource

# Local imports
from viz import create_visualization_video  # We'll assume we adapted viz.py for cross-attn
from dataset import VideoAudioDataset
from model import ValoAR  # <-- Your new autoregressive model (rename as needed)

DO_WANDB = True

# Increase file limit if needed
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
    vis_interval=1000,      # steps between visualizations
    checkpoint_interval=5000,
    output_dir="./outputs",
    resume_from=None,
    warmup_steps=5000
):
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Cosine learning rate scheduler with warmup
    steps_per_epoch = len(dataset) // batch_size
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs * steps_per_epoch,
        eta_min=learning_rate * 0.01
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=False,
        prefetch_factor=8
    )

    # Prepare a few samples for recurring visualization
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
            project="PoopyPants",  # update as needed
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
            # Warmup schedule
            if global_step < warmup_steps:
                lr = learning_rate * (global_step + 1) / warmup_steps
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            # Move batch to device
            frames = batch['frames'].to(device)   # [B, 3, 224, 224]
            audio = batch['audio'].to(device)     # [B, 40]

            # Forward pass
            # Returns:
            #   loss: scalar
            #   logits: [B, T, 4096]
            #   cross_attn: [B, T, 256] (if implemented in your model)
            loss, logits, cross_attn = model(frames, audio)
            epoch_losses.append(loss.item())

            # Optionally compute a simple token-level accuracy
            with torch.no_grad():
                preds = logits.argmax(dim=-1)          # [B, T]
                step_acc = (preds == audio).float().mean().item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Scheduler step after warmup
            if global_step >= warmup_steps:
                scheduler.step()

            # Log metrics
            if DO_WANDB:
                wandb.log({
                    "train_loss": loss.item(),
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "epoch": epoch,
                    "global_step": global_step,
                    "step_acc": step_acc
                }, step=global_step)

            # Visualization step
            if (global_step % vis_interval == 0) and (global_step > 0):
                model.eval()
                with torch.no_grad():
                    # Use the same teacher forcing approach on the visualization samples
                    _, viz_logits, viz_attn = model(vis_samples['frames'], vis_samples['audio'])
                    print("\nVisualization Sample Info:")
                    print(f" cross_attn shape: {viz_attn.shape}")   # [vis_batch, T, 256]?
                    print(f" logits shape: {viz_logits.shape}")     # [vis_batch, T, 4096]

                    # Show min/max of cross_attn for debugging
                    min_val = viz_attn.min().item()
                    max_val = viz_attn.max().item()
                    print(f" cross_attn range: {min_val:.4f} to {max_val:.4f}")

                    # Save a short video for the first sample
                    out_video_path = output_dir / f"visualization_step{global_step}.mp4"
                    create_visualization_video(
                        vis_samples['frames'][0],
                        viz_attn[0],         # shape [T, 256]
                        out_video_path,
                        fps=40
                    )

                model.train()
                torch.cuda.empty_cache()
                gc.collect()

            # Save checkpoint
            if (global_step % checkpoint_interval == 0) and (global_step > 0):
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

        # End of epoch summary
        epoch_time = time.time() - start_time
        avg_loss = np.mean(epoch_losses)
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Time: {epoch_time:.2f}s")
        print(f"  Steps: {global_step}")

        if DO_WANDB:
            wandb.log({
                "epoch_loss": avg_loss,
                "epoch_time": epoch_time
            }, step=global_step)


if __name__ == "__main__":
    # Training parameters
    NUM_EPOCHS = 50
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    VIS_INTERVAL = 1000
    CHECKPOINT_INTERVAL = 5000
    OUTPUT_DIR = "./outputs"
    WARMUP_STEPS = 5000

    # Create dataset and model
    dataset = VideoAudioDataset("/home/cis/VGGSound_Splits")
    model = ValoAR()  # Your new autoregressive model

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
        # or None if you don't want to resume
    )
