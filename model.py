import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor
import warnings
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
warnings.filterwarnings("ignore")

class ViTEmbedder(nn.Module):
    """
    Uses a DINOv2 model (vits14) to produce patch embeddings.
    Returns shape: [B, 256, D] (256 patches, D embedding dim).
    """
    def __init__(self):
        super().__init__()
        # Load DINOv2 VIT-S/14
        # This model outputs a list of patch embeddings
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        
        # This embed_dim depends on the pre-trained model
        self.embed_dim = self.model.embed_dim

        # Freeze or unfreeze as you wish
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, frames):
        """
        Args:
            frames: [B, 3, 224, 224] float
        Returns:
            patch_embs: [B, 256, D] patch embeddings
        """
        # get_intermediate_layers(..., n=1) returns a list with 1 item,
        # which is [B, 256, D] for ViT-S/14 (16x16 = 256 patches).
        patch_embs = self.model.get_intermediate_layers(frames, n=1)[0]  
        return patch_embs
    
class TokenEncoder(nn.Module):
    def __init__(self, vocab_size=4096, d_model=256, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, 40, d_model) * 0.02)
        self.d_model = d_model
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
    
    def forward(self, tokens):
        # tokens: [B, T] 
        x = self.embedding(tokens)  # [B, T, D]
        x = x + self.pos_embedding[:, :x.size(1), :]
        return self.transformer(x)  # [B, T, D]

class SequentialAudioVisualModel(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        # DINO feature dimension
        self.dino_dim = 384  # ViT-Small
        self.dino_model = ViTEmbedder()
        
        # Token processing
        self.token_encoder = TokenEncoder(d_model=d_model)
        
        # Fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.dino_dim + d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        # Final classifier
        self.classifier = nn.Linear(d_model, 4096)
    
    def fuse_patches_context(self, patches, context):
        # patches: [B, 256, dino_dim]
        # context: [B, 1, d_model]
        
        # Expand context to match patch dimension
        context = context.expand(-1, patches.size(1), -1)
        
        # Concatenate and fuse
        fused = torch.cat([patches, context], dim=-1)
        return self.fusion_mlp(fused)
    
    def _forward_selected_logits(self, patches, tokens):
        """
        Args:
            patches: [B, 256, dino_dim] - DINO features
            tokens: [B, 40] - Audio token sequence
        Returns:
            patch_probs: [B, 256, 40] - Per-patch probabilities for each timestep
        """
        batch_size = patches.size(0)
        all_patch_probs = []
        
        # First prediction with empty context
        empty_context = torch.zeros(batch_size, 1, self.token_encoder.d_model, device=patches.device)
        fused = self.fuse_patches_context(patches, empty_context)
        logits = self.classifier(fused)
        
        # Get probs for first token
        first_token = tokens[:, 0]  # [B]
        first_token = first_token.unsqueeze(1).expand(-1, 256)  # [B, 256]
        probs = torch.gather(
            torch.sigmoid(logits),
            dim=2,
            index=first_token.unsqueeze(-1)
        ).squeeze(-1)  # [B, 256]
        all_patch_probs.append(probs)
        
        # Rest of the sequence
        for t in range(tokens.size(1)-1):  # Now predicting for indices 1 to 39
            # Get context from previous tokens
            prev_tokens = tokens[:, :t+1]
            context = self.token_encoder(prev_tokens)[:, -1:]
            
            fused = self.fuse_patches_context(patches, context)
            logits = self.classifier(fused)
            
            next_token = tokens[:, t+1]
            next_token = next_token.unsqueeze(1).expand(-1, 256)
            
            probs = torch.gather(
                torch.sigmoid(logits),
                dim=2,
                index=next_token.unsqueeze(-1)
            ).squeeze(-1)
            
            all_patch_probs.append(probs)
        
        return torch.stack(all_patch_probs, dim=2)  # Should now be [B, 256, 40]
    
    def _forward_with_full_logits(self, patches, tokens):
        # Your training forward pass code here
        batch_size = patches.size(0)
        seq_length = tokens.size(1)
        all_logits = []

        # First prediction with empty context
        empty_context = torch.zeros(batch_size, 1, self.token_encoder.d_model, device=patches.device)
        fused = self.fuse_patches_context(patches, empty_context)
        logits = self.classifier(fused)
        all_logits.append(logits)

        # Rest of sequence
        for t in range(seq_length - 1):
            prev_tokens = tokens[:, :t+1]
            context = self.token_encoder(prev_tokens)[:, -1:]
            fused = self.fuse_patches_context(patches, context)
            logits = self.classifier(fused)
            all_logits.append(logits)

        return torch.stack(all_logits, dim=0).permute(1, 0, 2, 3)
    
    def forward(self, frames, tokens, return_all_logits=False):
        """
        Args:
            patches: [B, 256, dino_dim] - DINO features
            tokens: [B, 40] - Audio token sequence
            return_all_logits: If True, returns full logits for training
                             If False, returns only selected token probs for viz
        Returns:
            if return_all_logits:
                [B, T, 256, 4096] - Full logits for each timestep/patch
            else:
                [B, 256, 40] - Per-patch probabilities for selected tokens
        """
        patches = self.dino_model(frames)
        if return_all_logits:
            return self._forward_with_full_logits(patches, tokens)
        else:
            return self._forward_selected_logits(patches, tokens)
        
    def compute_loss(self, patches, tokens):
        """Convenience method for computing training loss"""
        all_logits = self(patches, tokens, return_all_logits=True)
        B, T, P, V = all_logits.shape
        
        all_losses = []
        for t in range(T):
            step_logits = all_logits[:, t, :, :]
            step_labels = tokens[:, t]
            step_labels_expanded = step_labels.unsqueeze(1).expand(-1, P)
            step_labels_flat = step_labels_expanded.reshape(-1)
            step_logits_flat = step_logits.reshape(-1, V)
            loss_t = F.cross_entropy(step_logits_flat, step_labels_flat)
            all_losses.append(loss_t)
            
        return torch.mean(torch.stack(all_losses))
    
if __name__ == "__main__":
    # Test setup
    batch_size = 2
    num_patches = 256
    dino_dim = 384
    seq_length = 40
    
    patches = torch.randn(batch_size, 3, 224, 224)
    tokens = torch.randint(0, 4096, (batch_size, seq_length))
    
    model = SequentialAudioVisualModel()
    
    # Test training
    loss = model.compute_loss(patches, tokens)
    print(f"Training loss: {loss.item():.4f}")
    
    # Test visualization
    patch_probs = model(patches, tokens, return_all_logits=False)
    print(f"Viz output shape: {patch_probs.shape}")  # Should be [B, 256, 40]

    # Test TokenEncoder
    token_encoder = TokenEncoder()
    test_tokens = torch.randint(0, 4096, (2, 10))  # Batch of 2, sequence of 10
    encoded = token_encoder(test_tokens)
    print("\nToken Encoder Test:")
    print(f"Input shape: {test_tokens.shape}")
    print(f"Output shape: {encoded.shape}")
    print(f"Output stats - mean: {encoded.mean():.3f}, std: {encoded.std():.3f}")
    
    # Test the full model with more detailed stats
    model = SequentialAudioVisualModel()
    patches = torch.randn(2, 3, 224, 224)
    tokens = torch.randint(0, 4096, (2, 40))
    
    print("\nFull Model Test:")
    # Test training mode
    model.train()
    loss = model.compute_loss(patches, tokens)
    print(f"Initial training loss: {loss.item():.4f}")
    
    # Test probability distribution
    patch_probs = model(patches, tokens, return_all_logits=False)
    print(f"\nProbability Stats:")
    print(f"Shape: {patch_probs.shape}")
    print(f"Mean prob: {patch_probs.mean():.4f}")
    print(f"Std dev: {patch_probs.std():.4f}")
    print(f"Min: {patch_probs.min():.4f}")
    print(f"Max: {patch_probs.max():.4f}")
    
    # Look at per-timestep stats
    for t in range(0, 40, 10):  # Check every 10th timestep
        print(f"\nTimestep {t}:")
        t_probs = patch_probs[:, :, t]
        print(f"Mean: {t_probs.mean():.4f}")
        print(f"Max per batch: {t_probs.max(dim=1)[0]}")