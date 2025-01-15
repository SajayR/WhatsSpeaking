import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor
import warnings
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
warnings.filterwarnings("ignore")

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
    
    def forward(self, patches, tokens):
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
    
if __name__ == "__main__":
    # Test dimensions
    batch_size = 2
    num_patches = 256
    dino_dim = 384
    seq_length = 40
    
    # Create dummy inputs
    patches = torch.randn(batch_size, num_patches, dino_dim)
    tokens = torch.randint(0, 4096, (batch_size, seq_length))
    
    # Initialize model
    model = SequentialAudioVisualModel()
    
    # Test forward pass
    patch_probs = model(patches, tokens)
    print("Input shapes:")
    print(f"Patches: {patches.shape}")
    print(f"Tokens: {tokens.shape}")
    print(f"Output patch probs: {patch_probs.shape}")
    
    # Test individual components
    print("\nTesting TokenEncoder:")
    token_encoder = TokenEncoder()
    context = token_encoder(tokens)
    print(f"Token encoder output: {context.shape}")
    
    # Test probability ranges
    print("\nTesting probability outputs:")
    print(f"Min prob: {patch_probs.min().item():.4f}")
    print(f"Max prob: {patch_probs.max().item():.4f}")
    print(f"Mean prob: {patch_probs.mean().item():.4f}")
    
    