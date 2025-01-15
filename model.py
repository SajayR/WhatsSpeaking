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
        # which is [B, 256, D] for ViT-S/14 (16x16=256 patches).
        patch_embs = self.model.get_intermediate_layers(frames, n=1)[0]  
        return patch_embs

class TokenEncoder(nn.Module):
    def __init__(self, vocab_size=4096, d_model=128, nhead=4, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, 40, d_model) * 0.02)
        self.d_model = d_model
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
    
    def forward(self, tokens):
        """
        In the updated version, we apply a causal mask so that
        output at time t only sees tokens up to t.
        
        tokens: [B, T]
        returns: [B, T, d_model], where index t has context up to t
        """
        B, T = tokens.shape
        x = self.embedding(tokens)  # [B, T, D]
        x = x + self.pos_embedding[:, :T, :]

        # Build a causal (triangular) mask for autoregressive attention
        # shape [T, T], True means "block" that attention
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()

        # The PyTorch Transformer Encoder can use 'src_mask'
        # We'll pass the same mask for every sample in the batch
        # shape [T, T] is broadcasted over B automatically
        x = self.transformer(x, mask=causal_mask)
        return x  # [B, T, D]

class SequentialAudioVisualModel(nn.Module):
    def __init__(self, d_model=128):
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
        # context: [B, T, d_model] or [B, 1, d_model]
        #   This function can handle either single-step or multi-step context
        #   Because we’ll do vectorization, context can be multiple steps at once.

        # We want to combine patches with context, so shape match:
        # If context is [B, T, d_model], expand patches to [B, T, 256, dino_dim]
        # Then cat along dim=-1

        B, P, _ = patches.shape
        B2, T, D = context.shape
        assert B == B2, "Batch size mismatch"

        # Expand patches to [B, T, P, dino_dim]
        patches_expanded = patches.unsqueeze(1).expand(-1, T, -1, -1)
        # Expand context to [B, T, P, d_model]
        context_expanded = context.unsqueeze(2).expand(-1, -1, P, -1)

        # Combine them
        fused = torch.cat([patches_expanded, context_expanded], dim=-1)  # [B, T, P, dino_dim + d_model]

        # Now run the MLP
        BTP = fused.shape[0] * fused.shape[1]
        fused_reshaped = fused.reshape(BTP, P, self.dino_dim + self.token_encoder.d_model)
        fused_out = self.fusion_mlp(fused_reshaped)  # [B*T, P, d_model]
        fused_out = fused_out.reshape(B, T, P, self.token_encoder.d_model)
        return fused_out  # [B, T, P, d_model]

    def _shift_tokens_right(self, tokens):
        """
        For autoregressive generation:
        The old logic predicted token t from context of tokens[:t].
        We replicate that with a shift and a causal mask.
        
        tokens: [B, T], e.g. t0, t1, t2, ...
        We return new_tokens: [B, T], where new_tokens[:, 0] is BOS (0),
        and new_tokens[:, t] = tokens[:, t-1] for t>=1.
        
        (If you have a real <BOS> ID, replace 0 below.)
        """
        B, T = tokens.shape
        bos = torch.zeros((B, 1), dtype=tokens.dtype, device=tokens.device)
        # Shift right: new_tokens[:, 1:] = tokens[:, :-1]
        new_tokens = torch.cat([bos, tokens[:, :-1]], dim=1)
        return new_tokens
    
    def _forward_with_full_logits(self, patches, tokens):
        """
        Vectorized version:
        1) Shift tokens to the right
        2) Encode with causal mask
        3) Fuse each time-step’s context with patches
        4) Classify for each time step in one shot
        Returns: [B, T, 256, 4096]
        """
        B, T = tokens.shape

        # 1) Shift the tokens so output at time t sees tokens up to t-1
        shifted_tokens = self._shift_tokens_right(tokens)  # [B, T]

        # 2) Encode them (causal mask is inside TokenEncoder)
        #    The output is [B, T, d_model], where row t is the context up to t
        encoded = self.token_encoder(shifted_tokens)  # [B, T, d_model]

        # 3) Fuse in a single pass for all time steps
        #    patches: [B, 256, dino_dim] -> expand to [B, T, 256, dino_dim]
        #    encoded: [B, T, d_model] -> broadcast to match
        fused = self.fuse_patches_context(patches, encoded)  # [B, T, 256, d_model]

        # 4) Classify
        logits = self.classifier(fused)  # [B, T, 256, vocab_size=4096]
        return logits

    def _forward_selected_logits(self, patches, tokens):
        """
        Vectorized version that gathers only the probabilities
        of the *actual* next token for each time step.
        
        Returns: [B, 256, T]
        """
        B, T = tokens.shape

        # Full logits: [B, T, 256, 4096], auto-regressive
        all_logits = self._forward_with_full_logits(patches, tokens)
        # Now we only want the prob for the "current" token at each step.
        # In the original code, the shape is [B, 256, T].

        # Convert logits -> probabilities
        probs = torch.sigmoid(all_logits)  # [B, T, 256, 4096]

        # Gather the correct token for each time step
        # tokens is [B, T], we want to pick tokens[:, t] from dimension -1 (vocab)
        # We'll expand tokens to match: index shape must be [B, T, 256, 1]
        tokens_expanded = tokens.unsqueeze(2).expand(-1, -1, 256).unsqueeze(-1)  # [B, T, 256, 1]

        selected_probs = torch.gather(probs, dim=-1, index=tokens_expanded)  # [B, T, 256, 1]
        selected_probs = selected_probs.squeeze(-1)  # [B, T, 256]

        # We want [B, 256, T] to match the original output
        selected_probs = selected_probs.permute(0, 2, 1)  # [B, 256, T]
        return selected_probs

    def forward(self, frames, tokens, return_all_logits=False):
        """
        Args:
            frames: [B, 3, 224, 224]
            tokens: [B, 40]
            return_all_logits: If True, returns full logits for training
                               If False, returns only selected token probs for viz
        Returns:
            if return_all_logits:
                [B, T, 256, 4096] - Full logits for each timestep/patch
            else:
                [B, 256, T] - Per-patch probabilities for selected tokens
        """
        patches = self.dino_model(frames)  # [B, 256, dino_dim]
        if return_all_logits:
            return self._forward_with_full_logits(patches, tokens)  # [B, T, 256, 4096]
        else:
            return self._forward_selected_logits(patches, tokens)   # [B, 256, T]
        
    def compute_loss(self, frames, tokens):
        """
        Convenience method for computing training loss:
        We get full logits [B, T, P, V], then do the usual cross-entropy,
        step by step, just like original code.
        """
        all_logits = self.forward(frames, tokens, return_all_logits=True)
        # all_logits: [B, T, 256, 4096]
        B, T, P, V = all_logits.shape
        
        all_losses = []
        for t in range(T):
            # step_logits: [B, 256, 4096]
            step_logits = all_logits[:, t, :, :]
            # step_labels: tokens[:, t], shape [B]
            step_labels = tokens[:, t]
            # expand step_labels to [B, 256]
            step_labels_expanded = step_labels.unsqueeze(1).expand(-1, P)
            # flatten
            step_labels_flat = step_labels_expanded.reshape(-1)         # [B*256]
            step_logits_flat = step_logits.reshape(-1, V)               # [B*256, V]
            loss_t = F.cross_entropy(step_logits_flat, step_labels_flat)
            all_losses.append(loss_t)
        
        return torch.mean(torch.stack(all_losses))

if __name__ == "__main__":
    # Test setup
    batch_size = 2
    seq_length = 40
    
    frames = torch.randn(batch_size, 3, 224, 224)
    tokens = torch.randint(0, 4096, (batch_size, seq_length))
    
    model = SequentialAudioVisualModel()
    
    # Test training
    loss = model.compute_loss(frames, tokens)
    print(f"Training loss: {loss.item():.4f}")
    
    # Test visualization
    patch_probs = model(frames, tokens, return_all_logits=False)
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
    model.train()
    loss = model.compute_loss(frames, tokens)
    print(f"\nFull Model Training Loss (again): {loss.item():.4f}")
    
    patch_probs = model(frames, tokens, return_all_logits=False)
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
