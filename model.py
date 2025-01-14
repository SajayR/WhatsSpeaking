import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import HubertModel, AutoProcessor
import warnings
warnings.filterwarnings("ignore")


class ViTEmbedder(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True):
        super().__init__()
        
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14') #torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        # Project to 4096 classes for classification
        self.classifier = nn.Linear(self.model.embed_dim, 4096)
        
        for param in self.model.parameters():
            param.requires_grad = True
            
    def forward(self, x):
        """
        Args:
            x: (batch_size, channels, height, width)
        Returns:
            patch_embeddings: (batch_size, num_patches, embedding_dim)
        """
        x = self.model.get_intermediate_layers(x, n=1)[0]
        x = self.classifier(x)
        return x

class Valo(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = ViTEmbedder()
        
    def compute_audio_visual_loss(self, patch_logits, audio_tokens, vocab_size=4096):
        """
        Args:
            patch_logits: [batch_size, 256, 4096] - logits for each patch
            audio_tokens: [batch_size, 40] - token indices
        Returns:
            loss: scalar
        """
        batch_size = patch_logits.size(0)
        
        # Create binary target vector
        target = torch.zeros(batch_size, vocab_size, device=patch_logits.device)
        # For each item in batch, set 1s at token indices
        for b in range(batch_size):
            target[b, audio_tokens[b]] = 1
        
        # Get max prediction across patches for each token
        # [batch_size, 256, 4096] -> [batch_size, 4096]
        max_logits_per_token = patch_logits.max(dim=1)[0]
        
        # Apply sigmoid after taking max to get probabilities
        token_probs = torch.sigmoid(max_logits_per_token)
        
        # Binary cross entropy between probabilities and targets
        loss = F.binary_cross_entropy(token_probs, target)
        
        return loss, token_probs
        
    def forward(self, video, audio):
        """
        Args:
            video: tensor of shape (batch_size, 3, 224, 224)
            audio: tensor of shape (batch_size, 40) containing token indices
        Returns:
            During training:
                loss: scalar tensor
                probs: tensor of shape (batch_size, 4096) - probabilities after sigmoid
            During inference:
                patch_probs: tensor of shape (batch_size, 256, 4096) - per-patch probabilities
        """
        patch_logits = self.vit(video)  # [batch_size, 256, 4096]
        
        # If we're in training mode, compute loss
        if self.training:
            loss, token_probs = self.compute_audio_visual_loss(patch_logits, audio)
            return loss, token_probs
        else:
            # During inference, return raw patch probabilities for visualization
            patch_probs = torch.sigmoid(patch_logits)  # [batch_size, 256, 4096]
            return patch_probs


if __name__ == "__main__":
    # Basic shape test for ViT
    model = ViTEmbedder()
    x = torch.randn(2, 3, 224, 224)
    patch_logits = model(x)
    print("\nViT Embedder Test:")
    print(f"Input shape: {x.shape}")
    print(f"Patch logits shape: {patch_logits.shape}")  # Should be [2, 256, 4096]
    
    # Test full Valo model
    valo = Valo()
    
    # Test training mode
    valo.train()
    dummy_video = torch.randn(2, 3, 224, 224)
    dummy_audio = torch.randint(0, 4096, (2, 40))  # Random token indices
    
    print("\nTraining Mode Test:")
    loss, token_probs = valo(dummy_video, dummy_audio)
    print(f"Loss: {loss.item()}")
    print(f"Token probs shape: {token_probs.shape}")  # Should be [2, 4096]
    print(f"Token probs range: ({token_probs.min().item():.3f}, {token_probs.max().item():.3f})")  # Should be (0,1)
    
    # Test inference mode
    valo.eval()
    print("\nInference Mode Test:")
    with torch.no_grad():
        patch_probs = valo(dummy_video, dummy_audio)
    print(f"Patch probs shape: {patch_probs.shape}")  # Should be [2, 256, 4096]
    print(f"Patch probs range: ({patch_probs.min().item():.3f}, {patch_probs.max().item():.3f})")  # Should be (0,1)
    
    # Test visualization mapping
    def get_token_heatmaps(patch_probs, audio_tokens):
        """
        Args:
            patch_probs: [batch_size, 256, 4096] - Full patch probabilities
            audio_tokens: [batch_size, 40] - Token indices
        Returns:
            heatmaps: [batch_size, 40, 256] - Probabilities for each token's presence in each patch
        """
        batch_size = patch_probs.size(0)
        heatmaps = []
        
        for b in range(batch_size):
            # Get probabilities for just the tokens that appear in the audio
            # [256, 40]
            token_patch_probs = patch_probs[b, :, audio_tokens[b]]
            heatmaps.append(token_patch_probs)
            
        return torch.stack(heatmaps).permute(0, 2, 1)  # [batch_size, 40, 256]
    
    print("\nVisualization Mapping Test:")
    heatmaps = get_token_heatmaps(patch_probs, dummy_audio)
    print(f"Heatmap shape: {heatmaps.shape}")  # Should be [2, 40, 256]
    print(f"Heatmap range: ({heatmaps.min().item():.3f}, {heatmaps.max().item():.3f})")  # Should be (0,1)
    
    # Test if gradients flow
    print("\nGradient Flow Test:")
    valo.train()
    loss, _ = valo(dummy_video, dummy_audio)
    loss.backward()
    
    # Check if gradients exist and are not zero
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                  for p in valo.parameters())
    print(f"Gradients exist and flow: {has_grad}")