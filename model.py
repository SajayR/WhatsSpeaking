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
def compute_prediction_stats(token_probs, target):
    """
    Args:
        token_probs: [batch_size, 4096] sigmoid probabilities
        target: [batch_size, 4096] binary targets
    """
    # For positive samples (where target = 1)
    pos_probs = token_probs[target == 1]
    pos_stats = {
        "pos_mean_prob": pos_probs.mean().item(),
        "pos_under_25": (pos_probs < 0.25).float().mean().item(), #bad
        "pos_over_75": (pos_probs > 0.75).float().mean().item(), #good
        "pos_correct": (pos_probs > 0.5).float().mean().item() #good
    }

    # For negative samples (where target = 0)
    neg_probs = token_probs[target == 0]
    neg_stats = {
        "neg_mean_prob": neg_probs.mean().item(),
        "neg_under_25": (neg_probs < 0.25).float().mean().item(), #good
        "neg_over_75": (neg_probs > 0.75).float().mean().item(), #bad
        "neg_incorrect": (neg_probs > 0.5).float().mean().item() #bad
    }

    # Overall prediction distribution
    all_stats = {
        "mean_prob": token_probs.mean().item(),
        "under_25": (token_probs < 0.25).float().mean().item(),
        "over_75": (token_probs > 0.75).float().mean().item(),
    }

    return {**pos_stats, **neg_stats, **all_stats}

class Valo(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = ViTEmbedder()
        
    def compute_audio_visual_loss(self, patch_logits, audio_tokens, vocab_size=4096):
        batch_size = patch_logits.size(0)
        
        # Create binary target vector
        target = torch.zeros(batch_size, vocab_size, device=patch_logits.device)
        for b in range(batch_size):
            target[b, audio_tokens[b]] = 1

        # Get max prediction across patches for each token [batch_size, 4096]
        max_logits_per_token = patch_logits.max(dim=1)[0]
        

        # Apply sigmoid after taking max to get probabilities
        token_probs = torch.sigmoid(max_logits_per_token)
        #print(f"Mean prediction probability: {token_probs.mean():.4f}")
        #print(f"Fraction of predictions >0.5: {(token_probs > 0.5).float().mean():.4f}")
        weights = torch.ones_like(token_probs)
        weights[target == 1] = 50
        # Binary cross entropy between probabilities and targets
        loss = F.binary_cross_entropy(token_probs, target, weight=weights)


        weights = torch.ones_like(target)
        weights[target == 1] = 50
        
        # Compute loss only on the selected tokens
        loss = F.binary_cross_entropy(token_probs, target, weight=weights)
        
        # Add top-k accuracy to stats
        top_k_accuracy = torch.zeros(batch_size, device=patch_logits.device)
        for b in range(batch_size):
            correct_tokens = set(audio_tokens[b].cpu().numpy())
            predicted_tokens = set(top_k_indices[b].cpu().numpy())
            top_k_accuracy[b] = len(correct_tokens.intersection(predicted_tokens)) / k
        
        stats = compute_prediction_stats(token_probs, target)
        stats['top_k_accuracy'] = top_k_accuracy.mean().item()
        
        return loss, token_probs, stats
        
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
        #print(f"Patch logits shape: {patch_logits.shape}")
        #print(f"Patch logits: {patch_logits[0]}")
        # If we're in training mode, compute loss
        if self.training:
            loss, token_probs, stats = self.compute_audio_visual_loss(patch_logits, audio)
            return loss, token_probs, stats
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
    print("Check")