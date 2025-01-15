import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

##################################################
# 1) Video Encoder (ViTEmbedder)
##################################################
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


##################################################
# 2) Autoregressive Audio Decoder
##################################################
class AudioARDecoder(nn.Module):
    """
    A small TransformerDecoder that autoregressively decodes audio tokens
    from visual patch embeddings.
    """
    def __init__(self, 
                 d_model=384,      # Must match VIT embed_dim for cross-attn
                 vocab_size=4096, 
                 nhead=8, 
                 num_layers=4, 
                 dim_feedforward=1024,
                 dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Embedding for audio tokens
        self.token_embed = nn.Embedding(vocab_size, d_model)
        
        # Positional embeddings for the audio tokens
        self.pos_embed = nn.Parameter(torch.randn(1, 1024, d_model) * 0.01)
        # 1024 is just an upper bound on sequence length; for 1s we only need 40

        # Standard nn.TransformerDecoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Final projection to vocab size
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, 
                tgt_tokens,  # [B, T] ground-truth audio tokens
                memory,      # [B, 256, d_model] from ViT
                tgt_mask=None):
        """
        Args:
            tgt_tokens: [B, T] (teacher-forced ground-truth tokens)
            memory: [B, 256, d_model] visual patch embeddings
            tgt_mask: optional causal mask for autoregressive decoding
        Returns:
            logits: [B, T, vocab_size]
            cross_attn_weights: [B, num_layers, T, 256] if you want them all
                                or just the last layer
        """
        B, T = tgt_tokens.shape
        
        # 1) Embed tokens
        token_embs = self.token_embed(tgt_tokens)  # [B, T, d_model]
        
        # 2) Add positional embeddings
        # we only need T positions out of pos_embed
        pos_embs = self.pos_embed[:, :T, :]  # [1, T, d_model]
        token_embs = token_embs + pos_embs
        
        # 3) Optionally create a causal mask (prevent future tokens)
        # If we want strict autoregressive, we do:
        #    shape: [T, T], filled with float('-inf') for j>i
        if tgt_mask is None:
            tgt_mask = self.generate_causal_mask(T, device=tgt_tokens.device)

        # 4) TransformerDecoder expects memory: [B, 256, d_model]
        #    token_embs shape: [B, T, d_model]
        #    Output: [B, T, d_model]
        decoded = self.decoder(
            token_embs, 
            memory, 
            tgt_mask=tgt_mask
        )
        
        # 5) Project to vocab
        logits = self.output_proj(decoded)  # [B, T, vocab_size]
        
        # 6) If we want cross-attn weights for heatmaps:
        #    We can grab them from the layers inside self.decoder
        #    by hooking into the forward hook or storing them
        #    in the forward pass. For brevity, let's store
        #    only the final layer's cross-attn.
        #    (If you want all layers, you'd store them all in a list.)
        
        # We'll assume we only want the final layer's cross-attn:
        # => We can modify the TransformerDecoderLayer to expose them.
        # Let's do a quick hacky approach:
        cross_attn_weights = None
        if hasattr(self.decoder.layers[-1], 'cross_attn_weights'):
            # shape [B, nhead, T, 256]
            # We'll average over heads => [B, T, 256]
            w = self.decoder.layers[-1].cross_attn_weights
            cross_attn_weights = w.mean(dim=1)  # average across heads

        return logits, cross_attn_weights

    def generate_causal_mask(self, size, device):
        """
        Generate a [size, size] mask for autoregressive decoding.
        Upper-triangular is True (or float('-inf')) to block future tokens.
        """
        mask = torch.full((size, size), float('-inf'), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask


##################################################
# Patch to store cross-attn in the last layer
# (Monkey-patch or custom TransformerDecoderLayer)
##################################################
from torch.nn import MultiheadAttention, Dropout, LayerNorm
class CustomTransformerDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cross_attn_weights = None

    def forward(self, tgt, memory, **kwargs):  # Just catch everything with kwargs
        # Self-attention
        x = tgt
        # Convert kwargs for self_attn
        self_attn_kwargs = {
            'attn_mask': kwargs.get('tgt_mask'),
            'key_padding_mask': kwargs.get('tgt_key_padding_mask'),
            'need_weights': False
        }
        if 'tgt_is_causal' in kwargs:
            self_attn_kwargs['is_causal'] = kwargs['tgt_is_causal']

        x2 = self.self_attn(x, x, x, **self_attn_kwargs)[0]
        
        x = x + self.dropout1(x2)
        x = self.norm1(x)

        # Cross-attention with correct kwarg names
        cross_attn_kwargs = {
            'attn_mask': kwargs.get('memory_mask'),
            'key_padding_mask': kwargs.get('memory_key_padding_mask'),
            'need_weights': True
        }
        x2, cross_attn_w = self.multihead_attn(x, memory, memory, **cross_attn_kwargs)
        self.cross_attn_weights = cross_attn_w

        x = x + self.dropout2(x2)
        x = self.norm2(x)

        # Feed-forward
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout3(x2)
        x = self.norm3(x)
        return x


##################################################
# 3) Final Model: ValoAR
##################################################
class ValoAR(nn.Module):
    """
    Full pipeline: 
      - Video -> Patch Embeddings
      - Autoregressive Audio Decoder -> [B, T, 4096]
      - Teacher-forced cross-entropy
    """
    def __init__(self, vocab_size=4096):
        super().__init__()
        
        # 1) Video Encoder
        self.video_encoder = ViTEmbedder()
        
        # 2) Build a custom TransformerDecoder with cross-attn storing:
        #    We replace the default layer in AudioARDecoder with our custom layer
        #    that can store cross_attn_weights.
        # Example: 4 layers
        d_model = self.video_encoder.embed_dim  # 384 for DINOv2_s14
        decoder_layer = CustomTransformerDecoderLayer(d_model=d_model, nhead=8, dim_feedforward=1024, dropout=0.1, batch_first=True)
        transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)
        
        self.audio_decoder = AudioARDecoder(
            d_model=d_model,
            vocab_size=vocab_size,
            nhead=8,
            num_layers=4,
            dim_feedforward=1024,
            dropout=0.1
        )
        # Overwrite the default decoder stack with our custom stack:
        self.audio_decoder.decoder = transformer_decoder

    def forward(self, video, audio_tokens):
        """
        Args:
            video: [B, 3, 224, 224]
            audio_tokens: [B, T] 
        Returns:
            loss: scalar
            logits: [B, T, vocab_size]
            cross_attn: [B, T, 256] (final layer)
        """
        # 1) Encode video frames
        patch_embs = self.video_encoder(video)  # [B, 256, d_model]
        
        # 2) Decode audio tokens autoregressively
        logits, cross_attn_weights = self.audio_decoder(audio_tokens, patch_embs)
        # logits: [B, T, vocab_size]
        # cross_attn_weights: [B, T, 256] (averaged over heads in final layer)
        
        # 3) Training loss (teacher forcing)
        # We'll compute cross-entropy:
        B, T, V = logits.shape
        # Flatten for CE
        logits_flat = logits.reshape(B*T, V)
        targets_flat = audio_tokens.reshape(B*T)
        
        loss = F.cross_entropy(logits_flat, targets_flat, reduction='mean')
        
        return loss, logits, cross_attn_weights
