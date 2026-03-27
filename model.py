import torch
import torch.nn as nn
import math


class PatchEmbed(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_channels=3, embed_dim=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Linear(patch_size * patch_size * in_channels, embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, self.img_size // self.patch_size, self.patch_size, self.img_size // self.patch_size, self.patch_size)
        x = x.permute(0, 2, 4, 3, 5, 1)
        x = x.reshape(B, self.num_patches, self.patch_size * self.patch_size * C)
        x = self.proj(x)
        return x


class MAEModel(nn.Module):
    def __init__(
        self,
        img_size=64,
        patch_size=8,
        in_channels=3,
        encoder_dim=128,
        encoder_layers=6,
        encoder_heads=4,
        encoder_ffn_dim=512,
        decoder_dim=64,
        decoder_layers=2,
        decoder_heads=2,
        mask_ratio=0.75,
        dropout=0.1
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.encoder_dim = encoder_dim
        self.mask_ratio = mask_ratio
        self.num_mask = int(mask_ratio * self.num_patches)
        self.num_visible = self.num_patches - self.num_mask
        self.decoder_dim = decoder_dim

        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, encoder_dim)

        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches, encoder_dim))
        nn.init.normal_(self.position_embedding, std=0.02)

        self.mask_token = nn.Parameter(torch.randn(1, 1, encoder_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=encoder_dim,
            nhead=encoder_heads,
            dim_feedforward=encoder_ffn_dim,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=encoder_layers)

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=decoder_dim,
            nhead=decoder_heads,
            dim_feedforward=decoder_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=decoder_layers)

        self.decoder_proj = nn.Linear(encoder_dim, decoder_dim)

        self.patch_to_pixel = nn.Linear(decoder_dim, patch_size * patch_size * in_channels)

        self.dropout = nn.Dropout(dropout)

    def get_random_indices(self, batch_size):
        rand_indices = torch.argsort(torch.rand(batch_size, self.num_patches), dim=-1)
        mask_indices = rand_indices[:, :self.num_mask]
        unmask_indices = rand_indices[:, self.num_mask:]
        return mask_indices, unmask_indices

    def _get_raw_patches(self, x):
        B, C, H, W = x.shape
        ph = pw = self.patch_size
        n_h = H // ph
        n_w = W // pw

        x = x.reshape(B, C, n_h, ph, n_w, pw)
        x = x.permute(0, 2, 4, 3, 5, 1)
        x = x.reshape(B, self.num_patches, ph * pw * C)
        return x

    def forward(self, x):
        B = x.shape[0]

        patch_embeddings = self.patch_embed(x)

        raw_patches = self._get_raw_patches(x)

        pos_emb = self.position_embedding.expand(B, -1, -1)

        patch_embeddings = patch_embeddings + pos_emb

        mask_indices, unmask_indices = self.get_random_indices(B)
        mask_indices = mask_indices.to(x.device)
        unmask_indices = unmask_indices.to(x.device)

        unmasked_embeddings = patch_embeddings.gather(
            dim=1,
            index=unmask_indices.unsqueeze(-1).expand(-1, -1, patch_embeddings.size(-1))
        )

        unmasked_positions = pos_emb.gather(
            dim=1,
            index=unmask_indices.unsqueeze(-1).expand(-1, -1, pos_emb.size(-1))
        )

        masked_positions = pos_emb.gather(
            dim=1,
            index=mask_indices.unsqueeze(-1).expand(-1, -1, pos_emb.size(-1))
        )

        mask_tokens = self.mask_token.expand(B, self.num_mask, -1)

        masked_embeddings = self.decoder_proj(mask_tokens) + self.decoder_proj(masked_positions)

        visible_embeddings = self.encoder(unmasked_embeddings)
        visible_embeddings = self.decoder_proj(visible_embeddings) + self.decoder_proj(unmasked_positions)

        all_decoder_tokens = torch.zeros(B, self.num_patches, self.decoder_dim).to(x.device)
        all_decoder_tokens.scatter_(1, unmask_indices.unsqueeze(-1).expand(-1, -1, self.decoder_dim), visible_embeddings)
        all_decoder_tokens.scatter_(1, mask_indices.unsqueeze(-1).expand(-1, -1, self.decoder_dim), masked_embeddings)

        decoder_output = self.decoder(all_decoder_tokens, visible_embeddings)

        all_pred_patches = self.patch_to_pixel(decoder_output)

        pred_patches = all_pred_patches.gather(
            dim=1,
            index=mask_indices.unsqueeze(-1).expand(-1, -1, all_pred_patches.size(-1))
        )

        target_patches = raw_patches.gather(
            dim=1,
            index=mask_indices.unsqueeze(-1).expand(-1, -1, raw_patches.size(-1))
        )

        return pred_patches, target_patches, mask_indices

    def encode(self, x):
        B = x.shape[0]
        patch_embeddings = self.patch_embed(x)
        pos_emb = self.position_embedding.expand(B, -1, -1)
        patch_embeddings = patch_embeddings + pos_emb

        encoder_output = self.encoder(patch_embeddings)

        return encoder_output

    def compute_loss(self, pred, target):
        loss_fn = nn.MSELoss()
        return loss_fn(pred, target)

    def reconstruct_from_patches(self, patches, B):
        C = self.patch_embed.proj.in_features // (self.patch_size * self.patch_size)
        H = W = self.img_size
        ph = pw = self.patch_size

        n_h = H // ph
        n_w = W // pw

        patches = patches.reshape(B, n_h, n_w, ph, pw, C)
        patches = patches.permute(0, 5, 1, 3, 2, 4)
        patches = patches.reshape(B, C, H, W)

        return patches


class DownstreamLSTM(nn.Module):
    def __init__(self, pretrain_model, num_classes=16, lstm_hidden=128,
                 lstm_layers=2, lstm_dropout=0.5, freeze_encoder=True):
        super().__init__()

        self.pretrain_model = pretrain_model
        self.freeze_encoder = freeze_encoder

        self.temporal_pos_embed = nn.Parameter(torch.randn(7, lstm_hidden) * 0.02)

        self.lstm = nn.LSTM(
            input_size=lstm_hidden,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=lstm_dropout if lstm_layers > 1 else 0,
            batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

        self.set_encoder_grad(freeze=freeze_encoder)

    def set_encoder_grad(self, freeze=True):
        self.freeze_encoder = freeze
        for param in self.pretrain_model.parameters():
            param.requires_grad = not freeze

    def get_param_groups(self, base_lr_encoder=1e-5, base_lr_head=5e-5):
        encoder_params = []
        head_params = []

        for name, param in self.named_parameters():
            if 'pretrain_model' in name:
                encoder_params.append(param)
            else:
                head_params.append(param)

        return [
            {'params': encoder_params, 'lr': base_lr_encoder},
            {'params': head_params, 'lr': base_lr_head}
        ]

    def forward(self, x):
        B, seq_len, C, H, W = x.size()

        frame_features = []
        for i in range(seq_len):
            frame = x[:, i, :, :, :]
            encoded = self.pretrain_model.encode(frame)
            pooled = encoded.mean(dim=1)
            frame_features.append(pooled)

        frame_features = torch.stack(frame_features, dim=1)

        pos_emb = self.temporal_pos_embed.unsqueeze(0).expand(B, -1, -1).to(x.device)
        frame_features = frame_features + pos_emb

        lstm_out, _ = self.lstm(frame_features)
        last_hidden = lstm_out[:, -1, :]

        logits = self.classifier(last_hidden)

        return logits


# =============================================================================
# New Model 1: Transformer-based temporal classifier (alternative to LSTM)
# =============================================================================

class DownstreamTransformer(nn.Module):
    """Downstream classifier using Transformer self-attention for temporal modeling.

    Replaces the LSTM in DownstreamLSTM with a Transformer encoder that models
    temporal dependencies across the 7-frame sequence via multi-head self-attention.
    The interface (constructor signature pattern, forward input/output shapes,
    set_encoder_grad, get_param_groups) mirrors DownstreamLSTM so the two can be
    swapped in downstream_train.py with minimal changes.
    """

    def __init__(self, pretrain_model, num_classes=16, d_model=128,
                 num_heads=4, num_layers=2, ffn_dim=256, dropout=0.3,
                 sequence_length=7, freeze_encoder=True):
        super().__init__()

        self.pretrain_model = pretrain_model
        self.freeze_encoder = freeze_encoder
        self.sequence_length = sequence_length
        self.d_model = d_model

        # Learnable temporal position embedding (same shape convention as DownstreamLSTM)
        self.temporal_pos_embed = nn.Parameter(torch.randn(sequence_length, d_model) * 0.02)

        # CLS token aggregates sequence information for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Transformer encoder for temporal self-attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(d_model)

        # Classification head (same structure as DownstreamLSTM)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

        self.set_encoder_grad(freeze=freeze_encoder)

    def set_encoder_grad(self, freeze=True):
        self.freeze_encoder = freeze
        for param in self.pretrain_model.parameters():
            param.requires_grad = not freeze

    def get_param_groups(self, base_lr_encoder=1e-5, base_lr_head=5e-5):
        encoder_params = []
        head_params = []
        for name, param in self.named_parameters():
            if 'pretrain_model' in name:
                encoder_params.append(param)
            else:
                head_params.append(param)
        return [
            {'params': encoder_params, 'lr': base_lr_encoder},
            {'params': head_params, 'lr': base_lr_head}
        ]

    def forward(self, x):
        B, seq_len, C, H, W = x.size()

        # Extract per-frame features using pretrained MAE encoder
        frame_features = []
        for i in range(seq_len):
            frame = x[:, i, :, :, :]
            encoded = self.pretrain_model.encode(frame)
            pooled = encoded.mean(dim=1)  # (B, d_model)
            frame_features.append(pooled)

        frame_features = torch.stack(frame_features, dim=1)  # (B, seq_len, d_model)

        # Add temporal position embedding
        pos_emb = self.temporal_pos_embed.unsqueeze(0).expand(B, -1, -1).to(x.device)
        frame_features = frame_features + pos_emb

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1).to(x.device)
        tokens = torch.cat([cls_tokens, frame_features], dim=1)  # (B, 1+seq_len, d_model)

        # Temporal self-attention
        tokens = self.temporal_encoder(tokens)
        tokens = self.norm(tokens)

        # Use CLS token output for classification
        cls_output = tokens[:, 0, :]  # (B, d_model)

        logits = self.classifier(cls_output)
        return logits


# =============================================================================
# New Model 2: MAE with Inter-Layer Attention (inspired by Kimi/MoonshotAI)
# =============================================================================

class InterLayerAttention(nn.Module):
    """Inter-layer attention module that allows each encoder layer to attend
    to the outputs of all previous layers.

    Inspired by the Kimi (MoonshotAI) approach of adding cross-layer attention
    so that deeper layers can selectively retrieve and integrate information
    from any earlier layer, not just the immediately preceding one. This helps
    with gradient flow, feature reuse, and learning richer representations.

    For each layer l, the module computes:
        query = current layer output
        key/value = stacked outputs from layers 0..l-1
        output = LayerNorm(current + Attention(query, key, value))
    """

    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        # Learnable gate to control how much inter-layer info to blend in
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, current, previous_layers):
        """
        Args:
            current: (B, N, D) - output of the current encoder layer
            previous_layers: list of (B, N, D) - outputs from all previous layers
        Returns:
            (B, N, D) - current output enriched with inter-layer attention
        """
        if len(previous_layers) == 0:
            return current

        # Stack all previous layer outputs: (B, L*N, D) where L = number of previous layers
        kv = torch.cat(previous_layers, dim=1)

        attn_out, _ = self.attn(current, kv, kv)
        gate = torch.sigmoid(self.gate)
        output = self.norm(current + gate * self.dropout(attn_out))
        return output


class MAEModelWithInterLayerAttention(nn.Module):
    """MAE model enhanced with inter-layer attention in the encoder.

    This extends the original MAEModel by inserting InterLayerAttention modules
    between encoder layers. Each layer can attend to all previous layers' outputs,
    enabling richer feature hierarchies and better gradient flow.

    The original MAEModel is preserved unchanged; this class implements the same
    interface (forward, encode, compute_loss, reconstruct_from_patches) so it
    can be used as a drop-in replacement.

    Architecture changes vs MAEModel:
        - Encoder uses manually unrolled TransformerEncoderLayers instead of
          nn.TransformerEncoder, to insert inter-layer attention between layers.
        - An InterLayerAttention module is applied after each encoder layer
          (except the first), attending to all preceding layer outputs.
        - All other components (patch embed, decoder, masking) are identical.
    """

    def __init__(
        self,
        img_size=64,
        patch_size=8,
        in_channels=3,
        encoder_dim=128,
        encoder_layers=6,
        encoder_heads=4,
        encoder_ffn_dim=512,
        decoder_dim=64,
        decoder_layers=2,
        decoder_heads=2,
        mask_ratio=0.75,
        dropout=0.1,
        inter_layer_heads=4
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_patches = (img_size // patch_size) ** 2
        self.encoder_dim = encoder_dim
        self.mask_ratio = mask_ratio
        self.num_mask = int(mask_ratio * self.num_patches)
        self.num_visible = self.num_patches - self.num_mask
        self.decoder_dim = decoder_dim

        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, encoder_dim)

        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches, encoder_dim))
        nn.init.normal_(self.position_embedding, std=0.02)

        self.mask_token = nn.Parameter(torch.randn(1, 1, encoder_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # Manually unrolled encoder layers (to insert inter-layer attention)
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=encoder_dim,
                nhead=encoder_heads,
                dim_feedforward=encoder_ffn_dim,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(encoder_layers)
        ])

        # Inter-layer attention modules (one per layer except the first)
        self.inter_layer_attns = nn.ModuleList([
            InterLayerAttention(encoder_dim, num_heads=inter_layer_heads, dropout=dropout)
            for _ in range(encoder_layers - 1)
        ])

        # Decoder (same as original MAEModel)
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=decoder_dim,
            nhead=decoder_heads,
            dim_feedforward=decoder_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=decoder_layers)

        self.decoder_proj = nn.Linear(encoder_dim, decoder_dim)
        self.patch_to_pixel = nn.Linear(decoder_dim, patch_size * patch_size * in_channels)
        self.dropout_layer = nn.Dropout(dropout)

    def _encode_with_inter_layer_attn(self, x):
        """Run encoder with inter-layer attention between layers."""
        layer_outputs = []

        hidden = x
        for i, layer in enumerate(self.encoder_layers):
            hidden = layer(hidden)
            if i > 0:
                # Apply inter-layer attention: current layer attends to all previous
                hidden = self.inter_layer_attns[i - 1](hidden, layer_outputs)
            layer_outputs.append(hidden)

        return hidden

    def get_random_indices(self, batch_size):
        rand_indices = torch.argsort(torch.rand(batch_size, self.num_patches), dim=-1)
        mask_indices = rand_indices[:, :self.num_mask]
        unmask_indices = rand_indices[:, self.num_mask:]
        return mask_indices, unmask_indices

    def _get_raw_patches(self, x):
        B, C, H, W = x.shape
        ph = pw = self.patch_size
        n_h = H // ph
        n_w = W // pw
        x = x.reshape(B, C, n_h, ph, n_w, pw)
        x = x.permute(0, 2, 4, 3, 5, 1)
        x = x.reshape(B, self.num_patches, ph * pw * C)
        return x

    def forward(self, x):
        B = x.shape[0]

        patch_embeddings = self.patch_embed(x)
        raw_patches = self._get_raw_patches(x)

        pos_emb = self.position_embedding.expand(B, -1, -1)
        patch_embeddings = patch_embeddings + pos_emb

        mask_indices, unmask_indices = self.get_random_indices(B)
        mask_indices = mask_indices.to(x.device)
        unmask_indices = unmask_indices.to(x.device)

        unmasked_embeddings = patch_embeddings.gather(
            dim=1,
            index=unmask_indices.unsqueeze(-1).expand(-1, -1, patch_embeddings.size(-1))
        )
        unmasked_positions = pos_emb.gather(
            dim=1,
            index=unmask_indices.unsqueeze(-1).expand(-1, -1, pos_emb.size(-1))
        )
        masked_positions = pos_emb.gather(
            dim=1,
            index=mask_indices.unsqueeze(-1).expand(-1, -1, pos_emb.size(-1))
        )

        mask_tokens = self.mask_token.expand(B, self.num_mask, -1)
        masked_embeddings = self.decoder_proj(mask_tokens) + self.decoder_proj(masked_positions)

        # Encode visible patches with inter-layer attention
        visible_embeddings = self._encode_with_inter_layer_attn(unmasked_embeddings)
        visible_embeddings = self.decoder_proj(visible_embeddings) + self.decoder_proj(unmasked_positions)

        all_decoder_tokens = torch.zeros(B, self.num_patches, self.decoder_dim).to(x.device)
        all_decoder_tokens.scatter_(1, unmask_indices.unsqueeze(-1).expand(-1, -1, self.decoder_dim), visible_embeddings)
        all_decoder_tokens.scatter_(1, mask_indices.unsqueeze(-1).expand(-1, -1, self.decoder_dim), masked_embeddings)

        decoder_output = self.decoder(all_decoder_tokens, visible_embeddings)
        all_pred_patches = self.patch_to_pixel(decoder_output)

        pred_patches = all_pred_patches.gather(
            dim=1,
            index=mask_indices.unsqueeze(-1).expand(-1, -1, all_pred_patches.size(-1))
        )
        target_patches = raw_patches.gather(
            dim=1,
            index=mask_indices.unsqueeze(-1).expand(-1, -1, raw_patches.size(-1))
        )

        return pred_patches, target_patches, mask_indices

    def encode(self, x):
        """Encode all patches (no masking) -- used for downstream tasks."""
        B = x.shape[0]
        patch_embeddings = self.patch_embed(x)
        pos_emb = self.position_embedding.expand(B, -1, -1)
        patch_embeddings = patch_embeddings + pos_emb
        encoder_output = self._encode_with_inter_layer_attn(patch_embeddings)
        return encoder_output

    def compute_loss(self, pred, target):
        loss_fn = nn.MSELoss()
        return loss_fn(pred, target)

    def reconstruct_from_patches(self, patches, B):
        C = self.in_channels
        H = W = self.img_size
        ph = pw = self.patch_size
        n_h = H // ph
        n_w = W // pw
        patches = patches.reshape(B, n_h, n_w, ph, pw, C)
        patches = patches.permute(0, 5, 1, 3, 2, 4)
        patches = patches.reshape(B, C, H, W)
        return patches
