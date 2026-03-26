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
        C = 3
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
