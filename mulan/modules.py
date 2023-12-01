"""Implementation of Light attention model"""

from typing import Union, Sequence, Optional, List
from dataclasses import dataclass
import torch
import torch.nn as nn

from mulan.config import MulanConfig


@dataclass
class OutputWithAttention:
    output: torch.Tensor = None
    attention: Union[torch.Tensor, Sequence[torch.Tensor]] = None


class AttentionMeanK(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.padding_value = config.padding_value
        kernel_sizes = list(config.kernel_sizes)
        self.n_attn = len(kernel_sizes)
        self.hidden_size = config.hidden_size
        self.last_hidden_size = config.last_hidden_size

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(config.conv_dropout)

        self.conv_heads = nn.ModuleList(
            [
                nn.LazyConv1d(self.hidden_size, kernel_size, stride=1, padding=kernel_size // 2)
                for kernel_size in kernel_sizes
            ]
        )

        self.attn_heads = nn.ModuleList(
            [
                nn.LazyConv1d(self.hidden_size, kernel_size, stride=1, padding=kernel_size // 2)
                for kernel_size in kernel_sizes
            ]
        )

        self.fc = nn.Sequential(
            nn.Linear(
                self.hidden_size * self.n_attn * 2, self.last_hidden_size * self.n_attn
            ),  # n_attn + concatenated maxpool
            nn.Dropout(config.hidden_dropout_prob),
            nn.LeakyReLU(0.5),
            nn.Linear(self.last_hidden_size * self.n_attn, self.last_hidden_size // 2),
            nn.LeakyReLU(0.5),
        )

    def forward(self, x):

        # feature convolution
        batch_size, length, hidden = x.shape
        mask = x[..., 0] != self.padding_value
        x = x.transpose(-1, -2)
        o = torch.stack([conv(x) for conv in self.conv_heads], dim=1)
        o = self.dropout(o)  # [batch_size, embeddings_dim, sequence_length]

        # attention weights
        attn_weights = torch.stack(
            [
                self.softmax(head(x).masked_fill(mask[:, None, :] == False, -1e9))
                for head in self.attn_heads
            ],
            dim=1,
        )
        # print(attn_weights.shape, o.shape)
        o1 = torch.sum(o * attn_weights, dim=-1).view(batch_size, -1)
        # print(o1.shape)

        # max pooling
        o2, _ = torch.max(o, dim=-1)
        o2 = o2.view(batch_size, -1)

        # mlp
        o = torch.cat([o1, o2], dim=-1)
        o = self.fc(o)
        # attn_mean = torch.softmax(attn_weigths.mean(dim=-1)
        output = OutputWithAttention(o, attn_weights)

        return output


class LightAttModel(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.encoder = AttentionMeanK(config)

        last_hidden_size = config.last_hidden_size
        if config.add_scores:
            last_hidden_size = last_hidden_size + 1

        self.linear = nn.Linear(last_hidden_size, 1)

    def forward(
        self,
        inputs_embeds: List[torch.FloatTensor],
        zs_scores: Optional[torch.FloatTensor] = None,
        output_attentions=False,
    ):

        batch_size = inputs_embeds[0].shape[0]
        # Siamese encoder
        encodings = [
            self.encoder(emb) for emb in inputs_embeds
        ]  # The order is [wt1, wt2, mut1, mut2]

        # features combination
        x_wt = torch.cat(
            [
                encodings[0].output * encodings[1].output,
                torch.abs(encodings[0].output - encodings[1].output),
            ],
            dim=1,
        )
        x_mut = torch.cat(
            [
                encodings[2].output * encodings[3].output,
                torch.abs(encodings[2].output - encodings[3].output),
            ],
            dim=1,
        )
        output = x_mut - x_wt

        if zs_scores is not None and (self.config.add_scores or self.config.add_columns_scores):
            output = torch.cat((output, zs_scores.view(batch_size, -1)), dim=-1)
        output = self.linear(output).squeeze(-1)

        if not output_attentions:
            return output
        else:
            return OutputWithAttention(output, tuple([enc.attention for enc in encodings[:2]]))

    @classmethod
    def from_pretrained(cls, pretrained_model_path, device=None, **kwargs):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(pretrained_model_path, map_location=device, weights_only=False)
        config = ckpt["config"]
        config.update(kwargs)
        config = MulanConfig(**config)
        state_dict = ckpt["state_dict"]
        model = cls(config).to(device)
        model.load_state_dict(state_dict, strict=False)
        return model
