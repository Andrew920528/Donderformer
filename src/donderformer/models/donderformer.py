"""DonderFormer: an encoder-decoder Transformer that writes a chart as event tokens.

    mel window (W x 80) -> conv front end -> encoder (self-attention over W frames)
                                                   | cross-attention
    BOS course level <past events> SEP  ->  causal decoder  ->  next-token logits (vocab)

Pre-LayerNorm layers with ``scaled_dot_product_attention``. Encoder frame ``j`` and the
time token ``TIME_j`` get the same sinusoidal code, so a time token and the audio frame it
names share a position signature. The output projection is tied to the token embedding.

``generate_chart`` slides the window over a whole song: each window's past region is
primed with the notes generated so far, and the target region is sampled token by token
(KV cache, grammar mask, temperature / top-p).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from donderformer.data.chart_dataset import WindowConfig
from donderformer.data.tokens import LONG_NOTES, NOTES, ONSET_NOTES, Vocab, build_sequence, events_to_classes, \
    tokens_to_events, window_events


@dataclass(frozen=True)
class DonderFormerConfig:
    d_model: int = 256
    n_heads: int = 4
    encoder_layers: int = 6
    decoder_layers: int = 6
    ffn: int = 1024
    dropout: float = 0.1
    n_mels: int = 80
    conv_stride: int = 1            # 2 halves the encoder length (46 ms tokens)

    @classmethod
    def preset(cls, name: str) -> "DonderFormerConfig":
        """``base`` (~11.5M parameters), ``small`` (~4.3M) or ``tiny`` (~1M)."""
        return {
            "base": cls(),
            "small": cls(d_model=192, encoder_layers=4, decoder_layers=4, ffn=768),
            "tiny": cls(d_model=128, encoder_layers=2, decoder_layers=2, ffn=512),
        }[name]

    def as_dict(self) -> dict:
        return asdict(self)


def sinusoid(n: int, d: int) -> torch.Tensor:
    position = torch.arange(n).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
    table = torch.zeros(n, d)
    table[:, 0::2] = torch.sin(position * div)
    table[:, 1::2] = torch.cos(position * div)
    return table


class Attention(nn.Module):
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.out = nn.Linear(d, d)

    def split(self, x):
        b, n, d = x.shape
        return x.view(b, n, self.heads, d // self.heads).transpose(1, 2)

    def forward(self, x, source=None, causal=False, cache=None):
        """Self-attention when ``source`` is None. ``cache`` (a dict) holds past keys/values:
        appended to for self-attention, computed once for cross-attention."""
        q = self.split(self.q(x))
        if cache is not None and source is not None and "k" in cache:
            k, v = cache["k"], cache["v"]
        else:
            k, v = (self.split(t) for t in self.kv(x if source is None else source).chunk(2, dim=-1))
            if cache is not None:
                if source is None and "k" in cache:
                    k, v = torch.cat([cache["k"], k], dim=2), torch.cat([cache["v"], v], dim=2)
                cache["k"], cache["v"] = k, v
        # A prompt (several queries) is masked causally; a single new query with a cache may
        # attend to every key before it.
        is_causal = causal and q.shape[2] > 1
        if is_causal and k.shape[2] != q.shape[2]:
            raise ValueError("a cached decoder takes one token at a time after the prompt")
        y = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal,
                                           dropout_p=self.dropout if self.training else 0.0)
        b, h, n, dh = y.shape
        return self.out(y.transpose(1, 2).reshape(b, n, h * dh))


class FeedForward(nn.Sequential):
    def __init__(self, d, ffn, dropout):
        super().__init__(nn.Linear(d, ffn), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn, d))


class EncoderLayer(nn.Module):
    def __init__(self, cfg: DonderFormerConfig):
        super().__init__()
        d = cfg.d_model
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = Attention(d, cfg.n_heads, cfg.dropout)
        self.ff = FeedForward(d, cfg.ffn, cfg.dropout)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        x = x + self.drop(self.attn(self.norm1(x)))
        return x + self.drop(self.ff(self.norm2(x)))


class DecoderLayer(nn.Module):
    def __init__(self, cfg: DonderFormerConfig):
        super().__init__()
        d = cfg.d_model
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.self_attn = Attention(d, cfg.n_heads, cfg.dropout)
        self.cross_attn = Attention(d, cfg.n_heads, cfg.dropout)
        self.ff = FeedForward(d, cfg.ffn, cfg.dropout)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, memory, cache=None):
        self_cache = cross_cache = None
        if cache is not None:
            self_cache, cross_cache = cache.setdefault("self", {}), cache.setdefault("cross", {})
        x = x + self.drop(self.self_attn(self.norm1(x), causal=True, cache=self_cache))
        x = x + self.drop(self.cross_attn(self.norm2(x), source=memory, cache=cross_cache))
        return x + self.drop(self.ff(self.norm3(x)))


class DonderFormer(nn.Module):
    def __init__(self, cfg: DonderFormerConfig = DonderFormerConfig(), window: WindowConfig = WindowConfig()):
        super().__init__()
        self.cfg, self.window = cfg, window
        self.vocab = Vocab(n_time=window.n_time)
        d = cfg.d_model

        self.front = nn.Sequential(
            nn.Conv1d(cfg.n_mels, d, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv1d(d, d, kernel_size=3, stride=cfg.conv_stride, padding=1), nn.GELU(),
        )
        self.register_buffer("frame_code", sinusoid(window.window_frames, d), persistent=False)
        self.encoder = nn.ModuleList(EncoderLayer(cfg) for _ in range(cfg.encoder_layers))
        self.encoder_norm = nn.LayerNorm(d)

        self.embed = nn.Embedding(self.vocab.size, d)
        self.position = nn.Embedding(window.max_tokens, d)
        self.decoder = nn.ModuleList(DecoderLayer(cfg) for _ in range(cfg.decoder_layers))
        self.decoder_norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(cfg.dropout)

        nn.init.normal_(self.embed.weight, std=d ** -0.5)
        nn.init.normal_(self.position.weight, std=0.02)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def encode(self, mel):
        """``(B, W, n_mels)`` -> ``(B, W / conv_stride, d)``."""
        x = self.front(mel.transpose(1, 2)).transpose(1, 2)
        x = x + self.frame_code[::self.cfg.conv_stride][:x.shape[1]]
        x = self.drop(x)
        for layer in self.encoder:
            x = layer(x)
        return self.encoder_norm(x)

    def embed_tokens(self, tokens, offset=0):
        x = self.embed(tokens) * math.sqrt(self.cfg.d_model)
        # Time tokens also carry the sinusoidal code of the frame they name.
        is_time = tokens >= self.vocab.TIME_BASE
        frames = (tokens - self.vocab.TIME_BASE).clamp(0, self.window.n_time - 1)
        x = x + self.frame_code[frames] * is_time.unsqueeze(-1)
        positions = torch.arange(offset, offset + tokens.shape[1], device=tokens.device)
        return self.drop(x + self.position(positions))

    def decode(self, tokens, memory, cache=None, offset=0):
        x = self.embed_tokens(tokens, offset)
        for i, layer in enumerate(self.decoder):
            x = layer(x, memory, None if cache is None else cache.setdefault(i, {}))
        return self.decoder_norm(x) @ self.embed.weight.T

    def forward(self, mel, tokens):
        """Teacher-forced logits ``(B, L, vocab)``; position t predicts token t + 1."""
        return self.decode(tokens, self.encode(mel))


def sequence_loss(logits, tokens, sep, label_smoothing=0.02, target_only=False):
    """Cross-entropy of next-token predictions after the 3 conditioning tokens (or after SEP)."""
    predicted, target = logits[:, :-1], tokens[:, 1:]
    positions = torch.arange(target.shape[1], device=tokens.device)
    first = (sep if target_only else torch.full_like(sep, 2)).to(tokens.device)
    mask = (positions.unsqueeze(0) >= first.unsqueeze(1)) & (target != Vocab.PAD)
    return F.cross_entropy(predicted[mask], target[mask], label_smoothing=label_smoothing)


# ---- generation

class Grammar:
    """Which tokens may come next in a target region: TIME then NOTE, times increasing inside
    the region, END only for an open long note, no onset while one is open."""

    def __init__(self, vocab: Vocab, lo: int, hi: int, long_open: bool):
        self.vocab, self.lo, self.hi = vocab, lo, hi
        self.long_open, self.last_time, self.pending_time = long_open, lo - 1, None
        self.just_opened = False

    def allowed(self, device) -> torch.Tensor:
        v = self.vocab
        mask = torch.zeros(v.size, dtype=torch.bool, device=device)
        if self.pending_time is None:
            first = self.last_time if self.just_opened else self.last_time + 1
            if first < self.hi:
                mask[v.TIME_BASE + first:v.TIME_BASE + self.hi] = True
            mask[v.EOS] = True
        elif self.long_open:
            mask[v.note("END")] = True
        else:
            for name in ONSET_NOTES + LONG_NOTES:
                mask[v.note(name)] = True
        return mask

    def step(self, token: int):
        v = self.vocab
        if v.is_time(token):
            self.pending_time = token - v.TIME_BASE
            return
        name = NOTES[token - v.NOTE_BASE]
        self.just_opened = name in LONG_NOTES
        self.long_open = name in LONG_NOTES or (self.long_open and name != "END")
        self.last_time, self.pending_time = self.pending_time, None


def sample_token(logits, temperature: float, top_p: float, generator) -> int:
    if temperature <= 0:
        return int(logits.argmax())
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    if top_p < 1.0:
        sorted_probs, order = probs.sort(descending=True)
        keep = sorted_probs.cumsum(0) - sorted_probs < top_p
        probs = torch.zeros_like(probs).scatter(0, order[keep], sorted_probs[keep])
    return int(torch.multinomial(probs / probs.sum(), 1, generator=generator))


@torch.inference_mode()
def generate_chart(model: DonderFormer, mel, course: str, level, temperature: float = 1.0,
                   top_p: float = 0.9, seed: int = 0, max_new_tokens: int = 400) -> np.ndarray:
    """Shared-format classes ``(n_frames,)`` for one song. ``mel``: ``(n_frames, n_mels)``, normalized."""
    model.eval()
    device = next(model.parameters()).device
    cfg, vocab = model.window, model.vocab
    P, T, W = cfg.past_frames, cfg.target_frames, cfg.window_frames
    n = len(mel)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    events = []                                   # global (frame, note), sorted

    for start in range(-P, n - P, T):
        window = np.zeros((W, mel.shape[1]), np.float32)
        lo, hi = max(start, 0), min(start + W, n)
        window[lo - start:hi - start] = mel[lo:hi]
        memory = model.encode(torch.from_numpy(window).unsqueeze(0).to(device))

        past = window_events(events, start, start + P)
        prompt, _ = build_sequence(vocab, course, level, past, None,
                                   max_tokens=cfg.max_tokens - min(max_new_tokens, cfg.max_tokens // 2))
        long_open = bool(events) and events[-1][1] in LONG_NOTES
        grammar = Grammar(vocab, P, min(P + T, n - start), long_open)

        cache = {}
        logits = model.decode(torch.tensor([prompt], device=device), memory, cache)[0, -1]
        length, new = len(prompt), []
        while length < cfg.max_tokens and len(new) < max_new_tokens:
            logits = logits.masked_fill(~grammar.allowed(logits.device), float("-inf"))
            token = sample_token(logits.cpu(), temperature, top_p, generator)
            if token == vocab.EOS:
                break
            new.append(token)
            grammar.step(token)
            logits = model.decode(torch.tensor([[token]], device=device), memory, cache, offset=length)[0, -1]
            length += 1
        events += [(start + f, note) for f, note in tokens_to_events(vocab, new)]

    return events_to_classes(events, n)
