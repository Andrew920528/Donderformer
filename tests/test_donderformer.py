import numpy as np
import torch

from donderformer.data.chart_dataset import WindowConfig, collate
from donderformer.data.tokens import Vocab, build_sequence, classes_to_events
from donderformer.models.donderformer import (DonderFormer, DonderFormerConfig, Grammar, generate_chart,
                                              sequence_loss)

WINDOW = WindowConfig(past_frames=64, target_frames=32, lookahead_frames=8, max_tokens=96)
CFG = DonderFormerConfig(d_model=32, n_heads=4, encoder_layers=1, decoder_layers=2, ffn=64, dropout=0.0)


def tiny_model(seed=0):
    torch.manual_seed(seed)
    return DonderFormer(CFG, WINDOW).eval()


def test_cached_decoding_matches_full_forward():
    model, vocab = tiny_model(), Vocab(n_time=WINDOW.n_time)
    mel = torch.randn(1, WINDOW.window_frames, 80)
    tokens, _ = build_sequence(vocab, "oni", 8, [(3, "don"), (10, "ka")], [(70, "DON"), (80, "ROLL"), (85, "END")])
    tokens = torch.tensor([tokens])
    with torch.no_grad():
        full = model(mel, tokens)[0]
        memory, cache = model.encode(mel), {}
        prompt = 6
        step = [model.decode(tokens[:, :prompt], memory, cache)[0, -1]]
        for i in range(prompt, tokens.shape[1]):
            step.append(model.decode(tokens[:, i:i + 1], memory, cache, offset=i)[0, -1])
    assert torch.allclose(torch.stack(step), full[prompt - 1:], atol=1e-4)


def test_grammar_only_allows_valid_targets():
    vocab = Vocab(n_time=WINDOW.n_time)
    g = Grammar(vocab, lo=64, hi=96, long_open=False)
    allowed = g.allowed("cpu")
    assert allowed[vocab.time(64)] and not allowed[vocab.time(63)] and allowed[vocab.EOS]
    assert not allowed[vocab.note("don")]
    g.step(vocab.time(70))
    allowed = g.allowed("cpu")
    assert allowed[vocab.note("don")] and allowed[vocab.note("ROLL")] and not allowed[vocab.note("END")]
    g.step(vocab.note("ROLL"))
    assert g.allowed("cpu")[vocab.time(70)]            # a one-frame roll may end on its start frame
    g.step(vocab.time(75))
    allowed = g.allowed("cpu")
    assert allowed[vocab.note("END")] and not allowed[vocab.note("don")]


def test_generate_chart_returns_shared_format():
    model = tiny_model()
    mel = np.random.default_rng(0).standard_normal((300, 80)).astype(np.float32)
    for temperature in (0.0, 1.0):
        classes = generate_chart(model, mel, "oni", 9, temperature=temperature, seed=1)
        assert classes.shape == (300,) and classes.min() >= 0 and classes.max() <= 6
        # Every long note is a contiguous run and the events round-trip.
        from donderformer.data.tokens import events_to_classes
        assert np.array_equal(events_to_classes(classes_to_events(classes), 300), classes)


def test_overfits_one_batch():
    model = tiny_model().train()
    vocab = Vocab(n_time=WINDOW.n_time)
    batch = []
    for k in range(4):
        tokens, sep = build_sequence(vocab, "oni", 8, [(5 + k, "don")], [(70 + k, "ka"), (90, "DON")])
        batch.append((torch.randn(WINDOW.window_frames, 80), torch.tensor(tokens), sep))
    mel, tokens, sep = collate(batch)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    losses = []
    for _ in range(150):
        optimizer.zero_grad()
        loss = sequence_loss(model(mel, tokens), tokens, sep, label_smoothing=0.0)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < 0.1 * losses[0]
