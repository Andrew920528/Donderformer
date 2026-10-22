import numpy as np
import pytest

from donderformer.data.tokens import (Vocab, build_sequence, classes_to_events, events_to_classes,
                                      events_to_tokens, tokens_to_events, window_events)

vocab = Vocab(n_time=768)


def random_chart(rng, n=3000):
    """Shared-format classes with onsets and long notes, like events_to_labels writes them."""
    classes = np.zeros(n, dtype=np.int64)
    frame = 0
    while frame < n:
        frame += int(rng.integers(1, 12))
        if frame >= n:
            break
        if rng.random() < 0.1:
            length = int(rng.integers(1, 60))
            classes[frame:frame + length] = rng.choice([5, 6])
            frame += length
        else:
            classes[frame] = rng.integers(1, 5)
    return classes


def test_vocab_layout():
    assert vocab.size == 795
    assert vocab.name(vocab.course("oni")) == "ONI"
    assert vocab.level(9) == vocab.LEVEL_BASE + 8
    assert vocab.level("") == vocab.level(None) == vocab.level(11) == vocab.LEVEL_UNK
    assert vocab.name(vocab.time(767)) == "TIME_767"


@pytest.mark.parametrize("seed", range(20))
def test_classes_round_trip(seed):
    classes = random_chart(np.random.default_rng(seed))
    assert np.array_equal(events_to_classes(classes_to_events(classes), len(classes)), classes)


def test_adjacent_long_notes_and_one_frame_roll():
    classes = np.array([0, 5, 5, 6, 6, 0, 5, 1, 0])
    events = classes_to_events(classes)
    assert events == [(1, "ROLL"), (2, "END"), (3, "BALLOON"), (4, "END"), (6, "ROLL"), (6, "END"), (7, "don")]
    assert np.array_equal(events_to_classes(events, len(classes)), classes)


@pytest.mark.parametrize("seed", range(5))
def test_windows_reassemble_the_chart(seed):
    """Cutting a chart into target regions and writing them back gives the chart again."""
    classes = random_chart(np.random.default_rng(seed))
    events = classes_to_events(classes)
    rebuilt = []
    for start in range(0, len(classes), 256):
        for frame, note in window_events(events, start, start + 256):
            if frame == 0 and note in ("ROLL", "BALLOON") and rebuilt and rebuilt[-1][1] != "END" \
                    and rebuilt[-1][1] == note:
                continue            # continuation of the long note opened in the previous window
            rebuilt.append((frame + start, note))
    assert np.array_equal(events_to_classes(rebuilt, len(classes)), classes)


def test_window_events_long_note_across_boundaries():
    events = [(10, "don"), (90, "ROLL"), (300, "END"), (400, "ka")]
    assert window_events(events, 100, 200) == [(0, "ROLL")]                  # open throughout
    assert window_events(events, 200, 350) == [(0, "ROLL"), (100, "END")]    # closes inside
    assert window_events(events, 50, 100) == [(40, "ROLL")]                  # opens inside


def test_tokens_round_trip_and_sequence():
    events = [(3, "don"), (520, "ROLL"), (530, "END"), (600, "KA")]
    tokens = events_to_tokens(vocab, events)
    assert tokens_to_events(vocab, tokens) == events
    seq, sep = build_sequence(vocab, "oni", 9, events[:1], events[1:])
    assert vocab.decode(seq) == ("BOS ONI LEVEL_9 TIME_3 don SEP TIME_520 ROLL TIME_530 END "
                                 "TIME_600 KA EOS")
    assert seq[sep] == vocab.SEP


def test_sequence_trims_oldest_past_pairs():
    past = [(k, "don") for k in range(100)]
    seq, sep = build_sequence(vocab, "oni", 9, past, [(600, "ka")], max_tokens=40)
    assert len(seq) <= 40 and seq[sep] == vocab.SEP
    kept = tokens_to_events(vocab, seq[3:sep])
    assert kept == past[-len(kept):]
