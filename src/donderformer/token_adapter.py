"""ASSUMED token format only: replace when the Transformer team defines its spec.

TODO(team): agree on the vocabulary, absolute/relative times, window offsets,
END marker, collision policy, and held-note duration tokens. No Transformer or
vocabulary exists in this repository yet; this is a strict adapter stub, not a
claim of compatibility with a future model.
"""
from pathlib import Path

import numpy as np


def tokens_to_frames(tokens, frame_ms):
    """ASSUMPTION: [absolute_ms, integer_class_1_to_6, ..., 'END'].

    A nonnegative millisecond timestamp is floored into a frame. Each token is
    one note start; held durations cannot be reconstructed from this assumed
    spec. Merge/offset multiple windows before calling. Duplicate frame starts,
    adjacent identical held starts, missing END and trailing tokens are errors.
    Length ends at the last event; pad to the human/audio length in the writer.
    """
    if not np.isfinite(frame_ms) or frame_ms <= 0:
        raise ValueError('frame_ms must be positive and finite')
    tokens = list(tokens)
    if not tokens or tokens[-1] != 'END' or len(tokens[:-1]) % 2:
        raise ValueError("Assumed spec requires alternating absolute_ms/type pairs followed by END")
    events = []
    for time, kind in zip(tokens[:-1:2], tokens[1:-1:2]):
        if isinstance(time, (bool, str)) or not np.isscalar(time) or not np.isfinite(time) or time < 0:
            raise ValueError('Time tokens must be nonnegative finite absolute milliseconds')
        if not isinstance(kind, (int, np.integer)) or isinstance(kind, bool) or not 1 <= kind <= 6:
            raise ValueError('Type tokens must be integer note classes 1..6')
        frame = int(time // frame_ms)
        if events and frame <= events[-1][0]:
            raise ValueError('Events must increase in time with at most one start per frame')
        if events and frame == events[-1][0]+1 and kind == events[-1][1] and kind in (5, 6):
            raise ValueError('Adjacent identical held starts cannot be distinguished in the frame format')
        events.append((frame, kind))
    sequence = np.zeros(events[-1][0]+1 if events else 0, dtype=np.int64)
    for frame, kind in events:
        sequence[frame] = kind
    return sequence


def save_token_chart(path, tokens, human, frame_ms):
    """Pad assumed token events to the human frame grid and write exporter keys."""
    from donderformer.evaluation import note_starts, remove_adjacent_notes
    human = np.asarray(human)
    note_starts(human)
    generated = tokens_to_frames(tokens, frame_ms)
    if len(generated) > len(human):
        raise ValueError('Token event extends beyond the reference frame sequence')
    generated = np.pad(generated, (0, len(human)-len(generated)))
    path = Path(path)
    if path.suffix != '.npz':
        raise ValueError('Generated chart path must end in .npz')
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite an existing generated chart: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, human=human, generated=generated, clean=remove_adjacent_notes(generated))
    return path
