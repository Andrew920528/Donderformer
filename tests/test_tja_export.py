import unittest

import numpy as np

from donderformer.data.tja_export import chart_to_tja


def measures(text):
    body = text.split("#START")[1].split("#END")[0]
    return [m for m in body.replace("\n", "").split(",") if m]


class SnapTests(unittest.TestCase):
    def setUp(self):
        # 16th notes at 150 BPM (0.1 s apart) on a 23 ms grid, with every third note a frame late
        self.frames = [int(i * 100 / 23) + (i % 3 == 1) for i in range(64)]   # frame holding the beat, or the next
        self.chart = np.zeros(400, dtype=np.int64)
        self.chart[self.frames] = 1

    def test_exact_keeps_frames_but_needs_fine_subdivisions(self):
        lengths = {len(m) for m in measures(chart_to_tja(self.chart, 23, 150.0, 0.0))}
        self.assertTrue(all(n > 16 for n in lengths))

    def test_snap_puts_jittered_notes_on_the_16th_grid(self):
        text = chart_to_tja(self.chart, 23, 150.0, 0.0, snap_frames=1)
        rows = measures(text)
        self.assertEqual({len(m) for m in rows}, {16})
        self.assertEqual(sum(m.count("1") for m in rows), len(self.frames))


if __name__ == "__main__":
    unittest.main()
