"""Synthetic checks; no raw ESE, cached full dataset, or training required."""
import ast
import csv
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from donderformer import evaluation as ev
from donderformer.data.taiko_dataset import Window
from donderformer.data.data_splitter import manifest_hash
from donderformer.data.feature_extractor import FeatureExtractor
from donderformer.models.taikonation import TaikoNationBaseline
from donderformer.paths import DataPaths, REPO_ROOT
from donderformer.runs import save_run


class MetricsTests(unittest.TestCase):
    def test_matching_and_type(self):
        human = np.array([0, 1, 0, 0, 2, 0])
        ai = np.array([1, 0, 0, 0, 3, 1])
        np.testing.assert_allclose(ev.onset_metrics(human, ai), (2/3, 1, .8, 2))
        np.testing.assert_allclose(ev.joint_onset_type_metrics(human, ai), (1/3, .5, .4, 1))
        # Nearest ties select the earlier human onset; predictions cannot reuse it.
        self.assertEqual(ev.onset_metrics(np.array([1, 0, 2, 0]), np.array([0, 1, 0, 2]))[3], 2)
        self.assertEqual(ev.onset_metrics(np.array([0, 1, 0]), np.array([1, 1, 1]))[3], 1)
        # Recall-style OCHuman deliberately permits reuse, unlike one-to-one F1.
        self.assertEqual(ev.ochuman_score(np.array([1, 0, 1]), np.array([0, 2, 0])), (2, 2))

    def test_empty_notes_and_boundaries(self):
        empty = np.zeros(8, dtype=int)
        self.assertEqual(ev.onset_metrics(empty, empty), (0, 0, 0, 0))
        self.assertEqual(ev.joint_onset_type_metrics(empty, empty), (0, 0, 0, 0))
        self.assertEqual(ev.score_from_counts(0, 0, 0), (0, 0, 0))
        self.assertEqual(ev.ochuman_score(empty, empty), (0, 0))
        self.assertEqual(ev.onset_metrics(np.array([1, 0, 0]), np.array([0, 0, 1]))[3], 0)
        paper = ev.calculate_paper_metrics(empty, empty)
        self.assertEqual(paper['DCHuman'], 100)
        self.assertEqual(paper['OCHuman'], 100)
        self.assertEqual(paper['Over_PSpace'], 100/256)
        self.assertEqual(paper['HI_PSpace'], 100)
        self.assertEqual(ev.extract_patterns([1, 0], window=8), set())

    def test_held_notes_cleaning_and_paper_tolerance(self):
        seq = np.array([1, 2, 3, 0, 5, 5, 6, 6, 0, 4])
        np.testing.assert_array_equal(ev.to_binary_chart(seq), [1, 1, 1, 0, 1, 0, 1, 0, 0, 1])
        np.testing.assert_array_equal(ev.remove_adjacent_notes(seq), [1, 0, 0, 0, 5, 0, 0, 0, 0, 4])
        self.assertAlmostEqual(ev.paper_ochuman([1, 0, 0], [0, 1, 0]), 2/3)

    def test_overlap_reconstruction_and_sampling(self):
        seq = np.array([0, 1, 2, 3, 4, 5])
        windows = np.eye(7, dtype=np.float32)[np.array([seq[i:i+4] for i in range(3)])]
        np.testing.assert_array_equal(ev.reconstruct_true_sequence(windows), seq)
        aggregated = ev.aggregate_overlapping_predictions(windows)
        np.testing.assert_array_equal(aggregated, np.eye(7)[seq])
        np.testing.assert_array_equal(ev.sample_chart(aggregated, np.random.default_rng(42)), seq)
        self.assertEqual(ev.aggregate_overlapping_predictions(windows[:0]).shape, (0, 7))
        self.assertEqual(ev.reconstruct_true_sequence(windows[:0]).shape, (0,))

    def test_ar_history_and_tail(self):
        class RecordingModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.inputs = []
            def forward(self, x):
                self.inputs.append(x.cpu().numpy().copy())
                logits = torch.full((len(x), 4, 7), -1000., device=x.device)
                logits[:, :, 2] = 0
                return logits
        x = np.full((3, 1385), 9, dtype=np.float32)
        model = RecordingModel()
        result = ev.generate_autoregressive(model, x, 'cpu', seed=43)
        np.testing.assert_array_equal(result, np.full(6, 2))
        self.assertEqual(len(model.inputs), 3)
        np.testing.assert_array_equal(model.inputs[0][0, 1280:], 0)
        history = model.inputs[2][0, 1280:].reshape(15, 7)
        np.testing.assert_array_equal(history[:-2], 0)
        np.testing.assert_array_equal(history[-2:], np.eye(7)[[2, 2]])
        changed = x.copy(); changed[:, 1280:] = -99
        np.testing.assert_array_equal(result, ev.generate_autoregressive(RecordingModel(), changed, 'cpu', seed=43))


    def test_ar_saved_window_and_zero_history(self):
        class RecordingModel(torch.nn.Module):
            def __init__(self, window):
                super().__init__()
                self.window = window
                self.inputs = []
            def forward(self, x):
                self.inputs.append(x.cpu().numpy().copy())
                logits = torch.full((len(x), 4, 7), -1000., device=x.device)
                logits[:, :, 2] = 0
                return logits
        for window in (Window(history=12, look_forward=3),
                       Window(audio_frames=20, history=0, look_forward=4)):
            with self.subTest(window=window):
                x = np.full((3, window.n_inputs), 9, dtype=np.float32)
                model = RecordingModel(window)
                result = ev.generate_autoregressive(model, x, 'cpu', seed=43)
                np.testing.assert_array_equal(result, np.full(6, 2))
                for i, values in enumerate(model.inputs):
                    self.assertEqual(values.shape, (1, window.n_inputs))
                    np.testing.assert_array_equal(values[0, :window.audio_frames*80], 9)
                    history = np.zeros((window.history, 7), dtype=np.float32)
                    length = min(i, window.history)
                    if length:
                        history[-length:] = np.eye(7)[np.full(length, 2)]
                    np.testing.assert_array_equal(values[0, window.audio_frames*80:], history.ravel())
                changed = x.copy(); changed[:, window.audio_frames*80:] = -99
                np.testing.assert_array_equal(result, ev.generate_autoregressive(RecordingModel(window), changed, 'cpu', seed=43))


class SavedRunTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.paths = DataPaths(self.root / 'data')
        self.extractor = FeatureExtractor()
        split = self.paths.split_dir('synthetic')
        split.mkdir(parents=True)
        with (split / 'manifest.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=['chart_id', 'audio_id', 'split', 'title', 'audio', 'bpm', 'offset', 'level', 'course'])
            writer.writeheader()
            writer.writerows([dict(chart_id=f'chart{i}', audio_id=f'audio{i}', split='test', title=f'Song {i}', audio=f'song{i}.ogg', bpm=120, offset=0, level=5, course='oni') for i in range(2)] + [dict(chart_id='chart_train', audio_id='audio_train', split='train', title='Training song', audio='train.ogg', bpm=120, offset=0, level=5, course='oni')])
        norm = self.extractor.normalization_path(self.paths, 'synthetic')
        norm.parent.mkdir(parents=True)
        np.savez(norm, mean=np.zeros(80), std=np.ones(80), manifest_sha256=manifest_hash(self.paths, 'synthetic'))
        for i in [0, 1, '_train']:
            mel = self.extractor.mel_path(self.paths, f'audio{i}')
            labels = self.extractor.labels_path(self.paths, f'audio{i}', f'chart{i}')
            labels.parent.mkdir(parents=True)
            np.save(mel, np.zeros((9, 80), dtype=np.float32))
            np.save(labels, np.eye(7, dtype=np.float32)[[0, 1, 0, 2, 0, 5, 5, 0, 0]])
        self.run = self.root / 'run'; self.run.mkdir()
        self.config = dict(model='taikonation', data=dict(split='synthetic', features='frame23ms',
                           extractor=asdict(self.extractor), audio_frames=16, history_frames=15, target_frames=4))
        # An untrained model is saved solely to test the checkpoint contract.
        self.model = TaikoNationBaseline()
        save_run(self.run, self.model, self.config, losses={'train': [], 'val': []})

    def test_checkpoint_and_legacy_config(self):
        model, config, dataset = ev.load_evaluation_run(self.run, self.paths)
        self.assertEqual(config['data'], self.config['data'])
        self.assertEqual(dataset.chart_ids, ['chart0', 'chart1'])
        for name, value in self.model.state_dict().items():
            torch.testing.assert_close(model.state_dict()[name], value)
        # Check embedded-config fallback and older feature-ID-only saved runs.
        checkpoint = torch.load(self.run/'model.pt', weights_only=True)
        del checkpoint['config']['data']['extractor']
        torch.save(checkpoint, self.run/'model.pt')
        (self.run/'config.json').unlink()
        ev.load_evaluation_run(self.run, self.paths)

    def test_window_config_and_oni_v2_split(self):
        # The newer split supplies LEVEL directly; no raw TJA or EDA is needed.
        split = self.paths.split_dir('oni_v2'); split.mkdir()
        (split/'manifest.csv').write_bytes((self.paths.split_dir('synthetic')/'manifest.csv').read_bytes())
        np.savez(self.extractor.normalization_path(self.paths, 'oni_v2'),
                 mean=np.zeros(80), std=np.ones(80), manifest_sha256=manifest_hash(self.paths, 'oni_v2'))
        from donderformer.chart_evaluation import load_reference_context
        for window in (Window(), Window(history=12, look_forward=3),
                       Window(audio_frames=20, history=0, look_forward=4)):
            with self.subTest(window=window):
                data = dict(split='oni_v2', features=self.extractor.id,
                            extractor=asdict(self.extractor), window=window.to_dict())
                save_run(self.run, TaikoNationBaseline(window, dropout=.35),
                         dict(model='taikonation', data=data, dropout=.35), losses={})
                model, config, dataset = ev.load_evaluation_run(self.run, self.paths)
                self.assertEqual(dataset.split, 'oni_v2')
                self.assertEqual(model.window, window)
                self.assertEqual(dataset.window, window)
                self.assertEqual(model.dropout.p, .35)
                x, y = dataset.chart_windows('chart0')
                probs = ev.predict_teacher_forced(model, x, 'cpu', batch_size=2)
                self.assertEqual(probs.shape, y.shape)
                self.assertEqual(len(ev.generate_autoregressive(model, x, 'cpu')), 9)
                context = load_reference_context(self.paths, dataset.split, dataset.extractor)
                self.assertEqual(set(context.test_humans), {'chart0', 'chart1'})
                self.assertFalse(self.paths.raw.exists())

    def test_incomplete_or_unsupported_saved_window(self):
        for settings, message in ((dict(audio_frames=16), 'Saved data.window'),
                                  (Window(targets=3).to_dict(), 'four-target')):
            config = dict(model='taikonation', data=dict(split='synthetic', features='frame23ms', window=settings))
            (self.run/'config.json').write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, message):
                ev.load_evaluation_run(self.run, self.paths)

    def test_missing_inputs(self):
        with self.assertRaisesRegex(FileNotFoundError, 'Set RUN_DIR'):
            ev.load_evaluation_run(self.root/'missing', self.paths)
        self.extractor.labels_path(self.paths, 'audio0', 'chart0').unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'Restore the matching feature cache'):
            ev.load_evaluation_run(self.run, self.paths)

    def test_notebook_runs_and_resumes(self):
        notebook = json.loads((REPO_ROOT/'notebooks/06_evaluation.ipynb').read_text())
        namespace = {}
        show_patch = patch.object(plt, 'show')
        show_patch.start()
        self.addCleanup(show_patch.stop)
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                source = source.replace('RUN_DIR = outputs_dir() / "runs" / "taikonation" / "<saved-run-folder>"', f'RUN_DIR = Path({str(self.run)!r})')
                source = source.replace('paths = DataPaths()', f'paths = DataPaths({str(self.paths.root)!r})')
                exec(compile(source, '<evaluation notebook>', 'exec'), namespace)
        out = self.run/'eval'
        self.assertEqual(set(json.loads((out/'teacher_forced/results.json').read_text())),
                         {'onset_precision', 'onset_recall', 'onset_f1', 'joint_precision', 'joint_recall', 'joint_f1'})
        with (out/'teacher_forced/scored/analysis/micro_summary.csv').open() as f:
            raw_onset = next(r for r in csv.DictReader(f) if r['scope']=='overall' and r['version']=='raw' and r['metric']=='onset')
        tf_aggregate = json.loads((out/'teacher_forced/results.json').read_text())
        self.assertAlmostEqual(float(raw_onset['f1']), tf_aggregate['onset_f1'])
        breakdown = json.loads((out/'teacher_forced/ochuman_by_difficulty.json').read_text())
        self.assertEqual({r['scope'] for r in breakdown}, {'overall', 'difficulty'})
        with np.load(out/'teacher_forced/ochuman.npz') as data:
            self.assertEqual(set(data.files), {'micro', 'macro', 'song_scores'})
        for chart in (out/'ar_test/charts').glob('*.npz'):
            with np.load(chart) as data:
                self.assertEqual(set(data.files), {'human', 'generated', 'clean'})
                self.assertEqual(data['human'].shape, (9,))
        with (out/'ar_test/summary.csv').open() as summary_file:
            self.assertEqual(len(list(csv.DictReader(summary_file))), 20)
        self.assertTrue((out/'plots/human_vs_autoregressive.png').is_file())
        self.assertTrue((out/'plots/taiko_chart_comparison.png').is_file())
        # Exercise the real exporter against the evaluation output schema.
        from donderformer.exporter import ChartExporter
        self.paths.raw.mkdir()
        for i in range(2):
            (self.paths.raw/f'song{i}.ogg').write_bytes(b'synthetic audio placeholder')
        exporter = ChartExporter(alias='synthetic', sources={'lstm': out/'ar_test/charts'},
                                 paths=self.paths, split='synthetic', include_human=True)
        with patch('donderformer.exporter.outputs_dir', return_value=self.root/'exports'):
            export_dir = exporter.export()
        self.assertEqual(len(list(export_dir.glob('*/*.tja'))), 4)
        before = (out/'ar_test/results.json').read_bytes()
        def fail_generation(*args, **kwargs):
            raise AssertionError('Existing charts should be reused')
        namespace['generate_autoregressive'] = fail_generation
        ar_cell = next(c for c in notebook['cells'] if c['cell_type']=='code' and 'all_results = []' in ''.join(c['source']))
        exec(''.join(ar_cell['source']), namespace)
        self.assertEqual(before, (out/'ar_test/results.json').read_bytes())
        plt.close('all')

    def test_training_save_contract(self):
        notebook = json.loads((REPO_ROOT/'notebooks/03_train_lstm.ipynb').read_text())
        code = '\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code')
        ast.parse(code)
        self.assertNotIn('test_dataset', code)
        self.assertNotIn('generate_autoregressive', code)
        values = {n.targets[0].id: ast.literal_eval(n.value)
                  for n in ast.parse(code).body if isinstance(n, ast.Assign)
                  and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.Constant)}
        self.assertEqual({k: values[k] for k in ('SPLIT','AUDIO_FRAMES','HISTORY','LOOK_FORWARD','DROPOUT')},
                         dict(SPLIT='oni_v2', AUDIO_FRAMES=16, HISTORY=15, LOOK_FORWARD=0, DROPOUT=.2))
        self.assertIn('"window": WINDOW.to_dict()', code)
        self.assertIn('dropout=DROPOUT', code)
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                self.assertEqual(cell['outputs'], [])
                self.assertIsNone(cell['execution_count'])
        save_cell = next(c for c in notebook['cells'] if c['cell_type']=='code' and 'checkpoint = save_run(' in ''.join(c['source']))
        namespace = dict(RUN_DIR=self.run, model=self.model, MODEL_NAME='taikonation', ALIAS='',
                         DATA_INFO=self.config['data'], DROPOUT=.2, HPARAMS={}, device='cpu', train_losses=[],
                         val_losses=[], finetune_train_losses=[], finetune_val_losses=[])
        exec(''.join(save_cell['source']), namespace)
        ev.load_evaluation_run(self.run, self.paths)


if __name__ == '__main__':
    unittest.main()
