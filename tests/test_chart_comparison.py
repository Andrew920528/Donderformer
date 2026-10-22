"""Metric sanity, data provenance, random control, and cross-model pairing tests."""
import csv
import itertools
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import numpy as np

from donderformer import evaluation as ev
from donderformer.chart_evaluation import (
    build_reference_context, difficulty_statistics, evaluate_random_baseline,
    evaluate_saved_charts, load_reference_context, pattern_set, random_chart,
)
from donderformer.data.feature_extractor import FeatureExtractor
from donderformer.evaluation_comparison import compare_models, load_model_results, wilcoxon_signed_rank
from donderformer.evaluation_config import evaluation_config
from donderformer.paths import DataPaths
from donderformer.token_adapter import save_token_chart, tokens_to_frames


def references():
    human = np.zeros(128, dtype=np.int64)
    human[[5, 25, 45, 65, 85, 105]] = [1, 2, 3, 4, 1, 2]
    human[110:114] = 5
    human[120:124] = 6
    other = human.copy(); other[45] = 2
    metadata = {c: dict(difficulty='stars=7', level=7, course='oni') for c in ('a', 'b', 'train')}
    config = evaluation_config({'PATTERN_WINDOWS_FRAMES': [8, 16]})
    return build_reference_context({'a': human, 'b': other}, {'train': human}, metadata, config)


class SanityTests(unittest.TestCase):
    def setUp(self):
        self.context = references()
        self.human = self.context.test_humans['a']

    def test_self_shift_and_type_shuffle(self):
        self.assertEqual(ev.onset_metrics(self.human, self.human)[2], 1)
        self.assertEqual(ev.joint_onset_type_metrics(self.human, self.human)[2], 1)
        shifted = np.r_[np.zeros(3, dtype=int), self.human[:-3]]
        self.assertLess(ev.onset_metrics(self.human, shifted)[2], 1)
        changed = self.human.copy()
        # Swap don/ka and big don/ka without changing onset times or holds.
        changed[self.human == 1] = 2; changed[self.human == 2] = 1
        changed[self.human == 3] = 4; changed[self.human == 4] = 3
        self.assertEqual(ev.onset_metrics(self.human, changed)[2], 1)
        self.assertLess(ev.joint_onset_type_metrics(self.human, changed)[2], 1)

    def test_greedy_matching_agrees_with_naive_start_reference(self):
        rng = np.random.default_rng(123)
        for _ in range(30):
            human = ev.note_starts(rng.integers(0,7,128))
            predicted = ev.note_starts(rng.integers(0,7,128))
            for tolerance in (0,1,3):
                for typed,metric in [(False,ev.onset_metrics),(True,ev.joint_onset_type_metrics)]:
                    available=set(np.flatnonzero(human)); matches=0
                    for position in np.flatnonzero(predicted):
                        candidates=[p for p in sorted(available) if abs(p-position)<=tolerance and (not typed or human[p]==predicted[position])]
                        if candidates:
                            nearest=min(candidates,key=lambda p:abs(p-position))
                            available.remove(nearest); matches+=1
                    self.assertEqual(metric(human,predicted,tolerance)[3],matches)

    def test_held_note_before_after_and_paper_unchanged(self):
        human = np.array([0, 5, 5, 5, 0, 2, 0, 0], dtype=int)
        generated = np.array([0, 5, 0, 0, 0, 2, 0, 0], dtype=int)
        # Legacy frame-count F1: TP=2, predicted=2, actual=4 -> 2/3.
        self.assertAlmostEqual(ev.score_from_counts(2, 2, 4)[2], 2/3)
        self.assertEqual(ev.onset_metrics(human, generated)[2], 1)
        self.assertEqual(ev.joint_onset_type_metrics(human, generated)[2], 1)
        np.testing.assert_array_equal(ev.note_starts(human), generated)
        legacy = ev.calculate_paper_metrics(human, generated)
        self.assertEqual(legacy['DCHuman'], 100)
        self.assertEqual(legacy['Over_PSpace'], 100/256)
        self.assertEqual(legacy['HI_PSpace'], 100)
        self.assertEqual(np.count_nonzero(ev.note_starts(np.array([5,5,6,6,1,1]))), 4)

    def test_repeating_pattern_and_density_band(self):
        from donderformer.chart_evaluation import score_chart
        rich = np.random.default_rng(0).integers(0, 3, 128)
        repeated = np.tile([1,0,0,0,0,0,0,0], 16)
        self.assertLess(len(pattern_set(repeated,8,'binary')), len(pattern_set(rich,8,'binary')))
        result = score_chart(self.human, repeated, 'a', self.context)
        self.assertLess(result['pspace_binary_w8_overall'], 100)
        dense = self.human.copy(); dense[[15,35,55,75,95,100,115,125]] = 1
        self.assertGreater(difficulty_statistics(dense,23)['note_density'], self.context.bands['stars=7']['note_density']['upper'])
        self.assertFalse(score_chart(self.human,dense,'a',self.context)['note_density_in_band'])
        self.assertIsNone(difficulty_statistics(np.zeros(8,dtype=int),23)['note_density'])
        self.assertIsNone(difficulty_statistics(np.ones(8,dtype=int),23)['don_ka_ratio'])

    def test_pools_include_other_humans_and_type_variant(self):
        from donderformer.chart_evaluation import score_chart
        expected = pattern_set(self.human,8,'type') | pattern_set(self.context.test_humans['b'],8,'type')
        self.assertEqual(self.context.pools[('stars=7',8,'type')], expected)
        result = score_chart(self.human,self.human,'a',self.context)
        self.assertAlmostEqual(result['pspace_type_w8_hi'], len(pattern_set(self.human,8,'type'))/len(expected)*100)
        self.assertEqual(result['pspace_type_w8_ratio'], 1)
        # Different held durations produce identical start patterns.
        short = self.human.copy(); short[111:114]=0; short[121:124]=0
        self.assertEqual(pattern_set(self.human,8,'type'),pattern_set(short,8,'type'))
        self.assertEqual(self.context.bands['stars=7']['note_density']['n'], 1)

    def test_difficulty_pools_are_separate_and_long_windows_null(self):
        from donderformer.chart_evaluation import score_chart
        a = np.array([1,0,1,0,1,0,1,0], dtype=int)
        b = np.array([0,2,0,2,0,2,0,2], dtype=int)
        metadata = {c: dict(difficulty=d, course='oni', level=int(d.split('=')[1])) for c,d in [('a','stars=7'),('ta','stars=7'),('b','stars=8'),('tb','stars=8')]}
        context = build_reference_context({'a':a,'b':b},{'ta':a,'tb':b},metadata, {'MIN_CHARTS_PER_BAND_GROUP':1})
        self.assertEqual(context.pools[('stars=7',8,'type')],pattern_set(a,8,'type'))
        self.assertEqual(context.pools[('stars=8',8,'type')],pattern_set(b,8,'type'))
        result = score_chart(a,a,'a',context)
        self.assertEqual(result['pspace_type_w8_hi'],100)
        self.assertIsNone(result['pspace_type_w16_hi'])
        self.assertIsNone(result['pspace_binary_w64_ratio'])

    def test_random_preserves_starts_and_order_independence(self):
        probabilities=self.context.type_probabilities['stars=7']
        for seed in self.context.config['RANDOM_BASELINE_SEEDS']:
            generated=random_chart(self.human,probabilities,seed,'a')
            self.assertEqual(np.count_nonzero(ev.note_starts(generated)),np.count_nonzero(ev.note_starts(self.human)))
            np.testing.assert_array_equal(generated,random_chart(self.human,probabilities,seed,'a'))
        dense=np.ones(128,dtype=int)
        generated=random_chart(dense,probabilities,0,'dense')
        self.assertEqual(np.count_nonzero(ev.note_starts(generated)),128)


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name); self.context=references()
        self.charts=self.root/'run/eval/ar_test/charts'; self.charts.mkdir(parents=True)
        human=self.context.test_humans['a']
        np.savez(self.charts/'a.npz',human=human,generated=human)  # clean optional

    def test_generic_outputs_and_random_seed_average(self):
        before=(self.charts/'a.npz').read_bytes()
        records=evaluate_saved_charts(self.charts,self.context)
        self.assertEqual((self.charts/'a.npz').read_bytes(),before)
        analysis=self.charts.parent/'analysis'
        self.assertTrue((analysis/'config.json').is_file())
        with (analysis/'summary.csv').open() as f: summary=list(csv.DictReader(f))
        self.assertEqual({r['scope'] for r in summary},{'overall','difficulty'})
        self.assertTrue(all(r['difficulty']=='stars=7' for r in summary if r['scope']=='difficulty'))
        # b is missing from generated charts but remains in the human pool.
        self.assertEqual(json.loads((analysis/'config.json').read_text())['provenance']['test_chart_ids'],['a','b'])
        random=evaluate_random_baseline(self.context,analysis/'random_baseline',['a'])
        seeds=json.loads((analysis/'random_baseline/seed_results.json').read_text())
        for metric in random[0]['raw']:
            values=[r['raw'][metric] for r in seeds if r['raw'][metric] is not None]
            if values: self.assertAlmostEqual(random[0]['raw'][metric],np.mean(values))
        models,meta=load_model_results({'lstm': self.root/'run','donderformer':None},self.context.config)
        self.assertEqual(set(models),{'lstm','random'})
        rows,tests=compare_models(models,self.root/'compare',self.context.config,meta)
        self.assertTrue((self.root/'compare/plots/hi_pspace.png').is_file())
        self.assertTrue(all(t['status']=='awaiting_donderformer' for t in tests))
        self.assertTrue(all(t['p_value'] is None for t in tests))
        self.assertEqual(records[0]['raw']['onset_f1'],1)
        # Execute the comparison notebook, including its optional-model path.
        from donderformer.paths import REPO_ROOT
        config = dict(self.context.config, MODEL_RUNS={'lstm': str(self.root/'run'), 'donderformer': None})
        notebook = json.loads((REPO_ROOT/'notebooks/07_compare_models.ipynb').read_text())
        namespace = {}
        with patch('donderformer.evaluation_config.evaluation_config', return_value=config), patch('donderformer.paths.outputs_dir', return_value=self.root/'outputs'):
            for cell in notebook['cells']:
                if cell['cell_type']=='code': exec(''.join(cell['source']), namespace)
        self.assertTrue((namespace['OUTPUT_DIR']/'metrics.csv').is_file())

    def test_model_loader_checks_transformer_provenance(self):
        import shutil
        evaluate_saved_charts(self.charts,self.context)
        analysis=self.charts.parent/'analysis'
        evaluate_random_baseline(self.context,analysis/'random_baseline',['a'])
        transformer=self.root/'transformer/eval/ar_test/analysis'
        shutil.copytree(analysis,transformer)
        models,_=load_model_results({'lstm':self.root/'run','donderformer':self.root/'transformer'},self.context.config)
        self.assertEqual(set(models),{'lstm','donderformer','random'})
        config=json.loads((transformer/'config.json').read_text())
        config['provenance']['train_label_sha256']['train']='changed'
        (transformer/'config.json').write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError,'different reference'):
            load_model_results({'lstm':self.root/'run','donderformer':self.root/'transformer'},self.context.config)

    def test_invalid_human_rejected_and_no_reference_leakage(self):
        with self.assertRaisesRegex(ValueError,'overlap'):
            build_reference_context({'a': self.context.test_humans['a']},{'a': self.context.test_humans['a']},self.context.metadata)
        np.savez(self.charts/'a.npz',human=np.zeros(128,dtype=int),generated=np.zeros(128,dtype=int))
        with self.assertRaisesRegex(ValueError,'disagrees'):
            evaluate_saved_charts(self.charts,self.context)

    def test_pair_by_chart_id_and_common_cohort(self):
        records=evaluate_saved_charts(self.charts,self.context)
        a=records[0]
        # Construct six known pairs with shuffled row order and one unmatched ID.
        import copy
        lstm=[]; transformer=[]; random=[]
        for i in range(6):
            left=copy.deepcopy(a); left['chart_id']=str(i); left['raw']={'onset_f1':0.2}
            right=copy.deepcopy(left); right['raw']={'onset_f1':0.2+(i+1)/10}
            lstm.append(left); transformer.insert(0,right); random.append(copy.deepcopy(left))
        extra=copy.deepcopy(a);extra['chart_id']='unmatched';lstm.append(extra)
        rows,tests=compare_models({'lstm':lstm,'donderformer':transformer,'random':random},self.root/'paired',self.context.config)
        test=next(t for t in tests if t['scope']=='overall' and t['metric']=='onset_f1')
        self.assertEqual(test['n_pairs'],6); self.assertEqual(test['p_value'],.03125)
        self.assertEqual(test['test_role'],'secondary')
        self.assertGreaterEqual(test['p_value_adjusted'],test['p_value'])
        self.assertEqual(json.loads((self.root/'paired/comparison.json').read_text())['excluded_chart_ids']['lstm'],['unmatched'])

    def test_labels_only_context_recovers_tja_level(self):
        paths=DataPaths(self.root/'labels'); extractor=FeatureExtractor()
        split=paths.split_dir('s');split.mkdir(parents=True)
        paths.raw.mkdir()
        (paths.raw/'song.tja').write_text('BPM:120\nCOURSE:Oni\nLEVEL:7\n#START\n1000,\n#END\n')
        with (split/'manifest.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=['chart_id','audio_id','split','course','level','tja'])
            writer.writeheader()
            writer.writerows([dict(chart_id=c,audio_id=c,split=part,course='oni',level='',tja='song.tja') for c,part in [('a','test'),('train','train')]])
        for c in ['a','train']:
            path=extractor.labels_path(paths,c,c);path.parent.mkdir(parents=True)
            np.save(path,np.eye(7,dtype=np.int32)[self.context.test_humans['a']])
        context=load_reference_context(paths,'s',extractor,self.context.config)
        self.assertEqual(context.metadata['a']['difficulty'],'stars=7')
        self.assertFalse(extractor.mel_path(paths,'a').exists())
        extractor.labels_path(paths,'train','train').unlink()
        with self.assertRaisesRegex(FileNotFoundError,'training AND test'):
            load_reference_context(paths,'s',extractor,self.context.config)


class SignedRankAndAdapterTests(unittest.TestCase):
    def test_exact_signed_rank_ties_zeros(self):
        self.assertEqual(wilcoxon_signed_rank(np.zeros(6),np.arange(1,7))['p_value'],.03125)
        self.assertEqual(wilcoxon_signed_rank([1,1],[1,1])['p_value'],1)
        self.assertIsNone(wilcoxon_signed_rank([],[])['p_value'])
        # Brute-force sign flips with tied mean ranks [1.5,1.5,3].
        ranks=np.array([1.5,1.5,3]); observed=1.5
        sums=[sum(ranks[np.array(signs,dtype=bool)]) for signs in itertools.product([0,1],repeat=3)]
        expected=min(1,2*sum(value<=observed for value in sums)/len(sums))
        self.assertEqual(wilcoxon_signed_rank([0,0,0,0],[1,-1,-2,0])['p_value'],expected)

    def test_token_stub_validation_and_exporter_keys(self):
        np.testing.assert_array_equal(tokens_to_frames([0,1,46,2,'END'],23),[1,0,2])
        for tokens in ([0,1], [0,1,0,2,'END'], [0,5,23,5,'END'], [-1,1,'END'], ['END',1,'END']):
            with self.assertRaises(ValueError): tokens_to_frames(tokens,23)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'a.npz'
            save_token_chart(path,[0,1,46,2,'END'],np.zeros(8,dtype=int),23)
            with np.load(path) as data:
                self.assertEqual(set(data.files),{'human','generated','clean'})
                self.assertEqual(len(data['generated']),8)
            with self.assertRaises(FileExistsError):
                save_token_chart(path,['END'],np.zeros(8,dtype=int),23)


if __name__=='__main__': unittest.main()
