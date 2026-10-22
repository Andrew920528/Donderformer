"""HI precision, long-window ratios, sparse Oni groups, and primary inference."""
import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from donderformer.chart_evaluation import (
    build_reference_context, score_chart, evaluate_saved_charts, evaluate_random_baseline,
    pattern_set,
)
from donderformer.difficulty_groups import merge_star_groups, save_group_report
from donderformer.evaluation_comparison import compare_models, hi_retention, holm_adjust
from donderformer.evaluation_config import evaluation_config


def context_for(human, config=None):
    return build_reference_context({'human':human}, {'train':human},
            {c:dict(course='oni',level=7) for c in ('human','train')}, config)


class PatternPrecisionTests(unittest.TestCase):
    def test_self_and_random_at_longest_window(self):
        human=np.tile(np.array([1,0,2,0,0,1,0,0],dtype=int),64)
        context=context_for(human)
        actual=score_chart(human,human,'human',context)
        for variant in ('binary','type'):
            for window in context.config['PATTERN_WINDOWS_FRAMES']:
                self.assertEqual(actual[f'pspace_{variant}_w{window}_hi_precision'],1)
                self.assertEqual(actual[f'pspace_{variant}_w{window}_overall_ratio'],1)
        with tempfile.TemporaryDirectory() as temp:
            random=evaluate_random_baseline(context,Path(temp))
            for variant in ('binary','type'):
                self.assertLess(random[0]['raw'][f'pspace_{variant}_w64_hi_precision'],1)

    def test_empty_pattern_never_contributes_to_precision(self):
        human=np.zeros(16,dtype=int); human[0]=1
        generated=np.zeros_like(human); generated[0]=2
        context=context_for(human, {'PATTERN_WINDOWS_FRAMES':[2,4]})
        result=score_chart(human,generated,'human',context)
        # The only match is the all-zero pattern: coverage=1/2, precision=0.
        self.assertEqual(result['pspace_type_w4_hi'],50)
        self.assertEqual(result['pspace_type_w4_hi_coverage'],50)
        self.assertEqual(result['pspace_type_w4_hi_precision'],0)
        self.assertEqual(result['pspace_type_w4_hi_primary'],0)
        empty=score_chart(human,np.zeros_like(human),'human',context)
        self.assertIsNone(empty['pspace_type_w4_hi_precision'])
        self.assertIsNone(empty['pspace_binary_w4_hi_precision'])
        self.assertEqual(empty['pspace_type_w4_hi_coverage'],50)
        coverage_config={**context.config,'HI_PRIMARY':'coverage'}
        coverage=context_for(human,coverage_config)
        self.assertEqual(score_chart(human,generated,'human',coverage)['pspace_type_w4_hi_primary'],.5)

    def test_repeated_patterns_lower_overall_ratio(self):
        human=np.random.default_rng(3).integers(0,3,512)
        repeated=np.tile([1,0,0,0,0,0,0,0],64)
        context=context_for(human)
        result=score_chart(human,repeated,'human',context)
        for variant in ('binary','type'):
            for window in context.config['PATTERN_WINDOWS_FRAMES']:
                self.assertLess(result[f'pspace_{variant}_w{window}_overall_ratio'],1)
                self.assertEqual(result[f'pspace_{variant}_w{window}_overall_ratio'],result[f'pspace_{variant}_w{window}_ratio'])

    def test_summary_headlines_and_random_density_exclusion(self):
        human=np.tile(np.array([1,0,2,0,0,1,0,0],dtype=int),16)
        context=context_for(human)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); charts=root/'charts';charts.mkdir()
            np.savez(charts/'human.npz',human=human,generated=human)
            records=evaluate_saved_charts(charts,context)
            with (root/'analysis/summary.csv').open() as stream: summary=list(csv.DictReader(stream))
            names={r['metric'] for r in summary}
            self.assertIn('pspace_type_w64_overall_ratio',names)
            self.assertNotIn('pspace_type_w64_overall',names)
            self.assertIn('pspace_binary_w8_overall',names)
            self.assertTrue(all(r['headline']=='True' for r in summary if r['metric'].endswith('_hi_precision')))
            # Existing raw JSON keys keep the exact old normalization/coverage.
            self.assertIn('pspace_type_w64_overall',records[0]['raw'])
            self.assertIn('pspace_type_w64_hi',records[0]['raw'])
            random=evaluate_random_baseline(context,root/'random')
            with (root/'random/summary.csv').open() as stream: random_summary=list(csv.DictReader(stream))
            self.assertFalse(any(r['metric'] in ('note_density','note_density_in_band') for r in random_summary))
            self.assertTrue(any(r['metric']=='don_ka_ratio' for r in random_summary))
            self.assertIn('note_density',random[0]['raw'])  # preserve existing per-chart keys
            self.assertTrue((root/'random/summary_notes.json').is_file())
            self.assertTrue((root/'analysis/difficulty_groups.json').is_file())


class GroupMergingTests(unittest.TestCase):
    def test_sparse_group_merges_and_reports_counts(self):
        metadata={};train=[];test=[]
        for level,count in [(1,2),(2,40),(3,30)]:
            for i in range(count):
                chart=f'train-{level}-{i}';train.append(chart);metadata[chart]=dict(course='oni',level=level)
            chart=f'test-{level}';test.append(chart);metadata[chart]=dict(course='oni',level=level)
        grouped,report=merge_star_groups(metadata,train,test,30)
        self.assertEqual(report['mapping'],{'1':'stars=1-2','2':'stars=1-2','3':'stars=3'})
        self.assertEqual(report['groups'][0]['train_charts'],42)
        self.assertEqual(report['groups'][0]['test_charts'],2)
        self.assertEqual(grouped['test-1']['difficulty'],'stars=1-2')
        self.assertNotIn('difficulty',metadata['test-1'])  # caller metadata preserved
        with tempfile.TemporaryDirectory() as temp:
            save_group_report(temp,report)
            self.assertEqual(json.loads((Path(temp)/'difficulty_groups.json').read_text())['mapping'],report['mapping'])

    def test_ties_choose_lower_and_all_small_groups_merge(self):
        metadata={c:dict(course='oni',level=l) for c,l in [('a',6),('b',7),('c',8)]}
        _,report=merge_star_groups(metadata,['a','b','c'],[],2)
        self.assertEqual(report['merge_history'][0]['merged'],[6,7])
        self.assertEqual(report['mapping'],{'6':'stars=6-8','7':'stars=6-8','8':'stars=6-8'})
        _,tiny=merge_star_groups(metadata,['a','b','c'],[],30)
        self.assertFalse(tiny['groups'][0]['meets_minimum'])

    def test_merged_pools_bands_and_type_distribution(self):
        a=np.array([1,0,1,0,0,0,0,0]*2,dtype=int)
        b=np.array([0,2,0,2,0,0,0,0]*2,dtype=int)
        train={'ta':a,'tb':b};test={'a':a,'b':b}
        metadata={c:dict(course='oni',level=l) for c,l in [('a',6),('ta',6),('b',7),('tb',7)]}
        context=build_reference_context(test,train,metadata,{'MIN_CHARTS_PER_BAND_GROUP':2})
        self.assertEqual(set(context.bands),{'stars=6-7'})
        self.assertEqual(context.bands['stars=6-7']['note_density']['n'],2)
        self.assertEqual(context.pools[('stars=6-7',8,'type')],pattern_set(a,8,'type')|pattern_set(b,8,'type'))
        np.testing.assert_array_equal(context.type_probabilities['stars=6-7'][:2],[.5,.5])


class RetentionAndHolmTests(unittest.TestCase):
    def test_retention_and_holm_small_examples(self):
        config=evaluation_config({'PATTERN_WINDOWS_FRAMES':[32,8,64]})
        metrics={'pspace_binary_w8_hi_precision':.8,'pspace_binary_w64_hi_precision':.4}
        self.assertEqual(hi_retention(metrics,'binary',config),.5)
        for missing in (0,None):
            self.assertIsNone(hi_retention({**metrics,'pspace_binary_w8_hi_precision':missing},'binary',config))
        self.assertIsNone(hi_retention({},'binary',config))
        np.testing.assert_allclose([p for p in holm_adjust([.01,.04,.03,None]) if p is not None],[.03,.06,.06])
        self.assertIsNone(holm_adjust([.01,.04,.03,None])[-1])
        self.assertEqual(holm_adjust([None,None]),[None,None])
        self.assertAlmostEqual(hi_retention({'pspace_binary_w8_hi_precision':.5,'pspace_binary_w64_hi_precision':.3},'binary',config),.6)

    def test_one_primary_and_global_secondary_family(self):
        config=evaluation_config({'PATTERN_WINDOWS_FRAMES':[8,64]})
        models={'lstm':[],'donderformer':[],'random':[]}
        for i in range(6):
            base={'onset_f1':.2,'note_density':4.,'note_density_in_band':True,
                  'pspace_binary_w8_hi_precision':.8,'pspace_binary_w64_hi_precision':.4,
                  'pspace_type_w8_hi_precision':.5,'pspace_type_w64_hi_precision':.2}
            left=dict(chart_id=str(i),difficulty='stars=7' if i<3 else 'stars=8',raw=base)
            right=copy.deepcopy(left);right['raw']['pspace_binary_w64_hi_precision']=.6
            right['raw']['onset_f1']=.5
            right['raw']['pspace_type_w64_hi_precision']=.3
            models['lstm'].append(left);models['donderformer'].insert(0,right)
            if i: models['random'].append(copy.deepcopy(left))
        with tempfile.TemporaryDirectory() as temp:
            rows,tests=compare_models(models,temp,config)
            primary=[t for t in tests if t['test_role']=='primary']
            self.assertEqual(len(primary),1)
            self.assertEqual(primary[0]['metric'],'hi_retention_binary')
            self.assertEqual(primary[0]['n_pairs'],6)  # independent of random's missing chart
            self.assertEqual(primary[0]['p_value'],.03125)
            self.assertEqual(primary[0]['p_value_adjusted'],.03125)
            self.assertTrue(primary[0]['significant'])
            secondary=[t for t in tests if t['test_role']=='secondary']
            self.assertEqual([t['p_value_adjusted'] for t in secondary],holm_adjust([t['p_value_raw'] for t in secondary]))
            self.assertTrue(all(r['metric'] not in ('note_density','note_density_in_band') for r in rows))
            self.assertTrue((Path(temp)/'primary_test.json').is_file())
            retention=json.loads((Path(temp)/'retention.json').read_text())
            self.assertEqual(next(r for r in retention if r['model']=='lstm')['hi_retention_binary'],.5)
            self.assertTrue((Path(temp)/'plots/hi_pspace.png').is_file())


if __name__=='__main__': unittest.main()
