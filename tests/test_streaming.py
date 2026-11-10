import tempfile
import unittest
from pathlib import Path
import numpy as np
import matplotlib as plt

from numcompute_stream import *
from numcompute_stream.io import read_csv_chunks
from numcompute_stream.visualize import plot_metric_over_time, compare_models, plot_predictions_vs_ground_truth


class TreeTests(unittest.TestCase):
    def setUp(self):
        self.X = np.arange(8.).reshape(-1, 1)
        self.y = np.array([0]*4+[1]*4)

    def test_gini(self):
        np.testing.assert_array_equal(DecisionTreeClassifier().fit(self.X, self.y).predict(self.X), self.y)

    def test_entropy(self):
        np.testing.assert_array_equal(DecisionTreeClassifier(criterion='entropy').fit(self.X, self.y).predict(self.X), self.y)

    def test_depth_zero_tie(self):
        self.assertTrue(np.all(DecisionTreeClassifier(max_depth=0).fit(self.X, self.y).predict(self.X) == 0))

    def test_min_samples(self):
        self.assertTrue(np.all(DecisionTreeClassifier(min_samples_split=9).fit(self.X, self.y).predict(self.X) == 0))

    def test_strings(self):
        y = np.where(self.y, 'yes', 'no')
        np.testing.assert_array_equal(DecisionTreeClassifier().fit(self.X, y).predict(self.X), y)

    def test_nan_split(self):
        X = [[np.nan], [1], [2]]
        np.testing.assert_array_equal(DecisionTreeClassifier().fit(X, [0,1,1]).predict(X), [0,1,1])

    def test_all_nan(self):
        self.assertEqual(DecisionTreeClassifier().fit([[np.nan],[np.nan]], [1,0]).predict([[np.nan]])[0], 0)

    def test_constant(self):
        self.assertEqual(DecisionTreeClassifier().fit([[1],[1],[1]], [1,0,1]).predict([[1]])[0], 1)

    def test_warmup(self):
        m = DecisionTreeClassifier(buffer_size=8).partial_fit(self.X[:4], self.y[:4])
        with self.assertRaises(ValueError): m.predict(self.X)
        self.assertEqual(m.buffer_count_, 4)

    def test_boundary(self):
        m = DecisionTreeClassifier(buffer_size=8).partial_fit(self.X[:4], self.y[:4]).partial_fit(self.X[4:], self.y[4:])
        self.assertEqual(m.buffer_count_, 0)
        self.assertEqual(m.n_retrains_, 1)
        np.testing.assert_array_equal(m.predict(self.X), self.y)

    def test_old_tree(self):
        m = DecisionTreeClassifier(buffer_size=8).fit(self.X, self.y)
        before = m.predict(self.X)
        m.partial_fit(self.X[:2], [1,1])
        np.testing.assert_array_equal(m.predict(self.X), before)

    def test_overshoot(self):
        m = DecisionTreeClassifier(buffer_size=3).partial_fit(self.X, self.y)
        self.assertEqual(m.n_retrains_, 1)
        self.assertEqual(m.buffer_count_, 0)

    def test_reset_fit(self):
        m = DecisionTreeClassifier().partial_fit(self.X, self.y).fit(self.X, self.y)
        self.assertEqual(m.buffer_count_, 0)

    def test_invalid_config(self):
        for kw in [dict(buffer_size=0), dict(max_depth=-1), dict(criterion='x'), dict(min_samples_split=1), dict(max_features=0)]:
            with self.assertRaises(ValueError): DecisionTreeClassifier(**kw)

    def test_shape(self):
        with self.assertRaises(ValueError): DecisionTreeClassifier().fit(self.X, [1])

    def test_infinity(self):
        with self.assertRaises(ValueError): DecisionTreeClassifier().fit([[np.inf]], [0])

    def test_missing_label(self):
        with self.assertRaises(ValueError): DecisionTreeClassifier().fit([[1]], [np.nan])

    def test_feature_mismatch(self):
        m = DecisionTreeClassifier().fit(self.X, self.y)
        with self.assertRaises(ValueError): m.predict([[1,2]])

    def test_max_features(self):
        for value in [1, 'sqrt', 'log2']:
            m = DecisionTreeClassifier(max_features=value).fit(self.X, self.y)
            np.testing.assert_array_equal(m.predict(self.X), self.y)


class TransformTests(unittest.TestCase):
    def test_scaler_matches_batch(self):
        X = np.array([[1,np.nan],[2,4],[3,8.]])
        m = StandardScaler().partial_fit(X[:1]).partial_fit(X[1:])
        np.testing.assert_allclose(m.mean_, np.nanmean(X, axis=0))
        np.testing.assert_allclose(m.var_, np.nanvar(X, axis=0))

    def test_large_offset(self):
        X = 1e10+np.arange(12.).reshape(6,2)
        m = StandardScaler().partial_fit(X[:3]).partial_fit(X[3:])
        np.testing.assert_allclose(m.var_, X.var(axis=0))

    def test_constant_scaling(self):
        np.testing.assert_array_equal(StandardScaler().fit_transform([[2],[2]]), [[0],[0]])

    def test_all_nan_then_valid(self):
        m = StandardScaler().partial_fit([[np.nan]]).partial_fit([[4]])
        self.assertEqual(m.mean_[0], 4)

    def test_minmax(self):
        m = MinMaxScaler().partial_fit([[2]]).partial_fit([[4]])
        np.testing.assert_array_equal(m.transform([[2],[4]]), [[0],[1]])

    def test_imputer_mean(self):
        m = Imputer().partial_fit([[2,np.nan]]).partial_fit([[4,8]])
        np.testing.assert_array_equal(m.transform([[np.nan,np.nan]]), [[3,8]])

    def test_imputer_fallback(self):
        self.assertEqual(Imputer(fill_value=-1).fit_transform([[np.nan]])[0,0], -1)

    def test_imputer_constant(self):
        self.assertEqual(Imputer(strategy='constant', fill_value=5).fit_transform([[np.nan]])[0,0], 5)

    def test_encoder_growth(self):
        m = OneHotEncoder().partial_fit([['b']]).partial_fit([['a']])
        np.testing.assert_array_equal(m.transform([['b'],['a'],['unknown']]), [[1,0],[0,1],[0,0]])

    def test_unfitted(self):
        for m in [StandardScaler(), MinMaxScaler(), Imputer(), OneHotEncoder()]:
            with self.assertRaises(ValueError): m.transform([[1]])

    def test_fit_resets(self):
        m = StandardScaler().fit([[100]]).fit([[2]])
        self.assertEqual(m.mean_[0], 2)


class MetricsStatsTests(unittest.TestCase):
    def test_accuracy_accumulates(self):
        m = ClassificationMetrics().update([0,1],[0,0]).update([1],[1])
        self.assertAlmostEqual(m.result()['accuracy'], 2/3)

    def test_confusion_and_macro(self):
        r = ClassificationMetrics().update(['a','b'], ['a','a']).result()
        np.testing.assert_array_equal(r['confusion_matrix'], [[1,0],[1,0]])
        self.assertEqual(r['precision'], .25)
        self.assertEqual(r['recall'], .5)

    def test_auc_ties(self):
        r = ClassificationMetrics().update([0,1],[0,1],[.5,.5]).result()
        self.assertEqual(r['auc'], .5)

    def test_auc_accumulates(self):
        m = ClassificationMetrics().update([0],[0],[.1]).update([1],[1],[.9])
        self.assertEqual(m.result()['auc'], 1)

    def test_undefined_auc(self):
        self.assertTrue(np.isnan(ClassificationMetrics().update([1],[1],[.9]).result()['auc']))

    def test_rolling(self):
        r = ClassificationMetrics(window_size=2).update([0,0,1],[1,0,1]).result()
        self.assertEqual(r['accuracy'], 1)
        self.assertEqual(r['n_samples'], 2)

    def test_reset(self):
        self.assertEqual(ClassificationMetrics().update([1],[1]).reset().result()['n_samples'], 0)

    def test_stats(self):
        m = StreamingStats().update_stats([[1,2]]).update([[3,4]])
        r = m.result(q=.5)
        np.testing.assert_array_equal(r['mean'], [2,3])
        np.testing.assert_array_equal(r['variance'], [1,1])
        np.testing.assert_array_equal(r['quantiles'], [2,3])
        self.assertEqual(r['histograms'][0][0].sum(), 2)

    def test_stats_window(self):
        r = StreamingStats(2).update([[100],[2],[4]]).result()
        self.assertEqual(r['mean'][0], 3)

    def test_stats_nan(self):
        r = StreamingStats().update([[np.nan]]).result()
        self.assertTrue(np.isnan(r['mean'][0]))
        self.assertEqual(r['histograms'][0][0].sum(), 0)


class IntegrationTests(unittest.TestCase):
    def test_pipeline_snapshot(self):
        pipe = Pipeline([('scale', StandardScaler()), ('tree', DecisionTreeClassifier(buffer_size=4))])
        X = np.array([[0],[1],[2],[3]])
        pipe.partial_fit(X, [0,0,1,1])
        before = pipe.predict(X)
        pipe.partial_fit([[1000]], [1])
        np.testing.assert_array_equal(pipe.predict(X), before)
        pipe.partial_fit([[1001],[1002],[1003]], [0,0,1])
        self.assertEqual(pipe.steps[-1][1].n_retrains_, 2)

    def test_category_growth_snapshot(self):
        pipe = Pipeline([('encode', OneHotEncoder()), ('tree', DecisionTreeClassifier(buffer_size=2))])
        pipe.partial_fit([['a'],['b']], [0,1])
        pipe.partial_fit([['c']], [1])
        np.testing.assert_array_equal(pipe.predict([['a'],['b']]), [0,1])
        pipe.partial_fit([['d']], [0])
        np.testing.assert_array_equal(pipe.predict([['c'],['d']]), [1,0])

    def test_trainer(self):
        trainer = StreamTrainer(DecisionTreeClassifier(buffer_size=2))
        trainer.fit_chunk([[0],[1]], [0,1])
        self.assertEqual(trainer.score_chunk([[0],[1]], [0,1])['accuracy'], 1)
        self.assertGreater(trainer.logs[0]['memory_bytes'], 0)

    def test_csv_chunks(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'x.csv'
            p.write_text('x,y\n0,0\n1,1\n2,1\n')
            chunks = list(read_csv_chunks(p, chunk_size=2, skip_header=1, dtype=float))
            self.assertEqual([a.shape for a in chunks], [(2,2),(1,2)])

    def test_plots(self):
        plt.use('Agg')
        with tempfile.TemporaryDirectory() as d:
            for i, (fn, args) in enumerate([(plot_metric_over_time, ([.5,1],)), (compare_models, ([1],[.5])), (plot_predictions_vs_ground_truth, ([0,1],[0,0]))]):
                path = Path(d)/f'{i}.png'
                fig, ax = fn(*args, save_path=path)
                self.assertTrue(path.exists())
                plt.close(fig)


if __name__ == '__main__':
    unittest.main()
