"""Self-tests for embryo_vc_bench.

Run:  python -m unittest -v test_embryo_vc_bench.py
   or: python test_embryo_vc_bench.py

All data here are SYNTHETIC. These tests verify the algebra of the framework,
not any biological result. Nothing in this file may be reported as a finding.
"""

from __future__ import annotations

import unittest

import numpy as np

import embryo_vc_bench as b


class TestDeltaSet(unittest.TestCase):
    def test_shape_and_fingerprint(self):
        ds = b.DeltaSet(np.zeros((3, 5)), ["a", "b", "c"], [f"g{i}" for i in range(5)], atlas="A")
        self.assertEqual(ds.delta.shape, (3, 5))
        self.assertTrue(ds.fingerprint.startswith("A:embryo:3x5:"))

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            b.DeltaSet(np.zeros(5), ["a"], ["g"])
        with self.assertRaises(ValueError):
            b.DeltaSet(np.array([[np.nan]]), ["a"], ["g"])
        with self.assertRaises(ValueError):
            b.DeltaSet(np.zeros((2, 2)), ["only_one"], ["g1", "g2"])


class TestBaselines(unittest.TestCase):
    def test_no_change_is_zero(self):
        t = np.array([[1.0, -2.0], [3.0, 4.0]])
        self.assertTrue(np.array_equal(b.no_change_baseline(t), np.zeros_like(t)))

    def test_train_mean_on_constant_effects_beats_no_change(self):
        # Every training perturbation has the same effect -> train_mean is exact.
        train = np.tile(np.array([2.0, -1.0, 0.5]), (6, 1))
        true = train[:3].copy()
        preds = b.all_baselines(train, true)
        self.assertLess(b.mae(preds["train_mean"], true), b.mae(preds["no_change"], true))
        self.assertAlmostEqual(b.mae(preds["train_mean"], true), 0.0)

    def test_no_change_wins_when_there_is_no_effect(self):
        train = np.zeros((6, 4))
        true = np.zeros((3, 4))
        preds = b.all_baselines(train, true)
        self.assertAlmostEqual(b.mae(preds["no_change"], true), 0.0)

    def test_additive_is_exact_on_additive_data(self):
        rng = np.random.default_rng(0)
        single = {"A": rng.normal(size=7), "B": rng.normal(size=7)}
        true = (single["A"] + single["B"])[None, :]
        pred = b.additive_baseline([("A", "B")], single, 7)
        self.assertAlmostEqual(b.mae(pred, true), 0.0)

    def test_additive_refuses_silent_imputation(self):
        with self.assertRaises(KeyError):
            b.additive_baseline([("A", "missing")], {"A": np.zeros(3)}, 3)


class TestMetrics(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.true = rng.normal(size=(12, 60))

    def test_mae_zero_for_perfect_prediction(self):
        self.assertAlmostEqual(b.mae(self.true, self.true), 0.0)

    def test_mae_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            b.mae(self.true, self.true[:, :5])

    def test_des_is_one_for_perfect_prediction_and_zero_for_no_change(self):
        self.assertAlmostEqual(b.des_score(self.true, self.true), 1.0)
        self.assertAlmostEqual(b.des_score(b.no_change_baseline(self.true), self.true), 0.0)

    def test_pds_is_one_for_perfect_prediction(self):
        self.assertAlmostEqual(b.pds_score(self.true, self.true), 1.0)

    def test_pds_constant_prediction_is_chance(self):
        const = np.ones_like(self.true)
        self.assertAlmostEqual(b.pds_score(const, self.true), 0.5)

    def test_score_all_has_all_three_metrics(self):
        self.assertEqual(set(b.score_all(self.true, self.true)), {"MAE", "DES", "PDS"})


class TestInference(unittest.TestCase):
    def test_permutation_pvalue_range(self):
        rng = np.random.default_rng(2)
        t = rng.normal(size=(10, 25))
        p = b.permutation_pvalue(b.des_score, t, t, B=99, seed=0)
        self.assertGreater(p, 0.0)
        self.assertLessEqual(p, 1.0)
        self.assertAlmostEqual(p, 0.01, places=2)  # perfect prediction is extreme

    def test_bootstrap_ci_ordered_and_covers_mean(self):
        x = [1.0, 1.1, 0.9, 1.05, 0.95]
        lo, hi = b.bootstrap_ci(x, B=500, seed=0)
        self.assertLessEqual(lo, np.mean(x))
        self.assertGreaterEqual(hi, np.mean(x))
        self.assertLess(lo, hi)

    def test_bh_monotone_and_bounded(self):
        adj = b.bh_adjust([0.001, 0.02, 0.4, 0.9])
        self.assertTrue(np.all(np.diff(adj[::-1]) <= 1e-12))
        self.assertTrue(np.all((adj >= 0) & (adj <= 1)))
        self.assertGreaterEqual(adj[3], adj[0])

    def test_bh_handles_nan_as_one(self):
        adj = b.bh_adjust([0.01, float("nan")])
        self.assertEqual(adj[1], 1.0)


class TestProtocol(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.atlases = {
            name: b.DeltaSet(rng.normal(size=(4, 12)), [f"{name}{i}" for i in range(4)],
                             [f"g{j}" for j in range(12)], atlas=name)
            for name in ("A", "B", "C", "D", "E")
        }

    def test_folds_count_and_leakage(self):
        folds = b.leave_one_atlas_out(self.atlases, seeds=(0, 1))
        self.assertEqual(len(folds), 10)
        self.assertTrue(all(b.check_fold_leakage(f) for f in folds))
        self.assertEqual({f.test_atlas for f in folds}, set(self.atlases))

    def test_too_few_atlases_raises(self):
        with self.assertRaises(ValueError):
            b.leave_one_atlas_out({"A": self.atlases["A"]})

    def test_surrogate_shape_and_sign(self):
        rng = np.random.default_rng(4)
        expr = rng.normal(size=(50, 20))
        eff = b.make_surrogate_effects(expr, [0, 3, 7])
        self.assertEqual(eff.shape, (3, 20))
        self.assertTrue(np.all(eff <= 0))

    def test_evaluate_one_fold_requires_mandatory_baselines(self):
        folds = b.leave_one_atlas_out(self.atlases, seeds=(0,))
        stacked = np.vstack([self.atlases[n].delta for n in sorted(self.atlases)])
        with self.assertRaises(KeyError):
            b.evaluate_one_fold(folds[0], stacked, {"some_model": np.zeros_like(stacked)}, B=20)

    def test_evaluate_one_fold_reports_models_with_baselines(self):
        folds = b.leave_one_atlas_out(self.atlases, seeds=(0,))
        stacked = np.vstack([self.atlases[n].delta for n in sorted(self.atlases)])
        preds = {
            "no_change": np.zeros_like(stacked),
            "train_mean": np.tile(stacked.mean(axis=0), (stacked.shape[0], 1)),
            "toy_model": stacked.copy(),  # deliberately perfect, for algebra check only
        }
        df = b.evaluate_one_fold(folds[0], stacked, preds, B=20)
        self.assertEqual(set(df["method"]), set(preds))
        self.assertTrue({"MAE", "DES", "PDS", "p_adj_DES", "p_adj_PDS"}.issubset(df.columns))
        toy = df.loc[df["method"] == "toy_model"].iloc[0]
        self.assertAlmostEqual(toy["MAE"], 0.0)
        self.assertAlmostEqual(toy["DES"], 1.0)


class TestOptionalDependencies(unittest.TestCase):
    def test_placeholder_raises_informative_error(self):
        with self.assertRaises(NotImplementedError):
            b._require_optional("scvi-tools")


if __name__ == "__main__":
    unittest.main(verbosity=2)
