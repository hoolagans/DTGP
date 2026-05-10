import unittest
from contextlib import redirect_stderr
from io import StringIO

from dtgp import DTGPClassifier


class TestDTGPClassifier(unittest.TestCase):
    def test_fit_predict_score(self):
        X = [
            [4.0, 1.0],
            [5.0, 2.0],
            [3.0, 1.0],
            [1.0, 3.0],
            [2.0, 5.0],
            [1.0, 4.0],
            [6.0, 2.0],
            [2.0, 6.0],
        ]
        y = [1 if row[0] > row[1] else 0 for row in X]

        clf = DTGPClassifier(
            num_models=40,
            generations=40,
            crossover_rate=0.5,
            mutation_rate=0.3,
            elitist_rate=0.2,
            max_depth=6,
            random_state=42,
        )
        clf.fit(X, y)

        pred = clf.predict(X)
        self.assertEqual(len(pred), len(y))
        self.assertGreaterEqual(clf.score(X, y), 0.75)

        proba = clf.predict_proba(X)
        self.assertEqual(len(proba), len(X))
        self.assertEqual(len(proba[0]), 2)

    def test_reproducible_with_random_state(self):
        X = [[4.0, 1.0], [1.0, 4.0], [5.0, 2.0], [2.0, 5.0], [3.0, 1.0], [1.0, 3.0]]
        y = [1 if row[0] > row[1] else 0 for row in X]

        clf1 = DTGPClassifier(random_state=7, num_models=25, generations=25)
        clf2 = DTGPClassifier(random_state=7, num_models=25, generations=25)

        clf1.fit(X, y)
        clf2.fit(X, y)

        self.assertListEqual(clf1.predict(X), clf2.predict(X))
        proba1 = clf1.predict_proba(X)
        proba2 = clf2.predict_proba(X)
        self.assertEqual(len(proba1), len(proba2))
        for row1, row2 in zip(proba1, proba2):
            self.assertEqual(len(row1), len(row2))
            for value1, value2 in zip(row1, row2):
                self.assertAlmostEqual(value1, value2, places=12)

    def test_get_set_params(self):
        clf = DTGPClassifier(num_models=10, generations=20, random_state=3)
        params = clf.get_params()
        self.assertEqual(params["num_models"], 10)
        self.assertEqual(params["generations"], 20)
        self.assertFalse(params["show_training_curve"])

        returned = clf.set_params(num_models=15, generations=30, show_training_curve=True)
        self.assertIs(returned, clf)
        self.assertEqual(clf.num_models, 15)
        self.assertEqual(clf.generations, 30)
        self.assertTrue(clf.show_training_curve)

    def test_view_model_interpretable_format(self):
        X = [[4.0, 1.0], [1.0, 4.0], [5.0, 2.0], [2.0, 5.0], [3.0, 1.0], [1.0, 3.0]]
        y = [1 if row[0] > row[1] else 0 for row in X]

        clf = DTGPClassifier(random_state=11, num_models=20, generations=20)
        clf.fit(X, y)

        one = clf.view_model()
        self.assertIsInstance(one, str)
        self.assertTrue(len(one) > 0)

        many = clf.view_model(3)
        self.assertIsInstance(many, list)
        self.assertEqual(len(many), 3)
        self.assertTrue(all(isinstance(expr, str) and len(expr) > 0 for expr in many))

    def test_view_model_tree_format(self):
        X = [[4.0, 1.0], [1.0, 4.0], [5.0, 2.0], [2.0, 5.0], [3.0, 1.0], [1.0, 3.0]]
        y = [1 if row[0] > row[1] else 0 for row in X]

        clf = DTGPClassifier(random_state=13, num_models=20, generations=20)
        clf.fit(X, y)

        tree_text = clf.view_model_tree()
        self.assertIsInstance(tree_text, str)
        self.assertIn("[Model 1]", tree_text)
        self.assertIn("└─", tree_text)

        trees = clf.view_model_tree(2)
        self.assertIsInstance(trees, list)
        self.assertEqual(len(trees), 2)
        self.assertTrue(all("[Model " in t for t in trees))

    def test_training_curve_history_and_live_output(self):
        X = [[4.0, 1.0], [1.0, 4.0], [5.0, 2.0], [2.0, 5.0], [3.0, 1.0], [1.0, 3.0]]
        y = [1 if row[0] > row[1] else 0 for row in X]

        clf = DTGPClassifier(random_state=17, num_models=10, generations=5, show_training_curve=True)
        stderr = StringIO()
        with redirect_stderr(stderr):
            clf.fit(X, y)

        self.assertTrue(hasattr(clf, "training_curve_"))
        self.assertEqual(len(clf.training_curve_), 6)
        live_text = stderr.getvalue()
        self.assertIn("Generation 0/5", live_text)
        self.assertIn("Generation 5/5", live_text)

    def test_multiclass_one_vs_rest_training(self):
        X = [
            [9.0, 1.0, 1.0],
            [8.0, 2.0, 1.0],
            [1.0, 9.0, 1.0],
            [2.0, 8.0, 1.0],
            [1.0, 1.0, 9.0],
            [1.0, 2.0, 8.0],
            [7.0, 2.0, 1.0],
            [2.0, 7.0, 1.0],
            [1.0, 2.0, 7.0],
        ]
        y = [0, 0, 1, 1, 2, 2, 0, 1, 2]

        clf = DTGPClassifier(random_state=19, num_models=16, generations=16)
        clf.fit(X, y)

        self.assertEqual(clf.multiclass_strategy_, "one_vs_rest")
        self.assertEqual(set(clf.classifiers_.keys()), set(clf.classes_))
        self.assertGreaterEqual(clf.parallel_workers_, 1)

        pred = clf.predict(X)
        self.assertEqual(len(pred), len(y))
        self.assertGreaterEqual(clf.score(X, y), 0.55)

    def test_multiclass_predict_proba_shape(self):
        X = [
            [10.0, 0.5, 0.5],
            [0.5, 10.0, 0.5],
            [0.5, 0.5, 10.0],
            [8.0, 1.5, 1.0],
            [1.0, 8.0, 1.5],
            [1.5, 1.0, 8.0],
        ]
        y = [0, 1, 2, 0, 1, 2]

        clf = DTGPClassifier(random_state=23, num_models=12, generations=12)
        clf.fit(X, y)
        proba = clf.predict_proba(X)

        self.assertEqual(len(proba), len(X))
        self.assertTrue(all(len(row) == 3 for row in proba))
        self.assertTrue(all(abs(sum(row) - 1.0) < 1e-9 for row in proba))


if __name__ == "__main__":
    unittest.main()
