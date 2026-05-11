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
        self.assertEqual(params["selection_method"], "pareto_tournament")
        self.assertEqual(params["fitness_method"], "pearson_r2")
        self.assertFalse(params["show_training_curve"])

        returned = clf.set_params(
            num_models=15,
            generations=30,
            selection_method="pareto_tournament",
            fitness_method="pearson_r2",
            show_training_curve=True,
        )
        self.assertIs(returned, clf)
        self.assertEqual(clf.num_models, 15)
        self.assertEqual(clf.generations, 30)
        self.assertEqual(clf.selection_method, "pareto_tournament")
        self.assertEqual(clf.fitness_method, "pearson_r2")
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

    def test_multiclass_view_model_includes_each_class(self):
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

        clf = DTGPClassifier(random_state=27, num_models=12, generations=12)
        clf.fit(X, y)

        one = clf.view_model()
        self.assertIsInstance(one, str)
        for class_label in clf.classes_:
            self.assertIn(f"[Class {class_label} | Model 1]", one)

        many = clf.view_model(2)
        self.assertIsInstance(many, list)
        self.assertEqual(len(many), len(clf.classes_) * 2)
        for class_label in clf.classes_:
            self.assertTrue(any(f"[Class {class_label} | Model 1]" in expr for expr in many))
            self.assertTrue(any(f"[Class {class_label} | Model 2]" in expr for expr in many))
            self.assertEqual(sum(1 for expr in many if f"[Class {class_label} | Model " in expr), 2)

    def test_multiclass_view_model_tree_includes_each_class(self):
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

        clf = DTGPClassifier(random_state=29, num_models=12, generations=12)
        clf.fit(X, y)

        one = clf.view_model_tree()
        self.assertIsInstance(one, str)
        for class_label in clf.classes_:
            self.assertIn(f"[Class {class_label} | Model 1]", one)

        many = clf.view_model_tree(2)
        self.assertIsInstance(many, list)
        self.assertEqual(len(many), len(clf.classes_) * 2)
        for class_label in clf.classes_:
            self.assertTrue(any(f"[Class {class_label} | Model 1]" in tree for tree in many))
            self.assertTrue(any(f"[Class {class_label} | Model 2]" in tree for tree in many))
            self.assertEqual(sum(1 for tree in many if f"[Class {class_label} | Model " in tree), 2)

    def test_view_model_tree_multi_model_rendering(self):
        X = [[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0]]
        y = [1, 0, 1, 0]
        clf = DTGPClassifier(random_state=41, num_models=8, generations=4)
        clf.fit(X, y)

        trees = clf.view_model_tree(2)
        self.assertIsInstance(trees, list)
        self.assertGreaterEqual(len(trees), 2)
        rendered = str(trees)
        self.assertIn("\n", rendered)
        self.assertIn("\n\n", rendered)
        self.assertNotIn("\\n", rendered)

    def test_view_model_multi_model_rendering(self):
        X = [[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0]]
        y = [1, 0, 1, 0]
        clf = DTGPClassifier(random_state=43, num_models=8, generations=4)
        clf.fit(X, y)

        models = clf.view_model(2)
        self.assertIsInstance(models, list)
        self.assertGreaterEqual(len(models), 2)
        rendered = str(models)
        self.assertIn("\n", rendered)
        self.assertIn("\n\n", rendered)
        self.assertNotIn("\\n", rendered)

    def test_seeded_tree_with_nested_math_operations(self):
        X = [
            [1.0, 1.0],
            [2.0, 1.0],
            [1.0, 3.0],
            [4.0, 2.0],
            [0.0, 5.0],
            [3.0, 2.0],
        ]
        y = [1 if (row[0] + 1.0) > abs(row[1] - 2.0) else 0 for row in X]
        seeded_tree = (
            "inter",
            "gt",
            ("math2", "add", ("var", 0), ("const", 1.0)),
            ("math1", "abs", ("math2", "sub", ("var", 1), ("const", 2.0))),
        )
        clf = DTGPClassifier(
            num_models=1,
            generations=0,
            crossover_rate=0.0,
            mutation_rate=0.0,
            elitist_rate=1.0,
            random_state=5,
            initial_population=[seeded_tree],
        )
        clf.fit(X, y)

        pred = clf.predict(X)
        self.assertEqual(len(pred), len(y))
        self.assertGreaterEqual(clf.score(X, y), 0.95)

        model_expr = clf.view_model()
        self.assertIn("x[0]", model_expr)
        self.assertIn("x[1]", model_expr)
        self.assertIn("abs(", model_expr)

    def test_invalid_selection_method_raises(self):
        X = [[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0]]
        y = [1, 0, 1, 0]
        clf = DTGPClassifier(selection_method="not_supported", random_state=31, num_models=6, generations=2)
        with self.assertRaises(ValueError):
            clf.fit(X, y)

    def test_invalid_fitness_method_raises(self):
        X = [[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0]]
        y = [1, 0, 1, 0]
        clf = DTGPClassifier(fitness_method="not_supported", random_state=33, num_models=6, generations=2)
        with self.assertRaises(ValueError):
            clf.fit(X, y)

    def test_pareto_tournament_returns_non_dominated_front(self):
        X = [[3.0, 1.0], [2.0, 2.5], [1.0, 2.0], [4.0, 0.5], [0.5, 3.0], [3.5, 2.0]]
        y_bool = [row[0] > row[1] for row in X]

        m1 = ("inter", "gt", ("var", 0), ("var", 1))
        m2 = ("inter", "gt", ("var", 0), ("const", 2.0))
        m3 = ("node", "and", m1, m2)
        models = [m1, m2, m3]
        clf = DTGPClassifier(
            tournament_size=3,
            selection_method="pareto_tournament",
            random_state=37,
            num_models=3,
            generations=0,
            initial_population=models,
        )
        clf.fit(X, [1 if v else 0 for v in y_bool])
        front = clf._pareto_tournament_select(models, X, y_bool)

        metrics = {m: (clf._fitness(m, X, y_bool), clf._model_complexity(m)) for m in models}

        def dominates(left, right):
            left_fitness, left_complexity = metrics[left]
            right_fitness, right_complexity = metrics[right]
            no_worse = left_fitness >= right_fitness and left_complexity <= right_complexity
            strictly_better = left_fitness > right_fitness or left_complexity < right_complexity
            return no_worse and strictly_better

        expected_front = [m for m in models if not any(dominates(other, m) for other in models if other is not m)]
        self.assertEqual(set(front), set(expected_front))

    def test_fit_with_pareto_selection_method(self):
        X = [[4.0, 1.0], [1.0, 4.0], [5.0, 2.0], [2.0, 5.0], [3.0, 1.0], [1.0, 3.0]]
        y = [1 if row[0] > row[1] else 0 for row in X]

        clf = DTGPClassifier(
            random_state=41,
            num_models=18,
            generations=18,
            selection_method="pareto_tournament",
        )
        clf.fit(X, y)
        pred = clf.predict(X)
        self.assertEqual(len(pred), len(y))

    def test_fit_with_pearson_r2_fitness_method(self):
        X = [[4.0, 1.0], [1.0, 4.0], [5.0, 2.0], [2.0, 5.0], [3.0, 1.0], [1.0, 3.0]]
        y = [1 if row[0] > row[1] else 0 for row in X]

        clf = DTGPClassifier(
            random_state=43,
            num_models=18,
            generations=18,
            fitness_method="pearson_r2",
        )
        clf.fit(X, y)
        pred = clf.predict(X)
        self.assertEqual(len(pred), len(y))
        self.assertGreaterEqual(clf.best_fitness_, 0.0)
        self.assertLessEqual(clf.best_fitness_, 1.0)

    def test_pareto_elite_layers_preserves_full_first_front(self):
        X = [[3.0, 1.0], [2.0, 2.5], [1.0, 2.0], [4.0, 0.5], [0.5, 3.0], [3.5, 2.0]]
        y_bool = [row[0] > row[1] for row in X]

        m1 = ("inter", "gt", ("var", 0), ("var", 1))
        m2 = ("inter", "gt", ("var", 0), ("const", 2.0))
        m3 = ("node", "and", m1, m2)
        models = [m1, m2, m3]
        clf = DTGPClassifier(
            tournament_size=3,
            selection_method="pareto_tournament",
            random_state=47,
            num_models=3,
            generations=0,
            initial_population=models,
        )
        clf.fit(X, [1 if v else 0 for v in y_bool])

        metrics = {m: (clf._fitness(m, X, y_bool), clf._model_complexity(m)) for m in models}

        def dominates(left, right):
            lf, lc = metrics[left]
            rf, rc = metrics[right]
            return (lf >= rf and lc <= rc) and (lf > rf or lc < rc)

        expected_front = [m for m in models if not any(dominates(other, m) for other in models if other is not m)]

        # With a budget equal to the full population the result must contain
        # at least the entire first Pareto front.
        elites = clf._pareto_elite_layers(models, X, y_bool, len(models))
        self.assertTrue(set(expected_front).issubset(set(elites)))

        # With budget equal to the first-front size, the result should be
        # exactly the first front.
        elites_tight = clf._pareto_elite_layers(models, X, y_bool, len(expected_front))
        self.assertEqual(set(elites_tight), set(expected_front))


if __name__ == "__main__":
    unittest.main()
