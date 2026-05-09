import unittest

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

        self.assertEqual(clf1.predict(X), clf2.predict(X))

    def test_get_set_params(self):
        clf = DTGPClassifier(num_models=10, generations=20, random_state=3)
        params = clf.get_params()
        self.assertEqual(params["num_models"], 10)
        self.assertEqual(params["generations"], 20)

        returned = clf.set_params(num_models=15, generations=30)
        self.assertIs(returned, clf)
        self.assertEqual(clf.num_models, 15)
        self.assertEqual(clf.generations, 30)


if __name__ == "__main__":
    unittest.main()
