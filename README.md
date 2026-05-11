# DTGP

Decision Tree Genetic Programming classifier with a scikit-learn-style API.

## Install

```bash
pip install .
```

## Quick start (binary classification)

```python
from dtgp import DTGPClassifier

X = [
    [4.0, 1.0],
    [5.0, 2.0],
    [1.0, 3.0],
    [2.0, 5.0],
]
y = [1 if row[0] > row[1] else 0 for row in X]

clf = DTGPClassifier(
    num_models=40,
    generations=40,
    max_depth=6,
    selection_method="pareto_tournament",
    fitness_method="pearson_r2",
    random_state=42,
)
clf.fit(X, y)

print(clf.predict(X))
print(clf.predict_proba(X))
print(clf.score(X, y))
```

## Multiclass usage

`DTGPClassifier` supports multiclass classification with one-vs-rest training.

```python
from dtgp import DTGPClassifier

X = [
    [9.0, 1.0, 1.0],
    [1.0, 9.0, 1.0],
    [1.0, 1.0, 9.0],
    [8.0, 2.0, 1.0],
    [1.0, 8.0, 2.0],
    [2.0, 1.0, 8.0],
]
y = [0, 1, 2, 0, 1, 2]

clf = DTGPClassifier(num_models=20, generations=20, random_state=7)
clf.fit(X, y)

pred = clf.predict(X)
proba = clf.predict_proba(X)
```

## Model inspection

You can inspect evolved models as expressions or tree-like text.

```python
expr = clf.view_model()            # one model as a readable expression
top3_expr = clf.view_model(3)      # top 3 models as expressions

tree = clf.view_model_tree()       # one model in tree-like format
top2_trees = clf.view_model_tree(2)
```

For multiclass one-vs-rest models, both inspection methods include class labels in the output
so it is clear which class each displayed model belongs to.

## Tree/value expression behavior

- Leaves are dataset variables (`x[i]`) and numeric constants.
- Comparison operands can include nested math layers (none, one, or many).
- Left and right sides are not required to be symmetric.

This allows regression-like value expressions at the bottom of boolean trees.

## Main training parameters

- `num_models`: population size
- `generations`: evolution steps
- `crossover_rate`: crossover fraction
- `mutation_rate`: mutation fraction
- `elitist_rate`: elite carryover fraction
- `max_depth`: maximum tree depth
- `tournament_size`: tournament selection size
- `selection_method`: parent selection strategy (`"tournament"` or `"pareto_tournament"`)
- `fitness_method`: fitness objective (`"accuracy"` or `"pearson_r2"`)
- `random_state`: reproducibility seed
- `show_training_curve`: print generation-by-generation best fitness

### Pareto tournament selection

Set `selection_method="pareto_tournament"` to optimize with two objectives during tournament
selection:

- maximize fitness (classification performance)
- minimize complexity (tree size)

For each tournament draw, DTGP computes the full non-dominated front and samples parents from
that front.

### Fitness methods

Set `fitness_method` to choose the optimization target used by evolutionary scoring:

- `"accuracy"`: maximize classification agreement (with inversion symmetry)
- `"pearson_r2"`: maximize squared Pearson correlation between predictions and labels
