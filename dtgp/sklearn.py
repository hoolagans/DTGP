"""SKLearn-compatible DTGP classifier implementation."""

from __future__ import annotations

import copy
import os
import sys
import math
import random
import statistics
import warnings
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Any, Iterable, List, Sequence, Tuple

MIN_FALLBACK_SCORE = 1e-12


class DTGPClassifier:
    """Decision Tree Genetic Programming classifier with sklearn-style API."""

    def __init__(
        self,
        num_models: int = 30,
        generations: int = 100,
        crossover_rate: float = 0.4,
        mutation_rate: float = 0.3,
        elitist_rate: float = 0.2,
        max_depth: int = 6,
        tournament_size: int = 5,
        random_state: int | None = None,
        initial_population: list | None = None,
        show_training_curve: bool = False,
    ) -> None:
        self.num_models = num_models
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitist_rate = elitist_rate
        self.max_depth = max_depth
        self.tournament_size = tournament_size
        self.random_state = random_state
        self.initial_population = [] if initial_population is None else initial_population
        self.show_training_curve = show_training_curve

    # --- sklearn-style parameter API ---
    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "num_models": self.num_models,
            "generations": self.generations,
            "crossover_rate": self.crossover_rate,
            "mutation_rate": self.mutation_rate,
            "elitist_rate": self.elitist_rate,
            "max_depth": self.max_depth,
            "tournament_size": self.tournament_size,
            "random_state": self.random_state,
            "initial_population": copy.deepcopy(self.initial_population) if deep else self.initial_population,
            "show_training_curve": self.show_training_curve,
        }

    def set_params(self, **params: Any) -> "DTGPClassifier":
        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(f"Invalid parameter '{key}' for DTGPClassifier")
            setattr(self, key, value)
        return self

    # --- fit/predict API ---
    def fit(self, X: Sequence[Sequence[float]], y: Sequence[Any]) -> "DTGPClassifier":
        X2 = _to_2d(X)
        y_list = list(y)
        if len(X2) != len(y_list):
            raise ValueError("X and y must have the same number of samples")
        if len(X2) == 0:
            raise ValueError("X and y must not be empty")

        classes = sorted(set(y_list))
        if len(classes) < 2:
            raise ValueError("DTGPClassifier requires at least two classes")

        self.classes_ = classes
        self.n_features_in_ = len(X2[0])

        if len(classes) > 2:
            return self._fit_multiclass_ova(X2, y_list)

        self.multiclass_strategy_ = None
        self.classes_ = classes
        positive_class = classes[1]
        y_bool = [label == positive_class for label in y_list]
        fitted = self._fit_binary_problem(X2, y_bool)
        self.invert_output_ = fitted["invert_output"]
        self.best_tree_ = fitted["best_tree"]
        self.population_ = fitted["population"]
        self.training_curve_ = fitted["training_curve"]
        self.best_fitness_ = fitted["best_fitness"]
        return self

    def predict(self, X: Sequence[Sequence[float]]) -> List[Any]:
        self._require_fitted()
        X2 = _to_2d(X)
        if any(len(row) != self.n_features_in_ for row in X2):
            raise ValueError("X has a different number of features than seen during fit")

        if self.multiclass_strategy_ == "one_vs_rest":
            proba = self.predict_proba(X2)
            return [self.classes_[max(range(len(row)), key=row.__getitem__)] for row in proba]

        bool_pred = [self._evaluate_model(self.best_tree_, row) for row in X2]
        if self.invert_output_:
            bool_pred = [not p for p in bool_pred]

        neg, pos = self.classes_[0], self.classes_[1]
        return [pos if p else neg for p in bool_pred]

    def predict_proba(self, X: Sequence[Sequence[float]]) -> List[List[float]]:
        self._require_fitted()
        X2 = _to_2d(X)
        if any(len(row) != self.n_features_in_ for row in X2):
            raise ValueError("X has a different number of features than seen during fit")

        if self.multiclass_strategy_ == "one_vs_rest":
            class_scores: List[List[float]] = []
            for class_label in self.classes_:
                model = self.classifiers_[class_label]
                pred = [self._evaluate_model(model["best_tree"], row) for row in X2]
                if model["invert_output"]:
                    pred = [not p for p in pred]
                class_scores.append([model["best_fitness"] if p else 0.0 for p in pred])

            fallback = [max(MIN_FALLBACK_SCORE, self.classifiers_[c]["best_fitness"]) for c in self.classes_]
            probs: List[List[float]] = []
            for sample_idx in range(len(X2)):
                row = [class_scores[class_idx][sample_idx] for class_idx in range(len(self.classes_))]
                if sum(row) <= 0.0:
                    row = fallback[:]
                total = sum(row)
                probs.append([v / total for v in row])
            return probs

        preds = self.predict(X)
        neg, pos = self.classes_[0], self.classes_[1]
        return [[1.0, 0.0] if label == neg else [0.0, 1.0] for label in preds]

    def score(self, X: Sequence[Sequence[float]], y: Sequence[Any]) -> float:
        y_true = list(y)
        y_pred = self.predict(X)
        if len(y_true) != len(y_pred):
            raise ValueError("X and y must have the same number of samples")
        return sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)

    def view_model(self, n_models: int = 1) -> str | List[str]:
        """Return interpretable representation(s) of evolved model(s)."""
        self._require_fitted()
        if n_models < 1:
            raise ValueError("n_models must be >= 1")

        limit = min(n_models, len(self.population_))
        models = self.population_[:limit]
        rendered: List[str] = []
        for i, model in enumerate(models):
            expr = self._tree_to_expression(model)
            if i == 0 and self.invert_output_:
                expr = f"NOT ({expr})"
            rendered.append(expr)
        return rendered[0] if n_models == 1 else rendered

    def view_model_tree(self, n_models: int = 1) -> str | List[str]:
        """Return tree-plot-like representation(s) of evolved model(s)."""
        self._require_fitted()
        if n_models < 1:
            raise ValueError("n_models must be >= 1")

        limit = min(n_models, len(self.population_))
        models = self.population_[:limit]
        rendered: List[str] = []
        for i, model in enumerate(models):
            lines = [f"[Model {i + 1}]"]
            if i == 0 and self.invert_output_:
                lines.append("└─ NOT")
                lines.extend(self._tree_plot_lines(model, "   "))
            else:
                lines.extend(self._tree_plot_lines(model, ""))
            rendered.append("\n".join(lines))
        return rendered[0] if n_models == 1 else rendered

    # --- DTGP internals ---
    def _leaf_ops(self):
        return {
            "avg": lambda d: statistics.fmean(d) if d else 0.0,
            "med": lambda d: statistics.median(d) if d else 0.0,
            "mn": lambda d: min(d) if d else 0.0,
            "mx": lambda d: max(d) if d else 0.0,
            "diff": lambda d: (d[-1] - d[0]) if len(d) >= 2 else 0.0,
            "diff2": lambda d: (d[1] - d[0]) if len(d) >= 2 else 0.0,
            "diff3": lambda d: (d[2] - d[1]) if len(d) >= 3 else 0.0,
            "chng": lambda d: (max(d) - min(d)) if d else 0.0,
            "dev": lambda d: statistics.pstdev(d) if len(d) >= 2 else 0.0,
            "getred": lambda d: d[0] if len(d) >= 1 else 0.0,
            "getgreen": lambda d: d[1] if len(d) >= 2 else 0.0,
            "getblue": lambda d: d[2] if len(d) >= 3 else 0.0,
        }

    def _inter_ops(self):
        return {
            "ge": lambda a, b: a >= b,
            "gt": lambda a, b: a > b,
            "le": lambda a, b: a <= b,
            "lt": lambda a, b: a < b,
            "eq": lambda a, b: a == b,
            "ne": lambda a, b: a != b,
        }

    def _node_ops(self):
        return {
            "and": lambda a, b: bool(a) and bool(b),
            "or": lambda a, b: bool(a) or bool(b),
            "nand": lambda a, b: not (bool(a) and bool(b)),
            "nor": lambda a, b: not (bool(a) or bool(b)),
            "xor": lambda a, b: bool(a) ^ bool(b),
        }

    def _random_leaf(self):
        choices = list(self._leaf_ops().keys()) + ["const"]
        op = self._rng.choice(choices)
        if op == "const":
            return ("const", self._rng.uniform(0.0, 1.0))
        return ("leaf", op)

    def _random_inter(self):
        op = self._rng.choice(list(self._inter_ops().keys()))
        return ("inter", op, self._random_leaf(), self._random_leaf())

    def _random_branch(self, depth: int, max_depth: int):
        if depth >= max_depth:
            return self._random_inter()
        if self._rng.randint(0, 2) == 1:
            op = self._rng.choice(list(self._node_ops().keys()))
            return (
                "node",
                op,
                self._random_branch(depth + 1, max_depth),
                self._random_branch(depth + 1, max_depth),
            )
        return self._random_inter()

    def _random_tree(self, max_depth: int):
        op = self._rng.choice(list(self._node_ops().keys()))
        tree = (
            "node",
            op,
            self._random_branch(1, max_depth),
            self._random_branch(1, max_depth),
        )
        while self._tree_depth(tree) > max_depth:
            op = self._rng.choice(list(self._node_ops().keys()))
            tree = (
                "node",
                op,
                self._random_branch(1, max_depth),
                self._random_branch(1, max_depth),
            )
        return tree

    def _tree_depth(self, tree) -> int:
        kind = tree[0]
        if kind == "inter":
            return 2
        return 1 + max(self._tree_depth(tree[2]), self._tree_depth(tree[3]))

    def _eval_leaf(self, leaf, data: Sequence[float]) -> float:
        if leaf[0] == "const":
            return float(leaf[1])
        value = self._leaf_ops()[leaf[1]](data)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
        return 0.0

    def _evaluate_model(self, tree, data: Sequence[float]) -> bool:
        kind = tree[0]
        if kind == "inter":
            _, op, left, right = tree
            return bool(self._inter_ops()[op](self._eval_leaf(left, data), self._eval_leaf(right, data)))
        _, op, left, right = tree
        return bool(self._node_ops()[op](self._evaluate_model(left, data), self._evaluate_model(right, data)))

    def _leaf_to_expression(self, leaf) -> str:
        if leaf[0] == "const":
            return f"{float(leaf[1]):.6g}"

        leaf_names = {
            "avg": "Avg",
            "med": "Med",
            "mn": "Mn",
            "mx": "Mx",
            "diff": "Diff",
            "diff2": "Diff2",
            "diff3": "Diff3",
            "chng": "Chng",
            "dev": "Dev",
            "getred": "GetRed",
            "getgreen": "GetGreen",
            "getblue": "GetBlue",
        }
        return f"{leaf_names[leaf[1]]}(data)"

    def _tree_to_expression(self, tree) -> str:
        if tree[0] == "inter":
            _, op, left, right = tree
            cmp_names = {"ge": ">=", "gt": ">", "le": "<=", "lt": "<", "eq": "==", "ne": "!="}
            return f"({self._leaf_to_expression(left)} {cmp_names[op]} {self._leaf_to_expression(right)})"

        _, op, left, right = tree
        node_names = {"and": "AND", "or": "OR", "nand": "NAND", "nor": "NOR", "xor": "XOR"}
        left_expr = self._tree_to_expression(left)
        right_expr = self._tree_to_expression(right)
        return f"({left_expr} {node_names[op]} {right_expr})"

    def _tree_plot_lines(self, tree, indent: str = "") -> List[str]:
        if tree[0] == "inter":
            _, op, left, right = tree
            cmp_names = {"ge": ">=", "gt": ">", "le": "<=", "lt": "<", "eq": "==", "ne": "!="}
            return [f"{indent}└─ {self._leaf_to_expression(left)} {cmp_names[op]} {self._leaf_to_expression(right)}"]

        _, op, left, right = tree
        node_names = {"and": "AND", "or": "OR", "nand": "NAND", "nor": "NOR", "xor": "XOR"}
        lines = [f"{indent}└─ {node_names[op]}"]
        lines.append(f"{indent}   ├─ LEFT")
        lines.extend(self._tree_plot_lines(left, f"{indent}   │  "))
        lines.append(f"{indent}   └─ RIGHT")
        lines.extend(self._tree_plot_lines(right, f"{indent}      "))
        return lines

    def _training_curve_line(self, generation: int, total_generations: int, best_fitness: float) -> str:
        width = 30
        filled = max(0, min(width, int(round(best_fitness * width))))
        bar = "#" * filled + "-" * (width - filled)
        return f"Generation {generation}/{total_generations} |{bar}| best_fitness={best_fitness:.4f}"

    def _fit_binary_problem(self, X, y_bool, curve_label: str | None = None):
        self._rng = random.Random(self.random_state)
        models, history = self._evolve(X, y_bool, curve_label=curve_label)
        best_tree = models[0]
        raw_fit = self._raw_fitness(best_tree, X, y_bool)
        invert_output = raw_fit < 0.5
        best_fitness = self._fitness(best_tree, X, y_bool)
        return {
            "best_tree": best_tree,
            "population": models,
            "invert_output": invert_output,
            "best_fitness": best_fitness,
            "training_curve": history,
        }

    def _fit_multiclass_ova(self, X, y_list):
        self.multiclass_strategy_ = "one_vs_rest"
        params = self.get_params(deep=True)
        base_seed = self.random_state
        classes = self.classes_
        n_workers = min(len(classes), os.cpu_count() or 1)

        tasks = []
        for idx, class_label in enumerate(classes):
            task_params = dict(params)
            task_params["random_state"] = None if base_seed is None else base_seed + idx + 1
            y_bool = [label == class_label for label in y_list]
            tasks.append((class_label, X, y_bool, task_params))

        if n_workers > 1:
            try:
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    results = list(executor.map(_train_binary_worker, tasks))
            except (BrokenProcessPool, OSError, RuntimeError):
                warnings.warn(
                    "Parallel one-vs-rest training failed; falling back to sequential execution.",
                    RuntimeWarning,
                )
                results = [_train_binary_worker(task) for task in tasks]
                n_workers = 1
        else:
            results = [_train_binary_worker(task) for task in tasks]

        self.parallel_workers_ = n_workers
        self.classifiers_ = {class_label: fitted for class_label, fitted in results}
        self.class_training_curves_ = {
            class_label: fitted["training_curve"] for class_label, fitted in results
        }
        self.class_best_fitness_ = {
            class_label: fitted["best_fitness"] for class_label, fitted in results
        }

        representative = self.classifiers_[classes[0]]
        self.invert_output_ = representative["invert_output"]
        self.best_tree_ = representative["best_tree"]
        self.population_ = representative["population"]
        self.training_curve_ = representative["training_curve"]
        self.best_fitness_ = representative["best_fitness"]
        return self

    def _raw_fitness(self, tree, X: Sequence[Sequence[float]], y_bool: Sequence[bool]) -> float:
        preds = [self._evaluate_model(tree, row) for row in X]
        return sum(a == b for a, b in zip(preds, y_bool)) / len(X)

    def _fitness(self, tree, X: Sequence[Sequence[float]], y_bool: Sequence[bool]) -> float:
        raw = self._raw_fitness(tree, X, y_bool)
        return max(raw, 1.0 - raw)

    def _collect_paths(self, tree, prefix: Tuple[int, ...] = ()) -> List[Tuple[int, ...]]:
        paths = [prefix]
        if tree[0] == "node":
            paths.extend(self._collect_paths(tree[2], prefix + (2,)))
            paths.extend(self._collect_paths(tree[3], prefix + (3,)))
        return paths

    def _get_subtree(self, tree, path: Tuple[int, ...]):
        cur = tree
        for idx in path:
            cur = cur[idx]
        return cur

    def _set_subtree(self, tree, path: Tuple[int, ...], replacement):
        if not path:
            return replacement
        idx = path[0]
        if tree[0] != "node":
            return replacement
        if idx == 2:
            return (tree[0], tree[1], self._set_subtree(tree[2], path[1:], replacement), tree[3])
        return (tree[0], tree[1], tree[2], self._set_subtree(tree[3], path[1:], replacement))

    def _mutate(self, tree):
        paths = self._collect_paths(tree)
        target = self._rng.choice(paths)
        replacement = self._random_branch(1, max(2, self.max_depth))
        return self._set_subtree(tree, target, replacement)

    def _crossover(self, tree1, tree2):
        paths1 = self._collect_paths(tree1)
        paths2 = self._collect_paths(tree2)
        p1 = self._rng.choice(paths1)
        p2 = self._rng.choice(paths2)

        s1 = self._get_subtree(tree1, p1)
        s2 = self._get_subtree(tree2, p2)

        new1 = self._set_subtree(tree1, p1, s2)
        new2 = self._set_subtree(tree2, p2, s1)
        return new1, new2

    def _tournament_select(self, models, X, y_bool):
        size = min(max(2, self.tournament_size), len(models))
        sample = self._rng.sample(models, size)
        scored = [(m, self._fitness(m, X, y_bool)) for m in sample]
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[0][0]

    def _evolve(self, X, y_bool, curve_label: str | None = None):
        models = list(self.initial_population)
        while len(models) < self.num_models:
            models.append(self._random_tree(self.max_depth))

        initial_best = max(self._fitness(m, X, y_bool) for m in models)
        history = [initial_best]
        if self.show_training_curve:
            line = self._training_curve_line(0, self.generations, initial_best)
            if curve_label:
                line = f"[{curve_label}] {line}"
            print(line, file=sys.stderr, flush=True)

        for gen_idx in range(self.generations):
            new_models = []

            cross_target = int(self.crossover_rate * self.num_models)
            mut_target = int((self.crossover_rate + self.mutation_rate) * self.num_models)
            elite_count = int(self.elitist_rate * self.num_models)

            while len(new_models) < cross_target:
                p1 = self._tournament_select(models, X, y_bool)
                p2 = self._tournament_select(models, X, y_bool)
                c1, c2 = self._crossover(p1, p2)
                new_models.extend([c1, c2])

            while len(new_models) < mut_target:
                p = self._tournament_select(models, X, y_bool)
                new_models.append(self._mutate(p))

            scored = sorted(((m, self._fitness(m, X, y_bool)) for m in models), key=lambda t: t[1], reverse=True)
            elites = [m for m, _ in scored[:elite_count]]
            new_models.extend(elites)

            deduped = list(dict.fromkeys(new_models))
            filtered = [m for m in deduped if self._tree_depth(m) <= self.max_depth]

            while len(filtered) < self.num_models:
                filtered.append(self._random_tree(self.max_depth))

            models = filtered[: self.num_models]
            best_now = max(self._fitness(m, X, y_bool) for m in models)
            history.append(best_now)
            if self.show_training_curve:
                line = self._training_curve_line(gen_idx + 1, self.generations, best_now)
                if curve_label:
                    line = f"[{curve_label}] {line}"
                print(line, file=sys.stderr, flush=True)

        final_scored = sorted(((m, self._fitness(m, X, y_bool)) for m in models), key=lambda t: t[1], reverse=True)
        return [m for m, _ in final_scored], history

    def _require_fitted(self):
        required = ["classes_", "n_features_in_"]
        if not all(hasattr(self, name) for name in required):
            raise ValueError("This DTGPClassifier instance is not fitted yet. Call 'fit' first.")
        if getattr(self, "multiclass_strategy_", None) == "one_vs_rest":
            if not hasattr(self, "classifiers_"):
                raise ValueError("This DTGPClassifier instance is not fitted yet. Call 'fit' first.")
        else:
            binary_required = ["best_tree_", "invert_output_"]
            if not all(hasattr(self, name) for name in binary_required):
                raise ValueError("This DTGPClassifier instance is not fitted yet. Call 'fit' first.")


def _train_binary_worker(task):
    class_label, X, y_bool, params = task
    model = DTGPClassifier(**params)
    fitted = model._fit_binary_problem(X, y_bool, curve_label=f"class={class_label}")
    return class_label, fitted


def _flatten_row(row: Iterable[Any]) -> List[float]:
    out: List[float] = []
    for val in row:
        if isinstance(val, (list, tuple)):
            out.extend(_flatten_row(val))
        else:
            out.append(float(val))
    return out


def _to_2d(X: Sequence[Sequence[float]]) -> List[List[float]]:
    if hasattr(X, "tolist"):
        X = X.tolist()  # type: ignore[assignment]

    rows = list(X)
    if not rows:
        return []

    first = rows[0]
    if isinstance(first, (int, float)):
        raise ValueError("X must be 2D (n_samples, n_features)")

    out = [_flatten_row(row) for row in rows]
    n_features = len(out[0])
    if n_features == 0:
        raise ValueError("X must contain at least one feature")
    if any(len(row) != n_features for row in out):
        raise ValueError("All samples in X must have the same number of features")
    return out
