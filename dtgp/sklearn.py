"""SKLearn-compatible DTGP classifier implementation."""

from __future__ import annotations

import copy
import os
import sys
import math
import random
import warnings
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Any, Iterable, List, Sequence, Tuple

MIN_FALLBACK_SCORE = 1e-12
DIV_EPSILON = 1e-12
DIV_FALLBACK = 0.0

# Module-level op-dispatch tables — built once instead of on every call.
_INTER_OPS = {
    "ge": lambda a, b: a >= b,
    "gt": lambda a, b: a > b,
    "le": lambda a, b: a <= b,
    "lt": lambda a, b: a < b,
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
}

_NODE_OPS = {
    "and": lambda a, b: bool(a) and bool(b),
    "or": lambda a, b: bool(a) or bool(b),
    "nand": lambda a, b: not (bool(a) and bool(b)),
    "nor": lambda a, b: not (bool(a) or bool(b)),
    "xor": lambda a, b: bool(a) ^ bool(b),
}

_MATH_UNARY_OPS = {
    "neg": lambda a: -a,
    "abs": lambda a: abs(a),
    "sqrt": lambda a: math.sqrt(abs(a)),
    "log1p": lambda a: math.log1p(abs(a)),
    "sin": lambda a: math.sin(a),
    "cos": lambda a: math.cos(a),
    "tanh": lambda a: math.tanh(a),
}

_MATH_BINARY_OPS = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b if abs(b) > DIV_EPSILON else DIV_FALLBACK,
    "min": lambda a, b: min(a, b),
    "max": lambda a, b: max(a, b),
}


class RenderableModelList(list):
    """List with readable multiline rendering for model inspection output.

    This is returned by view methods when multiple models are requested.
    Printing it (or displaying it in an interactive shell) renders each item
    separated by a blank line, instead of list-repr escaped newline sequences.
    """

    def __str__(self) -> str:
        return "\n\n".join(str(item) for item in self)

    def __repr__(self) -> str:
        return self.__str__()


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
        selection_method: str = "pareto_tournament",
        fitness_method: str = "pearson_r2",
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
        self.selection_method = selection_method
        self.fitness_method = fitness_method
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
            "selection_method": self.selection_method,
            "fitness_method": self.fitness_method,
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
        self._validate_selection_method()
        self._validate_fitness_method()
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
        if getattr(self, "multiclass_strategy_", None) == "one_vs_rest":
            return self._view_model_multiclass(n_models)

        limit = min(n_models, len(self.population_))
        models = self.population_[:limit]
        rendered: List[str] = []
        for i, model in enumerate(models):
            expr = self._tree_to_expression(model)
            if i == 0 and self.invert_output_:
                expr = f"NOT ({expr})"
            rendered.append(expr)
        return rendered[0] if n_models == 1 else RenderableModelList(rendered)

    def view_model_tree(self, n_models: int = 1) -> str | List[str]:
        """Return tree-plot-like representation(s) of evolved model(s)."""
        self._require_fitted()
        if n_models < 1:
            raise ValueError("n_models must be >= 1")
        if getattr(self, "multiclass_strategy_", None) == "one_vs_rest":
            return self._view_model_tree_multiclass(n_models)

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
        return rendered[0] if n_models == 1 else RenderableModelList(rendered)

    def _view_model_multiclass(self, n_models: int) -> str | List[str]:
        rendered: List[str] = []
        for class_label in self.classes_:
            fitted = self.classifiers_[class_label]
            limit = min(n_models, len(fitted["population"]))
            models = fitted["population"][:limit]
            for i, model in enumerate(models):
                expr = self._tree_to_expression(model)
                if i == 0 and fitted["invert_output"]:
                    expr = f"NOT ({expr})"
                rendered.append(f"[Class {class_label} | Model {i + 1}] {expr}")
        if n_models == 1:
            return "\n".join(rendered)
        return RenderableModelList(rendered)

    def _view_model_tree_multiclass(self, n_models: int) -> str | List[str]:
        rendered: List[str] = []
        for class_label in self.classes_:
            fitted = self.classifiers_[class_label]
            limit = min(n_models, len(fitted["population"]))
            models = fitted["population"][:limit]
            for i, model in enumerate(models):
                lines = [f"[Class {class_label} | Model {i + 1}]"]
                if i == 0 and fitted["invert_output"]:
                    lines.append("└─ NOT")
                    lines.extend(self._tree_plot_lines(model, "   "))
                else:
                    lines.extend(self._tree_plot_lines(model, ""))
                rendered.append("\n".join(lines))
        if n_models == 1:
            return "\n\n".join(rendered)
        return RenderableModelList(rendered)

    # --- DTGP internals ---
    def _inter_ops(self):
        return _INTER_OPS

    def _node_ops(self):
        return _NODE_OPS

    def _math_unary_ops(self):
        return _MATH_UNARY_OPS

    def _math_binary_ops(self):
        return _MATH_BINARY_OPS

    def _random_base_value(self):
        if self.n_features_in_ <= 0:
            return ("const", self._rng.uniform(-1.0, 1.0))
        if self._rng.random() < 0.65:
            return ("var", self._rng.randrange(self.n_features_in_))
        return ("const", self._rng.uniform(-1.0, 1.0))

    def _random_value_expr(self, depth: int = 0, max_depth: int = 3):
        if depth >= max_depth:
            return self._random_base_value()

        if depth > 0 and self._rng.random() < 0.45:
            return self._random_base_value()

        draw = self._rng.random()
        if draw < 0.35:
            return self._random_base_value()
        if draw < 0.65:
            op = self._rng.choice(list(self._math_unary_ops().keys()))
            return ("math1", op, self._random_value_expr(depth + 1, max_depth))
        op = self._rng.choice(list(self._math_binary_ops().keys()))
        return (
            "math2",
            op,
            self._random_value_expr(depth + 1, max_depth),
            self._random_value_expr(depth + 1, max_depth),
        )

    def _random_inter(self):
        op = self._rng.choice(list(self._inter_ops().keys()))
        return ("inter", op, self._random_value_expr(), self._random_value_expr())

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
        if kind in {"const", "var"}:
            return 1
        if kind == "math1":
            return 1 + self._tree_depth(tree[2])
        if kind == "math2":
            return 1 + max(self._tree_depth(tree[2]), self._tree_depth(tree[3]))
        if kind == "inter":
            return 1 + max(self._tree_depth(tree[2]), self._tree_depth(tree[3]))
        return 1 + max(self._tree_depth(tree[2]), self._tree_depth(tree[3]))

    def _sanitize_value(self, value: Any) -> float:
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            clipped = float(value)
            return max(-1e12, min(1e12, clipped))
        return 0.0

    def _eval_value(self, value_node, data: Sequence[float]) -> float:
        kind = value_node[0]
        if kind == "const":
            return self._sanitize_value(value_node[1])
        if kind == "var":
            idx = int(value_node[1])
            if 0 <= idx < len(data):
                return self._sanitize_value(data[idx])
            return 0.0
        if kind == "math1":
            _, op, child = value_node
            inner = self._eval_value(child, data)
            return self._sanitize_value(_MATH_UNARY_OPS[op](inner))
        _, op, left, right = value_node
        a = self._eval_value(left, data)
        b = self._eval_value(right, data)
        return self._sanitize_value(_MATH_BINARY_OPS[op](a, b))

    def _evaluate_model(self, tree, data: Sequence[float]) -> bool:
        kind = tree[0]
        if kind == "inter":
            _, op, left, right = tree
            return bool(_INTER_OPS[op](self._eval_value(left, data), self._eval_value(right, data)))
        _, op, left, right = tree
        return bool(_NODE_OPS[op](self._evaluate_model(left, data), self._evaluate_model(right, data)))

    def _value_to_expression(self, value_node) -> str:
        kind = value_node[0]
        if kind == "const":
            return f"{float(value_node[1]):.6g}"
        if kind == "var":
            return f"x[{int(value_node[1])}]"
        if kind == "math1":
            _, op, child = value_node
            unary_names = {
                "neg": "-",
                "abs": "abs",
                "sqrt": "sqrt",
                "log1p": "log1p",
                "sin": "sin",
                "cos": "cos",
                "tanh": "tanh",
            }
            child_expr = self._value_to_expression(child)
            if op == "neg":
                return f"(-{child_expr})"
            return f"{unary_names[op]}({child_expr})"
        _, op, left, right = value_node
        binary_names = {
            "add": "+",
            "sub": "-",
            "mul": "*",
            "div": "/",
            "min": "min",
            "max": "max",
        }
        left_expr = self._value_to_expression(left)
        right_expr = self._value_to_expression(right)
        if op in {"min", "max"}:
            return f"{binary_names[op]}({left_expr}, {right_expr})"
        return f"({left_expr} {binary_names[op]} {right_expr})"

    def _tree_to_expression(self, tree) -> str:
        if tree[0] == "inter":
            _, op, left, right = tree
            cmp_names = {"ge": ">=", "gt": ">", "le": "<=", "lt": "<", "eq": "==", "ne": "!="}
            return f"({self._value_to_expression(left)} {cmp_names[op]} {self._value_to_expression(right)})"

        _, op, left, right = tree
        node_names = {"and": "AND", "or": "OR", "nand": "NAND", "nor": "NOR", "xor": "XOR"}
        left_expr = self._tree_to_expression(left)
        right_expr = self._tree_to_expression(right)
        return f"({left_expr} {node_names[op]} {right_expr})"

    def _tree_plot_lines(self, tree, indent: str = "") -> List[str]:
        if tree[0] == "inter":
            _, op, left, right = tree
            cmp_names = {"ge": ">=", "gt": ">", "le": "<=", "lt": "<", "eq": "==", "ne": "!="}
            return [f"{indent}└─ {self._value_to_expression(left)} {cmp_names[op]} {self._value_to_expression(right)}"]

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
        self.n_features_in_ = len(X[0]) if X else 0
        self._rng = random.Random(self.random_state)
        models, history = self._evolve(X, y_bool, curve_label=curve_label)
        best_tree = models[0]
        _, raw_fit = self._predict_and_score(best_tree, X, y_bool)
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

    def _predict_and_score(self, tree, X: Sequence[Sequence[float]], y_bool: Sequence[bool]) -> Tuple[List[bool], float]:
        """Return (preds, raw_accuracy) to avoid recomputing predictions."""
        preds = [self._evaluate_model(tree, row) for row in X]
        return preds, sum(a == b for a, b in zip(preds, y_bool)) / len(X)

    def _fitness(self, tree, X: Sequence[Sequence[float]], y_bool: Sequence[bool]) -> float:
        # Use the per-generation cache when available.
        cache = getattr(self, "_fitness_cache", None)
        if cache is not None:
            key = id(tree)
            if key in cache:
                return cache[key]
        preds, raw = self._predict_and_score(tree, X, y_bool)
        if self.fitness_method == "accuracy":
            score = max(raw, 1.0 - raw)
        else:
            score = self._pearson_r_squared(preds, y_bool)
        if cache is not None:
            cache[key] = score
        return score

    def _pearson_r_squared(self, x: Sequence[bool], y: Sequence[bool]) -> float:
        x_float = [1.0 if v else 0.0 for v in x]
        y_float = [1.0 if v else 0.0 for v in y]
        n = len(x_float)
        if n == 0:
            return 0.0

        x_mean = sum(x_float) / n
        y_mean = sum(y_float) / n
        x_centered = [v - x_mean for v in x_float]
        y_centered = [v - y_mean for v in y_float]
        cov = sum(a * b for a, b in zip(x_centered, y_centered))
        x_var = sum(a * a for a in x_centered)
        y_var = sum(b * b for b in y_centered)
        if x_var <= 0.0 or y_var <= 0.0:
            return 0.0
        corr = cov / math.sqrt(x_var * y_var)
        return corr * corr

    def _validate_selection_method(self) -> None:
        valid = {"tournament", "pareto_tournament"}
        if self.selection_method not in valid:
            valid_list = ", ".join(sorted(valid))
            raise ValueError(f"selection_method must be one of: {valid_list}")

    def _validate_fitness_method(self) -> None:
        valid = {"accuracy", "pearson_r2"}
        if self.fitness_method not in valid:
            valid_list = ", ".join(sorted(valid))
            raise ValueError(f"fitness_method must be one of: {valid_list}")

    def _model_complexity(self, tree) -> int:
        """Return model complexity as total subtree path count (lower is simpler)."""
        return len(self._get_paths(tree))

    def _get_paths(self, tree) -> List[Tuple[int, ...]]:
        """Return cached paths list for *tree*, computing if not yet cached."""
        cache = getattr(self, "_paths_cache", None)
        if cache is not None:
            key = id(tree)
            if key not in cache:
                cache[key] = self._collect_paths(tree)
            return cache[key]
        return self._collect_paths(tree)

    def _child_indexes(self, node) -> Tuple[int, ...]:
        kind = node[0]
        if kind in {"node", "inter", "math2"}:
            return (2, 3)
        if kind == "math1":
            return (2,)
        return ()

    def _collect_paths(self, tree, prefix: Tuple[int, ...] = ()) -> List[Tuple[int, ...]]:
        paths = [prefix]
        for child_idx in self._child_indexes(tree):
            paths.extend(self._collect_paths(tree[child_idx], prefix + (child_idx,)))
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
        kind = tree[0]
        if kind in {"node", "inter", "math2"}:
            if idx == 2:
                return (tree[0], tree[1], self._set_subtree(tree[2], path[1:], replacement), tree[3])
            if idx == 3:
                return (tree[0], tree[1], tree[2], self._set_subtree(tree[3], path[1:], replacement))
            return tree
        if kind == "math1":
            if idx == 2:
                return (tree[0], tree[1], self._set_subtree(tree[2], path[1:], replacement))
            return tree
        return tree

    def _node_category(self, node) -> str:
        return "bool" if node[0] in {"node", "inter"} else "value"

    def _random_replacement_for(self, node):
        if self._node_category(node) == "bool":
            return self._random_branch(1, max(2, self.max_depth))
        return self._random_value_expr()

    def _mutate(self, tree):
        paths = self._get_paths(tree)
        target = self._rng.choice(paths)
        target_node = self._get_subtree(tree, target)
        replacement = self._random_replacement_for(target_node)
        return self._set_subtree(tree, target, replacement)

    def _crossover(self, tree1, tree2):
        paths1 = self._get_paths(tree1)
        paths2 = self._get_paths(tree2)
        p1 = self._rng.choice(paths1)
        s1 = self._get_subtree(tree1, p1)
        compatible_paths2 = [p for p in paths2 if self._node_category(self._get_subtree(tree2, p)) == self._node_category(s1)]
        if not compatible_paths2:
            return tree1, tree2
        p2 = self._rng.choice(compatible_paths2)

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

    def _pareto_elite_layers(self, models, X, y_bool, elite_count: int):
        """Return elites by preserving whole Pareto layers until the budget is filled.

        The first non-dominated front is always carried over in full.  Subsequent
        fronts are added whole until adding the next front would exceed *elite_count*,
        at which point models in that partial front are ranked by fitness (descending)
        then complexity (ascending) and the highest-ranked ones fill the remaining
        budget.
        """
        metrics = {m: (self._fitness(m, X, y_bool), self._model_complexity(m)) for m in models}

        def dominates(left, right) -> bool:
            lf, lc = metrics[left]
            rf, rc = metrics[right]
            return (lf >= rf and lc <= rc) and (lf > rf or lc < rc)

        remaining = list(models)
        elites: List = []
        while remaining and len(elites) < elite_count:
            front = [m for m in remaining if not any(dominates(other, m) for other in remaining if other is not m)]
            front.sort(key=lambda m: (-metrics[m][0], metrics[m][1]))
            budget_left = elite_count - len(elites)
            if len(front) <= budget_left:
                elites.extend(front)
            else:
                elites.extend(front[:budget_left])
            remaining = [m for m in remaining if m not in set(front)]
        return elites

    def _pareto_tournament_select(self, models, X, y_bool):
        """Return the full non-dominated front from one tournament sample.

        Dominance is computed on two objectives: maximize fitness and minimize
        model complexity.
        """
        size = min(max(2, self.tournament_size), len(models))
        sample = self._rng.sample(models, size)
        metrics = {m: (self._fitness(m, X, y_bool), self._model_complexity(m)) for m in sample}

        def dominates(left, right) -> bool:
            left_fitness, left_complexity = metrics[left]
            right_fitness, right_complexity = metrics[right]
            no_worse = left_fitness >= right_fitness and left_complexity <= right_complexity
            strictly_better = left_fitness > right_fitness or left_complexity < right_complexity
            return no_worse and strictly_better

        front = []
        for candidate in sample:
            if any(dominates(other, candidate) for other in sample if other is not candidate):
                continue
            front.append(candidate)
        front.sort(key=lambda m: (-metrics[m][0], metrics[m][1]))
        return front

    def _selection_candidates(self, models, X, y_bool):
        """Return tournament candidates under configured selection strategy."""
        if self.selection_method == "pareto_tournament":
            return self._pareto_tournament_select(models, X, y_bool)
        return [self._tournament_select(models, X, y_bool)]

    def _select_parent(self, models, X, y_bool):
        """Select one parent by sampling from strategy-specific tournament candidates."""
        candidates = self._selection_candidates(models, X, y_bool)
        return self._rng.choice(candidates)

    def _evolve(self, X, y_bool, curve_label: str | None = None):
        models = list(self.initial_population)
        while len(models) < self.num_models:
            models.append(self._random_tree(self.max_depth))

        # Per-generation caches; reset at the start of each generation.
        self._fitness_cache: dict = {}
        self._paths_cache: dict = {}

        initial_best = max(self._fitness(m, X, y_bool) for m in models)
        history = [initial_best]
        if self.show_training_curve:
            line = self._training_curve_line(0, self.generations, initial_best)
            if curve_label:
                line = f"[{curve_label}] {line}"
            print(line, file=sys.stderr, flush=True)

        for gen_idx in range(self.generations):
            # Reset per-generation caches so stale id()-keyed entries don't
            # accumulate or produce false hits if a new tree reuses an old id.
            self._fitness_cache = {}
            self._paths_cache = {}

            new_models = []

            cross_target = int(self.crossover_rate * self.num_models)
            mut_target = int((self.crossover_rate + self.mutation_rate) * self.num_models)
            elite_count = int(self.elitist_rate * self.num_models)

            while len(new_models) < cross_target:
                p1 = self._select_parent(models, X, y_bool)
                p2 = self._select_parent(models, X, y_bool)
                c1, c2 = self._crossover(p1, p2)
                new_models.extend([c1, c2])

            while len(new_models) < mut_target:
                p = self._select_parent(models, X, y_bool)
                new_models.append(self._mutate(p))

            if self.selection_method == "pareto_tournament":
                elites = self._pareto_elite_layers(models, X, y_bool, elite_count)
                # _pareto_elite_layers already scored all models; reuse the best.
                best_now = max(self._fitness(m, X, y_bool) for m in elites) if elites else 0.0
            else:
                scored = sorted(((m, self._fitness(m, X, y_bool)) for m in models), key=lambda t: t[1], reverse=True)
                elites = [m for m, _ in scored[:elite_count]]
                best_now = scored[0][1] if scored else 0.0
            new_models.extend(elites)

            seen: set = set()
            deduped = []
            for m in new_models:
                if m not in seen:
                    seen.add(m)
                    deduped.append(m)
            filtered = [m for m in deduped if self._tree_depth(m) <= self.max_depth]

            while len(filtered) < self.num_models:
                filtered.append(self._random_tree(self.max_depth))

            models = filtered[: self.num_models]
            history.append(best_now)
            if self.show_training_curve:
                line = self._training_curve_line(gen_idx + 1, self.generations, best_now)
                if curve_label:
                    line = f"[{curve_label}] {line}"
                print(line, file=sys.stderr, flush=True)

        # Clear caches after evolution to release tree references.
        del self._fitness_cache
        del self._paths_cache

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
