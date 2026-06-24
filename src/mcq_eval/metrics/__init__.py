# src/mcq_eval/metrics/__init__.py

from mcq_eval.metrics.base import BaseMetric
from mcq_eval.metrics.accuracy import Accuracy
from mcq_eval.metrics.mad import MAD

# Maps the metric name string used in config.yaml's metrics: list to the
# class that implements it. The eval runner instantiates from this registry
# — see config/experiment_template.yaml's metrics: section.
BUILTIN_METRICS: dict[str, type[BaseMetric]] = {
    "accuracy": Accuracy,
    "mad": MAD,
}

__all__ = [
    "BaseMetric",
    "Accuracy",
    "MAD",
    "BUILTIN_METRICS",
]
