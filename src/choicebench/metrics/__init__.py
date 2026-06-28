# src/choicebench/metrics/__init__.py

from choicebench.metrics.base import BaseMetric
from choicebench.metrics.accuracy import Accuracy
from choicebench.metrics.mad import MAD

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
