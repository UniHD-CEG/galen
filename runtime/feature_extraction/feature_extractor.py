import abc
import copy
from abc import abstractmethod

from runtime.model.model_handle import AModelReference, AExecutableModel
from runtime.sensitivity.sensitivity_analysis import SensitivityAnalysis, LayerSensitivityResults, SensitivityResults


class ModelFeatures:
    def __init__(self, sensitivity_results: SensitivityResults, model_metrics: dict[str, dict[str, float]]):
        self._features: dict[str, LayerFeatures] = dict()

        for layer_key in model_metrics.keys():
            self._features[layer_key] = LayerFeatures(layer_key, sensitivity_results[layer_key],
                                                      model_metrics[layer_key])

    @property
    def features(self):
        return self._features

    def update_metrics(self, model_metrics: dict[str, dict[str, float]]):
        for layer_key, layer_features in self._features.items():
            layer_features.metrics = model_metrics[layer_key]


class LayerFeatures:
    def __init__(self, layer_key: str, layer_sens: LayerSensitivityResults, layer_metrics: dict[str, float]):
        self._layer_key = layer_key
        self._layer_sens = layer_sens
        self._layer_metrics = layer_metrics

    @property
    def sensitivity(self):
        return self._layer_sens

    @property
    def metrics(self):
        return self._layer_metrics

    @metrics.setter
    def metrics(self, layer_metrics: dict[str, float]):
        self._layer_metrics = layer_metrics


class AMetricExtractor(metaclass=abc.ABCMeta):

    @abstractmethod
    def __call__(self, executable_model: AExecutableModel):
        pass

    @abstractmethod
    def __enter__(self):
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @abstractmethod
    def compute_metric_for_layer(self, layer_key: str) -> dict[str, float]:
        pass


class AFeatureExtractor(metaclass=abc.ABCMeta):
    def __init__(self, sensitivity_analysis: SensitivityAnalysis):
        self._sensitivity_analysis = sensitivity_analysis
        self._metric_extractors = self._register_extractors()

    def extract_metrics_and_sens(self, executable_model: AExecutableModel,
                                 model_reference: AModelReference) -> ModelFeatures:
        model_metrics = self._extract_model_metrics(executable_model)
        sens_analysis = self._sensitivity_analysis.analyse(model_reference, executable_model.applied_policy)

        return ModelFeatures(sens_analysis, model_metrics)

    def update_metrics(self, executable_model: AExecutableModel, model_features: ModelFeatures):
        new_metrics = self._extract_model_metrics(executable_model)
        new_features = copy.deepcopy(model_features)
        new_features.update_metrics(new_metrics)
        return new_features

    def _extract_model_metrics(self, current_model: AExecutableModel) -> dict[str, dict[str, float]]:
        layer_dict = dict()
        for extractor in self._metric_extractors:
            with extractor(current_model) as ex:
                for layer_key in current_model.all_layer_keys():
                    metric_result = ex.compute_metric_for_layer(layer_key)
                    if layer_key not in layer_dict:
                        layer_dict[layer_key] = metric_result
                    layer_dict[layer_key].update(metric_result)
        return layer_dict

    @abstractmethod
    def _register_extractors(self) -> list[AMetricExtractor]:
        pass
