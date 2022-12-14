import abc
import copy
from abc import abstractmethod

import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

from runtime.compress.compress_adapters import ACompressAdapter
from runtime.compress.compression_policy import CompressionPolicy, CompressionRatioParameter
from runtime.controller.algorithmic_mode import AlgorithmicMode
from runtime.evaluation.evaluator import AModelEvaluator
from runtime.model.model_handle import AModelReference, AModelFactory


class LayerSensitivityResults:
    def __init__(self, layer_key: str):
        self._policy_results: dict[str, float] = dict()
        self._layer_key = layer_key

    @property
    def layer_key(self):
        return self._layer_key

    @property
    def policy_results(self) -> dict[str, float]:
        return self._policy_results

    def __setitem__(self, policy_key: str, sensitivity: float):
        self._policy_results[policy_key] = sensitivity

    def __getitem__(self, policy_key):
        return self._policy_results[policy_key]


class SensitivityResults:
    def __init__(self):
        self._layer_results: dict[str, LayerSensitivityResults] = dict()

    @property
    def layer_results(self) -> dict[str, LayerSensitivityResults]:
        return self._layer_results

    def append(self, value: LayerSensitivityResults):
        self._layer_results[value.layer_key] = value

    def __getitem__(self, layer_key):
        if layer_key in self._layer_results:
            return self._layer_results[layer_key]
        return dict()


class APolicySupplier(metaclass=abc.ABCMeta):

    def generate_policies_for_layer(self, layer_key, merged_policy):
        test_policies = dict()
        # per sensitivity test a compression policy is created
        if layer_key not in merged_policy.layers:
            return test_policies
        for method_key, spec in merged_policy.layers[layer_key].items():
            if spec.has_parameters():
                # compression method with at least one compression ratio: apply delta to each and activate
                for parameter in spec.compression_parameters:
                    permuted_parameters = self._create_target_parameters(parameter)
                    for target_key, permuted_p in permuted_parameters.items():
                        target_policy = copy.deepcopy(merged_policy)
                        target_policy.layers[layer_key][method_key].replace_single_parameter(permuted_p)
                        target_policy.activate_compression(layer_key, method_key)
                        test_policies[f"{method_key}-{target_key}"] = target_policy
            else:
                # no compression ratio for compression method -> simply activate
                target_policy = copy.deepcopy(merged_policy)
                target_policy.activate_compression(layer_key, method_key)
                test_policies[method_key] = target_policy

        return test_policies

    @abstractmethod
    def _create_target_parameters(self, compression_parameter):
        pass

    @staticmethod
    def _create_param(test_ratio, compression_parameter):
        # no bounds check here - discretizers will anyway clip
        target = copy.deepcopy(compression_parameter)
        target.compression_ratio = test_ratio
        return target


class DeltaPolicySupplier(APolicySupplier):

    def __init__(self, relative_delta: float = 0.1):
        self._relative_delta = relative_delta

    def _create_target_parameters(self, compression_parameter: CompressionRatioParameter) -> dict[
        str, CompressionRatioParameter]:
        current_ratio = compression_parameter.compression_ratio

        parameter_key = compression_parameter.parameter_key
        delta = self._get_delta(current_ratio)
        return {
            f"{parameter_key}-inc": self._create_param(current_ratio + delta, compression_parameter),
            f"{parameter_key}-dec": self._create_param(current_ratio - delta, compression_parameter)
        }

    def _get_delta(self, current_ratio) -> float:
        if current_ratio == 0.0:
            return self._relative_delta
        return current_ratio * self._relative_delta


class SamplingPolicySupplier(APolicySupplier):

    def __init__(self, sampling_steps: int = 10):
        self._sampling_steps = sampling_steps

    def _create_target_parameters(self, c_param: CompressionRatioParameter):
        parameter_key = c_param.parameter_key
        ratios = np.linspace(c_param.bounds[0],
                             c_param.bounds[1],
                             num=self._sampling_steps,
                             endpoint=False)
        return {
            f"{parameter_key}-{idx}": self._create_param(ratio, c_param) for idx, ratio in enumerate(ratios)
        }


class SensitivityAnalysis:
    def __init__(self,
                 compress_adapter: ACompressAdapter,
                 model_factory: AModelFactory,
                 model_evaluator: AModelEvaluator,
                 alg_mode: AlgorithmicMode,
                 relative_delta: float = 0.1,
                 sampling_steps: int = 10,
                 disabled_analysis: bool = False):
        self._compress_adapter = compress_adapter
        self._model_factory = model_factory
        self._reference_policy = compress_adapter.get_reference_policy()
        self._evaluator = model_evaluator
        self._alg_mode = alg_mode
        self._disabled = disabled_analysis
        self._construct_policy_supplier(relative_delta, sampling_steps)

    def _construct_policy_supplier(self, relative_delta, sampling_steps):
        if self._alg_mode == AlgorithmicMode.ITERATIVE:
            self._policy_supplier = DeltaPolicySupplier(relative_delta)
        else:
            self._policy_supplier = SamplingPolicySupplier(sampling_steps)

    # delta most probably must be adapted for episodes
    def analyse(self, model_reference: AModelReference, last_policy: CompressionPolicy):
        reference_probabilities = self.get_reference_probabilities(model_reference)
        merged_policy = self._merge_policies(last_policy)

        sensitivity_results = SensitivityResults()
        for layer_key in tqdm(merged_policy.layers.keys(), desc="[Sensitivity Analysis]"):
            compression_policies = self._policy_supplier.generate_policies_for_layer(layer_key, merged_policy)
            if self._disabled:
                compression_policies = {}

            layer_results = LayerSensitivityResults(layer_key)
            for policy_key, policy in compression_policies.items():
                sensitivity = self._analyse_for_sensitivity(model_reference, policy, reference_probabilities)

                layer_results[policy_key] = sensitivity
            sensitivity_results.append(layer_results)

        return sensitivity_results

    def get_reference_probabilities(self, model_reference):
        uncompressed_model = self._model_factory.to_executable_model(model_reference)
        reference_probabilities = self._evaluator.sample_log_probabilities(uncompressed_model)
        return reference_probabilities

    def _merge_policies(self, last_policy) -> CompressionPolicy:
        if last_policy is None:
            return self._reference_policy

        merged_policy = copy.deepcopy(last_policy)
        for layer_key, compression_specifications in merged_policy.layers.items():
            reference_specifications = self._reference_policy.layers[layer_key]
            for spec_key, spec in reference_specifications.items():
                if spec_key not in compression_specifications:
                    merged_policy.layers[layer_key][spec_key] = spec
        return merged_policy

    def _analyse_for_sensitivity(self, reference_model, policy, reference_probabilities) -> float:
        _, compressed_model, _ = self._compress_adapter.do_compress(policy, reference_model)
        distorted_probabilities = self._evaluator.sample_log_probabilities(compressed_model)

        result = F.kl_div(distorted_probabilities, reference_probabilities, reduction='batchmean',
                          log_target=True).cpu().item()
        return result
