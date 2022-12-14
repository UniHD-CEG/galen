from abc import ABCMeta, abstractmethod

from runtime.agent.agent import AgentWrapper
from runtime.compress.compress_adapters import ACompressAdapter
from runtime.compress.compression_policy import CompressionPolicy
from runtime.evaluation.evaluator import AModelEvaluator, ALatencyEvaluator
from runtime.feature_extraction.feature_extractor import AFeatureExtractor
from runtime.log.logging import LoggingService
from runtime.model.model_handle import AModelFactory, AModelReference


class AController(metaclass=ABCMeta):
    def __init__(self,
                 feature_extractor: AFeatureExtractor,
                 agent_wrapper: AgentWrapper,
                 compress_adapter: ACompressAdapter,
                 model_evaluator: AModelEvaluator,
                 latency_evaluator: ALatencyEvaluator,
                 model_factory: AModelFactory,
                 logging_service: LoggingService,
                 reference_policy: CompressionPolicy,
                 step_mode: str,
                 eval_latency=False):
        self._feature_extractor = feature_extractor
        self._agent_wrapper = agent_wrapper
        self._compress_adapter = compress_adapter
        self._model_evaluator = model_evaluator
        self._latency_evaluator = latency_evaluator
        self._model_factory = model_factory
        self._logging_service = logging_service
        self._reference_policy = reference_policy
        self._step_mode = step_mode
        self._eval_latency = eval_latency
        self._steps_per_episode = self._compute_steps(reference_policy)

    @abstractmethod
    def search(self, number_of_episodes, model_reference):
        pass

    def _compress_and_evaluate(self, policy: CompressionPolicy, model_reference: AModelReference, episode=False):
        compression_protocol, executable_model = self._compress_adapter.compress(policy,
                                                                                 model_reference)
        evaluation = dict()
        if episode:
            self._model_evaluator.retrain(executable_model)
            evaluation.update(self._lat_eval(executable_model))

        if episode or self._agent_wrapper.requires_step_eval():
            evaluation.update(self._model_evaluator.evaluate(executable_model, compression_protocol))
        return executable_model, evaluation

    def _lat_eval(self, executable_model):
        lat_eval = dict()
        if self._eval_latency:
            lat_eval = self._latency_evaluator.measure_latency(executable_model)
        return lat_eval

    def _compute_steps(self, reference_policy: CompressionPolicy) -> int:
        if self._step_mode == "layer_passing":
            return len(reference_policy.layers.keys())
        return 1

    def _is_retraining_required(self, step):
        return step == (self._steps_per_episode - 1)
