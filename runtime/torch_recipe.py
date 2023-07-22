import ast

import torch.nn

from runtime.agent.agent import AAgent, AgentWrapper
from runtime.compress.compression_policy import load_policy
from runtime.compress.torch_compress.torch_adapters import TorchCompressAdapter
from runtime.controller.algorithmic_mode import AlgorithmicMode
from runtime.controller.controller import AController
from runtime.controller.independent_controller import IndependentController
from runtime.controller.iterative_controller import IterativeController
from runtime.data.data_provider import CIFAR10Provider
from runtime.data.imagenet_provider import ImageNetDataProvider
from runtime.evaluation.torch_evaluator import TorchOnlyEvaluator
from runtime.evaluation.tvm_evaluator import TvmConfig, TvmLatencyEvaluator
from runtime.feature_extraction.torch_extractor import TorchFeatureExtractor
from runtime.log.logging import LoggingService
from runtime.model.model_handle import AModelReference
from runtime.model.torch_model import TorchModelFactory
from runtime.sensitivity.sensitivity_analysis import SensitivityAnalysis


class TorchConfiguration:

    def __init__(self, target_device: torch.device, **kwargs):
        self.target_device = target_device
        # to use imagenet100 the data folder must contain a reduced dataset - the data provider works for both
        # + set num_classes
        self.data_set = kwargs.pop("data_set", "cifar10")
        self.split_ratio = float(kwargs.pop("split_ratio", 0.2))
        self.data_split_seed = int(kwargs.pop("split_seed", 42))
        self.data_dir = kwargs.pop("data_dir", "./data")
        self.sensitivity_sample_count = int(kwargs.pop("sensitivity_sample_count", 256))
        self.image_net_val_size = kwargs.pop("image_net_val_size", None)
        self.image_net_train_size = kwargs.pop("image_net_train_size", None)
        self.image_net_subset_file = kwargs.pop("image_net_subset_file", None)
        self.sensitivity_relative_delta = float(kwargs.pop("sensitivity_relative_delta", 0.1))
        self.sensitivity_sampling_steps = int(kwargs.pop("sensitivity_sampling_steps", 10))
        self.disable_sensitivity = kwargs.pop("disable_sensitivity", "False") == "True"
        self.num_workers = int(kwargs.pop("num_workers", 8))
        self.mixed_reference_bits = int(kwargs.pop("mixed_reference_bits", 5))

        self.retrain_batch_size = int(kwargs.pop("retrain_batch_size", 256))
        self.retrain_epochs = int(kwargs.pop("retrain_epochs", 1))
        self.retrain_lr = float(kwargs.pop("retrain_lr", 0.05))
        self.retrain_mom = float(kwargs.pop("retrain_mom", 0.9))
        self.retrain_weight_decay = float(kwargs.pop("retrain_weight_decay", 5e-4))

        self.search_identifier = kwargs.pop("search_identifier", "torch-cifar10")
        self.log_dir = kwargs.pop("log_dir", "./logs")
        self.frozen_layers = ast.literal_eval(
            kwargs.pop("frozen_layers", "{'p-lin': ['fc', 'classifier.1'], 'q-mixed': ['fc', 'classifier.1']}"))
        self.p_channel_round_to = int(kwargs.pop("p_channel_round_to", 1))
        self.num_classes = int(kwargs.pop("num_classes", 10))

        self.step_mode = kwargs.pop("step_mode", "layer_passing")
        self.enabled_methods = kwargs.pop("enabled_methods", tuple(["p-conv", "p-lin", "q-fp32", "q-int8", "q-mixed"]))
        self.eval_latency = kwargs.pop("enable_latency_eval", "True") == "True"
        self.alternative_reference_policy = kwargs.pop("alternative_reference_policy", None)
        self.tvm_config = TvmConfig(**kwargs)
        self.tvm_config_dict = vars(self.tvm_config)
        self.x86_mixed_check = kwargs.pop("x86_mixed_check", "False") == "True"


class TorchRecipe:

    def __init__(self,
                 agent: AAgent,
                 config: TorchConfiguration):
        self._agent = agent
        self._config = config

    def _get_controller(self, **args) -> AController:
        if self._agent.alg_mode() == AlgorithmicMode.ITERATIVE:
            return IterativeController(**args)
        return IndependentController(**args)

    def _get_reference_policy(self, compress_adapter):
        reference_policy = compress_adapter.get_reference_policy()
        if self._config.alternative_reference_policy:
            other_policy = load_policy(self._config.alternative_reference_policy)
            reference_policy.merge_policies(other_policy)
        return reference_policy

    def _get_enabled_compression_methods(self) -> tuple:
        enabled_methods = self._config.enabled_methods
        if self._config.alternative_reference_policy:
            additional_methods = load_policy(self._config.alternative_reference_policy).get_included_methods()
            return tuple(set(enabled_methods).union(set(additional_methods)))
        return enabled_methods

    def _create_data_provider(self, **kwargs):
        if self._config.data_set == "imagenet":
            return ImageNetDataProvider(**kwargs)
        if self._config.data_set == "cifar10":
            return CIFAR10Provider(**kwargs)
        raise Exception(f"Data set {self._config.data_set} not implemented")

    def construct_application(self, torch_model: torch.nn.Module) -> tuple[AController, AModelReference]:
        data_provider = self._create_data_provider(target_device=self._config.target_device,
                                                   data_dir=self._config.data_dir,
                                                   batch_size=self._config.retrain_batch_size,
                                                   sensitivity_sample_count=self._config.sensitivity_sample_count,
                                                   seed=self._config.data_split_seed,
                                                   num_workers=self._config.num_workers,
                                                   split_ratio=self._config.split_ratio,
                                                   train_size=self._config.image_net_train_size,
                                                   val_size=self._config.image_net_val_size,
                                                   subset_classes_file=self._config.image_net_subset_file
                                                   )
        logging_service = LoggingService(search_identifier=self._config.search_identifier,
                                         log_dir=self._config.log_dir)
        model_factory = TorchModelFactory(torch_model=torch_model,
                                          batch_input_shape=data_provider.batch_input_shape,
                                          target_device=self._config.target_device,
                                          frozen_layers=self._config.frozen_layers)
        model_reference = model_factory.get_reference_model()
        compress_adapter = TorchCompressAdapter(model_reference=model_reference,
                                                model_factory=model_factory,
                                                enabled_methods=self._get_enabled_compression_methods(),
                                                channel_round_to=self._config.p_channel_round_to,
                                                mixed_reference_bits=self._config.mixed_reference_bits,
                                                x86_mixed_check=self._config.x86_mixed_check)
        model_evaluator = TorchOnlyEvaluator(data_provider=data_provider,
                                             logging_service=logging_service,
                                             target_device=self._config.target_device,
                                             retrain_epochs=self._config.retrain_epochs,
                                             retrain_lr=self._config.retrain_lr,
                                             retrain_mom=self._config.retrain_mom,
                                             retrain_weight_decay=self._config.retrain_weight_decay)
        latency_evaluator = TvmLatencyEvaluator(data_provider, self._config.tvm_config)
        sensitivity_analysis = SensitivityAnalysis(compress_adapter=compress_adapter,
                                                   model_factory=model_factory,
                                                   model_evaluator=model_evaluator,
                                                   alg_mode=self._agent.alg_mode(),
                                                   relative_delta=self._config.sensitivity_relative_delta,
                                                   sampling_steps=self._config.sensitivity_sampling_steps,
                                                   disabled_analysis=self._config.disable_sensitivity)
        feature_extractor = TorchFeatureExtractor(data_provider=data_provider,
                                                  sensitivity_analysis=sensitivity_analysis)
        agent_wrapper = AgentWrapper(agent=self._agent)
        controller = self._get_controller(feature_extractor=feature_extractor,
                                          agent_wrapper=agent_wrapper,
                                          compress_adapter=compress_adapter,
                                          model_evaluator=model_evaluator,
                                          latency_evaluator=latency_evaluator,
                                          model_factory=model_factory,
                                          logging_service=logging_service,
                                          step_mode=self._config.step_mode,
                                          reference_policy=self._get_reference_policy(compress_adapter),
                                          eval_latency=self._config.eval_latency)
        return controller, model_reference
