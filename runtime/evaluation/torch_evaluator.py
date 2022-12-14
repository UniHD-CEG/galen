import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from runtime.compress.compression_policy import CompressionProtocolEntry
from runtime.data.data_provider import ADataProvider
from runtime.evaluation.evaluator import AModelEvaluator
from runtime.feature_extraction.torch_extractor import TorchMACsBOPsExtractor
from runtime.log.logging import LoggingService
from runtime.model.torch_model import TorchExecutableModel


class TorchOnlyEvaluator(AModelEvaluator):

    def __init__(self,
                 data_provider: ADataProvider,
                 logging_service: LoggingService,
                 target_device: torch.device,
                 retrain_epochs=10,
                 retrain_lr=0.001,
                 retrain_mom=0.4,
                 retrain_weight_decay=5e-4):
        self._data_provider = data_provider
        self._logging_service = logging_service
        self._target_device = target_device
        self._retrain_epochs = retrain_epochs
        self._retrain_lr = retrain_lr
        self._retrain_mom = retrain_mom
        self._retrain_weight_decay = retrain_weight_decay

        self._mac_extractor = TorchMACsBOPsExtractor(data_provider.get_random_tensor_with_input_shape())

    def retrain(self, executable_model: TorchExecutableModel):
        pytorch_model = executable_model.pytorch_model
        pytorch_model.to(self._target_device)
        pytorch_model.train()

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(pytorch_model.parameters(), lr=self._retrain_lr, momentum=self._retrain_mom,
                                    weight_decay=self._retrain_weight_decay)

        train_loader = self._data_provider.train_loader
        epoch_losses = torch.zeros(self._retrain_epochs)
        epoch_acc = torch.zeros(self._retrain_epochs)
        for epoch in tqdm(range(self._retrain_epochs), desc="[Retrain Epochs]", dynamic_ncols=True):
            batch_losses = torch.zeros(len(train_loader))
            batch_acc = torch.zeros((len(train_loader)))
            for batch_idx, data in tqdm(enumerate(train_loader), desc=f"[Retrain Epoch {epoch}]",
                                        total=len(train_loader), dynamic_ncols=True):
                inputs, targets = data
                inputs = inputs.to(self._target_device)
                targets = targets.to(self._target_device)

                optimizer.zero_grad()
                logits = pytorch_model(inputs)
                loss = criterion(logits, targets)

                loss.backward()
                optimizer.step()
                batch_acc[batch_idx] = self._acc(logits.detach(), targets.detach(), topk=(1,)).item()
                batch_losses[batch_idx] = loss.detach().cpu()
            epoch_losses[epoch] = torch.mean(batch_losses)
            epoch_acc[epoch] = torch.mean(batch_acc)
            self._logging_service.retrain_epoch_completed(batch_losses.numpy(), batch_acc.numpy())
        self._logging_service.retrain_completed(epoch_losses.numpy(), epoch_acc.numpy())

    @staticmethod
    def _acc(preds, targets, topk=(1, 5)):
        with torch.no_grad():
            maxk = max(topk)
            batch_size = targets.size(0)

            _, pred = preds.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(targets.view(1, -1).expand_as(pred))

            res = torch.zeros((len(topk)))
            for i, k in enumerate(topk):
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res[i] = correct_k / batch_size
            return res

    def evaluate(self, executable_model: TorchExecutableModel,
                 compression_protocol: list[CompressionProtocolEntry]) -> dict[str, float]:

        pytorch_model = executable_model.pytorch_model
        pytorch_model.to(self._target_device)
        pytorch_model.eval()

        with torch.no_grad():
            val_loader = self._data_provider.val_loader
            batch_acc = torch.zeros((2, len(val_loader)))
            for batch_idx, data in tqdm(enumerate(val_loader), total=len(val_loader), desc="[Validate Accuracy]",
                                        dynamic_ncols=True):
                inputs, targets = data
                inputs = inputs.to(self._target_device)
                targets = targets.to(self._target_device)

                logits = pytorch_model(inputs)

                batch_acc[:, batch_idx] = self._acc(logits.detach(), targets.detach())
            acc = torch.mean(batch_acc, dim=1)
        evaluation_metrics = {
            "acc": acc[0].item(),
            "acc_top5": acc[1].item()
        }
        evaluation_metrics.update(self._compute_with_extractor(executable_model, self._mac_extractor))
        # for now skip mac and bop extractions for phase steps
        return evaluation_metrics

    def sample_log_probabilities(self, executable_model: TorchExecutableModel) -> torch.Tensor:
        pytorch_model = executable_model.pytorch_model
        pytorch_model.to(self._target_device)

        pytorch_model.eval()
        with torch.no_grad():
            all_outputs = list()
            sens_loader = self._data_provider.sens_loader
            with torch.no_grad():
                for batch_idx, inputs in enumerate(sens_loader):
                    inputs = inputs[0].to(self._target_device)
                    logits = pytorch_model(inputs)
                    outputs = F.log_softmax(logits, dim=1).detach()
                    all_outputs.append(outputs)

                return torch.cat(all_outputs, dim=0)

    @staticmethod
    def _compute_with_extractor(executable_model: TorchExecutableModel, extractor):
        all_layer_metrics = dict()
        with extractor(executable_model) as ex:
            for layer_idx, layer_key in enumerate(executable_model.all_layer_keys()):
                layer_metrics = ex.compute_metric_for_layer(layer_key)
                for metric_key, metric_value in layer_metrics.items():
                    if metric_key not in all_layer_metrics:
                        all_layer_metrics[metric_key] = [metric_value]
                    else:
                        all_layer_metrics[metric_key].append(metric_value)
        return {m_key: np.sum(m_val) for m_key, m_val in all_layer_metrics.items()}
