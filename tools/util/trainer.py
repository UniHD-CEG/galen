import copy
import pickle
from pathlib import Path

import torch
import wandb
from tqdm import tqdm

from runtime.data.data_provider import ADataProvider


class Log:

    def __init__(self, log_dir: str, identifier: str):
        self._logs_dict = dict()
        self._log_dir = log_dir
        self._identifier = identifier

    def append(self, new_entry_dict):
        for key, value in new_entry_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.cpu().detach().item()
            if key in self._logs_dict:
                self._logs_dict[key].append(value)
            else:
                self._logs_dict[key] = [value]

    def store(self, additional_logs):
        self._logs_dict.update(additional_logs)
        path = Path(self._log_dir)
        path.mkdir(exist_ok=True, parents=True)
        path = path / (self._identifier + "_retrain.pickle")
        with open(path, "wb") as file:
            pickle.dump(self._logs_dict, file)


class Trainer:
    def __init__(self,
                 data_provider: ADataProvider,
                 target_device: torch.device,
                 train_epochs: int = 50,
                 train_lr=0.001,
                 train_mom=0.4,
                 weight_decay=5e-4,
                 use_adadelta=False,
                 store_dir="./results/checkpoints",
                 log_dir="./logs/train",
                 log_file_name="train.out",
                 add_identifier="",
                 model_name="model"):
        self._validate_episodes = 1
        self._data_provider = data_provider
        self._target_device = target_device
        self._train_epochs = train_epochs
        self._train_lr = train_lr
        self._train_mom = train_mom
        self._weight_decay = weight_decay
        self._criterion = torch.nn.CrossEntropyLoss()
        self._store_dir = store_dir
        self._add_identifier = add_identifier
        self._model_name = model_name
        self._use_adadelta = use_adadelta
        self._logs = Log(log_dir, log_file_name)

    def train(self, model: torch.nn.Module):
        if self._use_adadelta:
            optimizer = torch.optim.Adadelta(model.parameters())
            scheduler = None
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=self._train_lr, momentum=self._train_mom,
                                        weight_decay=self._weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self._train_epochs)

        train_loader = self._data_provider.train_loader
        best_acc = 0.0
        best_model = model
        model.to(self._target_device)
        for epoch in tqdm(range(self._train_epochs), desc="[Train Epochs]", dynamic_ncols=True):
            batch_losses = torch.zeros(len(train_loader))
            batch_acc = torch.zeros(len(train_loader))
            model.train()
            for batch_idx, data in tqdm(enumerate(train_loader), desc=f"[Train Epoch {epoch}]",
                                        total=len(train_loader), dynamic_ncols=True):
                inputs, targets = data
                inputs = inputs.to(self._target_device)
                targets = targets.to(self._target_device)

                optimizer.zero_grad()
                logits = model(inputs)
                loss = self._criterion(logits, targets)

                loss.backward()
                optimizer.step()
                preds = torch.argmax(logits, dim=1)
                batch_acc[batch_idx] = ((preds == targets).sum() / len(preds)).detach().cpu()
                batch_losses[batch_idx] = loss.detach().cpu()
                self.log_train(batch_acc, batch_idx, batch_losses, epoch, scheduler)

            if not self._use_adadelta:
                scheduler.step()
            if epoch % self._validate_episodes == 0:
                with torch.no_grad():
                    val_acc = self.validate(model)
                    if val_acc > best_acc:
                        best_model = copy.deepcopy(model)
                        best_acc = val_acc
                        self.store_model(model, epoch)
        return best_model

    def log_train(self, batch_acc, batch_idx, batch_losses, epoch, scheduler):
        if scheduler:
            lr = scheduler.get_lr()[0]
        else:
            lr = 0.0
        batch_logs = {
            'loss': batch_losses[batch_idx],
            'acc': batch_acc[batch_idx],
            'epoch': epoch,
            'lr': lr
        }
        self._logs.append(batch_logs)
        wandb.log(batch_logs)

    def store_model(self, model, epoch_idx):
        path = Path(self._store_dir)
        path.mkdir(exist_ok=True, parents=True)
        path = path / (self.create_identifier(epoch_idx) + ".pth")
        torch.save(model.state_dict(), path)

    def create_identifier(self, epoch_idx):
        return f"{self._model_name}_{self._add_identifier}_lr{self._train_lr}_mom{self._train_mom}_ep{epoch_idx}"

    def validate(self, model):
        return self._validate(model, self._data_provider.val_loader, "val")

    def test(self, model):
        return self._validate(model, self._data_provider.test_loader, "test")

    def store_logs(self, additional_logs):
        self._logs.store(additional_logs)

    def _validate(self, model, data_loader, prefix):
        model.to(self._target_device)
        model.eval()
        with torch.no_grad():
            batch_acc = torch.zeros(len(data_loader))
            batch_loss = torch.zeros(len(data_loader))
            for batch_idx, data in tqdm(enumerate(data_loader), total=len(data_loader), desc="[Validate]",
                                        dynamic_ncols=True):
                inputs, targets = data
                inputs = inputs.to(self._target_device)
                targets = targets.to(self._target_device)

                logits = model(inputs)
                preds = torch.argmax(logits, dim=1)

                batch_acc[batch_idx] = ((preds == targets).sum() / len(preds)).detach().cpu()
                batch_loss[batch_idx] = self._criterion(logits, targets)

            acc = torch.mean(batch_acc)
            loss = torch.mean(batch_loss)
            self.log_test(acc, loss, prefix)
            return acc

    def log_test(self, acc, loss, prefix):
        test_logs = {
            f"{prefix}-loss": loss,
            f"{prefix}-acc": acc
        }
        self._logs.append(test_logs)
        wandb.log(test_logs)
