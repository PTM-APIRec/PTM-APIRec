import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Loss, TopKCategoricalAccuracy
from ignite.contrib.handlers.tensorboard_logger import *
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.handlers import ModelCheckpoint, EarlyStopping, Checkpoint
from ignite.utils import setup_logger, convert_tensor

from typing import Dict, List
from argparse import Namespace


class Trainer:
    def __init__(self,
                 model: nn.Module,
                 api_dict: Dict[str, int],
                 class_dict: Dict[str, int],
                 class_to_api_dict: Dict[int, List[int]],
                 train_loader,
                 valid_loader,
                 args: Namespace):
        self.model: nn.Module = model
        self.api_dict: Dict = api_dict
        self.class_dict: Dict = class_dict
        self.class_to_api_dict: Dict[int, List[int]] = class_to_api_dict
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.args = args

    def train(self):
        train_loader = self.train_loader
        valid_loader = self.valid_loader
        device = torch.device(self.args.device)
        if self.args.data_parallel:
            self.model = nn.DataParallel(self.model)
        self.model.to(device)

        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        criterion = nn.CrossEntropyLoss()
        trainer = create_supervised_trainer(self.model, optimizer, criterion, device)
        pbar = ProgressBar()
        pbar.attach(trainer)

        metrics = {"top-1_acc": TopKCategoricalAccuracy(k=1),
                   "top-2_acc": TopKCategoricalAccuracy(k=2),
                   "top-3_acc": TopKCategoricalAccuracy(k=3),
                   "top-5_acc": TopKCategoricalAccuracy(k=5),
                   "top-10_acc": TopKCategoricalAccuracy(k=10),
                   "loss": Loss(criterion)}

        to_save = {
            "model": self.model,
            "optimizer": optimizer,
            "trainer": trainer
        }

        train_evaluator = create_supervised_evaluator(self.model, metrics, device)
        train_evaluator.logger = setup_logger("Train Evaluator")
        validation_evaluator = create_supervised_evaluator(self.model, metrics, device)
        validation_evaluator.logger = setup_logger("Validation Evaluator")

        model_sig = self.args.model_sig
        save_handler = Checkpoint(to_save,
                                  "my_models/" + model_sig,
                                  n_saved=10,
                                  global_step_transform=lambda e, _: e.state.epoch)
        trainer.add_event_handler(Events.EPOCH_COMPLETED(every=2), save_handler)
        early_stop_handler = EarlyStopping(patience=10, score_function=self.score_function, trainer=trainer)
        validation_evaluator.add_event_handler(Events.COMPLETED, early_stop_handler)

        @trainer.on(Events.EPOCH_COMPLETED)
        def compute_metrics(engine):
            train_evaluator.run(train_loader)
            validation_evaluator.run(valid_loader)

        tb_logger = TensorboardLogger(self.args.log_dir + "/" + model_sig + "/")
        tb_logger.attach_output_handler(trainer,
                                        event_name=Events.ITERATION_COMPLETED(every=100),
                                        tag="training",
                                        output_transform=lambda loss: {"batch_loss": loss},
                                        metric_names="all")

        for tag, evaluator in [('training', train_evaluator), ("validation", validation_evaluator)]:
            tb_logger.attach_output_handler(evaluator,
                                            event_name=Events.EPOCH_COMPLETED,
                                            tag=tag,
                                            metric_names=["top-1_acc",
                                                          "top-2_acc",
                                                          "top-3_acc",
                                                          "top-5_acc",
                                                          "top-10_acc",
                                                          "loss"],
                                            global_step_transform=global_step_from_engine(trainer))

        trainer.run(train_loader, max_epochs=self.args.max_epoch)
        with open('./signal_finished', 'w+') as fp:
            fp.write("OK")
        tb_logger.close()

    @staticmethod
    def score_function(engine):
        top_1_acc = engine.state.metrics['top-1_acc']
        return top_1_acc

