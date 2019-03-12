import os
import logging
import numpy as np
from tqdm.auto import tqdm
import warnings
from collections import OrderedDict
from batchgenerators.dataloading import MultiThreadedAugmenter
from .callbacks import AbstractCallback
from .abstract_trainer import AbstractNetworkTrainer
from functools import partial

from delira import get_backends

logger = logging.getLogger(__name__)

if "TORCH" in get_backends():
    import torch
    from .train_utils import torch_convert_to_numpy
    from .train_utils import create_optims_default_pytorch as create_optims_default
    from ..io.torch import load_checkpoint, save_checkpoint
    from ..models import AbstractPyTorchNetwork

    class PyTorchNetworkTrainer(AbstractNetworkTrainer):
        """
        Train and Validate a Network

        See Also
        --------
        :class:`AbstractNetwork`

        """

        def __init__(self,
                     network: AbstractPyTorchNetwork,
                     save_path: str,
                     losses=None,
                     optimizer_cls=None,
                     optimizer_params={},
                     train_metrics={},
                     val_metrics={},
                     val_dataset_metrics={},
                     lr_scheduler_cls=None,
                     lr_scheduler_params={},
                     gpu_ids=[],
                     save_freq=1,
                     optim_fn=create_optims_default,
                     fold=0,
                     callbacks=[],
                     start_epoch=1,
                     convert_batch_to_npy_fn=torch_convert_to_numpy,
                     mixed_precision=False,
                     mixed_precision_kwargs={"enable_caching": True,
                                             "verbose": False,
                                             "allow_banned": False},
                     criterions=None,
                     ** kwargs):
            """

            Parameters
            ----------
            network : :class:`AbstractPyTorchNetwork`
                the network to train
            save_path : str
                path to save networks to
            losses : dict
                dictionary containing the training losses
            optimizer_cls : subclass of torch.optim.Optimizer
                optimizer class implementing the optimization algorithm of choice
            optimizer_params : dict
                keyword arguments passed to optimizer during construction
            metrics : dict
                dictionary containing the validation metrics
            lr_scheduler_cls : Any
                learning rate schedule class: must implement step() method
            lr_scheduler_params : dict
                keyword arguments passed to lr scheduler during construction
            gpu_ids : list
                list containing ids of GPUs to use; if empty: use cpu instead
            save_freq : int
                integer specifying how often to save the current model's state.
                State is saved every state_freq epochs
            optim_fn : function
                creates a dictionary containing all necessary optimizers
            fold : int
                current cross validation fold (0 per default)
            callbacks : list
                initial callbacks to register
            start_epoch : int
                epoch to start training at
            mixed_precision : bool
                whether to use mixed precision or not (False per default)
            mixed_precision_kwargs : dict
                additional keyword arguments for mixed precision
            **kwargs :
                additional keyword arguments

            """

            if (criterions is not None) ^ (losses is not None):
                if losses is not None:
                    crits = losses
                elif criterions is not None:
                    warnings.warn(DeprecationWarning(
                        "The 'criterions' argument is deprecated and will be \
                        removed in next release to unify APIs across backends. \
                        Use 'losses' instead "))
                    crits = criterions

            else:
                crits = losses
                warnings.warn(
                    RuntimeWarning("'criterions' and 'losses' have \
                                    been specified.Using the values in \
                                    'losses' since 'criterions' is deprecated \
                                    and will be removed"))

            super().__init__(
                network, save_path, crits, optimizer_cls, optimizer_params,
                train_metrics, val_metrics, val_dataset_metrics,
                lr_scheduler_cls, lr_scheduler_params, gpu_ids, save_freq,
                optim_fn, fold, callbacks, start_epoch, convert_batch_to_npy_fn)

            self._setup(network, optim_fn, optimizer_cls, optimizer_params,
                        lr_scheduler_cls, lr_scheduler_params, gpu_ids,
                        convert_batch_to_npy_fn,
                        mixed_precision, mixed_precision_kwargs)

            for key, val in kwargs.items():
                setattr(self, key, val)

        def _setup(self, network, optim_fn, optimizer_cls, optimizer_params,
                   lr_scheduler_cls, lr_scheduler_params, gpu_ids,
                   convert_batch_to_npy_fn, mixed_precision,
                   mixed_precision_kwargs):
            """
            Defines the Trainers Setup

            Parameters
            ----------
            network : :class:`AbstractPyTorchNetwork`
                the network to train
            optim_fn : function
                creates a dictionary containing all necessary optimizers
            optimizer_cls : subclass of torch.optim.Optimizer
                optimizer class implementing the optimization algorithm of choice
            optimizer_params : dict
            lr_scheduler_cls : Any
                learning rate schedule class: must implement step() method
            lr_scheduler_params : dict
                keyword arguments passed to lr scheduler during construction
            gpu_ids : list
                list containing ids of GPUs to use; if empty: use cpu instead
            mixed_precision : bool
                whether to use mixed precision or not (False per default)
            mixed_precision_kwargs : dict
                additional keyword arguments for mixed precision

            """

            self.optimizers = optim_fn(network, optimizer_cls,
                                       **optimizer_params)

            super()._setup(network, lr_scheduler_cls, lr_scheduler_params,
                           gpu_ids, convert_batch_to_npy_fn,
                           network.prepare_batch)

            try:
                from apex import amp
                self._amp_handle = amp.init(mixed_precision,
                                            *mixed_precision_kwargs)
                wrap_fn = self._amp_handle.wrap_optimizer

            except ImportError:
                if mixed_precision:
                    logger.warning("Apex was not found found, trying to continue \
                                    in full precision instead")
                from ..utils.context_managers import DefaultOptimWrapperTorch
                wrap_fn = DefaultOptimWrapperTorch

            # wrap optimizers by half_precision_optimizer via apex if necessary
            self.optimizers = {k: wrap_fn(
                v, num_loss=len(self.losses)) for k, v
                in self.optimizers.items()}

            # Load latest epoch file if available
            if os.path.isdir(self.save_path):
                # check all files in directory starting with "checkpoint" and not
                # ending with "_best.pth"
                files = [x for x in os.listdir(self.save_path)
                         if os.path.isfile(os.path.join(self.save_path, x))
                         and x.startswith("checkpoint")
                         and not (x.endswith("_best.pth")
                                  or x.endswith("_best.pt"))]

                # if list is not empty: load previous state
                if files:

                    latest_epoch = max([
                        int(x.rsplit("_", 1)[-1].rsplit(".", 1)[0])
                        for x in files])

                    latest_state_path = os.path.join(self.save_path,
                                                     "checkpoint_epoch_%d.pth"
                                                     % latest_epoch)

                    # if pth file does not exist, load pt file instead
                    if not os.path.isfile(latest_state_path):
                        latest_state_path = latest_state_path[:-1]

                    logger.info("Attempting to load state from previous \
                                training from %s" % latest_state_path)
                    try:
                        self.update_state(latest_state_path)
                    except KeyError:
                        logger.warn("Previous State could not be loaded, \
                                    although it exists.Training will be \
                                    restarted")

            if gpu_ids and torch.cuda.is_available():
                self.use_gpu = True
                if (len(gpu_ids) > 1) and (torch.cuda.device_count() > 1):
                    # use GPU 0 as default input GPU
                    self.input_device = torch.device("cuda:%d" % gpu_ids[0])

                    # Train on multiple GPUs and use GPU 0 as output device
                    self.module = torch.nn.DataParallel(self.module.to(
                        self.input_device),
                        device_ids=gpu_ids,
                        output_device=gpu_ids[1])

                    # use GPU 1 as default output GPU for balanced GPU usage
                    self.output_device = torch.device("cuda:%d" % gpu_ids[1])
                else:
                    # use the only available GPU as input device
                    self.input_device = torch.device("cuda:%d" % gpu_ids[0])
                    self.module = self.module.to(self.input_device)

                    # use GPU 0 as output device as output device
                    self.output_device = torch.device("cuda:%d" % gpu_ids[0])
            else:
                self.use_gpu = False
                self.input_device = torch.device("cpu")
                self.output_device = torch.device("cpu")
                self.module = self.module.to(self.input_device)

            self._prepare_batch = partial(
                self._prepare_batch, input_device=self.input_device,
                output_device=self.output_device)

        def _at_training_begin(self, *args, **kwargs):
            """
            Defines behaviour at beginning of training

            Parameters
            ----------
            *args :
                positional arguments
            **kwargs :
                keyword arguments

            """
            self.save_state(os.path.join(
                self.save_path, "checkpoint_epoch_0"), 0)

        def _at_training_end(self):
            """
            Defines Behaviour at end of training: Loads best model if available

            Returns
            -------
            :class:`AbstractPyTorchNetwork`
                best network

            """
            if os.path.isfile(os.path.join(self.save_path, 'checkpoint_best.pt')):

                # load best model and return it
                self.update_state(os.path.join(self.save_path,
                                               'checkpoint_best.pt'))

            return self.module

        def _at_epoch_begin(self, metrics_val, val_score_key, epoch, num_epochs,
                            **kwargs):
            """
            Defines behaviour at beginning of each epoch: Executes all callbacks's
            `at_epoch_begin` method

            Parameters
            ----------
            metrics_val : dict
                validation metrics
            val_score_key : str
                validation score key
            epoch : int
                current epoch
            num_epochs : int
                total number of epochs
            **kwargs :
                keyword arguments

            """

            # execute all callbacks
            for cb in self._callbacks:
                self._update_state(cb.at_epoch_begin(self, val_metrics=metrics_val,
                                                     val_score_key=val_score_key,
                                                     curr_epoch=epoch))

        def _at_epoch_end(self, metrics_val, val_score_key, epoch, is_best,
                          **kwargs):
            """
            Defines behaviour at beginning of each epoch: Executes all callbacks's
            `at_epoch_end` method and saves current state if necessary

            Parameters
            ----------
            metrics_val : dict
                validation metrics
            val_score_key : str
                validation score key
            epoch : int
                current epoch
            num_epochs : int
                total number of epochs
            is_best : bool
                whether current model is best one so far
            **kwargs :
                keyword arguments

            """

            for cb in self._callbacks:
                self._update_state(cb.at_epoch_end(self, val_metrics=metrics_val,
                                                   val_score_key=val_score_key,
                                                   curr_epoch=epoch))

            if epoch % self.save_freq == 0:
                self.save_state(os.path.join(self.save_path,
                                             "checkpoint_epoch_%d.pt" % epoch),
                                epoch)

            if is_best:
                self.save_state(os.path.join(self.save_path,
                                             "checkpoint_best.pt"),
                                epoch)

        def _train_single_epoch(self, batchgen: MultiThreadedAugmenter, epoch,
                                verbose=False):
            """
            Trains the network a single epoch

            Parameters
            ----------
            batchgen : MultiThreadedAugmenter
                Generator yielding the training batches
            epoch : int
                current epoch

            """

            self.module.train()

            return super()._train_single_epoch(batchgen, epoch, verbose=verbose)

        def save_state(self, file_name, epoch, **kwargs):
            """
            saves the current state via :func:`delira.io.torch.save_checkpoint`

            Parameters
            ----------
            file_name : str
                filename to save the state to
            epoch : int
                current epoch (will be saved for mapping back)
            *args :
                positional arguments
            **kwargs :
                keyword arguments

            """
            if not (file_name.endswith(".pth") or file_name.endswith(".pt")):
                file_name = file_name + ".pt"
            save_checkpoint(file_name, self.module, self.optimizers,
                            **kwargs)

        @staticmethod
        def load_state(file_name, **kwargs):
            """
            Loads the new state from file via :func:`delira.io.torch.load_checkpoint`

            Parameters
            ----------
            file_name : str
                the file to load the state from
            **kwargs : keyword arguments

            Returns
            -------
            dict
                new state

            """

            if not (file_name.endswith(".pth") or file_name.endswith(".pt")):
                file_name = file_name + ".pt"

            return load_checkpoint(file_name, **kwargs)

        def update_state(self, file_name, *args, **kwargs):
            """
            Update internal state from a loaded state

            Parameters
            ----------
            file_name : str
                file containing the new state to load
            *args :
                positional arguments
            **kwargs :
                keyword arguments

            Returns
            -------
            :class:`AbstractNetworkTrainer`
                the trainer with a modified state

            """
            self._update_state(self.load_state(file_name, *args, **kwargs))

        def _update_state(self, new_state):
            """
            Update the state from a given new state

            Parameters
            ----------
            new_state : dict
                new state to update internal state from

            Returns
            -------
            :class:`PyTorchNetworkTrainer`
                the trainer with a modified state

            """

            if "model" in new_state:
                self.module.load_state_dict(new_state.pop("model"))

            if "optimizer" in new_state and new_state["optimizer"]:
                optim_state = new_state.pop("optimizer")
                for key in self.optimizers.keys():
                    self.optimizers[key].load_state_dict(
                        optim_state[key])

            if "epoch" in new_state:
                self.start_epoch = new_state.pop("epoch")

            return super()._update_state(new_state)
