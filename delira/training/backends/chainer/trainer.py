import os
import logging
from functools import partial
logger = logging.getLogger(__name__)

from batchgenerators.dataloading import MultiThreadedAugmenter
import chainer

from delira.training.base_trainer import BaseNetworkTrainer
from delira.models import AbstractChainerNetwork, DataParallelChainerNetwork, \
    DataParallelChainerOptimizer
from delira.io.chainer import load_checkpoint, save_checkpoint

from .utils import convert_to_numpy, create_optims_default


class ChainerNetworkTrainer(BaseNetworkTrainer):
    """
    Train and Validate a Network

    See Also
    --------
    :class:`AbstractNetwork`

    """

    def __init__(self,
                 network: AbstractChainerNetwork,
                 save_path: str,
                 key_mapping,
                 losses=None,
                 optimizer_cls=None,
                 optimizer_params={},
                 train_metrics={},
                 val_metrics={},
                 lr_scheduler_cls=None,
                 lr_scheduler_params={},
                 gpu_ids=[],
                 save_freq=1,
                 optim_fn=create_optims_default,
                 logging_type="tensorboardx",
                 logging_kwargs={},
                 fold=0,
                 callbacks=[],
                 start_epoch=1,
                 metric_keys=None,
                 convert_batch_to_npy_fn=convert_to_numpy,
                 mixed_precision=False,
                 val_freq=1,
                 ** kwargs):
        """

        Parameters
        ----------
        network : :class:`AbstractChainerNetwork`
            the network to train
        save_path : str
            path to save networks to
        key_mapping : dict
            a dictionary containing the mapping from the ``data_dict`` to
            the actual model's inputs.
            E.g. if a model accepts one input named 'x' and the data_dict
            contains one entry named 'data' this argument would have to
            be ``{'x': 'data'}``
        losses : dict
            dictionary containing the training losses
        optimizer_cls : subclass of chainer.Optimizer
            optimizer class implementing the optimization algorithm of
            choice
        optimizer_params : dict
            keyword arguments passed to optimizer during construction
        train_metrics : dict, optional
            metrics, which will be evaluated during train phase
            (should work on framework's tensor types)
        val_metrics : dict, optional
            metrics, which will be evaluated during test phase
            (should work on numpy arrays)
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
        logging_type : str or callable
            the type of logging. If string: it must be one of
            ["visdom", "tensorboardx"]
            If callable: it must be a logging handler class
        logging_kwargs : dict
            dictionary containing all logging keyword arguments
        fold : int
            current cross validation fold (0 per default)
        callbacks : list
            initial callbacks to register
        start_epoch : int
            epoch to start training at
        metric_keys : dict
            dict specifying which batch_dict entry to use for which metric
            as target; default: None, which will result in key "label" for
            all metrics
        convert_batch_to_npy_fn : type, optional
            function converting a batch-tensor to numpy, per default this
            is a function, which detaches the tensor, moves it to cpu and
            then calls ``.array`` on it
        mixed_precision : bool
            whether to use mixed precision or not (False per default)
        val_freq : int
            validation frequency specifying how often to validate the
            trained model (a value of 1 denotes validating every epoch,
            a value of 2 denotes validating every second epoch etc.);
            defaults to 1
        **kwargs :
            additional keyword arguments

        """

        super().__init__(
            network, save_path, losses, optimizer_cls, optimizer_params,
            train_metrics, val_metrics, lr_scheduler_cls,
            lr_scheduler_params, gpu_ids, save_freq, optim_fn, key_mapping,
            logging_type, logging_kwargs, fold, callbacks, start_epoch,
            metric_keys, convert_batch_to_npy_fn, val_freq)

        self._setup(network, optim_fn, optimizer_cls, optimizer_params,
                    lr_scheduler_cls, lr_scheduler_params, gpu_ids,
                    key_mapping, convert_batch_to_npy_fn,
                    mixed_precision)

        for key, val in kwargs.items():
            setattr(self, key, val)

    def _setup(self, network, optim_fn, optimizer_cls, optimizer_params,
               lr_scheduler_cls, lr_scheduler_params, gpu_ids,
               key_mapping, convert_batch_to_npy_fn, mixed_precision):
        """
        Defines the Trainers Setup

        Parameters
        ----------
        network : :class:`AbstractChainerNetwork`
            the network to train
        optim_fn : function
            creates a dictionary containing all necessary optimizers
        optimizer_cls : subclass of torch.optim.Optimizer
            optimizer class implementing the optimization algorithm of
            choice
        optimizer_params : dict
        lr_scheduler_cls : Any
            learning rate schedule class: must implement step() method
        lr_scheduler_params : dict
            keyword arguments passed to lr scheduler during construction
        gpu_ids : list
            list containing ids of GPUs to use; if empty: use cpu instead
        convert_batch_to_npy_fn : type
            function converting a batch-tensor to numpy
        mixed_precision : bool
            whether to use mixed precision or not (False per default)

        """

        self.optimizers = optim_fn(network, optimizer_cls,
                                   **optimizer_params)

        super()._setup(network, None, lr_scheduler_params,
                       gpu_ids, key_mapping, convert_batch_to_npy_fn,
                       network.prepare_batch)

        if mixed_precision:
            # enable chainer mixed precision globally
            chainer.global_config.dtype = chainer.mixed16

        # Load latest epoch file if available
        if os.path.isdir(self.save_path):
            # check all files in directory starting with "checkpoint" and
            # not ending with "_best.pth"
            files = [x for x in os.listdir(self.save_path)
                     if os.path.isfile(os.path.join(self.save_path, x))
                     and x.startswith("checkpoint")
                     and not x.endswith("_best.chain")]

            # if list is not empty: load previous state
            if files:

                latest_epoch = max([
                    int(x.rsplit("_", 1)[-1].rsplit(".", 1)[0])
                    for x in files])

                latest_state_path = os.path.join(
                    self.save_path,
                    "checkpoint_epoch_%d.chain"
                    % latest_epoch)

                # if pth file does not exist, load pt file instead
                if not os.path.isfile(latest_state_path):
                    latest_state_path = latest_state_path[:-1]

                logger.info("Attempting to load state from previous \
                            training from %s" % latest_state_path)
                try:
                    self.update_state(latest_state_path)
                except KeyError:
                    logger.warning("Previous State could not be loaded, \
                                although it exists.Training will be \
                                restarted")

        if chainer.chainerx.is_available():
            gpu_device_prefix = "cuda:"
            cpu_device_prefix = "native"
        else:
            gpu_device_prefix = "@cupy:"
            cpu_device_prefix = "@numpy"

        if gpu_ids:
            try:
                if chainer.cuda.check_cuda_available():
                    self.use_gpu = True
                    if len(gpu_ids) > 1:
                        # use GPU 0 as default input GPU

                        self.input_device = chainer.get_device(
                            gpu_device_prefix + str(gpu_ids[0]))

                        # Train on multiple GPUs and use GPU 0 as output
                        # device
                        self.module = DataParallelChainerNetwork(
                            self.module.to_device("@numpy"),
                            devices=[chainer.get_device(
                                gpu_device_prefix + str(_id))
                                for _id in gpu_ids])

                        # ToDo: Creating Multiple DataParallelOptimizers is
                        #  kinda tricky right now, since we need to add the
                        #  class itself to the parameters and use
                        #  DataParallelOptimizer as optimizer class.
                        #  Should look for other possibility,
                        #  but currently I don't know any
                        self.optimizers = optim_fn(
                            DataParallelChainerOptimizer,
                            {**optimizer_params,
                             "optim_cls": optimizer_cls})

                        self.output_device = chainer.get_device(
                            gpu_device_prefix + str(gpu_ids[0]))
                    else:
                        # use the only available GPU as input device
                        self.input_device = chainer.get_device(
                            cpu_device_prefix)
                        self.module = self.module.to_device(
                            self.input_device)

                        # use GPU 0 as output device as output device
                        self.output_device = chainer.get_device(
                            cpu_device_prefix)
                else:
                    # cuda unavailable -> no GPU support
                    self.use_gpu = False
                    self.input_device = chainer.get_device(
                        cpu_device_prefix)
                    self.output_device = chainer.get_device(
                        cpu_device_prefix)
                    self.module = self.module.to_device(self.input_device)

            # thrown if Cupy is unavailable -> no GPU support
            except RuntimeError as e:
                logging.exception(e)
                self.use_gpu = False
                self.input_device = chainer.get_device(cpu_device_prefix)
                self.output_device = chainer.get_device(cpu_device_prefix)
                self.module = self.module.to_device(self.input_device)

        # no gpu indices given
        else:
            self.use_gpu = False
            self.input_device = chainer.get_device(cpu_device_prefix)
            self.output_device = chainer.get_device(cpu_device_prefix)
            self.module = self.module.to_device(self.input_device)

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
        Defines Behaviour at end of training: Loads best model if
        available

        Returns
        -------
        :class:`AbstractPyTorchNetwork`
            best network

        """
        if os.path.isfile(os.path.join(self.save_path,
                                       'checkpoint_best.chain')):

            # load best model and return it
            self.update_state(os.path.join(self.save_path,
                                           'checkpoint_best.chain'))

        return self.module

    def _at_epoch_end(self, metrics_val, val_score_key, epoch, is_best,
                      **kwargs):
        """
        Defines behaviour at beginning of each epoch: Executes all
        callbacks's `at_epoch_end` method and saves current state if
        necessary

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

            self._update_state(cb.at_epoch_end(self,
                                               val_metrics=metrics_val,
                                               val_score_key=val_score_key,
                                               curr_epoch=epoch))

        if epoch % self.save_freq == 0:
            self.save_state(
                os.path.join(
                    self.save_path,
                    "checkpoint_epoch_%d.chain" %
                    epoch),
                epoch)

        if is_best:
            self.save_state(os.path.join(self.save_path,
                                         "checkpoint_best.chain"),
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

        chainer.global_config.train = True

        return super()._train_single_epoch(batchgen, epoch,
                                           verbose=verbose)

    def predict_data_mgr(self, datamgr, batchsize=None, metrics={},
                         metric_keys={}, verbose=False, **kwargs):
        """
        Defines a routine to predict data obtained from a batchgenerator

        Parameters
        ----------
        datamgr : :class:`BaseDataManager`
            Manager producing a generator holding the batches
        batchsize : int
            Artificial batchsize (sampling will be done with batchsize
            1 and sampled data will be stacked to match the artificial
            batchsize)(default: None)
        metrics : dict
            the metrics to calculate
        metric_keys : dict
            the ``batch_dict`` items to use for metric calculation
        verbose : bool
            whether to show a progress-bar or not, default: False
        **kwargs :
            additional keyword arguments

        Returns
        -------
        dict
            predictions
        dict
            calculated metrics

        """
        chainer.global_config.train = False

        return super().predict_data_mgr(datamgr, batchsize, metrics,
                                        metric_keys, verbose, **kwargs)

    def save_state(self, file_name, epoch, **kwargs):
        """
        saves the current state via
        :func:`delira.io.chainer.save_checkpoint`

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
        if not file_name.endswith(".chain"):
            file_name = file_name + ".chain"
        save_checkpoint(file_name, self.module, self.optimizers,
                        **kwargs)

    @staticmethod
    def load_state(file_name, **kwargs):
        """
        Loads the new state from file via
        :func:`delira.io.chainer.load_checkpoint`

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

        if not file_name.endswith(".chain"):
            file_name = file_name + ".chain"

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
        :class:`BaseNetworkTrainer`
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
        :class:`ChainerNetworkTrainer`
            the trainer with a modified state

        """

        if "model" in new_state:
            self.module = new_state.pop("model")

        if "optimizer" in new_state and new_state["optimizer"]:
            optim_state = new_state.pop("optimizer")
            for key in self.optimizers.keys():
                self.optimizers[key] = optim_state[key]

        if "epoch" in new_state:
            self.start_epoch = new_state.pop("epoch")

        return super()._update_state(new_state)