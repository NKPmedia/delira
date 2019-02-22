from .parameters import Parameters
from .experiment import AbstractExperiment
from .abstract_trainer import AbstractNetworkTrainer

from delira import get_backends

if "TORCH" in get_backends():
    from .experiment import PyTorchExperiment
    from .pytorch_trainer import PyTorchNetworkTrainer

if "TF" in get_backends():
    from .experiment import TfExperiment
    from .tf_trainer import TfNetworkTrainer
