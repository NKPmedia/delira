import jax
from delira.models.abstract_network import AbstractNetwork
from abc import abstractmethod
from delira.models.backends.jax.utils import channels_to_back


class AbstractJaxNetwork(AbstractNetwork):
    def __init__(self, init_params, predict_fun=None):

        super().__init__()

        if predict_fun is not None:
            def predict(*args, **kwargs):
                return {"pred": predict_fun(*args, **kwargs)}
            self.forward = predict

        self._params = init_params

    @property
    def parameters(self):
        return self._params

    @parameters.setter
    def parameters(self, new_params):
        self._params = new_params

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs) -> dict:
        raise NotImplementedError

    @staticmethod
    @jax.jit
    def _accumulate_gradients(*gradients):
        grads = gradients[0]
        for _grads in gradients[1:]:
            grads = grads + _grads

        return grads

    @staticmethod
    def closure(model, data_dict: dict, optimizers: dict, losses={},
                metrics={}, fold=0, **kwargs):

        predictions = model(data_dict["data"])
        gradients = []
        loss_vals = {}
        metric_vals = {}

        for key, loss_fn in losses:
            loss_val = loss_fn(predictions["pred"], data_dict["label"])
            loss_vals[key] = loss_val
            gradients.append(jax.grad(loss_fn)(predictions["pred"],
                                               data_dict["label"]))

        gradients = AbstractJaxNetwork._accumulate_gradients(*gradients)

        optimizers["default"].apply_update(gradients)

        for key, metric_fn in metrics.items():
            metric_vals[key] = metric_fn(predictions["pred"],
                                         data_dict["label"])

        return metric_vals, loss_vals, predictions

    @staticmethod
    def prepare_batch(batch: dict, input_device, output_device):
        return {"data": channels_to_back(batch["data"]),
                "label": channels_to_back(batch["label"])}
