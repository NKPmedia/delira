import jax
from delira.models.backends.jax.abstract_network import AbstractJaxNetwork


class Optimizer(object):
    def __init__(self, opt_state, opt_update, get_params):
        self._state = opt_state
        self._update = opt_update
        self._get_params = get_params
        self._curr_step = 0

    @classmethod
    def from_model(cls, model: AbstractJaxNetwork,
                   opt_init, opt_update, get_params):
        opt_state = opt_init(model.parameters)
        cls(opt_state, opt_update, get_params)

    @jax.jit
    def apply_update(self, gradients):
        self._update(self._curr_step, gradients, self._state)

        self._curr_step += 1

