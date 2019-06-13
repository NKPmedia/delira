from jax import numpy as np
from jax import jit
from delira.training.utils import recursively_convert_elements


@jit
def _channels_to_back_single_sample(sample: np.ndarray):
    # push first dimension to last -> comnvert NCHW to NHWC
    if sample.ndim > 1:
        sample = np.moveaxis(sample, 1, -1)

    return sample


@jit
def channels_to_back(*args, **kwargs):
    args = recursively_convert_elements(args, np.ndarray,
                                        _channels_to_back_single_sample)

    kwargs = recursively_convert_elements(kwargs, np.ndarray,
                                          _channels_to_back_single_sample)

    return args, kwargs