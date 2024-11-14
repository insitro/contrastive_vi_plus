from dataclasses import dataclass, field
from typing import Dict, Iterable, Union

from torch import Tensor

LossRecord = Union[Dict[str, Tensor], Tensor]


@dataclass
class LossOutput:
    """Loss signature for models.

    This class provides an organized way to record the model loss, as well as
    the components of the ELBO. This may also be used in MLE, MAP, EM methods.
    The loss is used for backpropagation during inference. The other parameters
    are used for logging/early stopping during inference.

    Parameters
    ----------
    loss
        Tensor with loss for minibatch. Should be one dimensional with one value.
        Note that loss should be in an array/tensor and not a float.
    reconstruction_loss
        Reconstruction loss for each observation in the minibatch. If a tensor, converted to
        a dictionary with key "reconstruction_loss" and value as tensor.
    kl_local
        KL divergence associated with each observation in the minibatch. If a tensor, converted to
        a dictionary with key "kl_local" and value as tensor.
    kl_global
        Global KL divergence term. Should be one dimensional with one value. If a tensor,
        converted to a dictionary with key "kl_global" and value as tensor.
    extra_metrics
        Additional metrics can be passed as arrays/tensors or dictionaries of
        arrays/tensors.
    n_obs_minibatch
        Number of observations in the minibatch. If None, will be inferred from
        the shape of the reconstruction_loss tensor.


    Examples
    --------
    >>> loss_output = LossOutput(
    ...     loss=loss,
    ...     reconstruction_loss=reconstruction_loss,
    ...     kl_local=kl_local,
    ...     extra_metrics={"x": scalar_tensor_x, "y": scalar_tensor_y},
    ... )
    """

    loss: LossRecord
    reconstruction_loss: Union[LossRecord, None] = None
    kl_local: Union[LossRecord, None] = None
    kl_global: Union[LossRecord, None] = None
    wasserstein_loss: Union[LossRecord, None] = None

    extra_metrics: dict[str, Tensor] = field(default_factory=dict)
    n_obs_minibatch: Union[int, None] = None

    reconstruction_loss_sum: Tensor = field(default=None, init=False)
    kl_local_sum: Tensor = field(default=None, init=False)
    kl_global_sum: Tensor = field(default=None, init=False)
    wasserstein_loss_sum: Tensor = field(default=None, init=False)

    def __post_init__(self):
        self.loss = self.dict_sum(self.loss)

        if self.n_obs_minibatch is None and self.reconstruction_loss is None:
            raise ValueError(
                "Must provide either n_obs_minibatch or reconstruction_loss"
            )

        default = 0 * self.loss
        if self.reconstruction_loss is None:
            self.reconstruction_loss = default
        if self.kl_local is None:
            self.kl_local = default
        if self.kl_global is None:
            self.kl_global = default
        if self.wasserstein_loss is None:
            self.wasserstein_loss = default

        self.reconstruction_loss = self._as_dict("reconstruction_loss")
        self.kl_local = self._as_dict("kl_local")
        self.kl_global = self._as_dict("kl_global")
        self.wasserstein_loss = self._as_dict("wasserstein_loss")

        self.reconstruction_loss_sum = self.dict_sum(self.reconstruction_loss).sum()
        self.kl_local_sum = self.dict_sum(self.kl_local).sum()
        self.kl_global_sum = self.dict_sum(self.kl_global)
        self.wasserstein_loss_sum = self.dict_sum(self.wasserstein_loss).sum()

        if self.reconstruction_loss is not None and self.n_obs_minibatch is None:
            rec_loss = self.reconstruction_loss
            self.n_obs_minibatch = list(rec_loss.values())[0].shape[0]

    @staticmethod
    def dict_sum(dictionary: Union[Dict[str, Tensor], Tensor]):
        """Sum over elements of a dictionary."""
        if isinstance(dictionary, dict):
            return sum(dictionary.values())
        else:
            return dictionary

    @property
    def extra_metrics_keys(self) -> Iterable[str]:
        """Keys for extra metrics."""
        return self.extra_metrics.keys()

    def _as_dict(self, attr_name: str):
        attr = getattr(self, attr_name)
        if isinstance(attr, dict):
            return attr
        else:
            return {attr_name: attr}
