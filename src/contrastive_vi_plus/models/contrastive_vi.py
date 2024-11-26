"""PyTorch module for Contrastive VI for single cell expression data."""
from __future__ import annotations

import logging
import warnings
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scvi import REGISTRY_KEYS, settings
from scvi.data import AnnDataManager
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
    NumericalObsField,
)
from scvi.dataloaders import AnnDataLoader
from scvi.distributions import ZeroInflatedNegativeBinomial
from scvi.model._utils import (
    _get_batch_code_from_category,
    _init_library_size,
    get_max_epochs_heuristic,
    scrna_raw_counts_properties,
    use_distributed_sampler,
)
from scvi.model.base import BaseModelClass
from scvi.model.base._de_core import _de_core
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from scvi.nn import DecoderSCVI, Encoder
from scvi.train import TrainingPlan, TrainRunner
from scvi.utils import setup_anndata_dsp
from scvi.utils._docstrings import devices_dsp
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl

from ._contrastive_data_splitting import ContrastiveDataSplitter
if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from anndata import AnnData

logger = logging.getLogger(__name__)
Number = int | float

torch.backends.cudnn.benchmark = True


class ContrastiveVAE(BaseModuleClass):
    """Variational inference for contrastive analysis of RNA-seq data.

    Implements the contrastiveVI model of :cite:p:`Weinberger23`.

    Parameters
    ----------
    n_input
        Number of input genes.
    n_batch
        Number of batches. If 0, no batch effect correction is performed.
    n_hidden
        Number of nodes per hidden layer.
    n_background_latent
        Dimensionality of the background latent space.
    n_salient_latent
        Dimensionality of the salient latent space.
    n_layers
        Number of hidden layers used for encoder and decoder NNs.
    dropout_rate
        Dropout rate for neural networks.
    use_observed_lib_size
        Use observed library size for RNA as scaling factor in mean of conditional
        distribution.
    library_log_means
        1 x n_batch array of means of the log library sizes. Parameterize prior on
        library size if not using observed library size.
    library_log_vars
        1 x n_batch array of variances of the log library sizes. Parameterize prior
        on library size if not using observed library size.
    wasserstein_penalty
        Weight of the Wasserstein distance loss that further discourages shared
        variations from leaking into the salient latent space.
    """

    def __init__(
        self,
        n_input: int,
        n_batch: int = 0,
        n_hidden: int = 128,
        n_background_latent: int = 10,
        n_salient_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        use_observed_lib_size: bool = True,
        library_log_means: np.ndarray | None = None,
        library_log_vars: np.ndarray | None = None,
        wasserstein_penalty: float = 0,
    ) -> None:
        super().__init__()
        self.n_input = n_input
        self.n_batch = n_batch
        self.n_hidden = n_hidden
        self.n_background_latent = n_background_latent
        self.n_salient_latent = n_salient_latent
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.latent_distribution = "normal"
        self.dispersion = "gene"
        self.px_r = torch.nn.Parameter(torch.randn(n_input))
        self.use_observed_lib_size = use_observed_lib_size
        self.wasserstein_penalty = wasserstein_penalty

        if not self.use_observed_lib_size:
            if library_log_means is None or library_log_vars is None:
                raise ValueError(
                    "If not using observed_lib_size, "
                    "must provide library_log_means and library_log_vars."
                )
            self.register_buffer("library_log_means", torch.from_numpy(library_log_means).float())
            self.register_buffer("library_log_vars", torch.from_numpy(library_log_vars).float())

        cat_list = [n_batch]
        # Background encoder.
        self.z_encoder = Encoder(
            n_input,
            n_background_latent,
            n_cat_list=cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=self.latent_distribution,
            inject_covariates=True,
            use_batch_norm=True,
            use_layer_norm=False,
            var_activation=None,
        )
        # Salient encoder.
        self.s_encoder = Encoder(
            n_input,
            n_salient_latent,
            n_cat_list=cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=self.latent_distribution,
            inject_covariates=True,
            use_batch_norm=True,
            use_layer_norm=False,
            var_activation=None,
        )
        # Library size encoder.
        self.l_encoder = Encoder(
            n_input,
            1,
            n_layers=1,
            n_cat_list=cat_list,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=True,
            use_batch_norm=True,
            use_layer_norm=False,
            var_activation=None,
        )
        # Decoder from latent variable to distribution parameters in data space.
        n_total_latent = n_background_latent + n_salient_latent
        self.decoder = DecoderSCVI(
            n_total_latent,
            n_input,
            n_cat_list=cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            inject_covariates=True,
            use_batch_norm=True,
            use_layer_norm=False,
        )

    @auto_move_data
    def _compute_local_library_params(
        self, batch_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes local library parameters.

        Compute two tensors of shape (batch_index.shape[0], 1) where each
        element corresponds to the mean and variances, respectively, of the
        log library sizes in the batch the cell corresponds to.
        """
        n_batch = self.library_log_means.shape[1]
        local_library_log_means = F.linear(
            F.one_hot(batch_index.squeeze(-1), n_batch).float(), self.library_log_means
        )
        local_library_log_vars = F.linear(
            F.one_hot(batch_index.squeeze(-1), n_batch).float(), self.library_log_vars
        )
        return local_library_log_means, local_library_log_vars

    @staticmethod
    def _get_min_batch_size(concat_tensors: dict[str, dict[str, torch.Tensor]]) -> int:
        return min(
            concat_tensors["background"][REGISTRY_KEYS.X_KEY].shape[0],
            concat_tensors["target"][REGISTRY_KEYS.X_KEY].shape[0],
        )

    @staticmethod
    def _reduce_tensors_to_min_batch_size(
        tensors: dict[str, torch.Tensor], min_batch_size: int
    ) -> None:
        for name, tensor in tensors.items():
            tensors[name] = tensor[:min_batch_size, :]

    @staticmethod
    def _get_inference_input_from_concat_tensors(
        concat_tensors: dict[str, dict[str, torch.Tensor]], index: str
    ) -> dict[str, torch.Tensor]:
        tensors = concat_tensors[index]
        x = tensors[REGISTRY_KEYS.X_KEY]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        input_dict = {"x": x, "batch_index": batch_index}
        return input_dict

    def _get_inference_input(
        self, concat_tensors: dict[str, dict[str, torch.Tensor]]
    ) -> dict[str, dict[str, torch.Tensor]]:
        background = self._get_inference_input_from_concat_tensors(concat_tensors, "background")
        target = self._get_inference_input_from_concat_tensors(concat_tensors, "target")
        # Ensure batch sizes are the same.
        min_batch_size = self._get_min_batch_size(concat_tensors)
        self._reduce_tensors_to_min_batch_size(background, min_batch_size)
        self._reduce_tensors_to_min_batch_size(target, min_batch_size)
        return {"background": background, "target": target}

    @staticmethod
    def _get_generative_input_from_concat_tensors(
        concat_tensors: dict[str, dict[str, torch.Tensor]], index: str
    ) -> dict[str, torch.Tensor]:
        tensors = concat_tensors[index]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        input_dict = {"batch_index": batch_index}
        return input_dict

    @staticmethod
    def _get_generative_input_from_inference_outputs(
        inference_outputs: dict[str, dict[str, torch.Tensor]], data_source: str
    ) -> dict[str, torch.Tensor]:
        z = inference_outputs[data_source]["z"]
        s = inference_outputs[data_source]["s"]
        library = inference_outputs[data_source]["library"]
        return {"z": z, "s": s, "library": library}

    def _get_generative_input(
        self,
        concat_tensors: dict[str, dict[str, torch.Tensor]],
        inference_outputs: dict[str, dict[str, torch.Tensor]],
    ) -> dict[str, dict[str, torch.Tensor]]:
        background_tensor_input = self._get_generative_input_from_concat_tensors(
            concat_tensors, "background"
        )
        target_tensor_input = self._get_generative_input_from_concat_tensors(
            concat_tensors, "target"
        )
        # Ensure batch sizes are the same.
        min_batch_size = self._get_min_batch_size(concat_tensors)
        self._reduce_tensors_to_min_batch_size(background_tensor_input, min_batch_size)
        self._reduce_tensors_to_min_batch_size(target_tensor_input, min_batch_size)

        background_inference_outputs = self._get_generative_input_from_inference_outputs(
            inference_outputs, "background"
        )
        target_inference_outputs = self._get_generative_input_from_inference_outputs(
            inference_outputs, "target"
        )
        background = {**background_tensor_input, **background_inference_outputs}
        target = {**target_tensor_input, **target_inference_outputs}
        return {"background": background, "target": target}

    @staticmethod
    def _reshape_tensor_for_samples(tensor: torch.Tensor, n_samples: int):
        return tensor.unsqueeze(0).expand((n_samples, tensor.size(0), tensor.size(1)))

    @auto_move_data
    def _generic_inference(
        self,
        x: torch.Tensor,
        batch_index: torch.Tensor,
        n_samples: int = 1,
    ) -> dict[str, torch.Tensor]:
        x_ = x
        if self.use_observed_lib_size:
            library = torch.log(x.sum(1)).unsqueeze(1)
        x_ = torch.log(1 + x_)

        qz_m, qz_v, z = self.z_encoder(x_, batch_index)
        qs_m, qs_v, s = self.s_encoder(x_, batch_index)

        ql_m, ql_v = None, None
        if not self.use_observed_lib_size:
            ql_m, ql_v, library_encoded = self.l_encoder(x_, batch_index)
            library = library_encoded

        if n_samples > 1:
            qz_m = self._reshape_tensor_for_samples(qz_m, n_samples)
            qz_v = self._reshape_tensor_for_samples(qz_v, n_samples)
            z = self._reshape_tensor_for_samples(z, n_samples)
            qs_m = self._reshape_tensor_for_samples(qs_m, n_samples)
            qs_v = self._reshape_tensor_for_samples(qs_v, n_samples)
            s = self._reshape_tensor_for_samples(s, n_samples)

            if self.use_observed_lib_size:
                library = self._reshape_tensor_for_samples(library, n_samples)
            else:
                ql_m = self._reshape_tensor_for_samples(ql_m, n_samples)
                ql_v = self._reshape_tensor_for_samples(ql_v, n_samples)
                library = Normal(ql_m, ql_v.sqrt()).sample()

        outputs = {
            "z": z,
            "qz_m": qz_m,
            "qz_v": qz_v,
            "s": s,
            "qs_m": qs_m,
            "qs_v": qs_v,
            "library": library,
            "ql_m": ql_m,
            "ql_v": ql_v,
        }
        return outputs

    @auto_move_data
    def inference(
        self,
        background: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
        n_samples: int = 1,
    ) -> dict[str, dict[str, torch.Tensor]]:
        background_batch_size = background["x"].shape[0]
        target_batch_size = target["x"].shape[0]
        inference_input = {}
        for key in background.keys():
            inference_input[key] = torch.cat([background[key], target[key]], dim=0)
        outputs = self._generic_inference(**inference_input, n_samples=n_samples)
        batch_size_dim = 0 if n_samples == 1 else 1
        background_outputs, target_outputs = {}, {}
        for key in outputs.keys():
            if outputs[key] is not None:
                background_tensor, target_tensor = torch.split(
                    outputs[key],
                    [background_batch_size, target_batch_size],
                    dim=batch_size_dim,
                )
            else:
                background_tensor, target_tensor = None, None
            background_outputs[key] = background_tensor
            target_outputs[key] = target_tensor
        background_outputs["s"] = torch.zeros_like(background_outputs["s"])
        return {"background": background_outputs, "target": target_outputs}

    @auto_move_data
    def _generic_generative(
        self,
        z: torch.Tensor,
        s: torch.Tensor,
        library: torch.Tensor,
        batch_index: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        latent = torch.cat([z, s], dim=-1)
        px_scale, px_r, px_rate, px_dropout = self.decoder(
            self.dispersion,
            latent,
            library,
            batch_index,
        )
        px_r = torch.exp(self.px_r)
        return {
            "px_scale": px_scale,
            "px_r": px_r,
            "px_rate": px_rate,
            "px_dropout": px_dropout,
        }

    @auto_move_data
    def generative(
        self,
        background: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
    ) -> dict[str, dict[str, torch.Tensor]]:
        latent_z_shape = background["z"].shape
        batch_size_dim = 0 if len(latent_z_shape) == 2 else 1
        background_batch_size = background["z"].shape[batch_size_dim]
        target_batch_size = target["z"].shape[batch_size_dim]
        generative_input = {}
        for key in ["z", "s", "library"]:
            generative_input[key] = torch.cat([background[key], target[key]], dim=batch_size_dim)
        generative_input["batch_index"] = torch.cat(
            [background["batch_index"], target["batch_index"]], dim=0
        )
        outputs = self._generic_generative(**generative_input)
        background_outputs, target_outputs = {}, {}
        for key in ["px_scale", "px_rate", "px_dropout"]:
            if outputs[key] is not None:
                background_tensor, target_tensor = torch.split(
                    outputs[key],
                    [background_batch_size, target_batch_size],
                    dim=batch_size_dim,
                )
            else:
                background_tensor, target_tensor = None, None
            background_outputs[key] = background_tensor
            target_outputs[key] = target_tensor
        background_outputs["px_r"] = outputs["px_r"]
        target_outputs["px_r"] = outputs["px_r"]
        return {"background": background_outputs, "target": target_outputs}

    @staticmethod
    def reconstruction_loss(
        x: torch.Tensor,
        px_rate: torch.Tensor,
        px_r: torch.Tensor,
        px_dropout: torch.Tensor,
    ) -> torch.Tensor:
        """Computes likelihood loss for zero-inflated negative binomial distribution.

        Parameters
        ----------
        x
            Input data.
        px_rate
            Mean of distribution.
        px_r
            Inverse dispersion.
        px_dropout
            Logits scale of zero inflation probability.

        Returns
        -------
        Negative log likelihood (reconstruction loss) for each data point. If number of
        latent samples == 1, the tensor has shape (batch_size, ). If number of latent
        samples > 1, the tensor has shape (n_samples, batch_size).
        """
        recon_loss = (
            -ZeroInflatedNegativeBinomial(mu=px_rate, theta=px_r, zi_logits=px_dropout)
            .log_prob(x)
            .sum(dim=-1)
        )
        return recon_loss

    @staticmethod
    def latent_kl_divergence(
        variational_mean: torch.Tensor,
        variational_var: torch.Tensor,
        prior_mean: torch.Tensor,
        prior_var: torch.Tensor,
    ) -> torch.Tensor:
        """Computes KL divergence between a variational posterior and prior Gaussian.

        Parameters
        ----------
        variational_mean
            Mean of the variational posterior Gaussian.
        variational_var
            Variance of the variational posterior Gaussian.
        prior_mean
            Mean of the prior Gaussian.
        prior_var
            Variance of the prior Gaussian.

        Returns
        -------
        KL divergence for each data point. If number of latent samples == 1, the tensor
        has shape (batch_size, ). If number of latent samples > 1, the tensor has shape
        (n_samples, batch_size).
        """
        return kl(
            Normal(variational_mean, variational_var.sqrt()),
            Normal(prior_mean, prior_var.sqrt()),
        ).sum(dim=-1)

    def library_kl_divergence(
        self,
        batch_index: torch.Tensor,
        variational_library_mean: torch.Tensor,
        variational_library_var: torch.Tensor,
        library: torch.Tensor,
    ) -> torch.Tensor:
        """Computes KL divergence between library size variational posterior and prior.

        Both the variational posterior and prior are Log-Normal.

        Parameters
        ----------
        batch_index
            Batch indices for batch-specific library size mean and variance.
        variational_library_mean
            Mean of variational Log-Normal.
        variational_library_var
            Variance of variational Log-Normal.
        library
            Sampled library size.

        Returns
        -------
        KL divergence for each data point. If number of latent samples == 1, the tensor
        has shape (batch_size, ). If number of latent samples > 1, the tensor has shape
        (n_samples, batch_size).
        """
        if not self.use_observed_lib_size:
            (
                local_library_log_means,
                local_library_log_vars,
            ) = self._compute_local_library_params(batch_index)

            kl_library = kl(
                Normal(variational_library_mean, variational_library_var.sqrt()),
                Normal(local_library_log_means, local_library_log_vars.sqrt()),
            )
        else:
            kl_library = torch.zeros_like(library)
        return kl_library.sum(dim=-1)

    def _generic_loss(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor],
        generative_outputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        x = tensors[REGISTRY_KEYS.X_KEY]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]

        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]
        qs_m = inference_outputs["qs_m"]
        qs_v = inference_outputs["qs_v"]
        library = inference_outputs["library"]
        ql_m = inference_outputs["ql_m"]
        ql_v = inference_outputs["ql_v"]
        px_rate = generative_outputs["px_rate"]
        px_r = generative_outputs["px_r"]
        px_dropout = generative_outputs["px_dropout"]

        prior_z_m = torch.zeros_like(qz_m)
        prior_z_v = torch.ones_like(qz_v)
        prior_s_m = torch.zeros_like(qs_m)
        prior_s_v = torch.ones_like(qs_v)

        recon_loss = self.reconstruction_loss(x, px_rate, px_r, px_dropout)
        kl_z = self.latent_kl_divergence(qz_m, qz_v, prior_z_m, prior_z_v)
        kl_s = self.latent_kl_divergence(qs_m, qs_v, prior_s_m, prior_s_v)
        kl_library = self.library_kl_divergence(batch_index, ql_m, ql_v, library)
        return {
            "recon_loss": recon_loss,
            "kl_z": kl_z,
            "kl_s": kl_s,
            "kl_library": kl_library,
        }

    def loss(
        self,
        concat_tensors: dict[str, dict[str, torch.Tensor]],
        inference_outputs: dict[str, dict[str, torch.Tensor]],
        generative_outputs: dict[str, dict[str, torch.Tensor]],
        kl_weight: float = 1.0,
    ) -> LossOutput:
        """Computes loss terms for contrastiveVI.

        Parameters
        ----------
        concat_tensors
            Tuple of data mini-batch. The first element contains background data
            mini-batch. The second element contains target data mini-batch.
        inference_outputs
            Dictionary of inference step outputs. The keys are "background" and
            "target" for the corresponding outputs.
        generative_outputs
            Dictionary of generative step outputs. The keys are "background" and
            "target" for the corresponding outputs.
        kl_weight
            Importance weight for KL divergence of background and salient latent
            variables, relative to KL divergence of library size.

        Returns
        -------
        An scvi.module.base.LossOutput instance that records the following:
        loss
            One-dimensional tensor for overall loss used for optimization.
        reconstruction_loss
            Reconstruction loss with shape (n_samples, batch_size) if number of latent
            samples > 1, or (batch_size, ) if number of latent samples == 1.
        kl_local
            KL divergence term with shape (n_samples, batch_size) if number of latent
            samples > 1, or (batch_size, ) if number of latent samples == 1.
        """
        background_tensors = concat_tensors["background"]
        target_tensors = concat_tensors["target"]
        # Ensure batch sizes are the same.
        min_batch_size = self._get_min_batch_size(concat_tensors)
        self._reduce_tensors_to_min_batch_size(background_tensors, min_batch_size)
        self._reduce_tensors_to_min_batch_size(target_tensors, min_batch_size)

        background_losses = self._generic_loss(
            background_tensors,
            inference_outputs["background"],
            generative_outputs["background"],
        )
        target_losses = self._generic_loss(
            target_tensors,
            inference_outputs["target"],
            generative_outputs["target"],
        )
        reconst_loss = background_losses["recon_loss"] + target_losses["recon_loss"]
        kl_divergence_z = background_losses["kl_z"] + target_losses["kl_z"]
        kl_divergence_s = target_losses["kl_s"]
        kl_divergence_l = background_losses["kl_library"] + target_losses["kl_library"]

        wasserstein_loss = torch.norm(
            inference_outputs["background"]["qs_m"], dim=-1
        ) ** 2 + torch.sum(inference_outputs["background"]["qs_v"], dim=-1)

        kl_local_for_warmup = kl_divergence_z + kl_divergence_s
        kl_local_no_warmup = kl_divergence_l

        weighted_kl_local = (
            (self.wasserstein_penalty * wasserstein_loss + kl_local_for_warmup)
            + kl_local_no_warmup
        )

        loss = torch.mean(reconst_loss + weighted_kl_local)

        kl_local = {
            "kl_divergence_l": kl_divergence_l,
            "kl_divergence_z": kl_divergence_z,
            "kl_divergence_s": kl_divergence_s,
        }

        # LossOutput internally sums the reconst_loss, kl_local, and kl_global
        # terms before logging, so we do the same for the `wasserstein_loss` term.
        return LossOutput(
            loss=loss,
            reconstruction_loss=reconst_loss,
            kl_local=kl_local,
            extra_metrics={"wasserstein_loss_sum": torch.sum(wasserstein_loss)},
        )

    def sample(self):
        raise NotImplementedError("No sampling method for contrastiveVI")

class ContrastiveVI(BaseModelClass):
    """contrastive variational inference :cite:p:`Weinberger23`.

    Parameters
    ----------
    adata
        AnnData object that has been registered via
        :meth:`~scvi.model.ContrastiveVI.setup_anndata`.
    n_hidden
        Number of nodes per hidden layer.
    n_background_latent
        Dimensionality of the background shared latent space.
    n_salient_latent
        Dimensionality of the salient latent space.
    n_layers
        Number of hidden layers used for encoder and decoder NNs.
    dropout_rate
        Dropout rate for neural networks.
    use_observed_lib_size
        Use observed library size for RNA as scaling factor in mean of conditional distribution.
    wasserstein_penalty
        Weight of the Wasserstein distance loss that further discourages background
        shared variations from leaking into the salient latent space.

    Notes
    -----
    See further usage examples in the following tutorial:

    1. :doc:`/tutorials/notebooks/scrna/contrastiveVI_tutorial`
    """

    _module_cls = ContrastiveVAE
    _data_splitter_cls = ContrastiveDataSplitter
    _training_plan_cls = TrainingPlan
    _train_runner_cls = TrainRunner

    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 128,
        n_background_latent: int = 10,
        n_salient_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        use_observed_lib_size: bool = True,
        wasserstein_penalty: float = 0,
    ) -> None:
        super().__init__(adata)

        n_cats_per_cov = (
            self.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY).n_cats_per_key
            if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry
            else None
        )
        n_batch = self.summary_stats.n_batch

        library_log_means, library_log_vars = None, None
        if not use_observed_lib_size:
            library_log_means, library_log_vars = _init_library_size(self.adata_manager, n_batch)

        self.module = self._module_cls(
            n_input=self.summary_stats.n_vars,
            n_batch=n_batch,
            n_hidden=n_hidden,
            n_background_latent=n_background_latent,
            n_salient_latent=n_salient_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            use_observed_lib_size=use_observed_lib_size,
            library_log_means=library_log_means,
            library_log_vars=library_log_vars,
            wasserstein_penalty=wasserstein_penalty,
        )
        self._model_summary_string = (
            f"ContrastiveVI Model with the following params: \nn_hidden: {n_hidden}, "
            f"n_background_latent: {n_background_latent}, n_salient_latent: {n_salient_latent}, "
            f"n_layers: {n_layers}, dropout_rate: {dropout_rate}, "
            f"use_observed_lib_size: {use_observed_lib_size}, "
            f"wasserstein_penalty: {wasserstein_penalty}"
        )
        self.init_params_ = self._get_init_params(locals())

    @devices_dsp.dedent
    def train(
        self,
        background_indices: list[int],
        target_indices: list[int],
        max_epochs: int | None = None,
        accelerator: str = "auto",
        devices: int | list[int] | str = "auto",
        train_size: float = 0.9,
        validation_size: float | None = None,
        shuffle_set_split: bool = True,
        load_sparse_tensor: bool = False,
        batch_size: int = 128,
        early_stopping: bool = False,
        datasplitter_kwargs: dict | None = None,
        plan_kwargs: dict | None = None,
        **trainer_kwargs,
    ):
        """Train the model.

        Parameters
        ----------
        max_epochs
            Number of passes through the dataset. If `None`, defaults to
            `np.min([round((20000 / n_cells) * 400), 400])`
        %(param_accelerator)s
        %(param_devices)s
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the test set. If `None`, defaults to 1 - `train_size`. If
            `train_size + validation_size < 1`, the remaining cells belong to a test set.
        shuffle_set_split
            Whether to shuffle indices before splitting. If `False`, the val, train, and test set
            are split in the sequential order of the data according to `validation_size` and
            `train_size` percentages.
        load_sparse_tensor
            ``EXPERIMENTAL`` If ``True``, loads data with sparse CSR or CSC layout as a
            :class:`~torch.Tensor` with the same layout. Can lead to speedups in data transfers to
            GPUs, depending on the sparsity of the data.
        batch_size
            Minibatch size to use during training.
        early_stopping
            Perform early stopping. Additional arguments can be passed in `**kwargs`.
            See :class:`~scvi.train.Trainer` for further options.
        datasplitter_kwargs
            Additional keyword arguments passed into
            :class:`~scvi.dataloaders.ContrastiveDataSplitter`.
        plan_kwargs
            Keyword args for :class:`~scvi.train.TrainingPlan`. Keyword arguments passed to
            `train()` will overwrite values present in `plan_kwargs`, when appropriate.
        **trainer_kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.
        """
        if max_epochs is None:
            max_epochs = get_max_epochs_heuristic(self.adata.n_obs)

        plan_kwargs = plan_kwargs or {}
        datasplitter_kwargs = datasplitter_kwargs or {}

        data_splitter = self._data_splitter_cls(
            self.adata_manager,
            background_indices=background_indices,
            target_indices=target_indices,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
            shuffle_set_split=shuffle_set_split,
            distributed_sampler=use_distributed_sampler(trainer_kwargs.get("strategy", None)),
            load_sparse_tensor=load_sparse_tensor,
            **datasplitter_kwargs,
        )
        training_plan = self._training_plan_cls(self.module, **plan_kwargs)

        es = "early_stopping"
        trainer_kwargs[es] = (
            early_stopping if es not in trainer_kwargs.keys() else trainer_kwargs[es]
        )
        runner = self._train_runner_cls(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            **trainer_kwargs,
        )
        return runner()

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: str | None = None,
        batch_key: str | None = None,
        labels_key: str | None = None,
        size_factor_key: str | None = None,
        categorical_covariate_keys: list[str] | None = None,
        continuous_covariate_keys: list[str] | None = None,
        **kwargs,
    ):
        """%(summary)s.

        Parameters
        ----------
        %(param_adata)s
        %(param_layer)s
        %(param_batch_key)s
        %(param_labels_key)s
        %(param_size_factor_key)s
        %(param_cat_cov_keys)s
        %(param_cont_cov_keys)s
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            NumericalObsField(REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key, required=False),
            CategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys),
            NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys),
        ]
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    @torch.inference_mode()
    def get_latent_representation(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        give_mean: bool = True,
        batch_size: int | None = None,
        representation_kind: str = "salient",
    ) -> np.ndarray:
        """Returns the background or salient latent representation for each cell.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`,
            defaults to the AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        give_mean
            Give mean of distribution or sample from it.
        batch_size
            Mini-batch size for data loading into model. Defaults to
            `scvi.settings.batch_size`.
        representation_kind
            Either "background" or "salient" for the corresponding representation kind.

        Returns
        -------
            A numpy array with shape `(n_cells, n_latent)`.
        """
        available_representation_kinds = ["background", "salient"]
        if representation_kind not in available_representation_kinds:
            raise ValueError(
                f"representation_kind = {representation_kind} is not one of"
                f" {available_representation_kinds}"
            )

        adata = self._validate_anndata(adata)
        data_loader = self._make_data_loader(
            adata=adata,
            indices=indices,
            batch_size=batch_size,
            shuffle=False,
            data_loader_class=AnnDataLoader,
        )
        latent = []
        for tensors in data_loader:
            x = tensors[REGISTRY_KEYS.X_KEY]
            batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
            outputs = self.module._generic_inference(x=x, batch_index=batch_index, n_samples=1)

            if representation_kind == "background":
                latent_m = outputs["qz_m"]
                latent_sample = outputs["z"]
            else:
                latent_m = outputs["qs_m"]
                latent_sample = outputs["s"]

            if give_mean:
                latent_sample = latent_m

            latent += [latent_sample.detach().cpu()]
        return torch.cat(latent).numpy()

    @torch.inference_mode()
    def get_normalized_expression(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        transform_batch: Sequence[Number | str] | None = None,
        gene_list: Sequence[str] | None = None,
        library_size: float | str = 1.0,
        n_samples: int = 1,
        n_samples_overall: int | None = None,
        batch_size: int | None = None,
        return_mean: bool = True,
        return_numpy: bool | None = None,
    ) -> dict[str, np.ndarray | pd.DataFrame]:
        """Returns the normalized (decoded) gene expression.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`,
            defaults to the AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        transform_batch
            Batch to condition on. If transform_batch is:
            - None, then real observed batch is used.
            - int, then batch transform_batch is used.
        gene_list
            Return frequencies of expression for a subset of genes. This can save
            memory when working with large datasets and few genes are of interest.
        library_size
            Scale the expression frequencies to a common library size. This
            allows gene expression levels to be interpreted on a common scale of
            relevant magnitude. If set to `"latent"`, use the latent library size.
        n_samples
            Number of posterior samples to use for estimation.
        n_samples_overall
            The number of random samples in `adata` to use.
        batch_size
            Mini-batch size for data loading into model. Defaults to
            `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a `numpy.ndarray` instead of a `pandas.DataFrame`.
            DataFrame includes gene names as columns. If either `n_samples=1` or
            `return_mean=True`, defaults to `False`. Otherwise, it defaults to `True`.

        Returns
        -------
            A dictionary with keys "background" and "salient", with value as follows.
            If `n_samples` > 1 and `return_mean` is `False`, then the shape is
            `(samples, cells, genes)`. Otherwise, shape is `(cells, genes)`. In this
            case, return type is `pandas.DataFrame` unless `return_numpy` is `True`.
        """
        adata = self._validate_anndata(adata)
        if indices is None:
            indices = np.arange(adata.n_obs)
        if n_samples_overall is not None:
            indices = np.random.choice(indices, n_samples_overall)
        data_loader = self._make_data_loader(
            adata=adata,
            indices=indices,
            batch_size=batch_size,
            shuffle=False,
            data_loader_class=AnnDataLoader,
        )

        transform_batch = _get_batch_code_from_category(
            self.get_anndata_manager(adata, required=True), transform_batch
        )

        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = adata.var_names
            gene_mask = [True if gene in gene_list else False for gene in all_genes]

        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn(
                    "return_numpy must be True if n_samples > 1 and"
                    " return_mean is False, returning np.ndarray",
                    stacklevel=settings.warnings_stacklevel,
                )
            return_numpy = True
        if library_size == "latent":
            generative_output_key = "px_rate"
            scaling = 1
        else:
            generative_output_key = "px_scale"
            scaling = library_size

        background_exprs = []
        salient_exprs = []
        for tensors in data_loader:
            x = tensors[REGISTRY_KEYS.X_KEY]
            batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
            background_per_batch_exprs = []
            salient_per_batch_exprs = []
            for batch in transform_batch:
                if batch is not None:
                    batch_index = torch.ones_like(batch_index) * batch
                inference_outputs = self.module._generic_inference(
                    x=x, batch_index=batch_index, n_samples=n_samples
                )
                z = inference_outputs["z"]
                s = inference_outputs["s"]
                library = inference_outputs["library"]
                background_generative_outputs = self.module._generic_generative(
                    z=z, s=torch.zeros_like(s), library=library, batch_index=batch_index
                )
                salient_generative_outputs = self.module._generic_generative(
                    z=z, s=s, library=library, batch_index=batch_index
                )
                background_outputs = self._preprocess_normalized_expression(
                    background_generative_outputs,
                    generative_output_key,
                    gene_mask,
                    scaling,
                )
                background_per_batch_exprs.append(background_outputs)
                salient_outputs = self._preprocess_normalized_expression(
                    salient_generative_outputs,
                    generative_output_key,
                    gene_mask,
                    scaling,
                )
                salient_per_batch_exprs.append(salient_outputs)

            background_per_batch_exprs = np.stack(
                background_per_batch_exprs
            )  # Shape is (len(transform_batch) x batch_size x n_var).
            salient_per_batch_exprs = np.stack(salient_per_batch_exprs)
            background_exprs += [background_per_batch_exprs.mean(0)]
            salient_exprs += [salient_per_batch_exprs.mean(0)]

        if n_samples > 1:
            # The -2 axis correspond to cells.
            background_exprs = np.concatenate(background_exprs, axis=-2)
            salient_exprs = np.concatenate(salient_exprs, axis=-2)
        else:
            background_exprs = np.concatenate(background_exprs, axis=0)
            salient_exprs = np.concatenate(salient_exprs, axis=0)
        if n_samples > 1 and return_mean:
            background_exprs = background_exprs.mean(0)
            salient_exprs = salient_exprs.mean(0)

        if return_numpy is None or return_numpy is False:
            genes = adata.var_names[gene_mask]
            samples = adata.obs_names[indices]
            background_exprs = pd.DataFrame(background_exprs, columns=genes, index=samples)
            salient_exprs = pd.DataFrame(salient_exprs, columns=genes, index=samples)
        return {"background": background_exprs, "salient": salient_exprs}

    @torch.inference_mode()
    def get_salient_normalized_expression(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        transform_batch: Sequence[Number | str] | None = None,
        gene_list: Sequence[str] | None = None,
        library_size: float | str = 1.0,
        n_samples: int = 1,
        n_samples_overall: int | None = None,
        batch_size: int | None = None,
        return_mean: bool = True,
        return_numpy: bool | None = None,
    ) -> np.ndarray | pd.DataFrame:
        """Returns the normalized (decoded) gene expression.

        Gene expressions are decoded from both the background and salient latent space.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`,
            defaults to the AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        transform_batch
            Batch to condition on. If transform_batch is:
            - None, then real observed batch is used.
            - int, then batch transform_batch is used.
        gene_list
            Return frequencies of expression for a subset of genes. This can
            save memory when working with large datasets and few genes are of interest.
        library_size
            Scale the expression frequencies to a common library size. This
            allows gene expression levels to be interpreted on a common scale of
            relevant magnitude. If set to `"latent"`, use the latent library size.
        n_samples
            Number of posterior samples to use for estimation.
        n_samples_overall
            The number of random samples in `adata` to use.
        batch_size
            Mini-batch size for data loading into model. Defaults to
            `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a `numpy.ndarray` instead of a `pandas.DataFrame`.
            DataFrame includes gene names as columns. If either `n_samples=1` or
            `return_mean=True`, defaults to `False`. Otherwise, it defaults to `True`.

        Returns
        -------
            If `n_samples` > 1 and `return_mean` is `False`, then the shape is
            `(samples, cells, genes)`. Otherwise, shape is `(cells, genes)`. In this
            case, return type is `pandas.DataFrame` unless `return_numpy` is `True`.
        """
        exprs = self.get_normalized_expression(
            adata=adata,
            indices=indices,
            transform_batch=transform_batch,
            gene_list=gene_list,
            library_size=library_size,
            n_samples=n_samples,
            n_samples_overall=n_samples_overall,
            batch_size=batch_size,
            return_mean=return_mean,
            return_numpy=return_numpy,
        )
        return exprs["salient"]

    @torch.inference_mode()
    def get_specific_normalized_expression(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        transform_batch: Sequence[Number | str] | None = None,
        gene_list: Sequence[str] | None = None,
        library_size: float | str = 1,
        n_samples: int = 1,
        n_samples_overall: int | None = None,
        batch_size: int | None = None,
        return_mean: bool = True,
        return_numpy: bool | None = None,
        expression_type: str | None = None,
        indices_to_return_salient: Sequence[int] | None = None,
    ):
        """Returns the normalized (decoded) gene expression.

        Gene expressions are decoded from either the background or salient latent space.
        One of `expression_type` or `indices_to_return_salient` should have an input
        argument.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`,
            defaults to the AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        transform_batch
            Batch to condition on. If transform_batch is:
            - None, then real observed batch is used.
            - int, then batch transform_batch is used.
        gene_list
            Return frequencies of expression for a subset of genes. This can
            save memory when working with large datasets and few genes are of interest.
        library_size
            Scale the expression frequencies to a common library size. This
            allows gene expression levels to be interpreted on a common scale of
            relevant magnitude. If set to `"latent"`, use the latent library size.
        n_samples
            Number of posterior samples to use for estimation.
        n_samples_overall
            The number of random samples in `adata` to use.
        batch_size
            Mini-batch size for data loading into model. Defaults to
            `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a `numpy.ndarray` instead of a `pandas.DataFrame`.
            DataFrame includes gene names as columns. If either `n_samples=1` or
            `return_mean=True`, defaults to `False`. Otherwise, it defaults to `True`.
        expression_type
            One of {"salient", "background"} to specify the type of
            normalized expression to return.
        indices_to_return_salient
            If `indices` is a subset of `indices_to_return_salient`, normalized
            expressions derived from background and salient latent embeddings are
            returned. If `indices` is not `None` and is not a subset of
            `indices_to_return_salient`, normalized expressions derived only from
            background latent embeddings are returned.

        Returns
        -------
            If `n_samples` > 1 and `return_mean` is `False`, then the shape is
            `(samples, cells, genes)`. Otherwise, shape is `(cells, genes)`. In this
            case, return type is `pandas.DataFrame` unless `return_numpy` is `True`.
        """
        is_expression_type_none = expression_type is None
        is_indices_to_return_salient_none = indices_to_return_salient is None
        if is_expression_type_none and is_indices_to_return_salient_none:
            raise ValueError(
                "Both expression_type and indices_to_return_salient are None! "
                "Exactly one of them needs to be supplied with an input argument."
            )
        elif (not is_expression_type_none) and (not is_indices_to_return_salient_none):
            raise ValueError(
                "Both expression_type and indices_to_return_salient have an input "
                "argument! Exactly one of them needs to be supplied with an input "
                "argument."
            )
        else:
            exprs = self.get_normalized_expression(
                adata=adata,
                indices=indices,
                transform_batch=transform_batch,
                gene_list=gene_list,
                library_size=library_size,
                n_samples=n_samples,
                n_samples_overall=n_samples_overall,
                batch_size=batch_size,
                return_mean=return_mean,
                return_numpy=return_numpy,
            )
            if not is_expression_type_none:
                return exprs[expression_type]
            else:
                if indices is None:
                    indices = np.arange(adata.n_obs)
                if set(indices).issubset(set(indices_to_return_salient)):
                    return exprs["salient"]
                else:
                    return exprs["background"]

    def differential_expression(
        self,
        adata: AnnData | None = None,
        groupby: str | None = None,
        group1: Iterable[str] | None = None,
        group2: str | None = None,
        idx1: Sequence[int] | (Sequence[bool] | str) | None = None,
        idx2: Sequence[int] | (Sequence[bool] | str) | None = None,
        mode: str = "change",
        delta: float = 0.25,
        batch_size: int | None = None,
        all_stats: bool = True,
        batch_correction: bool = False,
        batchid1: Iterable[str] | None = None,
        batchid2: Iterable[str] | None = None,
        fdr_target: float = 0.05,
        silent: bool = False,
        target_idx: Sequence[int] | None = None,
        n_samples: int = 1,
        **kwargs,
    ) -> pd.DataFrame:
        r"""Performs differential expression analysis.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`,
            defaults to the AnnData object used to initialize the model.
        groupby
            The key of the observations grouping to consider.
        group1
            Subset of groups, e.g. ["g1", "g2", "g3"], to which comparison shall be
            restricted, or all groups in `groupby` (default).
        group2
            If `None`, compare each group in `group1` to the union of the rest of
            the groups in `groupby`. If a group identifier, compare with respect to this
            group.
        idx1
            `idx1` and `idx2` can be used as an alternative to the AnnData keys.
            Custom identifier for `group1` that can be of three sorts:
            (1) a boolean mask, (2) indices, or (3) a string. If it is a string, then
            it will query indices that verifies conditions on adata.obs, as described
            in `pandas.DataFrame.query()`. If `idx1` is not `None`, this option
            overrides `group1` and `group2`.
        idx2
            Custom identifier for `group2` that has the same properties as `idx1`.
            By default, includes all cells not specified in `idx1`.
        mode:
            Method for differential expression. See
            https://docs.scvi-tools.org/en/0.14.1/user_guide/background/differential_expression.html
            for more details.
        delta
            Specific case of region inducing differential expression. In this case,
            we suppose that R\[-delta, delta] does not induce differential expression
            (change model default case).
        batch_size
            Mini-batch size for data loading into model. Defaults to
            scvi.settings.batch_size.
        all_stats
            Concatenate count statistics (e.g., mean expression group 1) to DE
            results.
        batch_correction
            Whether to correct for batch effects in DE inference.
        batchid1
            Subset of categories from `batch_key` registered in `setup_anndata`,
            e.g. ["batch1", "batch2", "batch3"], for `group1`. Only used if
            `batch_correction` is `True`, and by default all categories are used.
        batchid2
            Same as `batchid1` for `group2`. `batchid2` must either have null
            intersection with `batchid1`, or be exactly equal to `batchid1`. When the
            two sets are exactly equal, cells are compared by decoding on the same
            batch. When sets have null intersection, cells from `group1` and `group2`
            are decoded on each group in `group1` and `group2`, respectively.
        fdr_target
            Tag features as DE based on posterior expected false discovery rate.
        silent
            If `True`, disables the progress bar. Default: `False`.
        target_idx
            If not `None`, a boolean or integer identifier should be used for
            cells in the contrastive target group. Normalized expression values derived
            from both salient and background latent embeddings are used when
            {group1, group2} is a subset of the target group, otherwise background
            normalized expression values are used.
        kwargs: Keyword args for
            `scvi.model.base.DifferentialComputation.get_bayes_factors`.

        Returns
        -------
        Differential expression DataFrame.
        """
        adata = self._validate_anndata(adata)
        col_names = adata.var_names

        if target_idx is not None:
            target_idx = np.array(target_idx)
            if target_idx.dtype is np.dtype("bool"):
                assert (
                    len(target_idx) == adata.n_obs
                ), "target_idx mask must be the same length as adata!"
                target_idx = np.arange(adata.n_obs)[target_idx]
            model_fn = partial(
                self.get_specific_normalized_expression,
                return_numpy=True,
                n_samples=n_samples,
                batch_size=batch_size,
                expression_type=None,
                indices_to_return_salient=target_idx,
            )
        else:
            model_fn = partial(
                self.get_specific_normalized_expression,
                return_numpy=True,
                n_samples=n_samples,
                batch_size=batch_size,
                expression_type="salient",
                indices_to_return_salient=None,
            )

        result = _de_core(
            self.get_anndata_manager(adata, required=True),
            model_fn,
            representation_fn=None,
            groupby=groupby,
            group1=group1,
            group2=group2,
            idx1=idx1,
            idx2=idx2,
            all_stats=all_stats,
            all_stats_fn=scrna_raw_counts_properties,
            col_names=col_names,
            mode=mode,
            batchid1=batchid1,
            batchid2=batchid2,
            delta=delta,
            batch_correction=batch_correction,
            fdr=fdr_target,
            silent=silent,
            **kwargs,
        )
        return result

    @staticmethod
    @torch.inference_mode()
    def _preprocess_normalized_expression(
        generative_outputs: dict[str, torch.Tensor],
        generative_output_key: str,
        gene_mask: list | slice,
        scaling: float,
    ) -> np.ndarray:
        output = generative_outputs[generative_output_key]
        output = output[..., gene_mask]
        output *= scaling
        output = output.cpu().numpy()
        return output

    @torch.inference_mode()
    def get_latent_library_size(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        give_mean: bool = True,
        batch_size: int | None = None,
    ) -> np.ndarray:
        r"""Returns the latent library size for each cell.

        This is denoted as :math:`\ell_n` in the scVI paper.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`,
            defaults to the AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        give_mean
            Return the mean or a sample from the posterior distribution.
        batch_size
            Minibatch size for data loading into model. Defaults to
            `scvi.settings.batch_size`.
        """
        self._check_if_trained(warn=False)

        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)
        libraries = []
        for tensors in scdl:
            x = tensors[REGISTRY_KEYS.X_KEY]
            batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
            outputs = self.module._generic_inference(x=x, batch_index=batch_index)

            library = outputs["library"]
            if not give_mean:
                library = torch.exp(library)
            else:
                ql = (outputs["ql_m"], outputs["ql_v"])
                if ql is None:
                    raise RuntimeError(
                        "The module for this model does not compute the posterior"
                        "distribution for the library size. Set `give_mean` to False"
                        "to use the observed library size instead."
                    )
                library = torch.distributions.LogNormal(ql[0], ql[1]).mean
            libraries += [library.cpu()]
        return torch.cat(libraries).numpy()