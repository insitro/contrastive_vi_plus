"""Data preprocessing utilities."""

from typing import Any, Dict

import pytorch_lightning as pl
import torch


def gram_matrix(x, y, gammas):
    gammas = gammas.unsqueeze(1)
    pairwise_distances = torch.cdist(x, y, p=2.0)

    pairwise_distances_sq = torch.square(pairwise_distances)
    tmp = torch.matmul(gammas, torch.reshape(pairwise_distances_sq, (1, -1)))
    tmp = torch.reshape(torch.sum(torch.exp(-tmp), 0), pairwise_distances_sq.shape)
    return tmp


def mmd(x, y, gammas, device):
    gammas = gammas.to(device)

    cost = torch.mean(gram_matrix(x, x, gammas=gammas)).to(device)
    cost += torch.mean(gram_matrix(y, y, gammas=gammas)).to(device)
    cost -= 2 * torch.mean(gram_matrix(x, y, gammas=gammas)).to(device)

    if cost < 0:
        return torch.tensor(0).to(device)
    return cost


def gumbel_sigmoid(
    logits: torch.Tensor, tau: float = 1, hard: bool = False, threshold: float = 0.5
) -> torch.Tensor:
    """
    Samples from the Gumbel-Sigmoid distribution and optionally discretizes.
    The discretization converts the values greater than `threshold` to 1 and the rest to 0.
    The code is adapted from the official PyTorch implementation of gumbel_softmax:
    https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#gumbel_softmax

    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized,
            but will be differentiated as if it is the soft sample in autograd
     threshold: threshold for the discretization,
                values greater than this will be set to 1 and the rest to 0

    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Sigmoid distribution.
      If ``hard=True``, the returned samples are descretized according to `threshold`,
      otherwise they will be probability distributions.

    """
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
        .exponential_()
        .log()
    )  # ~Gumbel(0, 1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits, tau)
    y_soft = gumbels.sigmoid()

    if hard:
        # Straight through.
        indices = (y_soft > threshold).nonzero(as_tuple=True)
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format)
        y_hard[indices[0], indices[1]] = 1.0
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


# Below are helper functions for training on IMLS
class SaveContrastiveVIParamsCallback(pl.Callback):
    def on_save_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: Dict[str, Any],
    ):
        checkpoint["var_names"] = pl_module.var_names
        checkpoint["attr_dict"] = pl_module.user_attributes


class RunValidation(pl.Callback):
    def __init__(self, validation_dataloader: pl.LightningDataModule):
        """Instantiate this class with the above `DuringTrainingEvalConfig` object."""
        self.validation_dataloader = validation_dataloader

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        pl_module.module.eval()
        device = pl_module.device
        for batch_idx, batch in enumerate(self.validation_dataloader):
            # Move all tensors in the batch to device
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            pl_module.validation_step(batch, batch_idx=batch_idx)

        pl_module.on_validation_epoch_end()
        pl_module.module.train()
