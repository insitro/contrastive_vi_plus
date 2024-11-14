<div style="text-align: center">
<h1>Contrastive Variational Inference+</h1>
</div>

Code accompanying ["Modeling variable guide efficiency in pooled CRISPR screens with ContrastiveVI+"](https://arxiv.org/abs/2411.08072)

ConstrastiveVI+ is a generative modeling framework for analyzing genetic screen data with single cell readout (e.g PerturbSeq). ContrastiveVI+ disentangles perturbation-induced from non-perturbation-related variations, while also accounting for varying perturbation efficiency rates and estimating the probability each cell was perturbed.

## Usage

### Installation

First create environment with dependencies using [pixi](https://prefix.dev/):

```bash
curl -fsSL https://pixi.sh/install.sh | bash
pixi install
```

### Training a contrastiveVI+ model

ContrastiveVI+ is implemented using the [scvi-tools](https://scvi-tools.org/) framework and follows the framework's workflow and conventions.

First, the data needs to be loaded, filtered, and cleaned. We recommend using [scanpy](https://scanpy.readthedocs.io/en/stable/) and following the available tutorials.

We demonstrate the rest of the workflow using the [Replogle et. al.](https://pubmed.ncbi.nlm.nih.gov/35688146/) dataset, after it was processed and filtered following as in the [SAMS-VAE repo](https://github.com/insitro/sams-vae/tree/main/paper/experiments/replogle_filtered).

First, we load the anndata object using the `get_dataset` function, identify the indices of the background and perturbed cells, and prepare the object for training using the `setup_anndata` function
```python
adata = get_dataset("replogle_2022")
pert_label = "gene"
control_label = "non-targeting"
background_indices = np.where(adata.obs[pert_label] == control_label)[0]
target_indices = np.where(adata.obs[pert_label] != control_label)[0]
ContrastiveVIPlusModel.setup_anndata(adata, layer="counts", labels_key=pert_label)
```

We can now create and train the ContrastiveVI+ Model
```python
model = ContrastiveVIPlusModel(adata)
model.train(
    background_indices=background_indices,
    target_indices=target_indices,
    max_epochs=500,
    use_gpu=True,
)
```

After the model is trained, we can extract the predicted probabilities of each cell being perturbed using `model.predict`, and the two latent representations using the `get_latent_representation` method:
```python
pert_probs = model.predict()
salient_latent_rep = model.get_latent_representation(
    adata, representation_kind="salient"
)
background_latent_rep = model.get_latent_representation(
    adata, representation_kind="background"
)
```


### Replicate paper results

Reproducing the paper results is done in two steps:
1. Train models and produce the resulting embeddings by using the `run_experiment.py` and `run_experiment_all_seeds.py` scripts. Each run trains a single model on a specified dataset.
2. Reproduce benchmarks and figures by running the corresponding notebooks for each dataset from the `notebooks` directory.
