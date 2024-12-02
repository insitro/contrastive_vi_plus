<div style="text-align: center">
<h1>Contrastive Variational Inference+</h1>
</div>

Package containing ContrastiveVI+ model originally described in ["Modeling variable guide efficiency in pooled CRISPR screens with ContrastiveVI+"](https://arxiv.org/abs/2411.08072)

ConstrastiveVI+ is a generative modeling framework for analyzing genetic screen data with single cell readout (e.g PerturbSeq). ContrastiveVI+ disentangles perturbation-induced from non-perturbation-related variations, while also accounting for varying perturbation efficiency rates and estimating the probability each cell was perturbed. Links to Colab notebooks demonstrating ContrastiveVI+ on the datasets in the paper can be found below.

## Usage

### Installation

The latest version of ContrastiveVI+ can be installed from Github using pip:

```bash
pip install git+
```

### What you can do with ContrastiveVI+

* Explore novel perturbation-induced variations in single-cell perturbation screens without confounding from nuisance variations shared with control cells (e.g. cell-cycle-related variations).
* Predict and filter out cells that are labeled with a guide RNA but which do not exhibit functional consequences of perturbations (e.g. due to guide efficiency issues).


### Colab Notebook Examples

* [Validating ContrastiveVI+ on ECCITE-seq data from Papalexi et al. 2021](https://colab.research.google.com/drive/1H4H76Bb-6w5bcNfeFhGM-Rijxi_qqB7k?usp=sharing)
* [Applying ContrastiveVI+ to a large-scale CRISPRa experiment from Norman et al., 2019](https://colab.research.google.com/drive/1Rlnftxcm0bgBr_8lx9ts1XnvdI5tSLHs?usp=sharing)