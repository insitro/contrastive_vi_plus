import argparse
import os

import anndata2ri
import numpy as np
import rpy2.robjects as ro
import scanpy as sc
import scvi
from constants import METHODS
from rpy2.robjects import r
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

from .data.datasets import available_datasets, get_dataset
from .models.contrastive_vi import ContrastiveVIModel
from .models.contrastive_vi_plus import ContrastiveVIPlusModel

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=list(available_datasets.keys()))
parser.add_argument("--method", choices=METHODS)
parser.add_argument("--mmd_penalty", type=float, default=1000)
parser.add_argument("--n_classifier_layers", type=int, default=3)
parser.add_argument("--seed", type=int, default=123)
parser.add_argument(
    "--inference_strategy", type=str, choices=["marginalize", "gumbel_sigmoid"]
)
parser.add_argument("--learn_basal_mean", action="store_true")
parser.add_argument("--early_stopping", action="store_true")

args = parser.parse_args()
scvi.settings.seed = args.seed
r(f"set.seed({args.seed})")

if args.dataset == "papalexi_2021":
    mdata = get_dataset(args.dataset)
    adata = mdata["rna"]

    # Manually obtained genes with non-trivial effect sizes from the mixscape paper.
    # This filtering wasn't necessary for other datasets, as those gense were already
    # selected for strong effect sizes.
    strong_effect_genes = [
        "JAK2",
        "STAT1",
        "IFNGR1",
        "IFNGR2",
        "IRF1",
        "SMAD4",
        "STAT2",
        "BRD4",
        "MYC",
        "CUL3",
        "SPI1",
    ]

    background_indices = np.where(~adata.obs["gene"].isin(strong_effect_genes))[0]
    target_indices = np.where(adata.obs["gene"].isin(strong_effect_genes))[0]

    pert_label = "gene"
    control_label = "NT"
elif args.dataset == "norman_2019":
    adata = get_dataset(args.dataset)
    pert_label = "guide_merged"
    control_label = "ctrl"
    background_indices = np.where(adata.obs["gene_program"] == "Ctrl")[0]
    target_indices = np.where(adata.obs["gene_program"] != "Ctrl")[0]

elif args.dataset == "replogle_2022":
    adata = get_dataset(args.dataset)
    pert_label = "gene"
    control_label = "non-targeting"
    background_indices = np.where(adata.obs[pert_label] == control_label)[0]
    target_indices = np.where(adata.obs[pert_label] != control_label)[0]

results_dir = os.path.join("results", args.dataset, args.method, f"seed_{args.seed}")

if args.method == "mixscape":
    seurat = importr("Seurat")
    importr("purrr")

    # First convert AnnData to SingleCellExperiment object
    with localconverter(anndata2ri.converter):
        sce = ro.conversion.get_conversion().py2rpy(adata)

    # Convert SingleCellExperiment object to Seurat object
    seurat_obj = r("partial(as.Seurat, data=NULL)")(sce)

    # Pass the Seurat object into the R global environment
    ro.globalenv["seurat_obj"] = seurat_obj

    # Replace the 'orignalexp' assay produced by SCE with a more standard label ('RNA')
    r("seurat_obj[['RNA']] = seurat_obj[['originalexp']]")
    r("DefaultAssay(seurat_obj) <- 'RNA'")
    r("seurat_obj[['originalexp']] = NULL")

    # Flag all features as variable for Seurat since we already subsetted to HVGs during preprocessing
    r("VariableFeatures(seurat_obj) <- rownames(seurat_obj)")

    # Normalize + calculate PCA as done in Seurat
    r("seurat_obj <- NormalizeData(object = seurat_obj) %>% ScaleData()")
    r("seurat_obj <- RunPCA(object = seurat_obj)")

    # Calculate "perturbation signature" (PRTB) as defined by Papalexi et al.
    r(
        f"seurat_obj <- CalcPerturbSig(object = seurat_obj, assay = 'RNA', slot = 'data', gd.class = '{pert_label}',  nt.cell.class = '{control_label}', reduction = 'pca', num.neighbors = 20, new.assay.name = 'PRTB')"
    )

    # Prepare PRTB assay for dimensionality reduction
    r("DefaultAssay(object = seurat_obj) <- 'PRTB'")

    # Run mixscape
    r(
        f"seurat_obj <- RunMixscape(object = seurat_obj, assay = 'PRTB', slot = 'scale.data', labels = '{pert_label}', nt.class.name = '{control_label}', min.de.genes = 5, iter.num = 100, de.assay = 'RNA', verbose = F, prtb.type = 'KO')"
    )

    pert_probs = np.array(r("seurat_obj$mixscape_class_p_ko"))
    adata_pert = sc.AnnData(X=np.array(r('seurat_obj[["PRTB"]]$data')).T, obs=adata.obs)
    sc.pp.pca(adata_pert)
    salient_latent_rep = adata_pert.obsm["X_pca"]
    background_latent_rep = None

elif args.method == "contrastive_vi":
    ContrastiveVIModel.setup_anndata(
        adata,
        layer="counts",
    )

    model = ContrastiveVIModel(adata)
    model.train(
        background_indices=background_indices,
        target_indices=target_indices,
        max_epochs=500,
        use_gpu=True,
        early_stopping=args.early_stopping,
    )

    salient_latent_rep = model.get_latent_representation(
        adata, representation_kind="salient"
    )
    background_latent_rep = model.get_latent_representation(
        adata, representation_kind="background"
    )
    pert_probs = None

    results_dir = os.path.join(results_dir, f"early_stopping_{args.early_stopping}")

elif args.method == "contrastive_vi_plus":
    ContrastiveVIPlusModel.setup_anndata(adata, layer="counts", labels_key=pert_label)

    model = ContrastiveVIPlusModel(
        adata,
        learn_basal_mean=args.learn_basal_mean,
        n_classifier_layers=args.n_classifier_layers,
        mmd_penalty=args.mmd_penalty,
        inference_strategy=args.inference_strategy,
    )

    model.train(
        background_indices=background_indices,
        target_indices=target_indices,
        max_epochs=500,
        use_gpu=True,
        early_stopping=args.early_stopping,
    )

    pert_probs = model.predict()
    salient_latent_rep = model.get_latent_representation(
        adata, representation_kind="salient"
    )
    background_latent_rep = model.get_latent_representation(
        adata, representation_kind="background"
    )

    results_dir = os.path.join(
        results_dir,
        f"inference_{args.inference_strategy}",
        f"early_stopping_{args.early_stopping}",
        f"learn_basal_mean_{args.learn_basal_mean}",
        f"n_classifier_layers_{args.n_classifier_layers}",
        f"mmd_penalty_{args.mmd_penalty}",
    )

os.makedirs(results_dir, exist_ok=True)
np.save(os.path.join(results_dir, "salient_latent_rep.npy"), salient_latent_rep)
if background_latent_rep is not None:
    np.save(
        os.path.join(results_dir, "background_latent_rep.npy"), background_latent_rep
    )
if pert_probs is not None:
    np.save(os.path.join(results_dir, "pert_probs.npy"), pert_probs)

print(f"Results saved at {results_dir}")
