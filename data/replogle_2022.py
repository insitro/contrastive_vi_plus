import os

import scanpy as sc
from anndata import AnnData

from .utils import download_from_s3

REPLOGLE_S3_URL = "s3://insitro-research-2023-sams-vae/data/replogle.h5ad"


def download_replogle_2022(output_path: str) -> None:
    """
    Download Replogle et al. 2022 data from insitro S3.

    Args:
    ----
        output_path: Output path to store the downloaded and unzipped
        directories.

    Returns
    -------
        None. Files downloaded to output_path.
    """
    download_from_s3(
        REPLOGLE_S3_URL, save_path=os.path.join(output_path, "replogle_2022_raw.h5ad")
    )


def preprocess_replogle_2022(download_path: str) -> AnnData:
    """
    Preprocess expression data from Replogle et al 2022.

    Args:
    ----
        download_path: Path containing the downloaded Replogle et al. 2022 data files.

    Returns
    -------
        An AnnData object containing single-cell expression data. The layer
        "counts" contains the count data for the most variable genes. The .X
        variable contains the normalized and log-transformed data.
    """
    adata = sc.read_h5ad(os.path.join(download_path, "replogle_2022_raw.h5ad"))

    # Annotate each gene perturbation with pathway that it belongs to
    genes_to_pathways = {}
    for pathway in adata.uns["pathways"].keys():
        for gene in adata.uns["pathways"][pathway]:
            genes_to_pathways[gene] = pathway
    genes_to_pathways["non-targeting"] = "Ctrl"

    # For our benchmarking experiments we subset to the genes with annotated
    # pathways.
    adata = adata[adata.obs["gene"].isin(genes_to_pathways.keys())]
    adata.obs["pathway"] = [genes_to_pathways[x] for x in adata.obs["gene"]]

    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    return adata


def download_and_preprocess_replogle_2022(download_path) -> AnnData:
    """
    Download and preprocess expression data from Replogle et al., 2022.

    Args:
    ----
        download_path: Path for storing the downloaded Replogle et al. 2022 data files.

    Returns
    -------
        An AnnData object containing single-cell RNA and protein expression data.
        The layer "counts" contains the count data for the most variable genes. The .X
        variable contains the normalized and log-transformed data.
    """
    download_replogle_2022(download_path)
    return preprocess_replogle_2022(download_path)
