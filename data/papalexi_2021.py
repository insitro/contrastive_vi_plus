"""
Download, read, and preprocess Papalexi et al. (2021) expression data.

Single-cell expression data from Papalexi et al. Characterizing the molecular regulation
of inhibitory immune checkpoints with multimodal single-cell screens. (Nature Genetics
2021)
"""

import os
import shutil

import muon
import pandas as pd
import scanpy as sc
from anndata import AnnData
from mudata import MuData

from .utils import download_binary_file


def download_papalexi_2021(output_path: str) -> None:
    """
    Download Papalexi et al. 2021 data from the hosting URLs.

    Args:
    ----
        output_path: Output path to store the downloaded and unzipped
        directories.

    Returns
    -------
        None. File directories are downloaded to output_path.
    """

    counts_data_url = (
        "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE153056&format=file"
    )
    data_output_filename = os.path.join(output_path, "GSE153056_RAW.tar")
    download_binary_file(counts_data_url, data_output_filename)
    shutil.unpack_archive(data_output_filename, output_path)

    metadata_url = (
        "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE153056&"
        "format=file&file=GSE153056_ECCITE_metadata.tsv.gz"
    )
    metadata_filename = os.path.join(output_path, metadata_url.split("=")[-1])
    download_binary_file(metadata_url, metadata_filename)


def preprocess_papalexi_2021(download_path: str, n_top_genes: int) -> AnnData:
    """
    Preprocess expression data from Papalexi et al. 2021.

    Args:
    ----
        download_path: Path containing the downloaded Papalexi et al. 2021 data files.
        n_top_genes: Number of most variable genes to retain.

    Returns
    -------
        An AnnData object containing single-cell expression data. The layer
        "count" contains the count data for the most variable genes. The .X
        variable contains the normalized and log-transformed data for the most variable
        genes. A copy of data with all genes is stored in .raw.
    """

    df = pd.read_csv(
        os.path.join(download_path, "GSM4633614_ECCITE_cDNA_counts.tsv.gz"),
        sep="\t",
        index_col=0,
    )

    # Switch dataframe from gene rows and cell columns to cell rows and gene columns
    df = df.transpose()

    metadata = pd.read_csv(
        os.path.join(download_path, "GSE153056_ECCITE_metadata.tsv.gz"),
        sep="\t",
        index_col=0,
    )

    # Note: By initializing the anndata object from a dataframe, variable names
    # are automatically stored in adata.var
    rna_adata = AnnData(df)
    rna_adata.obs = metadata

    rna_adata.raw = rna_adata
    rna_adata.layers["counts"] = rna_adata.X.copy()

    if n_top_genes is not None:
        sc.pp.highly_variable_genes(
            rna_adata,
            flavor="seurat_v3",
            n_top_genes=n_top_genes,
            layer="counts",
            subset=True,
        )

    sc.pp.normalize_total(rna_adata)
    sc.pp.log1p(rna_adata)

    # Quantifies upregulation of the endoplasmic reticulum stress module
    # described in Papalexi 2021.
    sc.tl.score_genes(
        rna_adata,
        gene_list=[
            "FTH1",
            "HSPA5",
            "PDIA4",
            "HSP90B1",
            "SDF2L1",
            "NUCB2",
            "CRELD2",
            "HYOU1",
            "MANF",
            "HERPUD1",
            "TRIB3",
            "DDIT3",
            "SEC11C",
            "DNAJB11",
            "VIMP",
            "OSTC",
            "SELM",
            "SERP1",
            "CDK2AP2",
            "TMED2",
            "PPIB",
            "CALR",
            "P4HB",
            "SELK",
            "DNAJB9",
            "DNAJC3",
            "SEC61G",
            "CCPG1",
        ],
        score_name="stress_score",
    )

    # Protein measurements also collected as part of CITE-Seq
    protein_counts_df = pd.read_csv(
        os.path.join(download_path, "GSM4633615_ECCITE_ADT_counts.tsv.gz"),
        sep="\t",
        index_col=0,
    )

    # Switch dataframe from protein rows and cell columns to cell rows and protein
    # columns
    protein_counts_df = protein_counts_df.transpose()

    # Normalize protein counts using the centered-log-ratio transform
    # as implemented in Muon.
    protein_adata = AnnData(protein_counts_df)
    protein_adata.layers["counts"] = protein_adata.X.copy()

    # Need to explicitly cast to floating point or numpy will
    # throw errors during the normalization step.
    protein_adata.X = protein_adata.X.astype("float64")

    # Matching original normalization performed by Papalexi et al.
    muon.prot.pp.clr(protein_adata, axis=1)

    mdata = MuData({"rna": rna_adata, "protein": protein_adata})
    return mdata


def download_and_preprocess_papalexi_2021(download_path, n_top_genes=2000) -> AnnData:
    """
    Download and preprocess expression data from Papalexi et al., 2021.

    Args:
    ----
        download_path: Path for storing the downloaded Papalexi et al. 2021 data files.
        n_top_genes: Number of most variable genes to retain.

    Returns
    -------
        An AnnData object containing single-cell RNA and protein expression data.
        The layer "count" contains the count data for the most variable genes. The .X
        variable contains the normalized and log-transformed data for the most variable
        genes. Protein counts are stored in the "protein" obsm field.
    """
    download_papalexi_2021(download_path)
    return preprocess_papalexi_2021(download_path, n_top_genes)
