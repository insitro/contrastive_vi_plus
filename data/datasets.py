import os
import tempfile
from dataclasses import dataclass
from typing import Callable, Dict

import anndata
import mudata

from .norman_2019 import (
    download_and_preprocess_norman_2019,
)
from .papalexi_2021 import (
    download_and_preprocess_papalexi_2021,
)
from .replogle_2022 import (
    download_and_preprocess_replogle_2022,
)


@dataclass
class Dataset:
    """Provides basic metadata for individual datasets and location in S3"""

    description: str
    citation: str
    local_path: str
    preprocessing_function: Callable


available_datasets: Dict[str, Dataset] = {
    "papalexi_2021": Dataset(
        description="CITE-seq data from THP-1 cells. Cells were perturbed via CRISPRn to knock "
        "out genes associated with PD-L1 protein expression.",
        citation="Characterizing the molecular regulation of inhibitory immune checkpoints "
        "with multimodal single-cell screens, Nature Genetics (2021)",
        local_path="/data/constractive_plus_data/papalexi_2021.h5mu",
        preprocessing_function=download_and_preprocess_papalexi_2021,
    ),
    "norman_2019": Dataset(
        description="scRNA-seq data from K562 cells perturbed to upregulated various genes "
        "via CRISPRa.",
        citation="Exploring genetic interaction manifolds constructed from rich single-cell "
        "phenotypes, Science (2019)",
        local_path="/data/constractive_plus_data/norman_2019.h5ad",
        preprocessing_function=download_and_preprocess_norman_2019,
    ),
    "replogle_2022": Dataset(
        description="scRNA-seq data from K562 cells perturbed to downregulate various genes via "
        "CRISPRi.",
        citation="Mapping information-rich genotype-phenotype landscapes with genome-scale "
        "Perturb-seq, Cell (2022)",
        local_path="/data/constractive_plus_data/replogle_2022.h5ad",
        preprocessing_function=download_and_preprocess_replogle_2022,
    ),
}


def get_dataset(dataset_name: str) -> Dataset:
    """
    Get dataset metadata for a given dataset name.

    Args:
    ----
        dataset_name: Name of the dataset.

    Returns
    -------
        Dataset: Dataset metadata.
    """
    if dataset_name not in available_datasets:
        raise ValueError(f"Dataset {dataset_name} not found.")
    if os.path.exists(available_datasets[dataset_name].local_path):
        print(
            f"Found local copy of {dataset_name} at {available_datasets[dataset_name].local_path}"
        )
        if dataset_name == "papalexi_2021":
            dataset = mudata.read_h5mu(available_datasets[dataset_name].local_path)
        else:
            dataset = anndata.read_h5ad(available_datasets[dataset_name].local_path)
    else:
        print(f"No local copy found; downloading {dataset_name}...")
        with tempfile.TemporaryDirectory() as tempdir:
            dataset = available_datasets[dataset_name].preprocessing_function(tempdir)
    return dataset
