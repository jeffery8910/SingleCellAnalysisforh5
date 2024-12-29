# SingleCellAnalysisforh5
# Brain Cell Type Analysis with GPU Acceleration

This Python script performs GPU-accelerated analysis of brain cell types using single-cell RNA sequencing data.

## Prerequisites

### Hardware Requirements
- NVIDIA GPU with CUDA support
- Minimum 16GB RAM recommended

### Software Requirements

bash
Create a new conda environment
conda create -n gpu-env python=3.10
Activate the environment
conda activate gpu-env
Install required packages
conda install -c conda-forge cupy cudatoolkit=11.8
conda install -c conda-forge scanpy
conda install -c conda-forge pandas numpy scipy
conda install -c conda-forge matplotlib seaborn
conda install -c conda-forge h5py
conda install tqdm

## Input Files Required

1. **10X Genomics H5 File**
   - Format: H5 file containing single-cell RNA sequencing data
   - Default path: `./neuron_10k_v3_raw_feature_bc_matrix.h5`

2. **Marker Genes CSV File**
   - Format: CSV file with columns 'Name' (gene names) and 'List' (cell types)
   - Default path: `./mm_brain_markers.csv`

## Usage

1. **Place your input files in the same directory as the script**
   - H5 data file
   - Marker genes CSV file

2. **Run the script**
bash
python brainintegrated_gpu.py
