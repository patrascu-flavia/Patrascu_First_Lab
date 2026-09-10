# Multi-Source Pavement Crack Segmentation

Public research resources for multi-source pavement crack segmentation using ground-level and UAV imagery.

## Dataset and Trained Models

The full dataset and trained model checkpoints are archived on Zenodo.

**DOI:** https://doi.org/10.5281/zenodo.22697998

The public release contains:

- 3,564 pavement images
- 3,564 binary crack segmentation masks
- 3,564 image-mask pairs
- Train, validation, and test split files
- U-Net checkpoint
- U-Net++ checkpoint
- DeepLabV3+ checkpoint
- SegFormer checkpoint
- Exact PyTorch model architecture definitions
- Model-loading utility
- SHA-256 checksums

## Repository Contents

### `dataset/`

Contains documentation and the exact train, validation, and test split files used in the study.

The full image-mask dataset is hosted on Zenodo.

### `models/`

Contains the exact PyTorch architecture definitions required to reconstruct the four trained models, together with a model-loading utility.

The trained `.pt` checkpoints are hosted on Zenodo.

## Dataset Split

| Split | Images |
|---|---:|
| Training | 2,526 |
| Validation | 542 |
| Test | 496 |
| Total | 3,564 |

## Models

| Model | Best Epoch | Best Validation Dice | Final Threshold |
|---|---:|---:|---:|
| U-Net | 195 | 0.6250 | 0.85 |
| U-Net++ | 159 | 0.6244 | 0.85 |
| DeepLabV3+ | 390 | 0.6082 | 0.80 |
| SegFormer | 353 | 0.5595 | 0.85 |

All released architecture definitions were verified against the corresponding checkpoints using strict PyTorch state-dictionary loading.

## Zenodo

Dataset and trained models:

https://doi.org/10.5281/zenodo.22697998

## Authors

**Chaudhry Abu Bakar Imran**  
PhD Student  
Bert S. Turner Department of Construction Management  
College of Engineering  
Louisiana State University

**Flavia-Ioana Patrascu**  
Assistant Professor  
Bert S. Turner Department of Construction Management  
College of Engineering  
Louisiana State University

## License

The released dataset, model weights, metadata, and documentation are made available under the Creative Commons Attribution 4.0 International License.

See `LICENSE` for details.

## Citation

If you use the dataset or trained models, please cite:

Imran, Chaudhry Abu Bakar, and Flavia-Ioana Patrascu. (2026). *Multi-Source Pavement Crack Segmentation Dataset and Trained Deep Learning Models from Ground-Level and UAV Imagery*. Zenodo. https://doi.org/10.5281/zenodo.22697998

Please also cite the associated journal article when available.
