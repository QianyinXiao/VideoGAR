# Video-GAR: Generation-Augmented Video Corpus Moment Retrieval

Official PyTorch implementation of **“Generation-Augmented Video Corpus Moment Retrieval”**, published at the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2026).

- **Authors:** Mingjin Kuai, Qianyin Xiao, Juncheng Li, Jin Peng, Lizi Liao, and Wei Ji
- **Paper:** [ACM Digital Library](https://doi.org/10.1145/3805712.3809742)
- **Code:** [github.com/QianyinXiao/VideoGAR](https://github.com/QianyinXiao/VideoGAR)

Video-GAR addresses video corpus moment retrieval (VCMR) with a Bi-Mamba backbone and a training-only generative query-reconstruction objective. The generative decoder regularizes video-text alignment during training and adds no decoder overhead at inference time.

## Implementation scope

The released inference path uses the model's learned start/end boundary predictors directly. An experimental post-processing scorer that was not used in the paper is not part of this release and is not required to train or evaluate Video-GAR.

The implementation is based on [ReLoCLNet](https://github.com/26hzhang/ReLoCLNet). Dataset features and evaluation data follow the setup described in the paper and the ReLoCLNet repository.

## Requirements

Video-GAR requires Python 3.9–3.11 and a CUDA-capable GPU. The environment follows the dependency baseline used by the authors' related [VERSE](https://github.com/QianyinXiao/VERSE) repository.

| Package | Version |
|---|---|
| PyTorch | 2.4.0 (CUDA 11.8) |
| torchvision | 0.19.0 |
| mamba-ssm | 2.3.0 |
| transformers | >=4.38.2, <5 |
| tokenizers | >=0.15.2, <0.20 |
| NumPy | >=1.24, <2 |
| h5py | >=3.10, <4 |
| TensorBoard | >=2.15, <3 |
| easydict | >=1.13, <2 |
| tqdm | >=4.66, <5 |

The complete resolved environment is recorded in `uv.lock`. On Linux x86-64, the lock uses the official prebuilt `mamba-ssm` wheel for Python 3.9–3.11, so a local CUDA toolkit and `nvcc` are not required for installation.

## Installation

The recommended installation uses [uv](https://docs.astral.sh/uv/) and the CUDA 11.8 wheels declared in `pyproject.toml`.

```bash
git clone https://github.com/QianyinXiao/VideoGAR.git
cd VideoGAR

pip install uv
uv python install 3.9
uv sync --extra gpu-cu118 --extra bimamba
source .venv/bin/activate
source setup.sh
```

If PyTorch is already managed by your cluster, install the core dependencies with `uv sync` and install a PyTorch/`mamba-ssm` build compatible with the cluster's CUDA toolchain separately.

Verify the installation and run the synthetic GPU smoke tests with:

```bash
python -m unittest discover -s tests -v
```

The GPU tests are skipped automatically when CUDA or the optional Bi-Mamba dependencies are unavailable.

## Repository structure

```text
VideoGAR/
├── method_core/
│   ├── model.py              # VideoGARModel and fusion encoder
│   ├── bimamba.py            # Bidirectional Mamba encoder layer
│   ├── query_decoder.py      # Training-only query decoder
│   ├── config.py             # Training and evaluation options
│   ├── train.py              # Training entry point
│   └── inference.py          # VCMR/SVMR/VR inference
├── evaluation/
│   └── eval.py               # Retrieval metrics
└── utils/                    # Data, tensor, and model utilities
```

## Main components

- **Bi-Mamba backbone:** efficient bidirectional sequence modeling for long video contexts.
- **Generative augmentation:** query reconstruction acts as an auxiliary training objective.
- **Fusion encoder:** provides query-conditioned video memory to the training-only decoder.
- **Boundary-aware moment localization:** predicts start and end boundaries for retrieved moments.

## Citation

If this repository or the paper is useful in your research, please cite:

```bibtex
@inproceedings{kuai2026generation,
  author    = {Mingjin Kuai and Qianyin Xiao and Juncheng Li and Jin Peng and Lizi Liao and Wei Ji},
  title     = {Generation-Augmented Video Corpus Moment Retrieval},
  booktitle = {Proceedings of the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  publisher = {ACM},
  year      = {2026},
  pages     = {857--868},
  doi       = {10.1145/3805712.3809742},
  url       = {https://doi.org/10.1145/3805712.3809742}
}
```

## License

This project is released under the [MIT License](LICENSE).

**Keywords:** Video-GAR, VideoGAR, video corpus moment retrieval, VCMR, temporal video grounding, moment localization, video-text retrieval, generative augmentation, Bi-Mamba, SIGIR 2026.
