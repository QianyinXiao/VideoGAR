# VideoGAR

This is the anonymous code repository for the paper "VideoGAR".

## Structure

```
VideoGAR/
├── method_core/      # Core model implementation
│   ├── model.py           # Main VideoGARModel
│   ├── bimamba.py         # BiMamba encoder layer
│   ├── query_decoder.py   # Query decoder for generative augmentation
│   ├── tfvtg_scoring.py   # TFVTG scoring module
│   ├── config.py          # Configuration
│   ├── train.py           # Training script
│   ├── inference.py       # Inference script
│   └── ...
├── utils/            # Utility functions
│   ├── basic_utils.py
│   ├── model_utils.py
│   ├── tensor_utils.py
│   └── ...
└── evaluation/       # Evaluation tools
    └── eval.py
```

## Key Components

### VideoGARModel

The main model class that implements:

- BiMamba encoder for efficient sequence modeling
- Fusion encoder for query-video interaction
- Query decoder for generative augmentation

### Core Features

- **BiMamba Backbone**: Efficient bidirectional Mamba encoder
- **Generative Augmentation**: Query reconstruction with fusion encoder

## Requirements

See the original paper for detailed requirements and setup instructions.
