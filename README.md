# Oxford-IIIT Pet Segmentation: A Comparative Deep Learning Study

This project compares several deep learning models for semantic segmentation of the Oxford-IIIT Pet Dataset. We implement and evaluate U-Net, convolutional autoencoders, and CLIP-based models — including a point-based prompt segmentation model. We explore performance, generalization, and robustness to various perturbations, and offer a practical user interface for interactive inference.

## Project Structure

```bash
.
├── src/
│   ├── main/                  # Training and testing logic for all models
│   │   └── main.py            # Entry point for training, validation, testing
│   │   └── ...
│   ├── evaluation/         # Data loading, resizing, augmentation, prompt creation
│   │   └── evaluation.py   # Robustness and evaluation metrics
│   │   └── ...
│   ├── preprocessing/         # Data loading, resizing, augmentation, prompt creation
│   │   └── preprocessing.py   # CLI to preprocess the dataset (resize, augment, add prompt dots)
│   │   └── ...
│   ├── utils.py               # Utility functions. Metrics, plotting, and others
│   └── models/                # PyTorch model architectures
│       ├── unet.py
│       ├── autoencoder.py
│       ├── clip_decoder.py
│       └── ...

```

## Getting Started

### 1. Preprocess Dataset

Resize images, apply augmentations, or generate prompt-based variants using:

```bash
python src/preprocessing/preprocessing.py --resize --augment --prompt
```

Options:
- `--resize`: Resize all images and masks to model-specific dimensions
- `--augment`: Apply controlled color jitter, elastic transform, etc.
- `--prompt`: Add a red dot (Gaussian blurred) at the barycentre of the animal

More information with ```-h``` as is a CLI script.

### 2. Train a Model

Train and validate your model (U-Net, Autoencoder, CLIP) from `src/main/`:

```bash
python src/main/main.py --model unet
```

Options:
- `--model`: Select from `unet`, `autoencoder`, `clip`, `prompt`

More information with ```-h``` as is a CLI script.

### 3. Evaluate

Evaluate model outputs using IoU, Dice, and accuracy:

```bash
python src/evaluation.py --model unet
```

More information with ```-h``` as is a CLI script.

## Models Implemented

- **U-Net**: CNN-based encoder-decoder with skip connections
- **Autoencoder**: Fully convolutional architecture with bottleneck
- **CLIP-based**: Uses pretrained CLIP vision encoder, custom transformer-based decoder
- **Prompt-based CLIP**: Adds point-based conditioning via red-dot prompts

## Robustness Testing

We apply a variety of real-world perturbations (e.g., Gaussian noise, blur, contrast shifts, occlusion) to test model resilience. Functions for perturbation are located in `src/evaluation.py`.

## Interactive Segmentation UI

An interactive GUI allows users to click on an image to place a prompt point and receive an instant mask prediction using the prompt-based model.

## Metrics

Models are compared using:
- **IoU (Intersection over Union)**
- **Dice Score (F1)**
- **Pixel Accuracy**

## Paper

For a detailed explanation of methodology, results, and analysis, refer to the accompanying report:  
`report.pdf`

## Contributors

- [Stanislas Castellana](https://github.com/StanCDev)
- [Rhodri Thomas](https://github.com/RTGT2021)

## Future Work

- Experiment with deeper/regularized decoders for CLIP
- Improve boundary reconstruction with hybrid transformer-CNN decoders
- Add multi-modal prompts (point + text)
- Extend robustness suite and visualization tools
