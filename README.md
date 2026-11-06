# Steel Surface Defect Classification (NEU-DET)
An end-to-end Deep Learning project comparing a **Baseline CNN** against **MobileNetV3Small (Transfer Learning + Fine-tuning)** to identify 6 defect types on steel surfaces.

## Project Architecture
- **Baseline CNN**: 3 Convolutional blocks with Global Average Pooling (110,534 parameters).
- **MobileNetV3Small**: Transfer learning + Fine-tuning the top 30 layers on the NEU-DET classification dataset (1,013,750 parameters).

## Dataset
The dataset is based on the **NEU Surface Defect Database** consisting of 1,800 images distributed evenly across 6 categories:
- Crazing
- Inclusion
- Patches
- Pitted Surface
- Rolled-in Scale
- Scratches

Link to Dataset: [Kaggle NEU Surface Defect Database](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database)

## Directory Structure
```
├── models/
│   ├── mobilenetv3_neu.keras      # Best performing fine-tuned model
│   └── baseline_cnn_neu.keras     # Custom baseline CNN model
├── outputs/
│   ├── training_curves.png        # Training progression curves
│   ├── cm_Baseline_CNN.png        # Confusion matrix for Baseline CNN
│   ├── cm_MobileNetV3Small.png    # Confusion matrix for MobileNetV3Small
│   ├── misclassified.png          # Error analysis plot
│   └── results.csv                # Tabulated final metric comparisons
├── src/
│   └── app.py                     # Gradio application server script
├── requirements.txt               # Dependencies list
└── README.md                      # Documentation
```

## Performance Comparison
| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Parameters |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline CNN** | 91.85% | 92.01% | 91.85% | 91.66% | 110,534 |
| **MobileNetV3Small** | **99.63%** | **99.64%** | **99.63%** | **99.63%** | 1,013,750 |

## Getting Started & Execution

### Prerequites
Make sure python 3.9+ is installed.

### 1. Clone the repo and install dependencies
```bash
git clone <your-github-repo-url>
cd <your-github-repo-name>
pip install -r requirements.txt
```

### 2. Launching the Gradio App
You can run the interactive user interface locally by running:
```bash
python src/app.py
```

### 3. Model Outputs
All evaluation plots, error analysis charts, and comparative metrics are saved inside the `outputs/` folder.
