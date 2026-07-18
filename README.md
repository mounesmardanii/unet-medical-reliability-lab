# U-Net Medical Reliability Lab

A reproducible deep-learning project for medical image segmentation and reliability analysis using U-Net.

## Project Goals

- Generate a deterministic ultrasound-like lesion dataset
- Train a U-Net model with Dice-BCE loss
- Select the segmentation threshold using validation data
- Estimate predictive uncertainty using MC Dropout
- Analyze risk-coverage behavior
- Evaluate robustness against noise, blur, and intensity shifts
- Perform overlap-tile inference on large images
- Export the trained model with TorchScript
- Document limitations in a model card

## Repository Structure

```text
unet-medical-reliability-lab/
├── data/raw/
├── notebooks/
├── src/
├── models/
├── reports/
│   ├── figures/
│   └── artifacts/
├── tests/
├── README.md
├── requirements.txt
└── .gitignore