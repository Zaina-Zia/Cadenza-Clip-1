# Lyric Intelligibility Baselines

This project implements four baseline models for lyric intelligibility prediction:

- **STOI baseline**  
- **Whisper-based baseline**  
- **CNN + MFCC baseline**  
- **Wav2Vec2 baseline**

Each baseline predicts intelligibility scores on the validation split of the training data, ensuring that ground-truth targets are available for evaluation.  
The evaluation metrics include:

- **RMSE**  
- **Standard Deviation**  
- **Normalized Cross-Correlation (NCC)**  
- **Kendall’s Tau**

---

## Running the Baselines

To run any of the baselines:

1. Verify dataset paths inside each script.  
2. Install all required dependencies (see `requirements.txt`).  
3. Run the corresponding training or prediction script for the baseline.  
4. Run the evaluation script to compute **RMSE, STD, NCC, and Kendall's Tau**.

Outputs:

- Prediction files are saved to the **`results/`** directory.  
- Any generated visualizations are stored under **`plots/`**.

---

## Repository Structure

```
data/            - Dataset files and loaders
preprocessing/   - MFCC extraction, resampling, and audio processing
models/          - Model definitions for all baselines
training/        - Training and prediction scripts
evaluation/      - Metric computation utilities
results/         - Stored predictions from baselines
plots/           - Figures and visualizations
```

The official **STOI** and **Whisper** baselines were cloned from the Clarity GitHub repository.  
The **CNN+MFCC** and **Wav2Vec2** baselines are implemented within this project.

---

## Assignment Repository

The original assignment repository is available here:  
https://github.com/Zaina-Zia/Cadenza-Clip-1
