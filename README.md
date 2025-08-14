<<<<<<< HEAD
# SSL-Industrial-Defect-Detection-FP-NEW-START
This project is a fresh restart of my final year project due to technical issue. The earlier version and commit history can be found here https://github.com/kashaf739/MSc-Final-Project
=======
>>>>>>> ca2ad72 (Initial comit research data with models)


# Self-Supervised Learning for Industrial Defect Detection

This project explores the application of self-supervised learning (SSL) for industrial defect detection using the MVTec AD dataset. The implementation includes two SSL models (experimental and advanced) trained with SimCLR and a supervised CNN baseline for comparison.

## 📋 Project Overview

### Problem Statement
Traditional defect detection systems rely on supervised learning, which requires extensive labeled datasets. This project investigates whether SSL models trained exclusively on normal samples can achieve competitive performance in defect detection without requiring labeled defect data.

### Key Achievements
- **Advanced SSL Model**: Achieved F1-score of 0.845 without using any labeled defect data
- **Experimental SSL Model**: ResNet18-based implementation trained for 55 epochs
- **CNN Baseline**: Supervised model achieving F1-score of 0.962 with full labels
- **Comprehensive Evaluation**: Quantitative metrics and qualitative visualizations (t-SNE, Grad-CAM)

## 📊 Dataset

### MVTec AD Dataset
- **Categories Used**: bottle, screw, metal_nut, capsule, cable (5 out of 15)
- **Training Data**: Only normal samples from `train/good` directories
- **Test Data**: Both normal and defective samples from `test` directories
- **Total Test Samples**: 316 (116 normal, 200 defective)

### Data Statistics
| Category   | Normal Test | Defective Test | Total |
|------------|-------------|----------------|-------|
| Bottle     | 20          | 63             | 83    |
| Screw      | 39          | 39             | 78    |
| Metal Nut  | 18          | 28             | 46    |
| Capsule    | 19          | 30             | 49    |
| Cable      | 20          | 40             | 60    |

## 🤖 Models

### 1. Experimental SSL Model (SimCLR)
- **Backbone**: ResNet18
- **Feature Dimension**: 128
- **Training Epochs**: 55
- **Best Performance**: F1-score of 0.761 (cosine similarity method)
- **Key Features**: Basic SimCLR implementation with standard augmentations

### 2. Advanced SSL Model (SimCLR)
- **Backbone**: ResNet50
- **Feature Dimension**: 256
- **Training Epochs**: 4 (selected from interrupted training)
- **Best Performance**: F1-score of 0.845 (cosine similarity on projected features)
- **Advanced Techniques**:
  - MixUp and CutMix augmentations
  - Exponential Moving Average (EMA)
  - Cosine annealing with warm restarts
  - Mixed precision training

### 3. Supervised CNN Baseline
- **Architecture**: Custom CNN with dual output heads
- **Training**: Full supervision with labeled data
- **Best Performance**: F1-score of 0.962
- **Output Heads**:
  - Category classification (15 classes)
  - Defect classification (binary)

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- PyTorch 1.9+
- TensorFlow 2.8+
- Google Colab Pro account (recommended for GPU access)t4

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/ssl-industrial-defect-detection.git
cd ssl-industrial-defect-detection

# Install dependencies
pip install -r requirements.txt
```

### Dataset Preparation
1. Download the MVTec AD dataset from [official source](https://www.mvtec.com/company/research/datasets/mvtec-ad/)
2. Extract the dataset to `data/mvtec_ad/`
3. Ensure the following structure:
   ```
   data/mvtec_ad/
   ├── bottle/
   ├── screw/
   ├── metal_nut/
   ├── capsule/
   └── cable/
   ```

## 📈 Training

### SSL Model Training
```bash
# Experimental SSL Model
python train_experimental_ssl.py

# Advanced SSL Model
python train_advanced_ssl.py
```

### CNN Baseline Training
```bash
python train_cnn_baseline.py
```

## 🔍 Evaluation

### SSL Model Evaluation
```bash
# Experimental Model Evaluation
python evaluate_experimental_ssl.py

# Advanced Model Evaluation
python evaluate_advanced_ssl.py
```

### CNN Baseline Evaluation
```bash
python evaluate_cnn_baseline.py
```

### Comprehensive Evaluation
```bash
# Run all evaluations and generate comparison report
python run_comprehensive_evaluation.py
```

## 📊 Results Summary

### Performance Comparison
| Model                | F1-Score | Accuracy | ROC-AUC | Training Data       |
|----------------------|-----------|----------|---------|--------------------|
| Advanced SSL         | 0.845     | 0.838    | 0.912   | Normal samples only |
| Experimental SSL     | 0.761     | 0.754    | 0.841   | Normal samples only |
| CNN Baseline         | 0.962     | 0.968    | N/A     | Full labels all 15 categories|

### Key Findings
1. **SSL Effectiveness**: The advanced SSL model achieved competitive performance (84.5% F1-score) without any labeled defect data
2. **Training Efficiency**: SSL models achieved good performance in fewer epochs (4-55) compared to CNN (20 epochs)
3. **Feature Quality**: Projected features outperformed encoder features in SSL models
4. **Anomaly Detection**: Cosine similarity and KNN were the most effective anomaly detection methods

## 📁 Project Structure

ssl-industrial-defect-detection/
├── data/
│   └── mvtec_ad/                 # MVTec AD dataset
├── models/
│   ├── BIG5_epoch_55/         # Experimental model checkpoints
│   ├── UltraEfficient_SimCLR_epoch_004(the-best-model)/  # Advanced model checkpoints
│   └── best_cnn_model/        # CNN model checkpoints
│       └── label_encoders_cnn_20250801-114707
├── Notebooks/
│   ├── SIM_CLR/
│   │   ├── sim-clr-best-model.py
│   │   └── sim-clr-best-model-evaluations.py
│   ├── CNN/
│   │   ├── cnn_model.py
│   │   ├── accuracy-recall-f1.py
│   │   ├── Grad_CAM Evaluations.py
│   │   └── t-SNE Evaluations.py
│   ├── CNN-EVALUATION/
│   │   ├── confusion_metrics/     # Evaluation results and metrics
│   │   ├── visualizations/        # t-SNE plots, Grad-CAM heatmaps
│   │   └── reports/               # Evaluation reports
│   └── SIM_CLR-EVALUATION/
│       ├── cluster_analysis/
│       ├── feature_analysis/
│       ├── metrics/
│       ├── model_comparisons/
│       └── tsne_plots/
├── sim_clr-0-to-50-epoch-train-script.py         # Experimental model
├── sim_clr-50-to-100-epoch-train-script.py        # Experimental model
├── sim_clr-55-epoch-model_evaluation-script.py   # Experimental model evaluation script
├── requirements.txt
├── README.md
├── .gitignore
├── sim_clr anomaly score distribution (mahalanobis) 55 epoch.png          # Experimental model evaluation
├── sim_clr confusion matrix (mahalanobis) 55 epoch.png                     # Experimental model visualization
├── sim_clr method comparison 55 epoch evaluation.png                    # Experimental model visualization
├── sim_clr PCA and Commulative Explained Variance 55 epoch.png               # Experimental model evaluation
├── sim_clr performance matrics 55 epoch.png                  # Experimental model evaluation
├── sim_clr roc curves comparison 55 epoch.png                    # Experimental model evaluation
├── sim_clr t-SNE embeddings (normal vs anomaly) 55 epoch.png # Experimental model visualization
├── sim_clr t-SNE evaluation 55 epoch.png   # Experimental model visualization
├── sim_clr-0-to-20-train-eval.png   # Experimental model visualization
├── sim_clr-0-to-40-train-eval.png    # Experimental model visualization
├── sim_clr-0-to-50-train-eval.png   # Experimental model visualization
└── ultra-efficient models drive screenshot.png    # Experimental model visualization
##Key Scripts

##Training Scripts

 sim_clr-0-to-50-epoch-train-script.py: Trains experimental ResNet18-based SimCLR model (epochs 0-50)
 sim_clr-50-to-100-epoch-train-script.py: Continues training experimental ResNet18-based SimCLR model (epochs 50-100)
 Notebooks/SIM_CLR/sim-clr-best-model.py: Trains advanced ResNet50-based SimCLR model with advanced techniques
 Notebooks/CNN/cnn_model.py: Trains supervised CNN baseline
 
 ##Evaluation Scripts

 sim_clr-55-epoch-model_evaluation-script.py: Evaluates experimental SSL model (at epoch 55)
 Notebooks/SIM_CLR/sim-clr-best-model-evaluations.py: Evaluates advanced SSL model
 Notebooks/CNN/accuracy-recall-f1.py: Evaluates CNN baseline (accuracy, recall, F1 metrics)
 Notebooks/CNN/Grad_CAM Evaluations.py: Generates Grad-CAM visualizations for CNN baseline
 Notebooks/CNN/t-SNE Evaluations.py: Generates t-SNE visualizations for CNN baseline
 Visualization & Analysis Files

##The following experimental model evaluation outputs are located in the root directory:

 sim_clr anomaly score distribution (mahalanobis) 55 epoch.png
 sim_clr confusion matrix (mahalanobis) 55 epoch.png
 sim_clr method comparison 55 epoch evaluation.png
 sim_clr PCA and Commulative Explained Variance 55 epoch.png
 sim_clr performance matrics 55 epoch.png
 sim_clr roc curves comparison 55 epoch.png
 sim_clr t-SNE embeddings (normal vs anomaly) 55 epoch.png
 sim_clr t-SNE evaluation 55 epoch.png
 sim_clr-0-to-20-train-eval.png
 sim_clr-0-to-40-train-eval.png
 sim_clr-0-to-50-train-eval.png
 ultra-efficient models drive screenshot.png
## 📈 Visualizations

The project includes comprehensive visualizations:

1. **t-SNE Plots**: Visualize feature space clustering
2. **Grad-CAM Heatmaps**: Show model attention regions
3. **Confusion Matrices**: Display classification performance
4. **Training Curves**: Track loss and accuracy over epochs
5. **Score Distributions**: Compare normal vs. anomalous sample scores

## 🎯 Success Criteria Achievement

| Success Criteria                          | Status     | Result |
|------------------------------------------|------------|--------|
| SSL models achieve F1 ≥ 75%             | ✅ Achieved | 84.5%  |
| Embedding space visually separable        | ✅ Achieved | Clear separation |
| BYOL convergence (not implemented)      | ❓ Not Done(optional mentioned) | N/A    |
| Comprehensive evaluation completed       | ✅ Achieved | Full analysis |

## ⚠️ Limitations

1. **Dataset Scope**: Only 5 out of 15 MVTec AD categories were used for SSL Model training
2. **BYOL Implementation**: Not implemented due to computational constraints also in proposal already mentioned optional
3. **Real-wold Testing**: Evaluation was performed in simulated environment no TESTING on any dataset just evaluations on the basis of model performance (training results)
4. **Computational Resources**: Training was interrupted for advanced model and previous so many experimental models (available in git hub old directory not included in final project)

## 🔮 Future Work

1. **Extended Categories**: Evaluate on all 15 MVTec AD categories
2. **BYOL Implementation**: Compare different SSL frameworks
3. **Real-world Deployment**: Test in industrial setting
4. **Few-shot Learning**: Evaluate with minimal labeled data
5. **Model Optimization**: Reduce computational requirements

## 📝 Citation

If you use this code or findings in your research, please cite:

```bibtex
@misc{kashaf2024_ssl_industrial_defect,
  title={Self-Supervised Learning for Industrial Defect Detection},
  author={Kashaf, Wajeeha},
  year={2024},
  howpublished={\url{https://github.com/yourusername/ssl-industrial-defect-detection}}
}
```

## 📄 License

This project is acaademic achievement.

## 🙏 Acknowledgments

- **MVTec** for providing the industrial anomaly detection dataset
- **Google Colab** for providing GPU resources
- **Supervisors**: Yash Bardararanshokouhi and Dr. Xin Lu for their guidance and support

## 📞 Contact

For questions or suggestions, please open an issue or contact:
- **Name**: Wajeeha Kashaf
- **Email**: wajeehakashaf739@gmail.com
<<<<<<< HEAD
- **Student ID**: 2404372
=======
- **Student ID**: 2404372
>>>>>>> ca2ad72 (Initial comit research data with models)
