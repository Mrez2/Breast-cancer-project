# Breast Cancer Analysis Project

## Overview
This project analyzes the Breast Cancer Wisconsin (Diagnostic) Dataset to explore relationships between tumor characteristics and diagnoses. The analysis includes data exploration, visualization, and insights into key features that distinguish malignant (M) from benign (B) tumors.

## Dataset
The dataset contains 33 features computed from digitized images of breast mass fine needle aspirates (FNA), including:

- **Diagnosis**: Target variable (M = malignant, B = benign)
- **Radius**: Mean distance from center to perimeter
- **Texture**: Standard deviation of gray-scale values
- **Perimeter**: Tumor circumference
- **Area**: Tumor area
- **Smoothness**: Local variation in radius lengths
- **Compactness**: Perimeter² / area - 1.0
- **Concavity**: Severity of concave portions
- **Symmetry**: 
- **Fractal dimension**: "Coastline approximation"

## Key Visualizations

### 1. Radius vs. Area Relationship
![Radius vs Area](scatter_plot.png)
*Shows strong positive correlation between tumor radius and area, as expected in circular shapes*

### 2. Average Area by Diagnosis
![Diagnosis Comparison](bar_plot.png)
*Malignant tumors show significantly larger mean area compared to benign tumors*

## Key Findings
- Malignant tumors have larger average area (mean: 978 vs 462 units²)
- Higher concavity and compactness values correlate with malignancy
- Unnecessary column "Unnamed: 32" was removed during preprocessing
- Radius and area show strong positive correlation (R² ≈ 0.99)

## Dependencies
- Python 3.7+
- pandas
- numpy
- seaborn
- matplotlib
- jupyter
  
breast_cancer_analysis/
├── breast_cancer.ipynb          # Main analysis notebook
├── breast_cancer_data.csv        # Dataset
├── scatter_plot.png              # Radius vs. Area visualization
├── bar_plot.png                  # Diagnosis comparison
├── README.md                     # Project documentation
└── requirements.txt              # Python dependencies

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Mrez2/Breast-cancer-project.git
   cd Breast-cancer-project
