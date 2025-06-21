# Microplastics Health Risk Estimator:
## Risk Distribution Overview

The Microplastics Health Risk Estimator is a machine learning tool designed to assess health risk from microplastic exposure using simulated lifestyle and environmental data informed by scientific research. It classifies individuals into low, medium, or high risk categories based on factors such as water intake, seafood consumption, plastic packaging use, residence type, and awareness level. The project features data processing, model training, visualization, and actionable insights for reducing exposure.

## Key Features
- **Scientifically grounded exposure modeling**
- **Multiple machine learning models** (Decision Tree, Random Forest, Logistic Regression)
- **Visual analytics for risk distribution, feature importance, and exposure analysis**
- **Actionable risk mitigation insights**

## Data Sources
- **Water**: 3.57 microplastics/L (bottled), 9.24 microplastics/L (tap) (Kosuth et al., 2018)
- **Seafood**: 10 microplastics per 100g (Van Cauwenberghe & Janssen, 2014)
- **Urban living and awareness impact** (Dris et al., 2017; Schnurr et al., 2018)

## Technical Requirements
- **Python 3.8+**
- **Libraries:** pandas, numpy, scikit-learn, matplotlib, seaborn

## Installation
1. **Clone the repository:**
   
```bash
git clone https://github.com/yourusername/Microplastics-Health-Risk-Estimator.git
cd Microplastics-Health-Risk-Estimator
```

2. **Install dependencies:**

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

3. **Run the project:**

- Open and run the provided Jupyter notebook or Python script.

- Results will be saved as PNG images in the results/ directory.

## Project Structure
```text
microplastics-risk-estimator/
├── data_processing.py
├── model_training.py
├── visualizations.py
├── requirements.txt
├── README.md
└── results/
    ├── feature_importance.png
    ├── risk_distribution.png
    ├── exposure_vs_risk.png
    └── correlation_matrix.png
```

## Key Insights
- **Top risk factors**: Seafood consumption, plastic packaging use, water intake
- **Urban residents and low awareness increase risk**
- **Risk distribution: ~37% Low, ~49% Medium, ~14% High**

**Recommendations**
- Limit seafood and plastic packaging consumption.
- Prefer tap water over bottled water
- Increase awareness of microplastic sources

## Future Improvements
- Integrate real-world environmental data
- Develop a mobile app for exposure tracking
- Add image recognition for microplastic detection
- Collaborate with public health agencies for data validation

## References
- Kosuth et al. (2018). Human consumption of microplastics. Environmental Science & Technology

- Van Cauwenberghe & Janssen (2014). Microplastics in bivalves. Environmental Pollution

- Dris et al. (2017). Microplastic contamination in urban environments. Environmental Chemistry

- Schnurr et al. (2018). Reducing marine plastic pollution. Marine Policy

