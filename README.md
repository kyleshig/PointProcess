# PointProcess Tennis Match Predictor: From Standard ML to Stochastic Modeling

A sophisticated tennis match prediction system that evolves from traditional machine learning to advanced stochastic differential equation modeling.

## Project Overview

This project demonstrates the progression from standard machine learning approaches to mathematically sophisticated stochastic modeling for tennis match prediction. The system uses ML model outputs as drift terms in stochastic differential equations to capture the inherent uncertainty in tennis matches.

## Features

### Phase 1: Standard ML Predictor
- ELO rating system with surface-specific adjustments
- Recent form analysis with exponential weighting
- Head-to-head historical performance
- Ensemble machine learning models

### Phase 2: Stochastic Enhancement
- Stochastic Differential Equation (SDE) framework
- Monte Carlo simulation engine
- Point-by-point match progression modeling
- Uncertainty quantification and confidence intervals

## Technical Stack
- **Languages**: Python
- **ML Libraries**: scikit-learn, XGBoost
- **Mathematical Computing**: NumPy, SciPy
- **Data Processing**: Pandas
- **Web Framework**: Streamlit/Flask
- **Visualization**: Matplotlib, Plotly

## Mathematical Components
- Stochastic Differential Equations
- Monte Carlo Methods
- Time Series Analysis
- Probability Calibration
- Numerical Methods (Euler-Maruyama)

## Getting Started
```bash
# Clone repository
git clone https://github.com/kyleshig/PointProcess.git
cd PointProcess

# Set up environment
python3 -m venv tennis_env
source tennis_env/bin/activate  # On Windows: tennis_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run data collection
python src/data_collection.py
```
## Project Structure
tennis-match-predictor/
├── data/                   # Raw and processed data
├── notebooks/              # Jupyter notebooks for analysis
├── src/                    # Source code modules
│   ├── data_processor.py   # Data cleaning and processing
│   ├── feature_engineer.py # Feature engineering
│   ├── models/            # ML and stochastic models
│   └── web_app/           # Web application
├── tests/                 # Unit tests
├── docs/                  # Documentation
└── requirements.txt       # Python dependencies

## Author
Kyle Shigekawa - UCLA Mathematics of Computation

## License
MIT License