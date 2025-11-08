# credit-card-fraud-detection

credit-card-fraud-detection/
│
├── data/                    # All data files
│   ├── raw/                 # Original dataset
│   └── processed/           # Cleaned / split data
│
├── notebooks/               # Jupyter notebooks (EDA, experiments)
│   └── fraud_detection_eda.ipynb
│
├── src/                     # All main Python source code
│   ├── data_loader.py       # Load and preprocess data
│   ├── model_train.py       # Train and save ML model
│   ├── model_predict.py     # Load model and make predictions
│   ├── exception.py         # Custom exception handling
│   ├── logger.py            # Simple logging utility
│   └── config.py            # File paths and constants
│
├── models/                  # Saved ML model (.pkl)
│   └── model.pkl
│
├── app/                     # Flask app files
│   ├── app.py               # Main Flask application
│   └── templates/           # HTML templates for frontend
│       └── index.html
│
├── requirements.txt         # All dependencies
├── README.md                # Project description
└── .gitignore               # Ignore unnecessary files
