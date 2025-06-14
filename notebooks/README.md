
# Interactive notebooks

This folder is for short Jupyter demos of the package. They are not required
for training the models but can be handy for quick experiments.

## Available notebooks

- [loan_demo.ipynb](loan_demo.ipynb) – data loading, feature engineering
  and training example.
- [advanced_demo.ipynb](advanced_demo.ipynb) – grid search training,
  fairness evaluation and calibration.

Click the Binder badge at the bottom of this page to run these demos in the
cloud. Binder starts a temporary environment with all requirements so you can
execute the steps without installing anything locally.

Run the commands below to open a notebook:

```bash
# Ensure the Kaggle dataset is downloaded first
python scripts/download_data.py

# Then start Jupyter
jupyter notebook loan_demo.ipynb  # or advanced_demo.ipynb
```

You can also open the notebooks directly in Google Colab via the GitHub link.
Alternatively launch them in Binder:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/IvanStarostin1984/ML_classification/HEAD?labpath=notebooks%2Floan_demo.ipynb)
