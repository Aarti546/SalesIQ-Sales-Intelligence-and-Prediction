# SalesIQ - Sales Intelligence and Prediction

SalesIQ is an interactive sales analytics and machine learning dashboard built with Python and Streamlit. It explores Amazon India sales data, visualizes business trends, compares predictive models, and provides a cancellation prediction interface.

## Features

- Interactive Streamlit dashboard with a clean, professional light theme
- Sales overview with revenue, record count, cancellation rate, and average order value
- Category, fulfilment, B2B, distribution, and monthly trend analysis
- Colourful Plotly charts with consistent white backgrounds
- Regression model comparison for sales amount prediction
- Cancellation classification model comparison
- Confusion matrix and actual-versus-predicted visualizations
- Interactive cancellation prediction form
- Searchable dataset explorer

## Dashboard Pages

- **Overview** - High-level sales KPIs and category/cancellation summaries
- **Sales EDA** - Exploratory filters and visual analysis
- **Model Comparison** - Regression and classification performance tables
- **Model Details** - Individual model plots and evaluation details
- **Cancellation Prediction** - Predict whether an order is likely to be cancelled
- **Dataset Explorer** - Filter and download the cleaned dataset

## Dataset

The project uses `Amazon India Sales Data.csv`, which contains approximately 65,000 Amazon India order records from March-May 2022.

Relevant fields include:

- `date`
- `status`
- `fulfilment`
- `ship_service_level`
- `category`
- `size`
- `qty`
- `amount`
- `ship_state`
- `b2b`

The `is_cancelled` target is derived from `status`; the original `status` field is excluded from classification features to prevent target leakage.

## Machine Learning

### Regression

The regression models use order quantity (`qty`) to estimate sales amount (`amount`).

- Linear Regression
- Random Forest Regressor

Evaluation metrics:

- Mean Absolute Error (MAE)
- R-squared (R2)

Quantity alone has limited predictive power, so both regression models may have low R2 scores.

### Classification

The classification models predict whether an order is likely to be cancelled.

- Logistic Regression
- Decision Tree Classifier
- K-Nearest Neighbors

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

The training process uses an 80/20 stratified train-test split and preprocessing pipelines for numeric and categorical features.

## Installation

Requires Python 3.9 or later.

```bash
git clone https://github.com/Aarti546/SalesIQ-Sales-Intelligence-and-Prediction.git
cd SalesIQ-Sales-Intelligence-and-Prediction
python -m pip install -r requirements.txt
```

## Run the Dashboard

```bash
python -m streamlit run app.py
```

Open the local URL shown by Streamlit, normally `http://localhost:8501`.

The dashboard must be launched from the project directory because it reads `Amazon India Sales Data.csv` using a relative path.

## Run the Python Analysis

The standalone analysis script can be run with:

```bash
python main.py
```

`main.py` performs data cleaning, exploratory analysis, model training, evaluation, and static visualizations.

## Project Structure

```text
SalesIQ/
|-- app.py
|-- main.py
|-- README.md
|-- requirements.txt
|-- Amazon India Sales Data.csv
|-- Amazon India Sales DataExcel.xlsx
|-- .streamlit/
|   `-- config.toml
`-- images/
```

## Technologies

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- Scikit-learn
- Matplotlib
- Seaborn
- OpenPyXL

## Notes

- Model results describe historical data and should not be interpreted as guaranteed future performance.
- Recorded sales totals are not necessarily realized revenue.
- Improvements could include richer features, date-based seasonality, product attributes, hyperparameter tuning, and time-aware validation.
