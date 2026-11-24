# 5G Traffic Anomaly Detector

A machine learning project that trains a Random Forest model on 5G network traffic data and serves an interactive Streamlit dashboard.

## Project Structure
- `data/Combined.csv`: Dataset of 5G network traffic flows.
- `src/train.py`: Trains the Random Forest classifier and saves `src/rf_5g_model.joblib`.
- `src/predict.py`: Loads the saved model and runs sample traffic predictions in the terminal.
- `src/app.py`: Interactive Streamlit web dashboard with caching, traffic filtering, and CSV export.

## How to Run
1. Install dependencies:
   `pip install joblib pandas numpy scikit-learn streamlit`
2. Train the model:
   `python src/train.py`
3. Test predictions in terminal:
   `python src/predict.py`
4. Launch the dashboard:
   `python -m streamlit run src/app.py`