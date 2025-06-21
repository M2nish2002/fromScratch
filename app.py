import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Regression.models.BaseRegressionModel import Regression
from Regression.models.LassoRegressionModel import LassoRegression
from Regression.models.RidgeRegressionModel import RidgeRegression
from Regression.models.ElasticNetModel import ElasticNet

# Title
st.title("🔢 Regression from Scratch")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(df.head())

    # Input features and target selection
    columns = df.columns.tolist()
    X_cols = st.multiselect("Select Feature Columns", columns)
    y_col = st.selectbox("Select Target Column", columns)

    if X_cols and y_col:
        X = df[X_cols].values.astype(float)
        y = df[y_col].values.astype(float)

        # Regularization type
        model_type = st.selectbox("Choose Regularization Type", ["None", "L1 (Lasso)", "L2 (Ridge)", "L1 + L2 (Elastic Net)"])

        # Hyperparameters
        n_iterations = st.slider("Number of Iterations", 100, 5000, step=100, value=1000)
        learning_rate = st.number_input("Learning Rate", value=0.01)
        alpha = st.number_input("Alpha (Regularization Strength)", value=0.01)
        L1_ratio = 0.5
        if model_type == "L1 + L2 (Elastic Net)":
            L1_ratio = st.slider("L1 Ratio (0 = L2 only, 1 = L1 only)", 0.0, 1.0, 0.5, 0.05)

        if st.button("Train Model"):
            # Train appropriate model
            if model_type == "None":
                model = Regression(n_iterations=n_iterations, learn_rate=learning_rate)
                model.regularization = lambda w: 0
                model.regularization.grad = lambda w: np.zeros_like(w)
            elif model_type == "L1 (Lasso)":
                model = LassoRegression(alpha=alpha, n_iterations=n_iterations, learn_rate=learning_rate)
            elif model_type == "L2 (Ridge)":
                model = RidgeRegression(alpha=alpha, n_iterations=n_iterations, learn_rate=learning_rate)
            elif model_type == "L1 + L2 (Elastic Net)":
                model = ElasticNet(alpha=alpha, L1_ratio=L1_ratio, n_iterations=n_iterations, learn_rate=learning_rate)

            # Fit model
            model.fit(X, y)
            predictions = model.predict(X)

            # Plot predictions vs actual
            st.subheader("📊 Actual vs Predicted")
            fig, ax = plt.subplots()
            ax.scatter(y, predictions, alpha=0.6)
            ax.plot([min(y), max(y)], [min(y), max(y)], 'r--')
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title("Predicted vs Actual")
            st.pyplot(fig)

            # Plot training loss
            st.subheader("📉 Training Loss Over Iterations")
            fig2, ax2 = plt.subplots()
            ax2.plot(model.training_errors, label="Training Loss")
            ax2.set_xlabel("Iterations")
            ax2.set_ylabel("Loss")
            ax2.legend()
            st.pyplot(fig2)
