import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


def scale(X_new):
    mu = np.mean(X_new, axis=0)
    sigma = np.std(X_new, axis=0)
    X_norm = (X_new - mu) / sigma
    return mu, sigma, X_norm


def predict(X_new, w, b):
    return np.dot(X_new, w) + b


def cost(X_new, w, b, y):
    m = X_new.shape[0]
    error = predict(X_new, w, b) - y
    return np.sum(error**2) / (2 * m)


def gd(X_new, w, b, y):
    m = X_new.shape[0]
    error = predict(X_new, w, b) - y
    gdw = np.dot(X_new.T, error) / m
    gdb = np.sum(error) / m
    return gdw, gdb


def build_features(X):
    cost_per_unit = X[:, 1] / X[:, 0]
    efficiency = X[:, 2] / X[:, 0]
    interaction = X[:, 2] * X[:, 3]
    X_new = np.column_stack((X, cost_per_unit, efficiency, interaction))
    return X_new


def train_model():
    X = np.array([
        [1000, 50, 200, 2],
        [1500, 45, 220, 3],
        [2000, 40, 250, 5],
        [1200, 55, 210, 2],
        [1800, 42, 240, 4],
        [2200, 38, 260, 6],
        [1300, 52, 205, 2],
        [1700, 43, 230, 3]
    ], dtype=float)

    y = np.array([
        [50000],
        [62000],
        [71000],
        [54000],
        [68000],
        [76000],
        [52000],
        [66000]
    ], dtype=float)

    X_new = build_features(X)
    mu, sigma, X_norm = scale(X_new)

    w = np.zeros((7, 1))
    b = 0.0
    alpha = 0.001
    cost_history = []

    for i in range(10000):
        costv = cost(X_norm, w, b, y)
        cost_history.append(costv)
        gdw, gdb = gd(X_norm, w, b, y)
        w = w - alpha * gdw
        b = b - alpha * gdb

    predictions = predict(X_norm, w, b)
    error = predictions - y

    return X, y, w, b, mu, sigma, cost_history, predictions, error


st.title("Manufacturing Cost Prediction App")
st.write("Simple linear regression app based on your manual code.")

X, y, w, b, mu, sigma, cost_history, predictions, error = train_model()

st.subheader("Enter a new scenario")

volume = st.number_input("Production Volume", min_value=1.0, value=1000.0)
material = st.number_input("Material Cost", min_value=0.0, value=50.0)
labor = st.number_input("Labor Hours", min_value=0.1, value=200.0)
defect = st.number_input("Defect Rate", min_value=0.0, value=2.0)

if st.button("Predict Cost"):
    sample = np.array([[volume, material, labor, defect]], dtype=float)
    sample_new = build_features(sample)
    sample_norm = (sample_new - mu) / sigma
    prediction = predict(sample_norm, w, b)

    st.success(f"Predicted Total Cost: {prediction[0, 0]:,.2f}")

st.subheader("Training Cost History")
fig1, ax1 = plt.subplots()
ax1.plot(cost_history)
ax1.set_xlabel("Iteration")
ax1.set_ylabel("Cost")
ax1.set_title("Cost vs Iteration")
ax1.grid(True)
st.pyplot(fig1)

st.subheader("Actual vs Predicted")
fig2, ax2 = plt.subplots()
ax2.scatter(y, predictions)
ax2.set_xlabel("Actual")
ax2.set_ylabel("Predicted")
ax2.set_title("Actual vs Predicted")
ax2.grid(True)
st.pyplot(fig2)

st.subheader("Model Weights")
st.write(w)

st.subheader("Training Errors")
st.write(error)