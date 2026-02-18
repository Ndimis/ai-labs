# 📈 Linear Regression – From Scratch & Dataset

This project introduces **linear regression** through simple, practical examples.
The goal is to understand both **the theory** and **real-world usage** without hiding
the math behind heavy frameworks.

---

## 🧠 What is Linear Regression?

Linear regression is a supervised learning algorithm used to model the relationship
between a **continuous target variable** `y` and one or more **input features** `x`.

### Simple linear regression (1 feature)

\[
y = w \cdot x + b
\]

Where:

- `w` is the **weight** (slope)
- `b` is the **bias** (intercept)

The model learns `w` and `b` by minimizing the **Mean Squared Error (MSE)**.

---

## 🎯 When to use Linear Regression?

✅ Predicting prices (house, rent, salary)  
✅ Modeling trends  
✅ Understanding feature influence  
❌ Not suited for non‑linear relationships

---

## 📂 Project Structure

``

## 📊 Dataset – California Housing

This project uses the **California Housing dataset** (1990 census data).

- Source: scikit-learn / California Housing
- Samples: 20,640
- Features: 8 numerical features
- Target: median house value

The dataset is **not committed** to the repository.

### Download dataset

```bash
python src/download_dataset.py
```
