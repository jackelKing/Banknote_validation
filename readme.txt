# 🏦 Banknote Authentication Predictor

A **Gradio-based web app** that predicts whether a banknote is **Genuine** ✅ or **Fake** ❌ based on 4 key features: Variance, Skewness, Kurtosis, and Entropy.  

It also provides **interactive 2D and 3D visualizations** of the input and existing clusters.

---

## ✨ Features

- 🎛️ Interactive **sliders** for all 4 features.
- 🔮 Predicts **Genuine** or **Fake** banknotes.
- 📊 **2D Plot**: Variance vs Skewness.
- 🌐 **3D Plot**: Variance, Skewness, Kurtosis.
- ⚡ Uses **KMeans clustering** with automatic cluster-to-class mapping.
- ☁️ Deployment-ready for **Render** or any cloud platform.

---

## 📊 Dataset

- Features:

| Feature     | Description                                         | Range           |
|------------|----------------------------------------------------|----------------|
| Variance   | Measure of pixel value variation                  | 0.0 – 7.0      |
| Skewness   | Asymmetry of pixel value distribution            | -14.0 – 14.0   |
| Kurtosis   | Sharpness / peakedness of pixel distribution     | -10.0 – 20.0   |
| Entropy    | Randomness / complexity in pixel patterns        | -2.0 – 3.0     |
| Class      | 0 = Genuine, 1 = Fake                             | N/A            |

---



