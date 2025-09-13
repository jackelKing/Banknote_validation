import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score
import gradio as gr
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv("data_banknote_authentication.csv")
X = df[['variance','skewness','kurtosis','entropy']]
y = df['class']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=2, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

def adjust_clusters(df, cluster_col='Cluster', true_col='class'):
    df = df.copy()
    mapping = {}
    for cluster_id in np.unique(df[cluster_col]):
        majority_label = df[df[cluster_col] == cluster_id][true_col].mode()[0]
        mapping[cluster_id] = majority_label
    df['Cluster_mapped'] = df[cluster_col].map(mapping)
    
    conf_mat = confusion_matrix(df[true_col], df['Cluster_mapped'])
    accuracy = accuracy_score(df[true_col], df['Cluster_mapped'])
    return df, conf_mat, accuracy

df_adjusted, conf_mat, acc = adjust_clusters(df)
print("Adjusted Confusion Matrix:\n", conf_mat)
print(f"Clustering Accuracy: {acc:.3f}")


def predict_banknote_gradio(variance, skewness, kurtosis, entropy):
    new_data = [[variance, skewness, kurtosis, entropy]]
    new_data_scaled = scaler.transform(new_data)
    
    cluster_pred = kmeans.predict(new_data_scaled)[0]
    majority_label = df_adjusted[df_adjusted['Cluster'] == cluster_pred]['Cluster_mapped'].mode()[0]
    label = "Genuine" if majority_label == 0 else "Fake"

    # 2D Plot (Variance vs Skewness)
    fig2d, ax2d = plt.subplots(figsize=(6,5))
    sns.scatterplot(
        x=X_scaled[:,0], y=X_scaled[:,1],
        hue=df_adjusted['Cluster_mapped'], palette='Set1', alpha=0.6, ax=ax2d
    )
    ax2d.scatter(new_data_scaled[0][0], new_data_scaled[0][1],
                 color='black', s=150, marker='X', label='New Note')
    ax2d.set_xlabel("Variance (scaled)")
    ax2d.set_ylabel("Skewness (scaled)")
    ax2d.set_title("Banknote Classification (2D)")
    ax2d.legend()

    # 3D Plot (Variance, Skewness, Kurtosis)
    fig3d = plt.figure(figsize=(7,6))
    ax3d = fig3d.add_subplot(111, projection='3d')
    ax3d.scatter(
        X_scaled[:,0], X_scaled[:,1], X_scaled[:,2],
        c=df_adjusted['Cluster_mapped'], cmap='Set1', s=50, alpha=0.6
    )
    ax3d.scatter(new_data_scaled[0][0], new_data_scaled[0][1], new_data_scaled[0][2],
                 color='black', s=150, marker='X', label='New Note')
    ax3d.set_xlabel("Variance (scaled)")
    ax3d.set_ylabel("Skewness (scaled)")
    ax3d.set_zlabel("Kurtosis (scaled)")
    ax3d.set_title("Banknote Classification (3D)")
    ax3d.legend()

    return label, fig2d, fig3d


iface = gr.Interface(
    fn=predict_banknote_gradio,
    inputs=[
        gr.Number(label="Variance (Measure of how much the pixel values of the banknote image vary.) (0.0 – 7.0)"),
        gr.Number(label="Skewness (Asymmetry of the pixel value distribution.) (-14.0 – 14.0)"),
        gr.Number(label="Kurtosis (Sharpness or peakedness of the pixel value distribution..) (-10.0 – 20.0)"),
        gr.Number(label="Entropy (Randomness or complexity in the pixel patterns.) (-2.0 – 3.0)")
    ],
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Plot(label="2D Plot"),
        gr.Plot(label="3D Plot")
    ],
    title="Banknote Authentication Predictor",
    description="Input the 4 features of a banknote and predict whether it is Genuine or Fake. Shows 2D and 3D plots."
)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    iface.launch(server_name="0.0.0.0", server_port=port)
