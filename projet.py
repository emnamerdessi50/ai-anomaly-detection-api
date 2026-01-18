# ============================================
# Projet : Détection d'anomalies avec Autoencoder (version avancée)
# Auteur : Emna Merdessi
# ============================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# 1. Chargement du dataset
df = pd.read_csv("logsDB.logs.csv")
print("Dataset chargé :", df.shape)

# 2. Prétraitement
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype(str)

cols_unique = [col for col in df.columns if df[col].nunique() == len(df)]
df_ids = df[cols_unique]
df = df.drop(columns=cols_unique)

df = pd.get_dummies(df, columns=df.select_dtypes(include=['object']).columns)
df = df.fillna(0)

scaler = MinMaxScaler()
X = scaler.fit_transform(df)
print("Prétraitement terminé, shape :", X.shape)

# 3. Partition train/test
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
y_train, y_test = X_train, X_test

# 4. Construction de l'autoencoder
input_dim = X_train.shape[1]
autoencoder = models.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(16, activation="relu"),
    layers.Dense(8, activation="relu"),
    layers.Dense(4, activation="relu"),  # bottleneck
    layers.Dense(8, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(input_dim, activation="sigmoid")
])
autoencoder.compile(optimizer="adam", loss="mae")
autoencoder.summary()

# 5. Entraînement
history = autoencoder.fit(
    X_train, X_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# 6. Calcul du seuil d'anomalie (ajustable)
sigma_multiplier = 2.5  # <-- Tu peux modifier ce paramètre
recon_train = autoencoder.predict(X_train)
train_loss = np.mean(np.abs(recon_train - X_train), axis=1)
threshold = np.mean(train_loss) + sigma_multiplier * np.std(train_loss)
print(f"Seuil d'anomalie (μ + {sigma_multiplier}σ) :", threshold)

# 7. Détection sur le test
recon_test = autoencoder.predict(X_test)
test_loss = np.mean(np.abs(recon_test - X_test), axis=1)
anomalies = test_loss > threshold
anomalies_index = np.where(anomalies)[0]

print("Nombre d'anomalies détectées :", np.sum(anomalies))

# 8. Préparer le rapport d’anomalies
anomaly_report = []

if len(anomalies_index) > 0:
    for idx in anomalies_index:
        diff = np.abs(X_test[idx] - recon_test[idx])
        top_cols_idx = diff.argsort()[-5:][::-1]
        top_cols = df.columns[top_cols_idx]
        top_diffs = diff[top_cols_idx]

        anomaly_report.append({
            "anomaly_index": idx,
            "test_loss": test_loss[idx],
            "top_col_1": top_cols[0], "diff_1": top_diffs[0],
            "top_col_2": top_cols[1], "diff_2": top_diffs[1],
            "top_col_3": top_cols[2], "diff_3": top_diffs[2],
            "top_col_4": top_cols[3], "diff_4": top_diffs[3],
            "top_col_5": top_cols[4], "diff_5": top_diffs[4],
        })

    # Trier par test_loss décroissant (les plus anormales en premier)
    anomaly_df = pd.DataFrame(anomaly_report).sort_values(by="test_loss", ascending=False)
    anomaly_df.to_csv("anomalies_detectees.csv", index=False)
    print("Fichier 'anomalies_detectees.csv' sauvegardé avec", len(anomaly_df), "anomalies.")

# 9. Visualisation
plt.figure(figsize=(8, 4))
plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="val loss")
plt.legend()
plt.title("Courbe d'entraînement")
plt.xlabel("Epochs")
plt.ylabel("MAE loss")
plt.show()

# 10. Visualisation des 3 premières anomalies (triées par sévérité)
if len(anomaly_df) > 0:
    top_n = min(3, len(anomaly_df))
    for i in range(top_n):
        idx = anomaly_df.iloc[i]["anomaly_index"]
        plt.figure(figsize=(12, 3))
        plt.plot(X_test[idx], label="Original")
        plt.plot(recon_test[idx], label="Reconstruction")
        plt.legend()
        plt.title(f"Anomalie #{i+1} (idx={idx}, loss={test_loss[idx]:.4f})")
        plt.tight_layout()
        plt.show()