# train_nsl.py
import pandas as pd
import numpy as np
import pickle
import argparse

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

def train_kmeans_nsl_kdd(train_csv_path="../NSL_KDD-master/KDDTrain+.csv"):
    """
    NSL-KDD veri setinden K-Means eğitimi ve görselleştirme.
    -> kmeans_model.pkl, scaler.pkl, threshold_value.pkl, proto_labelencoder.pkl
    """

    print("[TRAIN] NSL-KDD üzerinden K-Means eğitimi başlıyor...")

    # --- 1) CSV'yi oku ve kolon isimlerini belirle
    df = pd.read_csv(train_csv_path, header=None)
    columns = [
        "duration", "protocol_type", "service", "flag",
        "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
        "hot", "num_failed_logins", "logged_in", "num_compromised", 
        "root_shell", "su_attempted", "num_root", "num_file_creations", 
        "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", 
        "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", 
        "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", 
        "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", 
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate", 
        "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate", "dst_host_srv_serror_rate", 
        "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
        "label", "difficulty_level"
    ]
    df.columns = columns

    # --- 2) Kullanılacak kolonları filtrele
    df = df[["protocol_type", "src_bytes", "dst_bytes", "label"]]

    # --- 3) 'protocol_type' kolonunu etiketle
    le_proto = LabelEncoder()
    df["protocol_type"] = le_proto.fit_transform(df["protocol_type"])

    # --- 4) 'label' kolonunu ayır (kullanılmayacak)
    X = df.drop(columns=["label"])

    # --- 5) Özellik ölçeklendirme
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- 6) KMeans eğitimi
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X_scaled)

    # --- 7) Örneklerin küme merkezlerine olan mesafeleri
    distances = pairwise_distances(X_scaled, kmeans.cluster_centers_, metric='euclidean')
    min_distances = distances.min(axis=1)

    # --- 8) 95. persentil değeri threshold olarak belirle
    threshold = np.percentile(min_distances, 95)

    # --- 9) Threshold üzerindeki noktaları anomaliler olarak işaretle
    anomalies = np.where(min_distances > threshold)[0]  
    inliers = np.where(min_distances <= threshold)[0]  

    # --- 10) Model objelerini pickle ile diske kaydet
    with open("kmeans_model.pkl", "wb") as mf:
        pickle.dump(kmeans, mf)
    with open("scaler.pkl", "wb") as sf:
        pickle.dump(scaler, sf)
    with open("threshold_value.pkl", "wb") as tf:
        pickle.dump(threshold, tf)
    with open("proto_labelencoder.pkl", "wb") as pf:
        pickle.dump(le_proto, pf)

    # --- Terminale çıktılar
    print("[i] K-Means eğitimi tamamlandı!")
    print(f"    -> Eşik (Threshold) değeri: {threshold:.4f}")
    print(f"    -> Toplam veri sayısı: {len(X_scaled)}")
    print(f"    -> Anomali (eşik üzeri) sayısı: {len(anomalies)}")
    print(f"    -> Normal (eşik altı) sayısı: {len(inliers)}")
    print("    -> Kaydedilen dosyalar: kmeans_model.pkl, scaler.pkl, threshold_value.pkl, proto_labelencoder.pkl")

    # --- 11) GÖRSEL ÇIKTILAR:  
    #     (a) KMeans küme dağılımı (2D PCA) + anomali noktaları
    #     (b) min_distances histogramı ve threshold

    # (a) 2D PCA ile dönüştürme
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    # Hangi cluster'a düştüğünü bul
    clusters = kmeans.predict(X_scaled)
    # Küme merkezlerini de PCA uzayına al
    centers_pca = pca.transform(kmeans.cluster_centers_)

    # --- Kümeleri ve anomalileri çiz
    plt.figure(figsize=(10, 7))  # Figür boyutu küçültüldü
    plt.title("K-Means Kümeleri ve Anomaliler (PCA ile İndirgenmiş)", fontsize=16)

    # Normal noktalar (inliers) - Siyah noktalar
    scatter_inliers = plt.scatter(
        X_pca[inliers, 0], X_pca[inliers, 1], 
        c='black', 
        marker='o', 
        s=10,  # Nokta boyutu sabit
        alpha=0.6, 
        label='Normal (Inliers)',
        edgecolor='blue',
        zorder=1
    )

    # Anomaliler (outliers) - Kırmızı yıldızlar
    scatter_anomalies = plt.scatter(
        X_pca[anomalies, 0], X_pca[anomalies, 1], 
        c='red', 
        marker='*', 
        s=50,  # Nokta boyutu sabit
        edgecolor='black',
        label='Anomali (Outliers)',
        zorder=3
    )

    # Küme merkezleri (Sarı yıldızlar)
    scatter_centers = plt.scatter(
        centers_pca[:, 0], centers_pca[:, 1], 
        c='yellow', 
        marker='*', 
        s=300,  # Küme merkezi boyutu sabit
        edgecolor='black', 
        label='Küme Merkezleri',
        zorder=4
    )

    # Eksen sınırlarını ayarla (Ölçeği düşürme)
    # Burada örnek olarak, PCA bileşenlerinin ±3 standart sapma içerisinde olmasını sağlıyoruz
    pca_std = np.std(X_pca, axis=0)
    plt.xlim(-3 * pca_std[0], 3 * pca_std[0])
    plt.ylim(-3 * pca_std[1], 3 * pca_std[1])

    plt.xlabel("PCA Bileşeni 1", fontsize=14)
    plt.ylabel("PCA Bileşeni 2", fontsize=14)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # (b) min_distances histogramı ve threshold
    plt.figure(figsize=(10, 5))  # Figür boyutu küçültüldü
    plt.title("Minimum Mesafelerin Dağılımı", fontsize=16)
    plt.hist(min_distances, bins=100, alpha=0.75, color='skyblue', edgecolor='black')
    plt.axvline(threshold, color='red', linestyle='--', linewidth=2, 
                label=f'Eşik (95. Persentil): {threshold:.2f}')
    plt.xlabel("Örneklerin En Yakın Küme Merkezine Uzaklığı", fontsize=14)
    plt.ylabel("Frekans", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------
# BAĞIMSIZ ÇALIŞMA: python train_nsl.py --nsl-path ...
# ---------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NSL-KDD Kullanarak K-Means Eğitimi")
    parser.add_argument("--nsl-path", default="../NSL_KDD-master/KDDTrain+.csv",
                        help="NSL-KDD eğitim CSV dosyasının yolu (varsayılan: ../NSL_KDD-master/KDDTrain+.csv)")
    args = parser.parse_args()

    train_kmeans_nsl_kdd(train_csv_path=args.nsl_path)
