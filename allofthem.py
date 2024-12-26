import sys
import time
import signal
import pandas as pd
import numpy as np
import pickle

from scapy.all import sniff
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

import matplotlib.pyplot as plt

############################
# 1) VERI TOPLAMA (Scapy)
############################
packet_list = []
stop_capture = False

def get_proto_str(proto_num):
    if proto_num == 6:
        return "tcp"
    elif proto_num == 17:
        return "udp"
    elif proto_num == 1:
        return "icmp"
    else:
        return "other"

def paket_isleme_toplama(paket):
    if paket.haslayer("IP"):
        ip_layer = paket["IP"]
        proto_num = ip_layer.proto
        src_ip = ip_layer.src
        dst_ip = ip_layer.dst
        pkt_len = len(paket)

        syn_flag = 0
        src_port = 0
        dst_port = 0

        if proto_num == 6 and paket.haslayer("TCP"):
            tcp_layer = paket["TCP"]
            src_port = tcp_layer.sport
            dst_port = tcp_layer.dport
            if (tcp_layer.flags & 0x02) != 0:
                syn_flag = 1
        elif proto_num == 17 and paket.haslayer("UDP"):
            udp_layer = paket["UDP"]
            src_port = udp_layer.sport
            dst_port = udp_layer.dport
        # ICMP port yok

        packet_info = {
            "kaynak_ip": src_ip,
            "hedef_ip": dst_ip,
            "proto_str": get_proto_str(proto_num),
            "syn_flag": syn_flag,
            "kaynak_port": src_port,
            "hedef_port": dst_port,
            "paket_uzunlugu": pkt_len
        }
        packet_list.append(packet_info)

def durdur_sinyal_topla(sig, frame):
    global stop_capture
    stop_capture = True

def veri_topla(iface="en0", sure=10):
    global stop_capture
    stop_capture = False

    print(f"[1] {sure} saniye boyunca IP trafiği toplanacak (iface={iface}).")
    signal.signal(signal.SIGINT, durdur_sinyal_topla)
    start_time = time.time()

    while True:
        sniff(
            iface=iface,
            prn=paket_isleme_toplama,
            filter="ip",
            store=False,
            timeout=1
        )
        if stop_capture:
            print("[!] Kullanıcı yakalamayı sonlandırdı (Ctrl + C).")
            break
        if (time.time() - start_time) >= sure:
            print(f"[i] Süre doldu ({sure} sn).")
            break

    if len(packet_list) > 0:
        df = pd.DataFrame(packet_list)
        df.to_csv("packets.csv", index=False)
        print(f"[i] {len(df)} paket kaydedildi -> packets.csv")
    else:
        print("[!] Hiç paket yakalanamadı veya liste boş.")


############################
# 2) K-MEANS EĞITİMİ (NSL-KDD)
############################
def train_kmeans_nsl_kdd(train_csv_path="./NSL_KDD-master/KDDTrain+.csv"):
    print("[2] NSL-KDD üzerinden K-Means eğitimi başlıyor...")

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

    df = df[["protocol_type", "src_bytes", "dst_bytes", "label"]]

    le_proto = LabelEncoder()
    df["protocol_type"] = le_proto.fit_transform(df["protocol_type"])

    X = df.drop(columns=["label"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X_scaled)

    from sklearn.metrics import pairwise_distances
    distances = pairwise_distances(X_scaled, kmeans.cluster_centers_, metric='euclidean')
    min_distances = distances.min(axis=1)
    threshold = np.percentile(min_distances, 95)

    with open("kmeans_model.pkl", "wb") as mf:
        pickle.dump(kmeans, mf)
    with open("scaler.pkl", "wb") as sf:
        pickle.dump(scaler, sf)
    with open("threshold_value.pkl", "wb") as tf:
        pickle.dump(threshold, tf)
    with open("proto_labelencoder.pkl", "wb") as pf:
        pickle.dump(le_proto, pf)

    print("[i] K-Means eğitimi tamamlandı!")
    print(f"    -> threshold={threshold:.4f}")
    print("    -> kmeans_model.pkl, scaler.pkl, threshold_value.pkl, proto_labelencoder.pkl")


##########################################
# 3) packets.csv'yi İnceleme (opsiyonel)
##########################################
def incele_toplanan_veri(csv_path="packets.csv"):
    print("[3] Toplanan packets.csv'yi inceleme...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"[!] {csv_path} bulunamadı.")
        return
    print(f"[i] Toplanan veri boyutu: {df.shape}")
    print(df.head(5))


############################
# 4) GERÇEK ZAMANLI TESPİT
############################
stop_detection = False

# Grafiksel sonuçlar için ek liste:
anomali_records = []

def map_proto_to_nslkdd(proto_str):
    if proto_str == "tcp":
        return 0
    elif proto_str == "udp":
        return 1
    elif proto_str == "icmp":
        return 2
    else:
        return 3

def paket_isleme_anomali(paket, kmeans_model, scaler, threshold):
    if paket.haslayer("IP"):
        ip_layer = paket["IP"]
        proto_num = ip_layer.proto
        proto_str = get_proto_str(proto_num)
        pkt_len = len(paket)

        syn_flag = 0
        if proto_str == "tcp" and paket.haslayer("TCP"):
            tcp_layer = paket["TCP"]
            if (tcp_layer.flags & 0x02) != 0:
                syn_flag = 1

        protocol_type_val = map_proto_to_nslkdd(proto_str)
        # "src_bytes" ~ pkt_len, "dst_bytes" ~ 0
        sample = np.array([[protocol_type_val, pkt_len, 0]], dtype=float)

        sample_scaled = scaler.transform(sample)
        dist = pairwise_distances(sample_scaled, kmeans_model.cluster_centers_, metric='euclidean').min(axis=1)[0]

        src_ip = ip_layer.src
        dst_ip = ip_layer.dst

        label_str = "NORMAL"
        if dist > threshold:
            label_str = "ANOMALİ"
        
        # Terminal çıktısı
        print(f"[{label_str}] proto={proto_str} SYN={syn_flag} {src_ip} -> {dst_ip} | dist={dist:.2f}")

        # Grafik için kaydet
        # anomali_records: mesafe, label, timestamp vs.
        anomali_records.append({
            "distance": dist,
            "label": label_str
        })

def durdur_sinyal_anomali(sig, frame):
    global stop_detection
    stop_detection = True

def anomali_tespiti(iface="en0", sure=10):
    print("[4] Gerçek zamanlı anomali tespiti başlıyor (IP).")
    try:
        with open("kmeans_model.pkl", "rb") as mf:
            kmeans_model = pickle.load(mf)
        with open("scaler.pkl", "rb") as sf:
            scaler = pickle.load(sf)
        with open("threshold_value.pkl", "rb") as tf:
            threshold = pickle.load(tf)
    except FileNotFoundError:
        print("[!] Model dosyaları bulunamadı. Önce K-Means eğitimi yapmalısınız.")
        return

    global stop_detection
    stop_detection = False
    signal.signal(signal.SIGINT, durdur_sinyal_anomali)

    start_time = time.time()
    while True:
        sniff(
            iface=iface,
            prn=lambda p: paket_isleme_anomali(p, kmeans_model, scaler, threshold),
            filter="ip",
            store=False,
            timeout=1
        )
        if stop_detection:
            print("[!] Kullanıcı durdurdu, anomali tespiti sonlandırılıyor.")
            break
        if (time.time() - start_time) >= sure:
            print(f"[i] Süre doldu ({sure} sn). Anomali tespiti sonlanıyor.")
            break

    # --- Grafiksel Çıktı Aşaması ---
    print("[i] Anomali tespiti tamamlandı. Şimdi grafik çiziliyor...")
    if len(anomali_records) > 0:
        df_plot = pd.DataFrame(anomali_records)
        # df_plot: "distance", "label"

        # İki grup: Normal, Anomali
        normal_mask = (df_plot["label"] == "NORMAL")
        anomaly_mask = (df_plot["label"] == "ANOMALİ")

        plt.figure(figsize=(10,6))
        plt.title("Gerçek Zamanlı Anomali Tespiti - Mesafe Grafiği")
        plt.xlabel("Paket Sırası")
        plt.ylabel("Uzaklık (Distance)")

        plt.scatter(df_plot.index[normal_mask], df_plot["distance"][normal_mask],
                    c="green", label="NORMAL")
        plt.scatter(df_plot.index[anomaly_mask], df_plot["distance"][anomaly_mask],
                    c="red", label="ANOMALİ")

        plt.legend()
        plt.show()
    else:
        print("[!] Hiç paket işlenmedi veya liste boş. Grafik çizilmedi.")


############################
# ANA AKIS
############################
if __name__ == "__main__":
    """
    Örnek akış:
    1) 10 sn IP trafiği topla (TCP/UDP/ICMP) -> packets.csv
    2) NSL-KDD K-Means Eğitimi -> (kmeans_model.pkl, scaler.pkl, threshold_value.pkl)
    3) (Opsiyonel) packets.csv'yi incele
    4) 10 sn gerçek zamanlı anomali tespiti -> Terminalde ANOMALİ/NORMAL
       Sonra matplotlib ile mesafe grafiği
    """
    import warnings
    warnings.filterwarnings("ignore")  # opsiyonel, estetik amaçlı

    iface_name = "en0"  # Linux için: "eth0" veya "wlan0"
    sure_toplama = 10

    print("===== [AŞAMA 1: IP Trafiği Toplama] =====")
    veri_topla(iface=iface_name, sure=sure_toplama)

    print("===== [AŞAMA 2: NSL-KDD K-Means Eğitimi] =====")
    train_kmeans_nsl_kdd("./NSL_KDD-master/KDDTrain+.csv")

    print("===== [AŞAMA 3: packets.csv'yi İnceleme (Opsiyonel)] =====")
    incele_toplanan_veri("packets.csv")

    print("===== [AŞAMA 4: Gerçek Zamanlı Anomali Tespiti + Grafik] =====")
    sure_anomali = 10
    anomali_tespiti(iface=iface_name, sure=sure_anomali)

    print("[i] Program akışı tamamlandı, çıkılıyor.")
