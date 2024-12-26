# detect.py
import sys
import time
import signal
import pickle
import pandas as pd
import numpy as np
import argparse

from scapy.all import sniff
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

anomali_records = []
stop_detection = False

def get_proto_str(proto_num):
    if proto_num == 6:
        return "tcp"
    elif proto_num == 17:
        return "udp"
    elif proto_num == 1:
        return "icmp"
    else:
        return "other"

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

        protocol_val = map_proto_to_nslkdd(proto_str)
        sample = np.array([[protocol_val, pkt_len, 0]], dtype=float)  # Basit: src_bytes ~ pkt_len, dst_bytes=0

        dist = pairwise_distances(
            scaler.transform(sample),
            kmeans_model.cluster_centers_,
            metric='euclidean'
        ).min(axis=1)[0]

        label_str = "NORMAL"
        if dist > threshold:
            label_str = "ANOMALİ"

        print(f"[{label_str}] proto={proto_str} SYN={syn_flag} | dist={dist:.2f}")
        anomali_records.append({"distance": dist, "label": label_str})

def durdur_sinyal_anomali(sig, frame):
    global stop_detection
    stop_detection = True

def detect_realtime(iface="en0", duration=10):
    try:
        with open("kmeans_model.pkl", "rb") as mf:
            kmeans_model = pickle.load(mf)
        with open("scaler.pkl", "rb") as sf:
            scaler = pickle.load(sf)
        with open("threshold_value.pkl", "rb") as tf:
            threshold = pickle.load(tf)
    except FileNotFoundError:
        print("[!] Model dosyaları yok. Önce train_nsl.py ile eğitimi yapın.")
        return

    global stop_detection
    stop_detection = False
    signal.signal(signal.SIGINT, durdur_sinyal_anomali)

    print(f"[DETECT] {duration} saniye IP trafiği analiz ediliyor (iface={iface}).")
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
            print("[!] Kullanıcı sonlandırma yaptı (Ctrl + C).")
            break
        if (time.time() - start_time) >= duration:
            print(f"[i] Süre doldu ({duration} sn).")
            break

    # Grafik
    print("[i] Anomali tespiti bitti, grafik çiziliyor...")
    if len(anomali_records) > 0:
        df_plot = pd.DataFrame(anomali_records)
        normal_mask = (df_plot["label"] == "NORMAL")
        anomaly_mask = (df_plot["label"] == "ANOMALİ")

        plt.figure(figsize=(10,6))
        plt.title("Gerçek Zamanlı Anomali Tespiti - Distance Grafiği")
        plt.xlabel("Paket Sırası")
        plt.ylabel("Distance")

        plt.scatter(df_plot.index[normal_mask], df_plot["distance"][normal_mask],
                    c="green", label="NORMAL")
        plt.scatter(df_plot.index[anomaly_mask], df_plot["distance"][anomaly_mask],
                    c="red", label="ANOMALİ")

        plt.legend()
        plt.show()
    else:
        print("[!] Hiç paket işlenmedi, grafik yok.")

# -----------------------------------------------------
# BAĞIMSIZ ÇALIŞMA: python detect.py --iface en0 --duration 10
# -----------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time Anomaly Detection")
    parser.add_argument("--iface", default="en0", help="Network interface (default en0)")
    parser.add_argument("--duration", type=int, default=10,
                        help="Detection duration in seconds (default 10)")
    args = parser.parse_args()

    detect_realtime(iface=args.iface, duration=args.duration)
