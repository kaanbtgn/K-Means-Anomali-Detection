import sys
import time
import signal
import pandas as pd
import numpy as np
import pickle
import platform
import logging
from collections import defaultdict

from scapy.all import sniff
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

############################
# LOGGING AYARLARI
############################
logging.basicConfig(
    filename='anomaly_detection.log',
    filemode='w',  # <-- 'w' modunda açılır, her yeniden çalıştırmada log dosyası sıfırlanır
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.DEBUG  # DEBUG seviyesi -> tüm loglar dosyada
)

############################
# 1) VERİ TOPLAMA (Scapy)
############################
packet_list = []
stop_capture = False
last_packet_time = defaultdict(lambda: 0)
packet_rate_counts = defaultdict(int)
RATE_WINDOW = 10

MAX_PACKET_STORAGE = 50000  # bellek koruması için örnek bir limit

def get_proto_str(proto_num):
    if proto_num == 6:
        return "tcp"
    elif proto_num == 17:
        return "udp"
    elif proto_num == 1:
        return "icmp"
    else:
        return "other"

def get_default_interface():
    system = platform.system().lower()
    if system == "windows":
        return "Ethernet"
    elif system == "darwin":
        return "en0"
    else:
        return "eth0"

def paket_isleme_toplama(paket):
    global last_packet_time, packet_rate_counts
    try:
        if len(packet_list) >= MAX_PACKET_STORAGE:
            logging.warning(f"Paket sayısı {MAX_PACKET_STORAGE} aştı, durduruluyor...")
            return

        if paket.haslayer("IP"):
            ip_layer = paket["IP"]
            proto_num = ip_layer.proto
            src_ip = ip_layer.src
            dst_ip = ip_layer.dst
            pkt_len = len(paket)
            timestamp = time.time()

            old_time = last_packet_time[src_ip]
            iat = timestamp - old_time

            if (timestamp - old_time) <= RATE_WINDOW:
                packet_rate_counts[src_ip] += 1
            else:
                packet_rate_counts[src_ip] = 1

            last_packet_time[src_ip] = timestamp
            pkt_rate = packet_rate_counts[src_ip]

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

            packet_info = {
                "source_ip": src_ip,
                "dest_ip": dst_ip,
                "proto_str": get_proto_str(proto_num),
                "syn_flag": syn_flag,
                "source_port": src_port,
                "dest_port": dst_port,
                "packet_len": pkt_len,
                "timestamp": timestamp,
                "inter_arrival_time": iat,
                "packet_rate": pkt_rate
            }
            packet_list.append(packet_info)

            # Her paketi INFO veya DEBUG seviyesinde loglayabiliriz
            logging.debug(f"Packet processed: {packet_info}")

    except Exception as e:
        logging.error(f"Paket işleme hatası: {str(e)}")

def durdur_sinyal_topla(sig, frame):
    global stop_capture
    stop_capture = True
    logging.info("Veri toplama durduruluyor...")
    print("\n[!] Veri toplama için durdurma sinyali alındı...")

def veri_topla(iface=None, sure=None, max_packets=None):
    global stop_capture, packet_list, last_packet_time, packet_rate_counts
    try:
        stop_capture = False
        packet_list = []
        last_packet_time = defaultdict(lambda: 0)
        packet_rate_counts = defaultdict(int)

        if iface is None:
            iface = get_default_interface()

        print(f"[1] Trafik dinlemeye başlıyor (arayüz={iface}).")
        logging.info(f"Başlangıç: Trafik dinlenmeye başlandı (arayüz={iface}).")

        signal.signal(signal.SIGINT, durdur_sinyal_topla)
        start_time = time.time()

        def stop_filter(_):
            if stop_capture:
                return True
            if max_packets and len(packet_list) >= max_packets:
                logging.info(f"Max paket sayısına ulaşıldı: {max_packets}")
                print(f"[!] Max paket sayısına ulaşıldı: {max_packets}")
                return True
            if sure and (time.time() - start_time) >= sure:
                logging.info(f"Dinleme süresi sona erdi: {sure} saniye")
                print(f"[!] Dinleme süresi ({sure} sn) doldu.")
                return True
            return False

        print("[i] Sniff başlıyor. Lütfen trafik veya saldırı simüle edin.")
        try:
            sniff(
                iface=iface,
                prn=paket_isleme_toplama,
                filter="ip",
                store=False,
                stop_filter=stop_filter
            )
        except Exception as e:
            logging.error(f"Sniff sırasında hata: {str(e)}")
            print(f"[!] Sniff sırasında hata oluştu: {str(e)}")

        if len(packet_list) > 0:
            df = pd.DataFrame(packet_list)
            df.to_csv("packets.csv", index=False)
            print(f"[i] {len(df)} paket kaydedildi -> packets.csv")
            logging.info(f"{len(df)} paket kaydedildi -> packets.csv")
            return df
        else:
            print("[!] Hiç paket yakalanamadı veya liste boş.")
            logging.warning("Paket yakalanamadı veya liste boş.")
            return None

    except Exception as e:
        logging.error(f"Veri toplama hatası: {str(e)}")
        print(f"[!] Veri toplama hatası: {str(e)}")
        return None

############################
# 2) K-MEANS EĞİTİMİ
############################
def train_kmeans_collected(df):
    """
    K-Means eğitimi yapar,
    sonuçlarını "kmeans_label" sütununa yazar (NORMAL / ANORMAL).
    """
    try:
        logging.info("K-Means eğitimi başlıyor...")
        if len(df) < 2:
            print("[!] K-Means için yeterli veri yok (en az 2 paket).")
            return None

        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.cluster import KMeans
        from sklearn.metrics import pairwise_distances

        features_df = df[[
            "proto_str", 
            "packet_len", 
            "syn_flag", 
            "source_port", 
            "dest_port", 
            "inter_arrival_time", 
            "packet_rate"
        ]].copy()

        le_proto = LabelEncoder()
        le_proto.fit(["tcp", "udp", "icmp", "other"])
        features_df["proto_num"] = le_proto.transform(features_df["proto_str"])

        # Port normalizasyonu
        features_df["source_port"] = features_df["source_port"] / 65535.0
        features_df["dest_port"]  = features_df["dest_port"]  / 65535.0

        max_iat = features_df["inter_arrival_time"].max()
        if max_iat > 0:
            features_df["inter_arrival_time"] = features_df["inter_arrival_time"] / max_iat
        else:
            features_df["inter_arrival_time"] = 0

        max_pr = features_df["packet_rate"].max()
        if max_pr > 0:
            features_df["packet_rate"] = features_df["packet_rate"] / max_pr
        else:
            features_df["packet_rate"] = 0

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features_df[[
            "proto_num", 
            "packet_len", 
            "syn_flag", 
            "source_port", 
            "dest_port", 
            "inter_arrival_time", 
            "packet_rate"
        ]])

        if len(X_scaled) < 2:
            print("[!] Tek paket ile K-Means olmaz.")
            return None

        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        logging.info(f"K-Means tamamlandı. Inertia={kmeans.inertia_:.2f}")

        distances = pairwise_distances(X_scaled, kmeans.cluster_centers_)
        min_distances = distances.min(axis=1)

        anomalous_threshold = np.percentile(min_distances, 95)
        kmeans_labels = ["ANORMAL" if dist > anomalous_threshold else "NORMAL" 
                         for dist in min_distances]
        df["kmeans_label"] = kmeans_labels

        # MODELLERİ KAYDET
        with open("kmeans_model_realtime.pkl", "wb") as mf:
            pickle.dump(kmeans, mf)
        with open("scaler_realtime.pkl", "wb") as sf:
            pickle.dump(scaler, sf)
        with open("thresholds_realtime.pkl", "wb") as tf:
            pickle.dump(anomalous_threshold, tf)
        with open("proto_labelencoder_realtime.pkl", "wb") as pf:
            pickle.dump(le_proto, pf)

        logging.info("K-Means eğitimi ve model kaydı tamamlandı.")
        print("\n[i] K-Means Model Bilgisi:")
        print(f"    -> Küme Merkezleri: {kmeans.cluster_centers_.shape}")
        print(f"    -> Inertia: {kmeans.inertia_:.2f}")
        print(f"    -> Anomali Eşiği (%95): {anomalous_threshold:.4f}")

        return kmeans, scaler, anomalous_threshold, le_proto

    except Exception as e:
        logging.error(f"K-Means eğitimi hatası: {str(e)}")
        print(f"[!] K-Means eğitimi sırasında hata: {str(e)}")
        return None

############################
# 3) CUSTOM SALDIRI TESPİTİ
############################
# Bu saldırı tespitlerinde özellikle saldırı durumunda logging.warning(...) kullanıyoruz
port_scan_counts = defaultdict(lambda: {"ports": set(), "first_seen": 0})
syn_flood_counts  = defaultdict(lambda: {"count": 0, "first_seen": 0})
icmp_flood_counts = defaultdict(lambda: {"bytes": 0, "first_seen": 0})

PORT_SCAN_WINDOW = 10
PORT_SCAN_THRESHOLD = 5
SYN_FLOOD_WINDOW = 2
SYN_FLOOD_THRESHOLD = 5
ICMP_FLOOD_WINDOW = 2
ICMP_FLOOD_THRESHOLD = 100

def check_port_scan(src_ip, dst_port):
    try:
        current_time = time.time()
        entry = port_scan_counts[src_ip]
        if current_time - entry["first_seen"] <= PORT_SCAN_WINDOW:
            entry["ports"].add(dst_port)
            if len(entry["ports"]) > PORT_SCAN_THRESHOLD:
                logging.warning(f"[PORT SCAN] {src_ip} - {len(entry['ports'])} port denendi!")
                return True
        else:
            port_scan_counts[src_ip] = {"ports": {dst_port}, "first_seen": current_time}
        return False
    except Exception as e:
        logging.error(f"Port tarama kontrol hatası: {str(e)}")
        return False

def check_syn_flood(src_ip):
    try:
        current_time = time.time()
        entry = syn_flood_counts[src_ip]
        if current_time - entry["first_seen"] <= SYN_FLOOD_WINDOW:
            entry["count"] += 1
            if entry["count"] > SYN_FLOOD_THRESHOLD:
                logging.warning(f"[SYN FLOOD] {src_ip} - {entry['count']} SYN paketi!")
                return True
        else:
            syn_flood_counts[src_ip] = {"count": 1, "first_seen": current_time}
        return False
    except Exception as e:
        logging.error(f"SYN flood kontrol hatası: {str(e)}")
        return False

def check_icmp_flood(src_ip, pkt_len):
    try:
        current_time = time.time()
        entry = icmp_flood_counts[src_ip]
        if current_time - entry["first_seen"] <= ICMP_FLOOD_WINDOW:
            entry["bytes"] += pkt_len
            if entry["bytes"] > ICMP_FLOOD_THRESHOLD:
                logging.warning(f"[ICMP FLOOD] {src_ip} - {entry['bytes']} byte!")
                return True
        else:
            icmp_flood_counts[src_ip] = {"bytes": pkt_len, "first_seen": current_time}
        return False
    except Exception as e:
        logging.error(f"ICMP flood kontrol hatası: {str(e)}")
        return False

############################
# 4) ANALİZ & RAPORLAMA (Grafik Dahil)
############################
def analyze_and_report(df):
    """
    1) Custom saldırı tespiti -> custom_label
    2) final_label = K-Means veya Custom Anormal ise ANORMAL
    3) Grafik çizimi
    """
    try:
        if "kmeans_label" not in df.columns:
            df["kmeans_label"] = "NORMAL"

        df["custom_label"] = "NORMAL"
        df["attack_type"] = "None"

        for i, row in df.iterrows():
            src_ip = row['source_ip']
            dst_port = row['dest_port']
            synf = row['syn_flag']
            proto = row['proto_str']
            pkt_len = row['packet_len']

            attacks = []

            if check_port_scan(src_ip, dst_port):
                attacks.append("Port Tarama")
            if synf == 1 and check_syn_flood(src_ip):
                attacks.append("SYN Flood")
            if proto == "icmp" and check_icmp_flood(src_ip, pkt_len):
                attacks.append("ICMP Flood")

            if attacks:
                df.loc[i, "custom_label"] = "ANORMAL"
                df.loc[i, "attack_type"] = ", ".join(attacks)

        # final_label
        df["final_label"] = df.apply(
            lambda x: "ANORMAL" if (x["kmeans_label"] == "ANORMAL" or x["custom_label"] == "ANORMAL")
                      else "NORMAL",
            axis=1
        )

        # Özet
        total_packets = len(df)
        anormal_packets = df[df["final_label"] == "ANORMAL"].shape[0]
        normal_packets  = df[df["final_label"] == "NORMAL"].shape[0]

        print(f"\nToplam Paket:   {total_packets}")
        print(f"Anormal Paket:  {anormal_packets}")
        print(f"Normal  Paket:  {normal_packets}")
        logging.info(f"Anormal Paket: {anormal_packets} / Toplam: {total_packets}")

        # Saldırı Türleri
        if anormal_packets > 0:
            attack_summary = df[df['final_label'] == "ANORMAL"]['attack_type'].value_counts().to_dict()
            if attack_summary:
                print("\n📊 Tespit Edilen Saldırı Türleri (Custom):")
                for atype, count in attack_summary.items():
                    print(f"  • {atype}: {count} adet")
            else:
                print("\n[i] K-Means kaynaklı anormal paketler var; custom attack_type boş.")

        # Örnek 5 anormal paket
        df_anormal = df[df["final_label"] == "ANORMAL"].head(5)
        if len(df_anormal) > 0:
            print("\n[Örnek 5 ANORMAL Paket]")
            print(df_anormal[[
                "source_ip","dest_ip","proto_str",
                "kmeans_label","custom_label","final_label",
                "attack_type"
            ]].to_string(index=False))

        # Grafik (final_label'e göre)
        # K-Means mesafesini tekrar hesaplamak için modeli yükleyelim
        print("\n[i] Anomali grafiği çiziliyor...")
        min_distances = np.zeros(len(df))
        threshold_val = 0

        try:
            with open("kmeans_model_realtime.pkl", "rb") as mf:
                kmeans = pickle.load(mf)
            with open("scaler_realtime.pkl", "rb") as sf:
                scaler = pickle.load(sf)
            with open("proto_labelencoder_realtime.pkl", "rb") as pf:
                le_proto = pickle.load(pf)
            with open("thresholds_realtime.pkl", "rb") as tf:
                threshold_val = pickle.load(tf)

            tmp_df = df[[
                "proto_str","packet_len","syn_flag",
                "source_port","dest_port","inter_arrival_time","packet_rate"
            ]].copy()
            tmp_df["proto_num"] = tmp_df["proto_str"].apply(lambda x: x if x in le_proto.classes_ else "other")
            tmp_df["proto_num"] = le_proto.transform(tmp_df["proto_num"])

            tmp_df["source_port"] = tmp_df["source_port"] / 65535.0
            tmp_df["dest_port"]   = tmp_df["dest_port"]   / 65535.0

            max_iat = tmp_df["inter_arrival_time"].max()
            if max_iat > 0:
                tmp_df["inter_arrival_time"] = tmp_df["inter_arrival_time"] / max_iat
            else:
                tmp_df["inter_arrival_time"] = 0

            max_pr = tmp_df["packet_rate"].max()
            if max_pr > 0:
                tmp_df["packet_rate"] = tmp_df["packet_rate"] / max_pr
            else:
                tmp_df["packet_rate"] = 0

            X_scaled = scaler.transform(tmp_df[[
                "proto_num","packet_len","syn_flag",
                "source_port","dest_port","inter_arrival_time","packet_rate"
            ]])
            from sklearn.metrics import pairwise_distances
            distances = pairwise_distances(X_scaled, kmeans.cluster_centers_)
            min_distances = distances.min(axis=1)

        except Exception as e:
            logging.error(f"Grafik için mesafe hesaplama hatası: {str(e)}")
            print(f"[!] Grafik için mesafe hesaplama hatası: {str(e)}")

        # final_label'e göre renk
        colors = ["red" if lbl == "ANORMAL" else "blue" for lbl in df["final_label"]]
        packet_indices = np.arange(1, len(df) + 1)

        plt.figure(figsize=(12, 6))
        plt.style.use('ggplot')
        plt.scatter(packet_indices, min_distances, c=colors, alpha=0.7, label='Paketler')
        if threshold_val > 0:
            plt.axhline(y=threshold_val, color='green', linestyle='--', linewidth=2, label='Threshold')

        plt.title('Final Anomaly Detection (K-Means + Custom)')
        plt.xlabel('Paket İndeksi')
        plt.ylabel('K-Means: Küme Merkezine Uzaklık')

        normal_patch   = mpatches.Patch(color='blue',  label='NORMAL')
        anormal_patch  = mpatches.Patch(color='red',   label='ANORMAL')
        threshold_line = mpatches.Patch(color='green', label='Threshold')
        plt.legend(handles=[normal_patch, anormal_patch, threshold_line], loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig("final_anomaly_distances.png", dpi=200)
        plt.show()

        logging.info("Final grafik oluşturuldu -> final_anomaly_distances.png")
        print("[i] Nihai grafik oluşturuldu -> final_anomaly_distances.png")

    except Exception as e:
        logging.error(f"Analiz & raporlama hatası: {str(e)}")
        print(f"[!] Analiz & raporlama hatası: {str(e)}")

############################
# ANA AKIŞ
############################
if __name__ == "__main__":
    """
    Program Akışı:
    1) Veri Toplama
    2) K-Means Eğitimi
    3) Analiz & Raporlama (Custom + Grafik)
    """
    try:
        import warnings
        warnings.filterwarnings("ignore")

        iface_name = get_default_interface()

        print("Trafik dinleme koşullarını seçiniz:")
        print("1. Belirli bir süre dinle (örn. 30 saniye)")
        print("2. Belirli bir paket sayısına ulaşıldığında dur (örn. 250)")
        print("3. Her ikisi (örn. 30 sn veya 250 paket)")

        while True:
            choice = input("Seçiminiz (1/2/3): ").strip()
            if choice == '1':
                sure = float(input("Dinleme süresi (saniye): ").strip())
                max_packets = None
                break
            elif choice == '2':
                max_packets = int(input("Paket sayısı: ").strip())
                sure = None
                break
            elif choice == '3':
                sure = float(input("Dinleme süresi (saniye): ").strip())
                max_packets = int(input("Paket sayısı: ").strip())
                break
            else:
                print("Lütfen 1, 2 veya 3 giriniz.")

        print(f"\nSeçilen Dinleme: Süre={sure if sure else 'Yok'}, "
              f"Paket Sayısı={max_packets if max_packets else 'Yok'}")

        # (1) Veri Toplama
        print("\n" + "="*50)
        print("📡 AŞAMA 1: IP Trafiği Toplama")
        print("="*50)
        df_collected = veri_topla(
            iface=iface_name,
            sure=sure,
            max_packets=max_packets
        )

        if df_collected is not None and len(df_collected) > 0:
            # (2) K-Means Eğitimi
            print("\n" + "="*50)
            print("🧠 AŞAMA 2: K-Means Eğitimi")
            print("="*50)
            model_components = train_kmeans_collected(df_collected)

            # (3) Analiz & Raporlama
            print("\n" + "="*50)
            print("📊 AŞAMA 3: Analiz & Raporlama (Grafik Çizimi)")
            print("="*50)
            analyze_and_report(df_collected)
        else:
            print("[!] Toplanan veri yok veya liste boş. K-Means ve analiz aşaması atlanıyor.")

        print("\n✅ Program akışı tamamlandı.")
        logging.info("Program başarıyla tamamlandı.")

    except Exception as e:
        logging.error(f"Ana program hatası: {str(e)}")
        print(f"[!] Ana program hatası: {str(e)}")
