# capture.py
import sys
import time
import signal
import pandas as pd
from scapy.all import sniff
import argparse

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
            # SYN bayrağı kontrolü (0x02)
            if (tcp_layer.flags & 0x02) != 0:
                syn_flag = 1
        elif proto_num == 17 and paket.haslayer("UDP"):
            udp_layer = paket["UDP"]
            src_port = udp_layer.sport
            dst_port = udp_layer.dport
        
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

def capture_traffic(iface="en0", duration=10, output_csv="packets.csv"):
    """
    'duration' saniye IP trafiği yakalar ve 'output_csv' dosyasına kaydeder.
    """
    global stop_capture
    stop_capture = False

    print(f"[CAPTURE] {duration} saniye IP trafiği toplanacak (iface={iface}).")
    signal.signal(signal.SIGINT, durdur_sinyal_topla)
    start_time = time.time()

    while True:
        sniff(
            iface=iface,
            prn=paket_isleme_toplama,
            filter="ip",  # Tüm IP trafiği (TCP,UDP,ICMP vs.)
            store=False,
            timeout=1
        )
        if stop_capture:
            print("[!] Kullanıcı yakalamayı sonlandırdı (Ctrl + C).")
            break
        if (time.time() - start_time) >= duration:
            print(f"[i] Süre doldu ({duration} sn).")
            break

    if len(packet_list) > 0:
        df = pd.DataFrame(packet_list)
        df.to_csv(output_csv, index=False)
        print(f"[i] {len(df)} paket kaydedildi -> {output_csv}")
    else:
        print("[!] Hiç paket yakalanamadı veya liste boş.")

# ------------------------------------------------
# BAĞIMSIZ ÇALIŞMA: python capture.py --iface en0 ...
# ------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture Network Traffic via Scapy")
    parser.add_argument("--iface", default="en0", help="Network interface (default: en0)")
    parser.add_argument("--duration", type=int, default=10, help="Capture duration in seconds")
    parser.add_argument("--output", default="packets.csv", help="Output CSV file (default: packets.csv)")
    args = parser.parse_args()

    capture_traffic(iface=args.iface, duration=args.duration, output_csv=args.output)
