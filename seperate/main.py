# main.py
import argparse
from capture import capture_traffic
from train_nsl import train_kmeans_nsl_kdd
from detect import detect_realtime

def main():
    parser = argparse.ArgumentParser(description="IDS Pipeline (Capture/Train/Detect)")
    parser.add_argument("--mode", choices=["capture", "train", "detect"], required=True,
                        help="Which step to run: capture, train, or detect")
    parser.add_argument("--iface", default="en0", help="Network interface (default: en0)")
    parser.add_argument("--duration", type=int, default=10, help="Time in seconds (default: 10)")
    parser.add_argument("--output", default="packets.csv", help="Output CSV for capture (default: packets.csv)")
    parser.add_argument("--nsl-path", default="./NSL_KDD-master/KDDTrain+.csv",
                        help="Path to NSL-KDD train CSV (default: ./NSL_KDD-master/KDDTrain+.csv)")

    args = parser.parse_args()

    if args.mode == "capture":
        capture_traffic(iface=args.iface, duration=args.duration, output_csv=args.output)

    elif args.mode == "train":
        train_kmeans_nsl_kdd(train_csv_path=args.nsl_path)

    elif args.mode == "detect":
        detect_realtime(iface=args.iface, duration=args.duration)

if __name__ == "__main__":
    main()
