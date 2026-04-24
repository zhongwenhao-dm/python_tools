#!/usr/bin/env python3
"""
ROS2 bag parser (.db3) — pure Python, no ROS install required.
Uses the `rosbags` library (pip install rosbags).

Extracts:
  - Image / CompressedImage topics → per-topic folders, filenames = ROS timestamp
  - All other topics              → per-topic .txt files

Usage:
    python rosbag_parser.py <bag_dir_or_db3> [--output_dir <dir>]
"""

import os
import argparse

import cv2
import numpy as np
from rosbags.rosbag2 import Reader
from tqdm import tqdm

# rosbags >= 0.9 uses get_typestore; older versions use deserialize_cdr directly
try:
    from rosbags.typesys import Stores, get_typestore
    _typestore = get_typestore(Stores.ROS2_GALACTIC)
    def deserialize(rawdata: bytes, msgtype: str):
        return _typestore.deserialize_cdr(rawdata, msgtype)
except ImportError:
    from rosbags.serde import deserialize_cdr
    def deserialize(rawdata: bytes, msgtype: str):
        return deserialize_cdr(rawdata, msgtype)


IMAGE_TYPES = {
    "sensor_msgs/msg/Image",
    "sensor_msgs/msg/CompressedImage",
}

_DEPTH_ENCODINGS = {"mono16", "16uc1", "32fc1"}
_MONO_ENCODINGS   = {"mono8", "8uc1"}


# ── helpers ──────────────────────────────────────────────────────────────────

def stamp_to_str(stamp) -> str:
    return f"{stamp.sec}_{stamp.nanosec:09d}"


def stamp_to_sec(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def topic_to_name(topic: str) -> str:
    return topic.lstrip("/").replace("/", "_")


# ── per-type CSV formatters ───────────────────────────────────────────────────

# Each entry: msgtype → (header_line, row_func)
# row_func(msg, ts_sec) → comma-separated string (no newline)

_FORMATTERS = {
    "sensor_msgs/msg/Imu": (
        "timestamp,ax,ay,az,wx,wy,wz,qx,qy,qz,qw",
        lambda msg, t: (
            f"{t:.9f},"
            f"{msg.linear_acceleration.x},{msg.linear_acceleration.y},{msg.linear_acceleration.z},"
            f"{msg.angular_velocity.x},{msg.angular_velocity.y},{msg.angular_velocity.z},"
            f"{msg.orientation.x},{msg.orientation.y},{msg.orientation.z},{msg.orientation.w}"
        ),
    ),
    "geometry_msgs/msg/Twist": (
        "timestamp,lx,ly,lz,ax,ay,az",
        lambda msg, t: (
            f"{t:.9f},"
            f"{msg.linear.x},{msg.linear.y},{msg.linear.z},"
            f"{msg.angular.x},{msg.angular.y},{msg.angular.z}"
        ),
    ),
    "geometry_msgs/msg/TwistStamped": (
        "timestamp,lx,ly,lz,ax,ay,az",
        lambda msg, t: (
            f"{t:.9f},"
            f"{msg.twist.linear.x},{msg.twist.linear.y},{msg.twist.linear.z},"
            f"{msg.twist.angular.x},{msg.twist.angular.y},{msg.twist.angular.z}"
        ),
    ),
    "geometry_msgs/msg/TwistWithCovarianceStamped": (
        "timestamp,lx,ly,lz,ax,ay,az",
        lambda msg, t: (
            f"{t:.9f},"
            f"{msg.twist.twist.linear.x},{msg.twist.twist.linear.y},{msg.twist.twist.linear.z},"
            f"{msg.twist.twist.angular.x},{msg.twist.twist.angular.y},{msg.twist.twist.angular.z}"
        ),
    ),
    "nav_msgs/msg/Odometry": (
        "timestamp,x,y,z,qx,qy,qz,qw,lx,ly,lz,ax,ay,az",
        lambda msg, t: (
            f"{t:.9f},"
            f"{msg.pose.pose.position.x},{msg.pose.pose.position.y},{msg.pose.pose.position.z},"
            f"{msg.pose.pose.orientation.x},{msg.pose.pose.orientation.y},"
            f"{msg.pose.pose.orientation.z},{msg.pose.pose.orientation.w},"
            f"{msg.twist.twist.linear.x},{msg.twist.twist.linear.y},{msg.twist.twist.linear.z},"
            f"{msg.twist.twist.angular.x},{msg.twist.twist.angular.y},{msg.twist.twist.angular.z}"
        ),
    ),
}


def decode_compressed(msg) :
    """Decompress a sensor_msgs/msg/CompressedImage message."""
    fmt = msg.format.lower()
    np_arr = np.frombuffer(bytes(msg.data), np.uint8)
    flag = cv2.IMREAD_UNCHANGED if "png" in fmt else cv2.IMREAD_COLOR
    return cv2.imdecode(np_arr, flag)


def decode_raw(msg) :
    """Convert a sensor_msgs/msg/Image message to an OpenCV image."""
    enc = msg.encoding.lower()
    data = np.frombuffer(bytes(msg.data), dtype=np.uint8)

    if enc in _DEPTH_ENCODINGS:
        if enc == "32fc1":
            arr = np.frombuffer(bytes(msg.data), dtype=np.float32)
        else:
            arr = np.frombuffer(bytes(msg.data), dtype=np.uint16)
        return arr.reshape((msg.height, msg.width))

    if enc in _MONO_ENCODINGS:
        return data.reshape((msg.height, msg.width))

    if enc in ("rgb8", "bgr8", "bayer_rggb8", "bayer_bggr8",
               "bayer_gbrg8", "bayer_grbg8"):
        img = data.reshape((msg.height, msg.width, 3))
        if enc == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    if enc == "rgba8":
        img = data.reshape((msg.height, msg.width, 4))
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    if enc == "bgra8":
        img = data.reshape((msg.height, msg.width, 4))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # fallback: try imdecode
    return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)


# ── main ─────────────────────────────────────────────────────────────────────

def parse_bag(bag_path: str, output_dir: str):
    # Accept either the .db3 file or its parent folder
    if bag_path.endswith(".db3"):
        bag_path = os.path.dirname(bag_path)

    os.makedirs(output_dir, exist_ok=True)

    with Reader(bag_path) as reader:
        # ── classify topics ──────────────────────────────────────────────────
        image_conns  = [c for c in reader.connections if c.msgtype in IMAGE_TYPES]
        other_conns  = [c for c in reader.connections if c.msgtype not in IMAGE_TYPES]

        image_topics = {c.topic for c in image_conns}
        other_topics = {c.topic for c in other_conns}

        print(f"Image topics : {sorted(image_topics)}")
        print(f"Other topics : {sorted(other_topics)}")

        # ── prepare output ───────────────────────────────────────────────────
        img_dirs = {}
        for c in image_conns:
            path = os.path.join(output_dir, topic_to_name(c.topic))
            os.makedirs(path, exist_ok=True)
            img_dirs[c.topic] = path

        txt_files  = {}   # topic → file handle
        topic_type = {}   # topic → msgtype
        for c in other_conns:
            if c.topic not in txt_files:
                fname = topic_to_name(c.topic) + ".txt"
                f = open(os.path.join(output_dir, fname), "w")
                # write CSV header if we have a known formatter
                if c.msgtype in _FORMATTERS:
                    f.write(_FORMATTERS[c.msgtype][0] + "\n")
                txt_files[c.topic]  = f
                topic_type[c.topic] = c.msgtype

        # ── iterate messages ─────────────────────────────────────────────────
        total = sum(c.msgcount for c in reader.connections)
        for conn, recv_ts, rawdata in tqdm(reader.messages(), total=total, unit="msg"):
            topic    = conn.topic
            msgtype  = conn.msgtype
            msg      = deserialize(rawdata, msgtype)

            # ── image ────────────────────────────────────────────────────────
            if topic in img_dirs:
                ts_str    = stamp_to_str(msg.header.stamp)
                save_path = os.path.join(img_dirs[topic], f"{ts_str}.png")

                if msgtype == "sensor_msgs/msg/CompressedImage":
                    cv_img = decode_compressed(msg)
                else:
                    cv_img = decode_raw(msg)

                if cv_img is not None:
                    cv2.imwrite(save_path, cv_img)

            # ── other ────────────────────────────────────────────────────────
            elif topic in txt_files:
                f  = txt_files[topic]
                mt = topic_type[topic]
                if hasattr(msg, "header"):
                    ts = stamp_to_sec(msg.header.stamp)
                else:
                    ts = recv_ts * 1e-9

                if mt in _FORMATTERS:
                    f.write(_FORMATTERS[mt][1](msg, ts) + "\n")
                else:
                    f.write(f"--- timestamp: {ts:.9f} ---\n")
                    f.write(str(msg))
                    f.write("\n\n")

    for f in txt_files.values():
        f.close()

    print(f"\nDone. Output written to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Parse a ROS2 .db3 bag (pure Python).")
    parser.add_argument("bag_path", help="Path to .db3 file or bag folder")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: <bag_stem>_parsed)")
    args = parser.parse_args()

    bag_path  = os.path.abspath(args.bag_path)
    bag_stem  = os.path.splitext(os.path.basename(bag_path))[0]
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(bag_path), f"{bag_stem}_parsed"
    )

    parse_bag(bag_path, output_dir)


if __name__ == "__main__":
    main()
