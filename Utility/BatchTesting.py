import subprocess
import json
import re
import time
import signal
import threading
from datetime import datetime
import sys
from xml.etree import ElementTree as ET
import argparse

LAUNCH_FILE_PATH = "../catkin_ws/src/control_node/launch/start_mock.launch"

def parse_results(output):
    patterns = {
        "DeepLab": r"\[DeepLab\] mIoU: ([\d.]+), Acc: ([\d.]+), ClassAcc: ([\d.]+)",
        "Pseudo": r"\[Pseudo\]\s+mIoU: ([\d.]+), Acc: ([\d.]+), ClassAcc: ([\d.]+)",
        "SAM": r"\[SAM\]\s+mIoU: ([\d.]+), Acc: ([\d.]+), ClassAcc: ([\d.]+)",
        "SAM_Avg": r"\[SAM Avg Changed Pixels\] ([\d.]+)%"
    }
    results = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        results[key] = [float(x) for x in match.groups()] if match else []
    return results

def format_scene(i):
    return f"{i:04d}_00"

def spinner_thread(stop_event):
    spinner = ['|', '/', '-', '\\']
    idx = 0
    while not stop_event.is_set():
        sys.stdout.write(f"\r⏳ Executing... {spinner[idx % len(spinner)]}")
        sys.stdout.flush()
        idx += 1
        time.sleep(0.1)
    sys.stdout.write("\r✅ Execution complete.            \n")

def extract_voxel_and_mode():
    try:
        tree = ET.parse(LAUNCH_FILE_PATH)
        root = tree.getroot()

        voxel_size = "unknown"
        automatic = False
        feed_w = None
        feed_h = None

        for arg in root.findall("arg"):
            if arg.attrib.get("name") == "voxel_size":
                voxel_size = arg.attrib.get("default")

        for node in root.findall("node"):
            if node.attrib.get("name") == "mocked_control_node":
                for param in node.findall("param"):
                    name = param.attrib.get("name")
                    value = param.attrib.get("value")

                    if name == "automatic":
                        automatic = value.lower() == "true"
                    elif name == "feed_w":
                        feed_w = int(value)
                    elif name == "feed_h":
                        feed_h = int(value)

        return voxel_size, automatic, feed_w, feed_h

    except Exception as e:
        print(f"⚠️ Error parsing launch file: {e}")
        return "unknown", False, None, None

def run_scene(scene_number):
    print(f"▶ Running scene {scene_number}...")

    launch_cmd = [
        "roslaunch", "control_node", "start_mock.launch",
        f"scene_number:={scene_number}"
    ]

    process = subprocess.Popen(
        launch_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        text=True,
        bufsize=1
    )

    full_output = ""
    spinner_stop = threading.Event()
    spinner = threading.Thread(target=spinner_thread, args=(spinner_stop,))
    spinner.start()

    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    termination_time = None

    try:
        for line in process.stdout:
            full_output += line
            if any(tag in line for tag in ["[DeepLab]", "[Pseudo]", "[SAM]", "SAM Avg Changed Pixels"]):
                print(line.strip())
            elif not line.startswith("Evaluating"):
                print(line.strip())
            if "Scene execution complete, shutting down node." in line:
                termination_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                spinner_stop.set()
                print("🔚 Node signaled completion. Sending Ctrl+C...")
                process.send_signal(signal.SIGINT)
                break

        for line in process.stdout:
            full_output += line

    except KeyboardInterrupt:
        print("⛔ Interrupted. Terminating node.")
        termination_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        spinner_stop.set()
        process.terminate()

    spinner_stop.set()
    spinner.join()
    process.wait()

    return {
        "scene": scene_number,
        "start_time": start_time,
        "termination_time": termination_time or "unknown",
        "results": parse_results(full_output)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True, help="Start scene index")
    parser.add_argument("--end", type=int, required=True, help="End scene index (exclusive)")
    args = parser.parse_args()

    voxel_size, automatic,feed_w, feed_h = extract_voxel_and_mode()
    mode = "auto" if automatic else "manual"

    output_file = f"scene_output_vs{voxel_size}_{mode}_{feed_w}x{feed_h}_{args.start}to{args.end - 1}.json"
    print(f"📂 Output will be saved to: {output_file}")
    print(f"📋 Params — voxel_size: {voxel_size}, mode: {mode}, resolution: {feed_w}x{feed_h}")

    all_data = []
    for i in range(args.start, args.end):
        scene = format_scene(i)
        result = run_scene(scene)
        result.update({
            "voxel_size": voxel_size,
            "mode": mode,
            "img_size": [feed_w, feed_h]
        })
        all_data.append(result)

        with open(output_file, "w") as f:
            json.dump(all_data, f, indent=4)

    print("✅ All scenes processed.")

if __name__ == "__main__":
    main()
