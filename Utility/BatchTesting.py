import subprocess
import argparse
import json
import re
import time
import signal
import threading
from datetime import datetime
import sys


def parse_results(output):
    results = {}
    patterns = {
        "DeepLab": r"\[DeepLab\] mIoU: ([\d.]+), Acc: ([\d.]+), ClassAcc: ([\d.]+)",
        "Pseudo": r"\[Pseudo\]\s+mIoU: ([\d.]+), Acc: ([\d.]+), ClassAcc: ([\d.]+)",
        "SAM": r"\[SAM\]\s+mIoU: ([\d.]+), Acc: ([\d.]+), ClassAcc: ([\d.]+)",
        "SAM_Avg": r"\[SAM Avg Changed Pixels\] ([\d.]+)%"
    }

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


def run_scene(scene_number, output_json):
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

    results = parse_results(full_output)

    entry = {
        "scene": scene_number,
        "start_time": start_time,
        "termination_time": termination_time or "unknown",
        "results": {
            "DeepLab": results.get("DeepLab", []),
            "Pseudo": results.get("Pseudo", []),
            "SAM": results.get("SAM", []),
            "SAM_Avg_Changed_Pixels": results.get("SAM_Avg", [None])[0]
        }
    }

    try:
        with open(output_json, "r") as f:
            all_data = json.load(f)
    except FileNotFoundError:
        all_data = []

    all_data.append(entry)
    with open(output_json, "w") as f:
        json.dump(all_data, f, indent=4)

    print(f"✅ Scene {scene_number} complete.\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True, help="Start scene index")
    parser.add_argument("--end", type=int, required=True, help="End scene index (exclusive)")
    parser.add_argument("--output", default="scene_results.json", help="Output JSON file")
    args = parser.parse_args()

    print(f"🎬 Processing scenes {args.start} to {args.end - 1}")
    for i in range(args.start, args.end):
        scene = format_scene(i)
        run_scene(scene, args.output)


if __name__ == "__main__":
    main()
