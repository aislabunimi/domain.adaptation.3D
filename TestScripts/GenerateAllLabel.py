import rospy
from sensor_msgs.msg import Image
import cv2
import os
from PILBridge import PILBridge

# Dizionario per tenere traccia dei timestamp e dei nomi dei file
pending_frames = {}

def deeplab_labels_callback(msg):
    try:
        # Converti immagine segmentata da ROS a NumPy
        sem_label = PILBridge.rosimg_to_numpy(msg)

        # Recupera timestamp
        ts = msg.header.stamp.to_sec()

        # Associa il timestamp a un filename se presente
        if ts in pending_frames:
            filename = pending_frames.pop(ts)
            print(f"[INFO] Ricevuta segmentazione per {filename}")

            # Salvataggio dell'immagine segmentata
            output_dir = "/home/michele/Desktop/Domain-Adaptation-Pipeline/10_deeplab_labels"
            os.makedirs(output_dir, exist_ok=True)

            # Opzionale: converti RGB→BGR se necessario per compatibilità OpenCV
            sem_label = cv2.cvtColor(sem_label, cv2.COLOR_RGB2BGR)

            out_path = os.path.join(output_dir, filename)
            cv2.imwrite(out_path, sem_label)
        else:
            rospy.logwarn(f"Nessun file associato al timestamp {ts:.6f}")

    except Exception as e:
        rospy.logerr(f"Errore nella callback deeplab_labels_callback: {e}")

def main():
    rospy.init_node("deeplab_sync_node")

    # Publisher per le immagini RGB
    deeplab_pub = rospy.Publisher('/deeplab/rgb', Image, queue_size=10)

    # Subscriber per i risultati segmentati
    rospy.Subscriber("/deeplab/segmented_image", Image, deeplab_labels_callback)

    input_dir = "/home/michele/Desktop/Domain-Adaptation-Pipeline/IO_pipeline/Scannet/scans/scene0002_00/color"
    img_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".jpg")])

    rospy.sleep(1.0)  # Aspetta inizializzazione nodi/pubblisher

    for i, fname in enumerate(img_files[:10]):
        img_path = os.path.join(input_dir, fname)
        rgb = cv2.imread(img_path)

        if rgb is None:
            rospy.logwarn(f"Immagine non trovata o non valida: {img_path}")
            continue

        # Converte in messaggio ROS
        msg = PILBridge.numpy_to_rosimg(rgb, encoding='bgr8')

        # Aggiunge timestamp
        ts = rospy.Time.now()
        msg.header.stamp = ts

        # Associa timestamp al nome del file
        pending_frames[ts.to_sec()] = fname

        deeplab_pub.publish(msg)
        rospy.loginfo(f"Inviata immagine {fname} con timestamp {ts.to_sec():.6f}")

        rospy.sleep(0.2)  # Lascia tempo per elaborazione

    rospy.spin()

if __name__ == "__main__":
    main()
