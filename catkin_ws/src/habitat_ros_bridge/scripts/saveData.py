#!/usr/bin/env python3
import rospy
import os
import numpy as np
import csv
from habitat_ros_bridge.msg import Sensors
from helper.PILBridge import PILBridge
from PIL import Image
from helper.resources import NYU40_MAPPING_FILE

#dir = "/media/adaptation/New_volume/Domain_Adaptation_Pipeline/ColomboHM3D/ReplicaDataset/room_0/"
dir = "/media/adaptation/New_volume/Domain_Adaptation_Pipeline/ColomboHM3D/DataSet/scene00808_1/"


dirs = ['RGB', 'DEPTH', 'GT', 'GT_colored', 'POSE']


def nyu40_color_dict():

    csv_file = os.path.normpath(NYU40_MAPPING_FILE)

    nyu40_colors = np.zeros((41, 3), dtype=np.uint8)

    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            nyu_id = int(row['id'])
            r = int(row['red'])
            g = int(row['green'])
            b = int(row['blue'])
            nyu40_colors[nyu_id] = [r, g, b]
        
    return nyu40_colors

def empty_folder(percorso):
    try:
        if os.path.isdir(percorso):
            
            files = os.listdir(percorso)
            for file in files:
                file_path = os.path.join(percorso, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        else:
            print(f"Il percorso {percorso} non è una cartella valida.")
    except Exception as e:
        print(f"Errore: {e}")

class saveData:
    def __init__(self):
        rospy.init_node('habitat_ros_bridge', anonymous=True)

        rospy.Subscriber("/habitat/scene/sensors", Sensors, self.sens_callback)

        self.counter = 0
        self.nyu40_color = nyu40_color_dict()

        if not os.path.exists(dir):
            os.makedirs(dir)

        for type in dirs:
            if not os.path.exists(dir + type):
                os.makedirs(dir + type)
            else:
                empty_folder(dir+type)

        self.run()
    
    def sens_callback(self, msg):
        
        rgb_arr = PILBridge.rosimg_to_numpy(msg.rgb)
        rgb_img = Image.fromarray(rgb_arr)
        rgb_img.save(f"{dir}RGB/{self.counter}.jpg")

        depth_arr = PILBridge.rosimg_to_numpy(msg.depth)
        depth_mm = (depth_arr * 1000).astype(np.uint16)

        # Salvataggio come PNG 16-bit
        Image.fromarray(depth_mm).save(f"{dir}/DEPTH/{self.counter}.png")

        sem_arr = PILBridge.rosimg_to_numpy(msg.sem)
        sem_img = Image.fromarray(sem_arr)
        sem_img.save(f"{dir}GT/{self.counter}.png")

        sem_array_colored = self.nyu40_color[sem_arr]
        sem_img_colored = Image.fromarray(sem_array_colored)
        sem_img_colored.save(f"{dir}GT_colored/{self.counter}.png")

        pose = np.array(msg.pose, dtype=np.float64).reshape((4,4))
        np.savetxt(f"{dir}POSE/{self.counter}.txt", pose, fmt="%.6f")

        self.counter +=1
        print(self.counter)

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            rate.sleep()

if __name__== "__main__":
    try:
        saveData()
    except rospy.ROSInterruptException:
        pass