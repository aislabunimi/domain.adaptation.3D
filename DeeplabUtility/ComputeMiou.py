from metrics.SemanticMeter import SemanticsMeter
import metrics.Miou as Miou

#pred_dir = "/media/adaptation/New_volume/Domain_Adaptation_Pipeline/ColomboHM3D/DataSet/scene0000_08/deeplab_optimized/"
#gt_dir = "/media/adaptation/New_volume/Domain_Adaptation_Pipeline/ColomboHM3D/DataSet/scene0000_08/GT/"
#pred_dir = "/media/adaptation/New_volume/Domain_Adaptation_Pipeline/ColomboHM3D/DataSet/scene0001_08/pseudo_labels_0.03/"
#pred_dir = "/media/adaptation/New_volume/Domain_Adaptation_Pipeline/ColomboHM3D/DataSet/scene0001_08/sam_labels_0.03_prompt_320x240/"
pred_dir = "/media/adaptation/New_volume/Domain_Adaptation_Pipeline/ColomboHM3D/ReplicaDataset/room_0/sam_labels_0.03_prompt_320x240"
gt_dir = "/media/adaptation/New_volume/Domain_Adaptation_Pipeline/ColomboHM3D/ReplicaDataset/room_0/GT"

meter_gt_dlab = SemanticsMeter(number_classes=40)

miou, acc, class_acc = Miou.calculate_metrics(pred_dir, gt_dir, meter_gt_dlab)
print(miou, acc, class_acc)