import cv2
import os
import random
import torch

from glob import glob
from torch.utils.data import Dataset

try:
    from .helper import AugmentationList
except Exception:
    from helper import AugmentationList

__all__ = ["ScanNetNGP"]


class ScanNetNGPFullSceneBothMethods(Dataset):

    def __init__(
        self,
        root,
        scene,
        deeplab: bool,
        pseudo3d: bool,
        voxel: int,
        imsize_sam: str = 'b',
        mode="train",
        output_trafo=None,
        output_size=(240, 320),
        degrees=10,
        flip_p=0.5,
        jitter_bcsh=[0.3, 0.3, 0.3, 0.05],
        sub=1,
        data_augmentation=False,
        label_setting="default",
        confidence_aux=0,
    ):
        """
        Dataset dosent know if it contains replayed or normal samples !

        Some images are stored in 640x480 other ins 1296x968
        Warning scene0088_03 has wrong resolution -> Ignored
        Parameters
        ----------
        root : str, path to the ML-Hypersim folder
        mode : str, option ['train','val]
        """

        super(ScanNetNGPFullSceneBothMethods, self).__init__()
        self._sub = sub
        self._mode = mode
        self._confidence_aux = confidence_aux

        self.H = output_size[0]
        self.W = output_size[1]

        self._label_setting = label_setting
        self.root = root

        if deeplab:
            label_folder = 'deeplab'
        elif pseudo3d:
            label_folder = f'pseudo{voxel}'
        else:
            label_folder = f'samC{imsize_sam}{voxel}'

        if '_' in scene:
            scene = scene.split('_')[0]

        if mode == 'train':
            self.labels_pths = self.get_label_pth(scene + '_00', label_folder='pseudo5')
            self.label_gt_pths = [
                p.replace('pseudo5', label_folder)
                for p in self.labels_pths
            ]

            self.image_pths = [
                p.replace(label_folder, 'color').replace("png", 'jpg')
                for p in self.label_gt_pths
            ]

            self.label_gt_pths = self.label_gt_pths + [
                p.replace('C', 'A')
                for p in self.label_gt_pths
            ]
            self.image_pths += self.image_pths

        else:
            self.label_gt_pths = self.get_label_pth(scene + '_01', label_folder='gt')
            self.image_pths = [
                p.replace('gt', 'color').replace("png", 'jpg')
                for p in self.label_gt_pths
            ]

        self.length = len(self.image_pths)

        self._augmenter = AugmentationList(output_size, degrees, flip_p,
                                           jitter_bcsh)
        self._output_trafo = output_trafo
        self._data_augmentation = data_augmentation

    def get_label_pth(self, scene, label_folder):

        # The path of images are always loaded from this folder because we do not render the labels (for kimera and SAM) without a valid pose
        all_labels = glob(f'{self.root}/{scene}/{label_folder}/*png')

        all_labels = sorted(all_labels,
                              key=lambda x: int(os.path.basename(x)[:-4]))

        return all_labels

    def __getitem__(self, index):
        # Read Image and Label

        img = cv2.imread(self.image_pths[index],
                                 cv2.IMREAD_UNCHANGED)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)
        img = (torch.from_numpy(img).type(torch.float32).permute(2, 0, 1)
              )  # C H W range 0-1

        label = cv2.imread(self.label_gt_pths[index],
                                   cv2.IMREAD_UNCHANGED)


        label = cv2.resize(label, (self.W, self.H),
                           interpolation=cv2.INTER_NEAREST)
        label = torch.from_numpy(label).type(
            torch.float32)[None, :, :]  # C H W -> contains 0-40

        label = [label]

        if self._data_augmentation:
            img, label = self._augmenter.apply(img, label)


        label[0] = label[0] - 1

        img_ori = img.clone()
        if self._output_trafo is not None:
            img = self._output_trafo(img)

        ret = (img, label[0].type(torch.int64)[0, :, :])
        ret += (img_ori,)


        current_scene_name = os.path.normpath(self.image_pths[index]).split(
                os.path.sep)[-3]
        current_image_name = os.path.normpath(self.image_pths[index]).split(
                os.path.sep)[-1]
        ret += (current_scene_name, current_image_name,)

        return ret

    def __len__(self):
        return self.length

    def __str__(self):
        string = "=" * 90
        string += "\nScannet Dataset: \n"
        length = len(self)
        string += f"    Total Samples: {length}"
        string += f"  »  Mode: {self._mode} \n"
        string += f"  »  DataAug: {self._data_augmentation}"
        string += "=" * 90
        return string
