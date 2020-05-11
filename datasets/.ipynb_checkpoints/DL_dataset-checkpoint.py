import numpy as np
import PIL.Image as pil
import os
import random
import torch
import torch.utils.data as data
from torchvision import transforms

from .mono_dataset import MonoDataset

# image size
img_width = 306
img_height = 256

image_size = [img_width, img_height]

class DL_dataset(MonoDataset):
    def __init__(self, *args, **kwargs):
        # Img_ext: jpeg
        super(DL_dataset, self).__init__(*args, **kwargs)

        self.K = [np.array([[879.03824732, 0, 613.17597314],
                            [0, 879.03824732, 524.14407205],
                            [0, 0, 1]], dtype=np.float32),
                  np.array([[882.61644117, 0, 621.63358525],
                            [0, 882.61644117, 524.38397862],
                            [0, 0, 1]], dtype=np.float32),
                  np.array([[880.41134027, 0, 618.9494972],
                            [0, 880.41134027, 521.38918482],
                            [0, 0, 1]], dtype=np.float32),
                  np.array([[881.28264688, 0, 612.29732111],
                            [0, 881.28264688, 521.77447199],
                            [0, 0, 1]], dtype=np.float32),
                  np.array([[882.93018422, 0, 616.45479905],
                            [0, 882.93018422, 528.27123027],
                            [0, 0, 1]], dtype=np.float32),
                  np.array([[881.63835671, 0, 607.66308183],
                            [0, 881.63835671, 525.6185326],
                            [0, 0, 1]], dtype=np.float32)]
        
        self.cam_list = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]
        self.full_res_shape = (img_width, img_height)

    def check_depth(self):
        return False
    
    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        line = self.filenames[index].split()
        folder = line[0]
        
        # Now we have a line of length 4
        scene_index = int(line[1])
        samp_index = int(line[2])
        cam_index = int(line[3])
        side = line[-1]

        for i in self.frame_idxs:
            inputs[("color", i, -1)] = self.get_color(folder, scene_index, samp_index + i, self.cam_list[cam_index], 
                                                      side, do_flip)

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            # Need to figure out which K are we talking about
            K = self.K[cam_index].copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1

            inputs["stereo_T"] = torch.from_numpy(stereo_T)

        return inputs

    def get_image_path(self, folder, scene_index, samp_index, cam, side):
        f_str = f"{folder}/scene_{scene_index}/sample_{samp_index}/{cam}{self.img_ext}"
        image_path = os.path.join(self.data_path, f_str)
        return image_path

    def get_color(self, folder, scene_index, samp_index, cam, side, do_flip):
        color = self.loader(self.get_image_path(folder, scene_index, samp_index, cam, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        
        return color