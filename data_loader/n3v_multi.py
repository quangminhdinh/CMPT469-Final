import os
import numpy as np
from PIL import Image
import einops
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from .ray_utils import *
from .pycolmap.pycolmap.scene_manager import SceneManager


class N3VMultiCameraDataset(Dataset):
    def __init__(
        self, datadir, split="train", downsample=4, is_stack=True, keep_hw=False, max_frame=None
    ):
        assert downsample in [1, 2, 4, 8]
        self.downsample = downsample
        self.keep_hw = keep_hw
        poses_arr = np.load(os.path.join(datadir, 'poses_bounds.npy'))
        poses = poses_arr[:, :-2].reshape([-1, 3, 5])
        self.extr = poses[..., :-1]
        self.intr = poses[..., -1].squeeze()
        assert self.extr.shape[-1] == 4
        self.max_frame = max_frame

        self.root_dir = datadir
        self.colmap_dir = os.path.join(datadir, "sparse/0/")
        self.split = split
        self.is_stack = is_stack
        self.define_transforms()

        manager = SceneManager(self.colmap_dir)
        manager.load_cameras()
        manager.load_images()
        manager.load_points3D()

        self.points3D = torch.tensor(manager.points3D, dtype=torch.float)
        self.points3D_color = (
            torch.tensor(manager.point3D_colors, dtype=torch.float) / 255.0
        )

        # Load images.
        if downsample > 1:
            image_dir_suffix = f"_{downsample}"
        else:
            image_dir_suffix = ""
        image_dir = os.path.join(datadir, "images" + image_dir_suffix)

        image_names = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
        inds = np.argsort(image_names)
        image_names = [image_names[i] for i in inds]
        self.image_paths = [os.path.join(image_dir, f) for f in image_names]

        # Select the split.
        all_indices = np.arange(len(self.image_paths))
        vid_indices = [int(image_names[idx][3:5]) for idx in all_indices]
        vid_frames = [int(image_names[idx][6:10]) for idx in all_indices]
        all_vid_indices = sorted(list(set(vid_indices)))
        self.total_frames = [vid_indices.count(idx) for idx in all_vid_indices]
        self.vid_indices_map = {all_vid_indices[_i]: _i for _i in range(len(all_vid_indices))}

        split_indices = {
            "train": [idx for idx in all_indices if int(vid_indices[idx]) > all_vid_indices[0]],
            "test": [idx for idx in all_indices if int(vid_indices[idx]) <= all_vid_indices[0]],
        }

        indices = split_indices[split]
        if max_frame is not None:
            t_i = [idx for idx in indices if int(vid_frames[idx]) < max_frame]
            indices = t_i
        self.image_paths = [self.image_paths[i] for i in indices]
        self.vid_indices = [vid_indices[i] for i in indices]
        self.vid_frames = [vid_frames[i] for i in indices]

        if self.split == "train" and self.keep_hw:
            split_name = "val"
        else:
            split_name = split

        print(f"Loaded {len(self.image_paths)} {split_name} samples!")

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        _vid_idx = self.vid_indices_map[self.vid_indices[idx]]
        _c2wr = self.extr[_vid_idx, ...].squeeze()
        _c2w = np.zeros_like(_c2wr)
        _c2w[:, 0] = _c2wr[:, 1]
        _c2w[:, 1] = _c2wr[:, 0]
        _c2w[:, 2] = -_c2wr[:, 2]
        _c2w[:, 3] = _c2wr[:, 3]
        _c2w = torch.FloatTensor(_c2w)
        _intr = self.intr[_vid_idx, ...].squeeze()

        im_h, im_w = _intr[0], _intr[1]
        fx, fy = _intr[2], _intr[2]
        cx, cy = im_w / 2, im_h / 2
        intr_mat = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        intr_mat[:2, :] /= self.downsample

        img = Image.open(self.image_paths[idx])
        self.img_wh = img.size
        img = self.transform(img)
        
        directions = get_ray_directions(
            self.img_wh[1],
            self.img_wh[0],
            [intr_mat[0, 0], intr_mat[1, 1]],
        )  # (h, w, 3)
        directions = directions / torch.norm(
            directions, dim=-1, keepdim=True
        )

        rgb = img.view(3, -1).permute(1, 0)  # (h*w, 3) RGB
        rgb = rgb.reshape(*self.img_wh[::-1], 3)

        rays_o, rays_d = get_rays(directions, _c2w)
        ray = torch.cat([rays_o, rays_d], -1)  # (h*w, 6)
        ray = ray.reshape(*self.img_wh[::-1], 6)
        t = self.vid_frames[idx] / self.max_frame

        if self.split == "train" and not self.keep_hw:
            ray = einops.rearrange(
                ray, "h w r -> (h w) r"
            )
            rgb = einops.rearrange(
                rgb, "h w c -> (h w) c"
            )
            gap = 1 / self.max_frame
            new_t = torch.normal(mean=torch.tensor(t), std=torch.tensor(gap / 4))
            t = torch.clip(new_t, t - gap / 2, t + gap / 2)

        return ray, rgb, t


if __name__ == "__main__":
    N3VMultiCameraDataset("data/N3V/coffee_smal", "train", 1)
