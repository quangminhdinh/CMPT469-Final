import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from .ray_utils import *
from .pycolmap.pycolmap.scene_manager import SceneManager


class N3VSimpleDataset(Dataset):
    def __init__(
        self, datadir, split="train", downsample=4, is_stack=True, cutoff_train=2, cutoff_test=7
    ):
        assert downsample in [1, 2, 4, 8]
        poses_arr = np.load(os.path.join(datadir, 'poses_bounds.npy'))
        poses = poses_arr[:, :-2].reshape([-1, 3, 5])
        extr = poses[..., :-1]
        intr = poses[..., -1].squeeze()
        assert extr.shape[-1] == 4

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

        # Assume shared intrinsics between all cameras.
        # cam = manager.cameras[1]
        # fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
        # self.intrinsics = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        # self.intrinsics[:2, :] /= downsample

        # Extract extrinsic matrices in world-to-camera format.
        imdata = manager.images
        # w2c_mats = []
        # bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
        # for k in imdata:
        #     im = imdata[k]
        #     rot = im.R()
        #     trans = im.tvec.reshape(3, 1)
        #     w2c = np.concatenate(
        #         [np.concatenate([rot, trans], 1), bottom], axis=0
        #     )
        #     w2c_mats.append(w2c)
        # w2c_mats = np.stack(w2c_mats, axis=0)

        # # Convert extrinsics to camera-to-world.
        # c2w_mats = np.linalg.inv(w2c_mats)

        # Image names from COLMAP. No need for permuting the poses according to
        # image names anymore.
        
        # c2w = c2w_mats[inds]

        # Load images.
        if downsample > 1:
            image_dir_suffix = f"_{downsample}"
        else:
            image_dir_suffix = ""
        colmap_image_dir = os.path.join(datadir, "images")
        image_dir = os.path.join(datadir, "images" + image_dir_suffix)

        # Downsampled images may have different names vs images used for COLMAP,
        # so we need to map between the two sorted lists of files.
        colmap_files = sorted(os.listdir(colmap_image_dir))
        image_files = sorted(os.listdir(image_dir))
        colmap_to_image = dict(zip(colmap_files, image_files))

        image_names = image_files#[imdata[k].name for k in imdata]

        # Previous Nerf results were generated with images sorted by filename,
        # ensure metrics are reported on the same test set.
        inds = np.argsort(image_names)
        image_names = [image_names[i] for i in inds]

        self.image_paths = [
            os.path.join(image_dir, colmap_to_image[f]) for f in image_names
        ]

        # img = Image.open(self.image_paths[0])
        # self.img_wh = img.size

        # ray directions for all pixels, same for all images (same H, W, focal)
        # self.directions = get_ray_directions(
        #     self.img_wh[1],
        #     self.img_wh[0],
        #     [self.intrinsics[0, 0], self.intrinsics[1, 1]],
        # )  # (h, w, 3)
        # self.directions = self.directions / torch.norm(
        #     self.directions, dim=-1, keepdim=True
        # )

        # Select the split.
        all_indices = np.arange(len(self.image_paths))
        vid_indices = [int(image_names[idx][3:5]) for idx in all_indices]
        vid_frames = [int(image_names[idx][6:10]) for idx in all_indices]
        self.total_frames = [vid_indices.count(idx) for idx in range(8)]
        all_vid_indices = sorted(list(set(vid_indices)))
        vid_indices_map = {all_vid_indices[_i]: _i for _i in range(len(all_vid_indices))}

        # cutoff = 7
        split_indices = {
            "train": [idx for idx in all_indices if int(image_names[idx][4]) < cutoff_train],
            "test": [idx for idx in all_indices if int(image_names[idx][4]) >= cutoff_test],
        }
        # assert len(split_indices["train"]) + len(split_indices["test"]) == len(self.image_paths)
        # assert len(split_indices["train"]) > len(split_indices["test"])

        indices = split_indices[split]
        # c2w = c2w[indices]
        self.image_paths = [self.image_paths[i] for i in indices]
        self.vid_indices = [vid_indices[i] for i in indices]
        self.vid_frames = [vid_frames[i] for i in indices]

        # self.poses = []
        # self.cameras = []
        self.all_rays = []
        self.all_rgbs = []
        self.all_times = []
        # num_rays = self.img_wh[1] * self.img_wh[0]
        for i in tqdm(
            range(len(self.image_paths)),
            desc=f"Loading data ({len(self.image_paths)})",
        ):
            _vid_idx = vid_indices_map[self.vid_indices[i]]
            _c2wr = extr[_vid_idx, ...].squeeze()
            _c2w = np.zeros_like(_c2wr)
            _c2w[:, 0] = _c2wr[:, 1]
            _c2w[:, 1] = _c2wr[:, 0]
            _c2w[:, 2] = -_c2wr[:, 2]
            _c2w = torch.FloatTensor(_c2w)
            _intr = intr[_vid_idx, ...].squeeze()

            im_h, im_w = _intr[0], _intr[1]
            fx, fy = _intr[2], _intr[2]
            cx, cy = im_w / 2, im_h / 2
            intr_mat = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            intr_mat[:2, :] /= downsample

            # _c2w = torch.FloatTensor(c2w[i])
            # self.poses += [_c2w]

            img = Image.open(self.image_paths[i])
            self.img_wh = img.size
            directions = get_ray_directions(
                self.img_wh[1],
                self.img_wh[0],
                [intr_mat[0, 0], intr_mat[1, 1]],
            )  # (h, w, 3)
            directions = directions / torch.norm(
                directions, dim=-1, keepdim=True
            )
            img = self.transform(img)
            img = img.view(3, -1).permute(1, 0)  # (h*w, 3) RGB
            self.all_rgbs += [img]

            rays_o, rays_d = get_rays(directions, _c2w)
            self.all_rays += [torch.cat([rays_o, rays_d], -1)]  # (h*w, 6)
            # self.cameras += [rays_o.view(-1, 3)[0]]
            # self.all_times += [torch.ones(num_rays, 1) * self.vid_frames[i]] # (h*w, 1)
            self.all_times.append(self.vid_frames[i])

        # self.poses = torch.stack(self.poses)
        # self.cameras = torch.stack(self.cameras)
        self.all_times = torch.Tensor(self.all_times) # N
        if not self.is_stack:
            self.all_rays = torch.cat(self.all_rays, 0)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)
            # self.all_times = torch.cat(self.all_times, 0)

        else:
            self.all_rays = torch.stack(self.all_rays, 0) # (N, h*w, 6)
            assert (
                self.all_rays[0].shape[-1] == 6
            ), "Rays should have 6 as last dim"
            self.all_rays = self.all_rays.reshape(-1, *self.img_wh[::-1], 6)

            self.all_rgbs = torch.stack(self.all_rgbs, 0)
            assert (
                self.all_rgbs[0].shape[-1] == 3
            ), "RGBs should have 3 as last dim"
            self.all_rgbs = self.all_rgbs.reshape(-1, *self.img_wh[::-1], 3)

        # self.all_times = torch.stack(self.all_times, 0)
            # assert (
            #     self.all_times[0].shape[-1] == 1
            # ), "Times should have 1 as last dim"
            # self.all_times = self.all_times.reshape(-1, *self.img_wh[::-1], 1)

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        return len(self.all_rgbs)

    def __getitem__(self, idx):

        if self.split == "train":  # use data in the buffers
            sample = {
                "rays": self.all_rays[idx],
                "rgbs": self.all_rgbs[idx],
                "ts": self.all_times[idx],
            }

        else:  # create data for each image separately

            img = self.all_rgbs[idx]
            rays = self.all_rays[idx]
            ts = self.all_times[idx]

            sample = {"rays": rays, "rgbs": img, "ts": ts}
        return sample


if __name__ == "__main__":
    N3VSimpleDataset("data/N3V/coffee_smal", "train", 1)
