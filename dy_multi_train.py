import os
import uuid
import yaml
import gc
import numpy as np
from PIL import Image
import configargparse
import tqdm
import warnings

warnings.filterwarnings("ignore")

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from data_loader import MultiDyDataHandler
from metrics import lpips, ssim
from configs import *
from radfoam_model.dscene_v1 import DynamicFoamScene
from radfoam_model.utils import psnr


seed = 42
torch.random.manual_seed(seed)
np.random.seed(seed)


def train(args, pipeline_args, model_args, optimizer_args, dataset_args):
    device = torch.device(model_args.device)
    # Setting up output directory
    if not pipeline_args.debug:
        if len(pipeline_args.experiment_name) == 0:
            unique_str = str(uuid.uuid4())[:8]
            experiment_name = f"{dataset_args.scene}@{unique_str}"
        else:
            experiment_name = pipeline_args.experiment_name
        out_dir = f"dy_output/{experiment_name}"
        writer = SummaryWriter(out_dir, purge_step=0)
        os.makedirs(f"{out_dir}/test", exist_ok=True)
        os.makedirs(f"{out_dir}/samp", exist_ok=True)

        def represent_list_inline(dumper, data):
            return dumper.represent_sequence(
                "tag:yaml.org,2002:seq", data, flow_style=True
            )

        yaml.add_representer(list, represent_list_inline)

        # Save the arguments to a YAML file
        with open(f"{out_dir}/config.yaml", "w") as yaml_file:
            yaml.dump(vars(args), yaml_file, default_flow_style=False)

    # Setting up dataset
    iter2downsample = dict(
        zip(
            dataset_args.downsample_iterations,
            dataset_args.downsample,
        )
    )
    train_data_handler = MultiDyDataHandler(
        dataset_args, batch_size=1, device=device
    )
    downsample = iter2downsample[0]
    train_data_handler.reload(split="train", downsample=downsample)

    val_data_handler = MultiDyDataHandler(
        dataset_args, batch_size=1, device=device
    )
    val_data_handler.reload(split="train", downsample=downsample, keep_hw=True)

    test_data_handler = MultiDyDataHandler(
        dataset_args, batch_size=1, device=device
    )
    test_data_handler.reload(
        split="test", downsample=max(dataset_args.downsample)
    )

    # Define viewer settings
    viewer_options = {
        "camera_pos": train_data_handler.viewer_pos,
        "camera_up": train_data_handler.viewer_up,
        "camera_forward": train_data_handler.viewer_forward,
    }

    # Setting up pipeline
    rgb_loss = nn.SmoothL1Loss(reduction="none")

    # Setting up model
    model = DynamicFoamScene(
        args=model_args,
        device=device,
        points=train_data_handler.points3D,
        points_colors=train_data_handler.points3D_colors,
    )
    # model.load_pt("/root/radfoam/dy_output/coffee_martini@f8ef4ecf/model.pt")

    # Setting up optimizer
    model.declare_optimizer(
        args=optimizer_args,
        warmup=pipeline_args.densify_from,
        max_iterations=pipeline_args.iterations,
    )
    # model.update_triangulation(torch.tensor(0.5), incremental=False)

    def test_render(
        loader, tg = None, debug=False, iter=-1, cutoff=None, full_metrics=False
    ):
        prev_t = -1
        psnr_list = []
        # lps = []
        # ssms = []
        with torch.no_grad():
            for i, (ray_batch, rgb_batch, t) in tqdm.tqdm(enumerate(loader), disable=False):
                if cutoff is not None and i > cutoff:
                    break
                ray_batch = ray_batch[0].to(device)
                rgb_batch = rgb_batch[0].to(device)
                t = t.squeeze().to(device)
                if t != prev_t:
                    model.update_triangulation(t, incremental=True)
                    prev_t = t

                output, _, _, _, _ = model(ray_batch, t)

                # White background
                opacity = output[..., -1:]
                rgb_output = output[..., :3] + (1 - opacity)
                rgb_output = rgb_output.reshape(*rgb_batch.shape).clip(0, 1)

                img_psnr = psnr(rgb_output, rgb_batch).mean()
                # if full_metrics:
                    # img_lpips = lpips(rgb_output, rgb_batch).mean()
                    # img_ssim = ssim(rgb_output, rgb_batch).mean()
                    # lps.append(img_lpips)
                    # ssms.append(img_ssim)
                psnr_list.append(img_psnr)
                torch.cuda.synchronize()

                if not debug:
                    error = np.uint8((rgb_output - rgb_batch).cpu().abs() * 255)
                    rgb_output = np.uint8(rgb_output.cpu() * 255)
                    rgb_batch = np.uint8(rgb_batch.cpu() * 255)

                    im = Image.fromarray(
                        np.concatenate([rgb_output, rgb_batch, error], axis=1)
                    )
                    im.save(
                        f"{out_dir}/test/rgb_{i:03d}_t_{int(t.item()):03d}_psnr_{img_psnr:.3f}.png"
                    )
                elif i == 0 and iter % 500 == 0:
                    error = np.uint8((rgb_output - rgb_batch).cpu().abs() * 255)
                    rgb_output = np.uint8(rgb_output.cpu() * 255)
                    rgb_batch = np.uint8(rgb_batch.cpu() * 255)

                    im = Image.fromarray(
                        np.concatenate([rgb_output, rgb_batch, error], axis=1)
                    )
                    im.save(
                        f"{out_dir}/samp/iter_{iter:05d}_psnr_{img_psnr:.3f}.png"
                    )

        average_psnr = sum(psnr_list) / len(psnr_list)
        # if full_metrics:
            # avg_ssim = sum(ssms) / len(ssms)
            # avg_lpips = sum(lps) / len(lps)
        if not debug:
            f = open(f"{out_dir}/metrics.txt", "w")
            f.write(f"Average PSNR: {average_psnr}")
            # if full_metrics:
                # f.write(f"Average SSIM: {avg_ssim}")
                # f.write(f"Average LPIPS: {avg_lpips}")
            f.close()

        return average_psnr

    def train_loop(viewer):
        print("Training")

        torch.cuda.synchronize()

        data_iterator = train_data_handler.get_iter()
        val_iterator = val_data_handler.get_iter()
        test_loader = test_data_handler.get_loader()
        ray_batch, rgb_batch, t_batch = next(data_iterator)
        ray_feat_dim, rgb_feat_dim = ray_batch.shape[-1], rgb_batch.shape[-1]
        ray_batch = ray_batch.reshape(-1, ray_feat_dim).to(device)
        rgb_batch = rgb_batch.reshape(-1, rgb_feat_dim).to(device)
        t = t_batch.squeeze().to(device)
        # model.update_triangulation(t)

        triangulation_update_period = 1
        iters_since_update = 1
        iters_since_densification = 0
        next_densification_after = 1

        with tqdm.trange(pipeline_args.iterations) as train:
            for i in train:
                if viewer is not None:
                    model.update_viewer(viewer, t)
                    viewer.step(i)
                
                model.zero_gradient_cache()
                for b_idx in range(dataset_args.bs):
                    if i in iter2downsample and i and b_idx == 0:
                        downsample = iter2downsample[i]
                        train_data_handler.reload(
                            split="train", downsample=downsample
                        )
                        data_iterator = train_data_handler.get_iter()
                        ray_batch, rgb_batch, t_batch = next(data_iterator)
                        ray_batch = ray_batch.reshape(-1, ray_feat_dim).to(device)
                        rgb_batch = rgb_batch.reshape(-1, rgb_feat_dim).to(device)
                        t = t_batch.squeeze().to(device)
                        model.update_triangulation(t, incremental=True)
                        # gc.collect()
                    else:
                        ray_batch, rgb_batch, t_batch = next(data_iterator)
                        ray_batch = ray_batch.reshape(-1, ray_feat_dim).to(device)
                        rgb_batch = rgb_batch.reshape(-1, rgb_feat_dim).to(device)
                        t = t_batch.squeeze().to(device)
                        if i % 1000 == 999 and b_idx == 0:
                            if i > 15000:
                                model.save_pt(f"{out_dir}/model_{int(i)}.pt")
                            model.update_triangulation(t, incremental=False)
                            gc.collect()
                        else:
                            model.update_triangulation(t, incremental=True)

                    depth_quantiles = (
                        torch.rand(*ray_batch.shape[:-1], 2, device=device)
                        .sort(dim=-1, descending=True)
                        .values
                    )

                    rgba_output, depth, _, _, _ = model(
                        ray_batch, t,
                        depth_quantiles=depth_quantiles,
                    )

                    # White background
                    opacity = rgba_output[..., -1:]
                    if pipeline_args.white_background:
                        rgb_output = rgba_output[..., :3] + (1 - opacity)
                    else:
                        rgb_output = rgba_output[..., :3]

                    color_loss = rgb_loss(rgb_batch, rgb_output)
                    opacity_loss = ((1 - opacity) ** 2).mean()

                    valid_depth_mask = (depth > 0).all(dim=-1)
                    quant_loss = (depth[..., 0] - depth[..., 1]).abs()
                    quant_loss = (quant_loss * valid_depth_mask).mean()
                    w_depth = pipeline_args.quantile_weight * min(
                        2 * i / pipeline_args.iterations, 1
                    )

                    f_out = rgb_output.reshape(-1, *train_data_handler.img_wh[::-1], 3)
                    f_gt = rgb_batch.reshape(-1, *train_data_handler.img_wh[::-1], 3)

                    Lssim = 1.0 - ssim(f_out, f_gt)

                    loss = color_loss.mean() + opacity_loss + 0.2 * Lssim + w_depth * quant_loss + Lssim

                    model.optimizer.zero_grad(set_to_none=True)

                    # Hide latency of data loading behind the backward pass
                    event = torch.cuda.Event()
                    event.record()
                    loss.backward()
                    model.cache_gradient()
                    event.synchronize()

                model.set_batch_gradient(dataset_args.bs)
                model.optimizer.step()
                model.update_learning_rate(i)

                train.set_postfix(color_loss=f"{color_loss.mean().item():.5f}")

                if i % 500 == 499 and not pipeline_args.debug:
                    tqdm.tqdm.write(f"Num points: {model.primal_points.shape[0]}")
                    tqdm.tqdm.write(f"Sum pos at time {t.item()}: {torch.sum(model.get_xyz(0)).item()}")
                    writer.add_scalar("train/rgb_loss", color_loss.mean(), i)
                    num_points = model.primal_points.shape[0]
                    writer.add_scalar("test/num_points", num_points, i)

                    test_psnr = test_render(test_loader, None, True, i + 1, 100)
                    writer.add_scalar("test/psnr", test_psnr, i)

                    writer.add_scalar(
                        "lr/points_lr", model.xyz_scheduler_args(i), i
                    )
                    writer.add_scalar(
                        "lr/density_lr", model.den_scheduler_args(i), i
                    )
                    writer.add_scalar(
                        "lr/attr_lr", model.attr_dc_scheduler_args(i), i
                    )
                    writer.add_scalar(
                        "lr/temp_center_lr", model.temporal_center_scheduler_args(i), i
                    )
                    writer.add_scalar(
                        "lr/temp_duration_lr", model.temporal_duration_scheduler_args(i), i
                    )

                # if iters_since_update >= triangulation_update_period:
                #     model.update_triangulation(t, incremental=True)
                #     iters_since_update = 0

                #     if triangulation_update_period < 100:
                #         triangulation_update_period += 2

                iters_since_update += 1
                if i + 1 >= pipeline_args.densify_from:
                    iters_since_densification += 1

                if (
                    iters_since_densification == next_densification_after
                    and model.primal_points.shape[0]
                    < 0.9 * model.num_final_points
                ):
                    point_error, point_contribution = model.collect_error_map_cpu(
                        val_iterator, t, pipeline_args.white_background
                    )
                    model.prune_and_densify(
                        t,
                        point_error,
                        point_contribution,
                        pipeline_args.densify_factor,
                    )

                    model.update_triangulation(t, incremental=False)
                    triangulation_update_period = 1
                    gc.collect()

                    # Linear growth
                    iters_since_densification = 0
                    next_densification_after = int(
                        (
                            (pipeline_args.densify_factor - 1)
                            * model.primal_points.shape[0]
                            * (
                                pipeline_args.densify_until
                                - pipeline_args.densify_from
                            )
                        )
                        / (model.num_final_points - model.num_init_points)
                    )
                    next_densification_after = max(
                        next_densification_after, 100
                    )

                # if i == optimizer_args.freeze_points:
                #     model.update_triangulation(t, incremental=False)

                if viewer is not None and viewer.is_closed():
                    break

        model.save_ply(f"{out_dir}/scene.ply")
        model.save_pt(f"{out_dir}/model.pt")
        del data_iterator

    if pipeline_args.viewer:
        model.show(
            train_loop, iterations=pipeline_args.iterations, **viewer_options
        )
    else:
        train_loop(viewer=None)
    if not pipeline_args.debug:
        writer.close()

    test_loader = test_data_handler.get_loader()

    t_psnr = test_render(
        test_loader,
        debug=pipeline_args.debug,
        cutoff=dataset_args.max_f,
        full_metrics=True
    )
    print("Avg PSNR:", t_psnr)


def main():
    parser = configargparse.ArgParser(
        default_config_files=["configs/n3v.yaml"]
    )

    model_params = ModelParams(parser)
    pipeline_params = PipelineParams(parser)
    optimization_params = OptimizationParams(parser)
    dataset_params = DatasetParams(parser)

    # Add argument to specify a custom config file
    parser.add_argument(
        "-c", "--config", is_config_file=True, help="Path to config file"
    )

    # Parse arguments
    args = parser.parse_args()

    torch.cuda.empty_cache()
    gc.collect()

    train(
        args,
        pipeline_params.extract(args),
        model_params.extract(args),
        optimization_params.extract(args),
        dataset_params.extract(args),
    )


if __name__ == "__main__":
    main()
