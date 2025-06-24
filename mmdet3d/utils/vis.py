import os
import torch
import numpy as np
import matplotlib
from PIL import Image

def colorize(value, vmin=None, vmax=None, cmap='gray_r', invalid_val=-99, invalid_mask=None,
             background_color=(128, 128, 128, 255), gamma_corrected=False, value_transform=None):
    """Converts a depth map to a color image.

    Args:
        value (torch.Tensor, numpy.ndarry): Input depth map. Shape: (H, W) or (1, H, W) or (1, 1, H, W). All singular dimensions are squeezed
        vmin (float, optional): vmin-valued entries are mapped to start color of cmap. If None, value.min() is used. Defaults to None.
        vmax (float, optional):  vmax-valued entries are mapped to end color of cmap. If None, value.max() is used. Defaults to None.
        cmap (str, optional): matplotlib colormap to use. Defaults to 'magma_r'.
        invalid_val (int, optional): Specifies value of invalid pixels that should be colored as 'background_color'. Defaults to -99.
        invalid_mask (numpy.ndarray, optional): Boolean mask for invalid regions. Defaults to None.
        background_color (tuple[int], optional): 4-tuple RGB color to give to invalid pixels. Defaults to (128, 128, 128, 255).
        gamma_corrected (bool, optional): Apply gamma correction to colored image. Defaults to False.
        value_transform (Callable, optional): Apply transform function to valid pixels before coloring. Defaults to None.

    Returns:
        numpy.ndarray, dtype - uint8: Colored depth map. Shape: (H, W, 4)
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    value = value.squeeze()
    if invalid_mask is None:
        invalid_mask = value == invalid_val
    mask = np.logical_not(invalid_mask)

    # normalize
    vmin = np.percentile(value[mask], 2) if vmin is None else vmin
    vmax = np.percentile(value[mask], 85) if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.

    # squeeze last dim if it exists
    # grey out the invalid values

    value[invalid_mask] = np.nan
    cmapper = matplotlib.cm.get_cmap(cmap)
    if value_transform:
        value = value_transform(value)
        # value = value / value.max()
    value = cmapper(value, bytes=True)  # (nxmx4)

    # img = value[:, :, :]
    img = value[...]
    img[invalid_mask] = background_color

    #     return img.transpose((2, 0, 1))
    if gamma_corrected:
        # gamma correction
        img = img / 255
        img = np.power(img, 2.2)
        img = img * 255
        img = img.astype(np.uint8)
    return img


def denormalize_input_img(x):
    """Reverses the imagenet normalization applied to the input.

    Args:
        x (torch.Tensor - shape(N,3,H,W)): input tensor

    Returns:
        torch.Tensor - shape(N,3,H,W): Denormalized input
    """
    mean = torch.Tensor([123.675, 116.28, 103.53]).view(3, 1, 1).to(x.device)
    std = torch.Tensor([58.395, 57.12, 57.375]).view(3, 1, 1).to(x.device)
    return (x * std + mean)


def denormalize_depth_img(x):
    """Reverses the depthnet normalization applied to the input.

    Args:
        x (torch.Tensor - shape(N,3,H,W)): input tensor

    Returns:
        torch.Tensor - shape(N,3,H,W): Denormalized input
    """
    mean = torch.Tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to(x.device)
    std = torch.Tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to(x.device)
    return (x * std + mean)


def vis_img_depth(imgs, depths, orig_size=None, tag='default', denorm_format="midas"):

    vis_dataloader_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                       "..", "..", "loader", 'input')
    if not os.path.exists(vis_dataloader_path):
        os.makedirs(vis_dataloader_path, exist_ok=True)

    B, N, C, H, W = imgs.shape
    if orig_size is None:
        orig_size = (W, H)
    for b in range(B):
        for n in range(N):
            is_midas = denorm_format == 'midas'
            denorm_func = denormalize_depth_img if is_midas else denormalize_input_img
            denorm = (denorm_func(imgs[b, n]).cpu().numpy() * (255 if is_midas else 1)).astype(np.uint8)
            img = Image.fromarray(np.transpose(denorm, (1, 2, 0))[:, :, ::-1])
            pred = Image.fromarray(colorize(depths[b, n].unsqueeze(dim=0)))
            # Stack img and pred side by side for comparison and save
            pred = pred.resize(orig_size, Image.ANTIALIAS)
            stacked = Image.new("RGB", (orig_size[0] * 2, orig_size[1]))
            stacked.paste(img, (0, 0))
            stacked.paste(pred, (orig_size[0], 0))
            stacked.save(os.path.join(vis_dataloader_path, "{}_depth_cmp_{}_{}.png".format(tag, b, n)))
    return

def vis_img_normal(imgs, orig_size=None, tag='default'):

    vis_dataloader_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                       "..", "..", "loader", 'input')
    if not os.path.exists(vis_dataloader_path):
        os.makedirs(vis_dataloader_path, exist_ok=True)

    B, N, C, H, W = imgs.shape
    if orig_size is None:
        orig_size = (W, H)
    for b in range(B):
        for n in range(N):
            img = imgs[b, n, :, :orig_size[1], :orig_size[0]]
            denorm = (denormalize_input_img(img).cpu().numpy()).astype(np.uint8)
            img = Image.fromarray(np.transpose(denorm, (1, 2, 0))[:,:,::-1])
            img.save(os.path.join(vis_dataloader_path, "{}_raw_{}_{}.png".format(tag, b, n)))
    return

def visualize_camera_images(
    image: Image.Image, folder: str = "default", occ_tag: str = "default"):
    """
    Visualize the segmentation result.
    Args:
        image (Image.Image): the input image
        sem_seg (np.ndarray): the segmentation result
        vocabulary (List[str]): the vocabulary used for the segmentation
        output_file (str): the output file path
        mode (str): the visualization mode, can be "overlay" or "mask"
    Returns:
        Image.Image: the visualization result. If output_file is not None, return None.
    """
    # add temporary metadata
    # set numpy seed to make sure the colors are the same
    vis_occ_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "..", "..", "loader", "occ", folder)
    if not os.path.exists(vis_occ_path):
        os.makedirs(vis_occ_path, exist_ok=True)

    def denormalize_img(x):
        """Reverses the imagenet normalization applied to the input.

        Args:
            x (torch.Tensor - shape(3,H,W)): input tensor

        Returns:
            torch.Tensor - shape(3,H,W): Denormalized input
        """
        # mmlab normalize
        mean = torch.Tensor([123.675, 116.28, 103.53]).view(3, 1, 1).to(x.device)
        std = torch.Tensor([58.395, 57.12, 57.375]).view(3, 1, 1).to(x.device)
        # ClipSAN normalize
        # mean = torch.Tensor([122.7709, 116.7460, 104.0937]).view(3, 1, 1).to(x.device)
        # std = torch.Tensor([68.5005, 66.6322, 70.3232]).view(3, 1, 1).to(x.device)
        denorm_tensor = x * std + mean
        denorm_img = (denorm_tensor.cpu().numpy()).astype(np.uint8)
        img = Image.fromarray(np.transpose(denorm_img, (1, 2, 0))[:, :, ::-1])
        return img

    image = denormalize_img(image)
    image.save(os.path.join(vis_occ_path, occ_tag + ".png"))



def vis_occ(pred_occ, occ_tag='default', occ_path=None, folder="default", use_semantic=True, heatmap=True, mode="nuscenes"):
    import open3d as o3d

    vis_occ_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                       "..", "..", "loader", "occ")
    if not os.path.exists(vis_occ_path):
        os.makedirs(vis_occ_path, exist_ok=True)

    # Color map
    NUSCENES_COLOR_MAP = np.array(
        [
            [175, 0, 75, 255],  # others                dark red
            [255, 120, 50, 255],  # barrier              orange
            [255, 192, 203, 255],  # bicycle              pink
            [255, 255, 0, 255],  # bus                  yellow
            [0, 255, 255, 255],  # car                  blue

            [255, 0, 255, 255],  # construction_vehicle dark pink
            [200, 180, 0, 255],  # motorcycle           dark orange
            [255, 0, 0, 255],  # pedestrian           red
            [255, 240, 150, 255],  # traffic_cone         light yellow
            [135, 60, 0, 255],  # trailer              brown

            [160, 32, 240, 255],  # truck                purple
            [139, 137, 137, 255],  # driveable_surface    grey
            # [175,   0,  75, 255],
            [0, 150, 245, 255],  # other_flat           cyan
            [75, 0, 75, 255],  # sidewalk             dark purple
            [150, 240, 80, 255],  # terrain              light green

            [230, 230, 250, 255],  # manmade              white
            [0, 175, 0, 255],  # vegetation           green
            [0, 0, 0, 255],  # ignore
        ]
    )

    SEMKITTI_COLOR_MAP = np.array(
        [
            [0, 0, 0, 255],  # ignore
            [100, 150, 245, 255],
            [100, 230, 245, 255],
            [30, 60, 150, 255],
            [80, 30, 180, 255],
            [100, 80, 250, 255],
            [255, 30, 30, 255],
            [255, 40, 200, 255],
            [150, 30, 90, 255],
            [255, 0, 255, 255],
            [255, 150, 255, 255],
            [75, 0, 75, 255],
            [175, 0, 75, 255],
            [255, 200, 0, 255],
            [255, 120, 50, 255],
            [0, 175, 0, 255],
            [135, 60, 0, 255],
            [150, 240, 80, 255],
            [255, 240, 150, 255],
            [255, 0, 0, 255],
        ]
    ).astype(np.uint8)

    color_map = NUSCENES_COLOR_MAP if mode == "nuscenes" else SEMKITTI_COLOR_MAP

    if heatmap and mode == "nuscenes":
        color_map[0] = np.array([45, 45, 45, 0])
        # color_map[1] = np.array([126, 158, 148, 255])

    class_num = 17 if mode == "nuscenes" else 19
    # if use_semantic:
    #     _, voxel = torch.max(torch.softmax(pred_occ, dim=1), dim=1)
    # else:
    #     voxel = torch.sigmoid(pred_occ[:, 0])
    voxel = pred_occ

    for i in range(voxel.shape[0]):
        x = torch.linspace(0, voxel[i].shape[0] - 1, voxel[i].shape[0])
        y = torch.linspace(0, voxel[i].shape[1] - 1, voxel[i].shape[1])
        z = torch.linspace(0, voxel[i].shape[2] - 1, voxel[i].shape[2])
        X, Y, Z = torch.meshgrid(x, y, z)
        vv = torch.stack([X, Y, Z], dim=-1)

        if mode == "nuscenes":
            vertices = vv[voxel[i] < class_num]
            # Revise here according to the dataset config
            x_range, y_range, z_range = [-40, 40], [-40, 40], [-1, 5.4]
        else:
            vertices = vv[voxel[i] > 0]
            x_range, y_range, z_range = [0, 51.2], [-25.6, 25.6], [-2, 4.4]

        vertices[:, 0] = (vertices[:, 0] + 0.5) * (x_range[1] - x_range[0]) / voxel[i].shape[0] + x_range[0]
        vertices[:, 1] = (vertices[:, 1] + 0.5) * (y_range[1] - y_range[0]) / voxel[i].shape[1] + y_range[0]
        vertices[:, 2] = (vertices[:, 2] + 0.5) * (z_range[1] - z_range[0]) / voxel[i].shape[2] + z_range[0]

        vertices = vertices.numpy()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices)
        if use_semantic:
            if mode == "nuscenes":
                semantics = (voxel[i][voxel[i] < class_num]).cpu().numpy()
            else:
                semantics = (voxel[i][voxel[i] > 0]).cpu().numpy()
            color = color_map[semantics] / 255.0
            pcd.colors = o3d.utility.Vector3dVector(color[..., :3])
            vertices = np.concatenate([vertices, semantics[:, None]], axis=-1)

        if occ_path is not None:
            save_dir = os.path.join(vis_occ_path, folder, occ_path.replace('.npy', '').split('/')[-1])
        else:
            save_dir = os.path.join(vis_occ_path, folder)
        os.makedirs(save_dir, exist_ok=True)

        o3d.io.write_point_cloud(os.path.join(save_dir, occ_tag + '.ply'), pcd)
        np.save(os.path.join(save_dir, occ_tag + '.npy'), vertices)

        # for cam_id, cam_path in enumerate(img_metas[i]['filename']):
        #     os.system('cp {} {}/{}.jpg'.format(cam_path, save_dir, cam_id))
