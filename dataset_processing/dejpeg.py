"""
This script is designed to be run with the SwinIR model provided at https://github.com/JingyunLiang/SwinIR.
Use the following weights:
- For color images: 006_colorCAR_DFWB_s126w7_SwinIR-M_jpeg40.pth
- For black and white images: 006_CAR_DFWB_s126w7_SwinIR-M_jpeg40.pth
"""

import argparse
import cv2
import glob
import numpy as np
from collections import OrderedDict
import os
import torch
from pathlib import Path

from models.network_swinir import SwinIR as net
from utils import util_calculate_psnr_ssim as util

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='color_jpeg_car', help='classical_sr, lightweight_sr, real_sr, '
                                                                     'gray_dn, color_dn, jpeg_car, color_jpeg_car')
    parser.add_argument('--scale', type=int, default=1, help='scale factor: 1, 2, 3, 4, 8') # 1 for dn and jpeg car
    parser.add_argument('--noise', type=int, default=15, help='noise level: 15, 25, 50')
    parser.add_argument('--jpeg', type=int, default=40, help='scale factor: 10, 20, 30, 40')
    parser.add_argument('--training_patch_size', type=int, default=128, help='patch size used in training SwinIR. '
                                       'Just used to differentiate two different settings in Table 2 of the paper. '
                                       'Images are NOT tested patch by patch.')
    parser.add_argument('--model_path', type=str,
                        default='006_colorCAR_DFWB_s126w7_SwinIR-M_jpeg40.pth')
    parser.add_argument('--folder_lq', type=str, default=None, help='input low-quality test image folder')
    parser.add_argument('--tile', type=int, default=756, help='Tile size, None for no tile during testing (testing as a whole)')
    parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--save_dir_base', type=str, required=True, help='Base directory to save processed images')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    if os.path.exists(args.model_path):
        print(f'loading model from {args.model_path}')
    else:
        print("Model not found")

    model = define_model(args)
    
    model.eval()
    model = model.to(device)
    model.mean = model.mean.to(device)

    model = torch.compile(model, dynamic=True, mode="reduce-overhead")

    data_path = args.data_path
    titles = [x for x in os.listdir(data_path) if 'ipynb' not in x and os.path.isdir(os.path.join(data_path, x))]
    
    print(titles)
    for title in titles:
        chapters = [x for x in os.listdir(os.path.join(data_path, title)) if os.path.isdir(os.path.join(data_path, title, x))]
        for chapter in chapters:
            args.folder_lq = os.path.join(data_path, title, chapter)
            print(f'Current chapter: {args.folder_lq}')

            # setup folder and path
            folder, save_dir, border, window_size = setup(args)
            os.makedirs(save_dir, exist_ok=True)

            for idx, img_name in enumerate(os.listdir(folder)):
                path = os.path.join(folder, img_name)
                print(path)
                
                if os.path.exists(f'{save_dir}/{os.path.splitext(img_name)[0]}.png'):
                    continue
                # read image
                try:
                    imgname, img_lq = get_image_pair(args, path)  # image to HWC-BGR, float32
                except KeyboardInterrupt:
                    1/0
                except:
                    continue
                    
                

                if min(img_lq.shape[0:2]) < 500:
                    continue

                img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
                img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)  # CHW-RGB to NCHW-RGB

                # inference
                with torch.no_grad():
                    # pad input image to be a multiple of window_size
                    _, _, h_old, w_old = img_lq.size()
                    h_pad = (h_old // window_size + 1) * window_size - h_old
                    w_pad = (w_old // window_size + 1) * window_size - w_old
                    img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
                    img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
                    output = test(img_lq, model, args, window_size)
                    output = output[..., :h_old * args.scale, :w_old * args.scale]
            
                print('Testing {:d} {:20s}'.format(idx, imgname))
                # save image
                output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
                if output.ndim == 3:
                    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
                output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
                cv2.imwrite(f'{save_dir}/{imgname}.png', output)

                print('Saved {:d} {:20s}'.format(idx, imgname))



def define_model(args):

    # 006 grayscale JPEG compression artifact reduction
    # use window_size=7 because JPEG encoding uses 8x8; use img_range=255 because it's sligtly better than 1
    if args.task == 'jpeg_car':
        model = net(upscale=1, in_chans=1, img_size=126, window_size=7,
                    img_range=255., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
        param_key_g = 'params'

    # 006 color JPEG compression artifact reduction
    # use window_size=7 because JPEG encoding uses 8x8; use img_range=255 because it's sligtly better than 1
    elif args.task == 'color_jpeg_car':
        model = net(upscale=1, in_chans=3, img_size=126, window_size=7,
                    img_range=255., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
        param_key_g = 'params'

    pretrained_model = torch.load(args.model_path)
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)

    return model


def setup(args):
    path_parts = Path(args.folder_lq).parts
    title, chapter = path_parts[-2], path_parts[-1]
    
    save_dir = os.path.join(args.save_dir_base, title, chapter)
    folder = args.folder_lq
    border = 0
    window_size = 7

    return folder, save_dir, border, window_size


def get_image_pair(args, path):
    (imgname, imgext) = os.path.splitext(os.path.basename(path))

    if args.task in ['jpeg_car']:
        img_lq = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img_lq.ndim != 2:
            img_lq = util.bgr2ycbcr(img_lq, y_only=True)
        img_lq = np.expand_dims(img_lq, axis=2).astype(np.float32) / 255.

    # 006 JPEG compression artifact reduction (load gt image and generate lq image on-the-fly)
    elif args.task in ['color_jpeg_car']:
        img_lq = cv2.imread(path)
        
        img_lq = img_lq.astype(np.float32)/ 255.

        h, w, _ = img_lq.shape
        min_dim = min(img_lq.shape[:2])
        
        if w == min_dim and min_dim >= 1029 or h == min_dim and min_dim >= 1715:
            if w == min_dim and min_dim >= 1029:
                ratio = 1029 / w
                new_h = int(h * ratio)

                new_shape = (1029, new_h)
            elif h == min_dim and min_dim >= 1715:
                ratio = 1715 / h
                new_w = int(w * ratio)

                new_shape = (new_w, 1715)

            img_lq = cv2.resize(img_lq, (new_shape))

        

    return imgname, img_lq


def test(img_lq, model, args, window_size):
    if args.tile is None:
        # test the image as a whole
        output = model(img_lq)
    else:
        # test the image tile by tile
        b, c, h, w = img_lq.size()
        tile = min(args.tile, h, w)
        assert tile % window_size == 0, "tile size should be a multiple of window_size"
        tile_overlap = args.tile_overlap
        sf = args.scale

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h*sf, w*sf).type_as(img_lq)
        
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]

                print(in_patch.shape)
                out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
        output = E.div_(W)

    return output

if __name__ == '__main__':
    main()
