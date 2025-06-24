# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import torch
import os

def parse_args():
    parser = argparse.ArgumentParser(
        description='Preprocess the downloaded SAN checkpoint.')
    parser.add_argument('--base_folder', help='base folder', 
                        default="./ckpts/clipsan")
    parser.add_argument('--input', nargs='+', help="the input files of SAN", 
                        default=['san_vit_b_16.pth', 'san_vit_large_14.pth'])
    parser.add_argument('--output', nargs='+', help='the output file for VEON training',
                        default=["SAN_ViT-B.pth", "SAN_ViT-L.pth"])
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # Processing base/large variants
    for input_file, output_file in zip(args.input, args.output):
        inpf = os.path.join(args.base_folder, input_file)
        oupf = os.path.join(args.base_folder, output_file)

        raw_dict = torch.load(inpf)
        new_dict = {'trainer': raw_dict['trainer'],
                    'iteration': raw_dict['iteration'],
                    'state_dict': raw_dict['model']}
        torch.save(new_dict, oupf)
    return


if __name__ == '__main__':
    main()
