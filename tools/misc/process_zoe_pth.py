# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import torch

def parse_args():
    parser = argparse.ArgumentParser(
        description='Preprocess the downloaded ZoeDepth checkpoint.')
    parser.add_argument('--input', help='downloaded raw ZoeDepth file in dict format', 
                        default="./ckpts/zoedepth/ZoeD_M12_NK.pt")
    parser.add_argument('--output', help='the output file for VEON training',
                        default="./ckpts/zoedepth/ZoeD_M12_NK_p.pt")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    raw_dict = torch.load(args.input)
    new_dict = {'epoch': raw_dict['epoch'],
                'optimizer': raw_dict['optimizer'],
                'state_dict': raw_dict['model']}
    torch.save(new_dict, args.output)
    return


if __name__ == '__main__':
    main()
