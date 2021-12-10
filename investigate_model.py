import argparse
from argparse import ArgumentParser

import torch

from lib import dataset
from lib.lightning.lightningmodel import LightningModel

import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def visualize(mat, rows, cols, start_idx):
    min, max = mat.min(axis=0), mat.max(axis=0)
    mat = (mat - min) / (max - min)
    #cmat = (mat - mat.min()) / (mat.max() - mat.min())
    for ix in range(1, rows*cols+1):
        ax = plt.subplot(rows, cols, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(mat[start_idx+ix-1, :, :], 'gray')
    plt.show()

def prepare_inputs(content, style):
    content = dataset.load(content)
    style = dataset.load(style)

    content = dataset.content_transforms()(content)
    style = dataset.style_transforms()(style)

    content = content.to(device).unsqueeze(0)
    style = style.to(device).unsqueeze(0)

    return content, style

def parse_args():
    parser = ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--content', type=str, default='./content.png')
    parser.add_argument('--style', type=str, default='./style.png')
    parser.add_argument('--output', type=str, default='./output.png')
    parser.add_argument('--model', type=str, default='./model.ckpt')

    return vars(parser.parse_args())

if __name__ == '__main__':
    args = parse_args()

    model = LightningModel.load_from_checkpoint(checkpoint_path=args['model'])
    model = model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    model.eval()

    with torch.no_grad():
        device = next(model.parameters()).device

        content, style = prepare_inputs(args['content'], args['style'])
        output, details = model(content, style, return_details=True)

    vgg_style_embeddings = details['vgg_style_embeddings']
    vgg_style_embeddings = vgg_style_embeddings.detach().numpy().squeeze(0)
    visualize(vgg_style_embeddings, rows=3, cols=3, start_idx=40)

    style_global_embeddings = details['style_global_embeddings']
    style_global_embeddings = style_global_embeddings.detach().numpy().squeeze(0)
    visualize(style_global_embeddings, rows=3, cols=3, start_idx=40)

    kernel_results = details['kernel_result']
    k = kernel_results[0].detach().numpy().squeeze(0)
    visualize(k, rows=3, cols=3, start_idx=40)

    kernel_results = details['kernel_result']
    k = kernel_results[1].detach().numpy().squeeze(0)
    visualize(k, rows=3, cols=3, start_idx=40)

    kernel_results = details['kernel_result']
    k = kernel_results[2].detach().numpy().squeeze(0)
    visualize(k, rows=3, cols=3, start_idx=40)

    kernel_results = details['kernel_result']
    k = kernel_results[3].detach().numpy().squeeze(0)
    visualize(k, rows=3, cols=3, start_idx=40)




