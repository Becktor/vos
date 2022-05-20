import pickle
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import pandas as pd
import seaborn as sns
import matplotlib

matplotlib.use('AGG')
import matplotlib.pyplot as plt
from metric_utils import *
import os

import ujson as json

recall_level_default = 0.95

parser = argparse.ArgumentParser(description='Evaluates an OOD Detector',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--energy', type=int, default=1, help='noise for Odin')
parser.add_argument('--T', default=1., type=float, help='temperature: energy|Odin')
parser.add_argument('--thres', default=1., type=float)
parser.add_argument('--name', default=1., type=str)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--model', default='faster-rcnn', type=str)
args = parser.parse_args()

concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()

# ID data
id_data = pickle.load(open('detection/data/VOC-Detection/' + args.model + '/' + args.name + '/random_seed' + '_' + str(
    args.seed) + '/inference/voc_custom_val/standard_nms/corruption_level_0/probabilistic_scoring_res_odd_' + str(
    args.thres) + '.pkl', 'rb'))
ood_data = pickle.load(open('detection/data/VOC-Detection/' + args.model + '/' + args.name + '/random_seed' + '_' + str(
    args.seed) + '/inference/coco_ood_val/standard_nms/corruption_level_0/probabilistic_scoring_res_odd_' + str(
    args.thres) + '.pkl', 'rb'))

pp = 'detection/data/VOC-Detection/' + args.model + '/' + args.name + '/random_seed' + '_' + str(
    args.seed) + '/inference/voc_custom_val/standard_nms/corruption_level_0/'
prediction_file_name = os.path.join(pp, 'coco_instances_results.json')
predicted_instances = json.load(open(prediction_file_name, 'r'))
scale = 0.75 #best so far
def calc_norms(lbls_score, total_score):
    norms = []
    for x, y in lbls_score.items():
        lbl_score = np.array(y)
        u = lbl_score.mean()
        s = lbl_score.std()
        norms.append(torch.distributions.normal.Normal(u, s))
        plt.hist(lbl_score * -1, bins=20, alpha=0.6, density=True)
        plt.title(f"{x} hist:")
        plt.savefig(f'plots/{x}_hist.png')
        plt.clf()

    u = total_score.mean()
    s = total_score.std()
    gd = torch.distributions.normal.Normal(u, s)
    plt.hist(total_score * -1, bins=20, alpha=0.6, density=True)
    plt.title(f"global hist:")
    plt.savefig(f'plots/global_hist.png')
    plt.clf()
    return gd, norms


def run_w_scale(scale):
    lbls_score = {}
    total_score = []
    for x in predicted_instances:
        iid, cid, bbox, score, i_f, _, _ = x.items()
        v = i_f[1][:-1]
        if score[1] <= args.thres * scale:
            continue
        neglogsum = -np.log(np.exp(v).sum())
        if cid[1] in lbls_score:
            lbls_score[cid[1]].append(neglogsum)
        else:
            lbls_score[cid[1]] = [neglogsum]
        total_score.append(neglogsum)

    total_score = np.array(total_score)
    gd, norms = calc_norms(lbls_score, total_score)
    max_mean = min([n.mean for n in norms])
    print(max_mean)
    print(gd)
    print(total_score.shape)
    id = 0
    T = 1
    id_score = []
    ood_score = []


    assert len(id_data['inter_feat'][0]) == 21  # + 1024
    if args.energy:
        class_lbl = id_data['predicted_cls_id'].unique().cpu().numpy()
        print(id_data.keys())
        id_score = -args.T * torch.logsumexp(torch.stack(id_data['inter_feat'])[:, :-1] / args.T, dim=1).cpu().data.numpy()
        pred_score = list(zip(id_data['predicted_cls_id'].cpu().data.numpy(), id_score))
        u = id_score.mean()
        s = id_score.std()
        certs = []
        for i in pred_score:
            idx, val = i
            idx = int(idx)
            val = torch.tensor(val)
            certs.append((gd.cdf(val * (norms[idx].stddev/norms[idx].mean) / (gd.stddev/gd.mean))).numpy())

        id_score2 = np.array(certs)

        ood_score = -args.T * torch.logsumexp(torch.stack(ood_data['inter_feat'])[:, :-1] / args.T,
                                              dim=1).cpu().data.numpy()
        ood_pred_score = list(zip(ood_data['predicted_cls_id'].cpu().data.numpy(), ood_score))

        certs = []
        for i in ood_pred_score:
            idx, val = i
            idx = int(idx)
            val = torch.tensor(val)
            certs.append((gd.cdf(val * (norms[idx].stddev/norms[idx].mean) / (gd.stddev/gd.mean))).numpy())

        ood_score2 = np.array(certs)

    else:
        id_score = -np.max(F.softmax(torch.stack(id_data['inter_feat'])[:, :-1], dim=1).cpu().data.numpy(), axis=1)
        ood_score = -np.max(F.softmax(torch.stack(ood_data['inter_feat'])[:, :-1], dim=1).cpu().data.numpy(), axis=1)

    ###########
    ########
    print(len(id_score))
    print(len(ood_score))

    measures = get_measures(-id_score, -ood_score, plot=False)
    measures2 = get_measures(-id_score2, -ood_score2, plot=False)

    if args.energy:
        print_measures(measures[0], measures[1], measures[2], 'energy')
        print_measures(measures2[0], measures2[1], measures2[2], 'energy')
    else:
        print_measures(measures[0], measures[1], measures[2], 'msp')


for x in np.arange(0.1, 0.9, 0.1):
    print("running Scale:", x)
    run_w_scale(x)