import numpy as np
import sys
import os
import pickle
import argparse

import scipy.special
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
# from models.wrn import WideResNet
# from models.densenet import DenseNet3
from models.wrn_virtual import WideResNet
from models.densenet import DenseNet3
from skimage.filters import gaussian as gblur
from PIL import Image as PILImage
import sys
from os import path
from data_loader import ShippingLabClassification, Cifar10_Imbalanced
import matplotlib
from tqdm import tqdm

matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import utils.display_results as dr
import utils.svhn_loader as svhn
import utils.lsun_loader as lsun_loader
import utils.score_calculation as lib
from scipy.stats import multivariate_normal
import sys

gettrace = getattr(sys, 'gettrace', None)

parser = argparse.ArgumentParser(description='Evaluates a CIFAR OOD Detector',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Setup
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--num_to_avg', type=int, default=1, help='Average measures across num_to_avg runs.')
parser.add_argument('--validate', '-v', action='store_true', help='Evaluate performance on validation distributions.')
parser.add_argument('--use_xent', '-x', action='store_true', help='Use cross entropy scoring instead of the MSP.')
parser.add_argument('--method_name', '-m', type=str, default='cifar10_allconv_baseline', help='Method name.')
# Loading details
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
parser.add_argument('--load', '-l', type=str, default='snapshots', help='Checkpoint path to resume / test.')
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')
# EG and benchmark details
parser.add_argument('--out_as_pos', action='store_true', help='OE define OOD data as positive.')
parser.add_argument('--score', default='MSP', type=str, help='score options: MSP|energy')
parser.add_argument('--T', default=1., type=float, help='temperature: energy|Odin')
parser.add_argument('--noise', type=float, default=0, help='noise for Odin')
parser.add_argument('--model_name', default='res', type=str)

args = parser.parse_args()
print(args)
# torch.manual_seed(1)
# np.random.seed(1)

# mean and standard deviation of channels of CIFAR-10 images
mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]
invNorm = trn.Normalize(-np.array(mean) / np.array(std), 1 / np.array(std))
inp_size = 32
test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

if 'cifar10_' in args.method_name:
    # test_data = dset.CIFAR10('nobackup-slow/dataset/my_xfdu/cifarpy', train=False, transform=test_transform)
    imbalance = [1, 1, 1, 1, 1, 0.3, 1, 1, 0.05, 0.1]
    test_data = Cifar10_Imbalanced('nobackup-slow/dataset/my_xfdu/cifarpy', train=False, transform=test_transform,
                                   imbalance=imbalance)
    label_mapping = {y: x for x, y in test_data.class_to_idx.items()}
    num_classes = 10
elif 'imbacifar_' in args.method_name:
    imbalance = [1, 1, 1, 1, 1, 0.05, 1, 1, 1, 0.1]
    test_data = Cifar10_Imbalanced('nobackup-slow/dataset/my_xfdu/cifarpy', train=False, transform=test_transform,
                                   imbalance=imbalance)
    label_mapping = {y: x for x, y in test_data.class_to_idx.items()}
    num_classes = 10
elif 'ships_' in args.method_name:
    inp_size = 128
    # mean and standard deviation of channels of CIFAR-10 images
    # mean = [x / 255 for x in [118.1, 113.5, 111.5]]
    # std = [x / 255 for x in [51.7, 48.5, 50.6]]
    val_dir = r'ship/val_set'
    test_transform = trn.Compose([trn.Resize((inp_size, inp_size)), trn.ToTensor(), trn.Normalize(mean, std)])

    test_data = ShippingLabClassification(root_dir=val_dir,
                                          transform=test_transform)
    label_mapping = {y: x for x, y in test_data.classes.items()}
    num_classes = len(test_data.classes)
else:
    test_data = dset.CIFAR100('nobackup-slow/dataset/my_xfdu/cifarpy', train=False, transform=test_transform)
    num_classes = 100

test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_bs, shuffle=False,
                                          num_workers=args.prefetch, pin_memory=True)

# Create model
if args.model_name == 'res':
    net = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate)
else:
    net = DenseNet3(100, num_classes, 12, reduction=0.5, bottleneck=True, dropRate=0.0, normalizer=None,
                    k=None, info=None)
start_epoch = 0

# Restore model
if args.load != '':
    for i in range(1000 - 1, -1, -1):
        if 'pretrained' in args.method_name:
            subdir = 'pretrained'
        elif 'oe_tune' in args.method_name:
            subdir = 'oe_tune'
        elif 'energy_ft' in args.method_name:
            subdir = 'energy_ft'
        elif 'baseline' in args.method_name:
            subdir = 'baseline'
        else:
            subdir = 'oe_scratch'

        # model_name = os.path.join(os.path.join(args.load, subdir), args.method_name + '_epoch_' + str(i) + '.pt')
        model_name = os.path.join(os.path.join(args.load, subdir), args.method_name + '.pt')
        if os.path.isfile(model_name):
            net.load_state_dict(torch.load(model_name))
            print('Model restored! Epoch:', i)
            start_epoch = i + 1
            break
    if start_epoch == 0:
        assert False, "could not resume " + model_name

net.eval()

if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.ngpu > 0:
    net.cuda()
    # torch.cuda.manual_seed(1)

cudnn.benchmark = True  # fire on all cylinders

# /////////////// Detection Prelims ///////////////

ood_num_examples = len(test_data) // 5
expected_ap = ood_num_examples / (ood_num_examples + len(test_data))

concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()


def calc_norms(label_mapping, lbls_score, scale=0.9):
    norms = []
    u = in_score.mean()
    s = in_score.std()
    gd = torch.distributions.normal.Normal(u, s)
    means = []
    var = []
    for x in range(len(label_mapping.keys())):
        walu = np.stack(lbls_score[lbls_score[:, 0] == x][:, 2])
        #n_walu = 2*(walu-np.min(walu, axis=1, keepdims=True))/np.ptp(walu, axis=1, keepdims=True)-1
        #s_walu = scipy.special.softmax(n_walu, axis=1)
        mm = np.mean(walu, axis=0)
        means.append(mm)
        diff = (walu - mm)
        covar_ = diff.T @ diff / len(diff)
        covar_ += np.eye(len(covar_)) * 1e-5
        var.append(covar_)

    mvns = []
    mvns2 = []
    for m, c in zip(means, var):
        mvn = torch.distributions.multivariate_normal.MultivariateNormal(torch.tensor(m), torch.tensor(c))
        mvns2.append(multivariate_normal(m, c))
        mvns.append(mvn)

    for x in range(len(label_mapping.keys())):
        lbl_score = lbls_score[lbls_score[:, 0] == x][:, 1]
        u = lbl_score.mean()
        s = lbl_score.std()
        norms.append(torch.distributions.normal.Normal(u, s))
        plt.hist(lbl_score * -1, bins=20, alpha=0.6, density=True)
        plt.title(f"{x} hist:")
        plt.savefig(f'plots/{x}_hist.png')
        plt.clf()

    plt.hist(in_score * -1, bins=20, alpha=0.6, density=True)
    plt.title(f"global hist:")
    plt.savefig(f'plots/global_hist.png')
    plt.clf()
    return gd, norms, mvns, mvns2


def flatten(t):
    return np.array([item for sublist in t for item in sublist], dtype=object)


def get_ood_scores(loader, in_dist=False):
    _score = []
    _targets = []
    _rs = []
    _soft_score = []
    _right_score = []
    _wrong_score = []
    _lbl_score = []
    _pred_score = []
    _outputs = []
    _softmaxs = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            if batch_idx >= ood_num_examples // args.test_bs and in_dist is False:
                break

            data = data.cuda()

            output = net(data)
            smax = to_np(F.softmax(output, dim=1))

            if args.use_xent:
                _score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1))))
            else:
                if args.score == 'energy':
                    _score.append(-to_np((args.T * torch.logsumexp(output / args.T, dim=1))))
                    _soft_score.append(-np.max(smax, axis=1))
                    _outputs.append(to_np(output))

                else:  # original MSP and Mahalanobis (but Mahalanobis won't need this returned)
                    _score.append(-np.max(smax, axis=1))
            preds = np.argmax(smax, axis=1)
            _pred_score.append(list(zip(preds, _score[-1], _outputs[-1])))
            _softmaxs.append(smax)
            if in_dist:
                targets = target.numpy().squeeze()
                right_indices = preds == targets
                wrong_indices = np.invert(right_indices)
                _lbl_score.append(
                    list(zip(targets[right_indices], _score[-1][right_indices], _outputs[-1][right_indices])))
                _targets.append(preds)
                _rs.append(right_indices)
                if args.use_xent:
                    _right_score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1)))[right_indices])
                    _wrong_score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1)))[wrong_indices])
                else:
                    _right_score.append(-np.max(smax[right_indices], axis=1))
                    _wrong_score.append(-np.max(smax[wrong_indices], axis=1))
                    test = False
                    if test:
                        ci = _score[batch_idx].argmin()
                        c = invNorm(data[ci]).cpu().numpy().transpose(1, 2, 0)
                        ui = _score[batch_idx].argmax()
                        u = invNorm(data[ui]).cpu().numpy().transpose(1, 2, 0)
                        plot_with_energy(c, smax[ci], int(target[ci]), _score[batch_idx][ci], f'{batch_idx}_cert')
                        plot_with_energy(u, smax[ui], int(target[ui]), _score[batch_idx][ui], f'{batch_idx}_uncert')
                        # print(f'mostuncert={_score[batch_idx].max()}, leastuncert{_score[batch_idx].min()}')

    if in_dist:

        return concat(_score).copy(), concat(_right_score).copy(), \
               concat(_wrong_score).copy(), flatten(_lbl_score).copy(), \
               concat(_targets).copy(), concat(_rs).copy(), concat(_outputs).copy()

    else:
        return concat(_score)[:ood_num_examples].copy(), flatten(_pred_score).copy()


def plot_with_energy(img, smax, lbl, energy, name):
    fig, axes = plt.subplots(2, 1, figsize=(5, 7), gridspec_kw={'height_ratios': [3, 1]})
    ax1, ax2 = axes.ravel()
    ax1.imshow(img)
    data_test = {y: x for x, y in test_data.classes.items()}
    # ax1.title.set_text(f'pred:  {n}\nlabel: {l}')
    ax1.set_title(f'pred: {data_test[np.argmax(smax)]} -- lbl: {data_test[lbl]}\nenergy: {energy}', loc='left')
    ids = [line.replace("_", " ") for line in list(test_data.classes.keys())]
    ax2.bar(ids, smax)
    ax2.title.set_text(f'bar of class prediciton')
    for label in ax2.get_xticklabels():
        label.set_rotation(45)
        label.set_ha('right')
    plt.savefig(f'plots/{name}.png')
    plt.clf()


if args.score == 'Odin':
    # separated because no grad is not applied
    in_score, right_score, wrong_score = lib.get_ood_scores_odin(test_loader, net, args.test_bs, ood_num_examples,
                                                                 args.T, args.noise, in_dist=True)
elif args.score == 'M':
    from torch.autograd import Variable

    _, right_score, wrong_score = get_ood_scores(test_loader, in_dist=True)

    if 'cifar10_' in args.method_name:
        train_data = dset.CIFAR10('/nobackup-slow/dataset/cifarpy', train=True, transform=test_transform)
    else:
        train_data = dset.CIFAR100('/nobackup-slow/dataset/cifarpy', train=True, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.test_bs, shuffle=False,
                                               num_workers=args.prefetch, pin_memory=True)
    num_batches = ood_num_examples // args.test_bs

    temp_x = torch.rand(2, 3, 32, 32)
    temp_x = Variable(temp_x)
    temp_x = temp_x.cuda()
    temp_list = net.feature_list(temp_x)[1]
    num_output = len(temp_list)
    feature_list = np.empty(num_output)
    count = 0
    for out in temp_list:
        feature_list[count] = out.size(1)
        count += 1

    print('get sample mean and covariance', count)
    sample_mean, precision = lib.sample_estimator(net, num_classes, feature_list, train_loader)
    in_score = lib.get_Mahalanobis_score(net, test_loader, num_classes, sample_mean, precision, count - 1, args.noise,
                                         num_batches, in_dist=True)
    print(in_score[-3:], in_score[-103:-100])
else:
    in_score, right_score, wrong_score, lbls_score, lbls, rs, outs = get_ood_scores(test_loader, in_dist=True)

gd, norms, mvn, mmvn = calc_norms(label_mapping, lbls_score)
max_mean = min([x.mean for x in norms])
print(max_mean)
certs = []
id_pred_certs = {}
id_gd_pred_certs = {}
ood_pred_certs = {}
ood_gd_pred_certs = {}
for x in np.unique(lbls):
    id_pred_certs[x] = []
    id_gd_pred_certs[x] = []
    ood_pred_certs[x] = []
    ood_gd_pred_certs[x] = []
mnv_nm = {}
for i, j in enumerate(mvn):
    mnv_nm_i = {}
    for v in range(j.mean.shape[0]):
        mnv_nm_i[v] = torch.distributions.normal.Normal(j.mean[v], j.covariance_matrix[v, v])
    mnv_nm[i] = mnv_nm_i


rss, nss = [], []
for i in tqdm(zip(lbls, in_score, outs)):
    idx, val, o = i
    idx = int(idx)
    sto = torch.tensor(o)
    val = torch.tensor(val)
    to_idx = (sto > 0).nonzero().squeeze(1)
    cdf_slice = []
    for j in to_idx:
        nm = mnv_nm[idx][int(j)]
        if j == idx:
            #cdf_slice.append(1-nm.cdf(sto[j]))
            continue
        cdf_slice.append(nm.cdf(sto[j]))
    if len(cdf_slice) == 0:
        conf_scale = 1
        sm_diff = 1
    else:
        conf_scale = torch.stack(cdf_slice).mean()
        sm_diff = torch.exp(mvn[idx].log_prob(sto) - mvn[idx].log_prob(mvn[idx].mean + mvn[idx].covariance_matrix[idx]))
    scale = (sm_diff + conf_scale)/2
    tv = gd.cdf(val * scale)
    inp_val = tv
    certs.append(inp_val)


in_score = np.array(certs)
num_right = len(right_score)
num_wrong = len(wrong_score)

print('Error Rate {:.2f}'.format(100 * num_wrong / (num_wrong + num_right)))
print('Precision {:.2f}'.format(100 * (num_right / (num_wrong + num_right))))
# /////////////// End Detection Prelims ///////////////

if 'cifar10_' in args.method_name:
    print('\nUsing CIFAR-10 as typical data')
elif 'imbacifar_' in args.method_name:
    print('\nUsing Imbalanced-CIFAR-10 as typical data')
elif 'ships_' in args.method_name:
    print('\nUsing Ships as typical data')
else:
    print('\nUsing CIFAR-100 as typical data')

# /////////////// Error Detection ///////////////

print('\n\nError Detection')
dr.show_performance_fpr(wrong_score, right_score, method_name=args.method_name)
# dr.show_performance_fpr(np.array(nss), np.array(rss), method_name=args.method_name)

# /////////////// OOD Detection ///////////////
auroc_list, aupr_list, fpr_list = [], [], []


def get_and_print_results(ood_loader, num_to_avg=args.num_to_avg, class_wise=False):
    aurocs, auprs, fprs = [], [], []
    for _ in tqdm(range(num_to_avg)):
        if args.score == 'Odin':
            out_score = lib.get_ood_scores_odin(ood_loader, net, args.test_bs, ood_num_examples, args.T, args.noise)
        elif args.score == 'M':
            out_score = lib.get_Mahalanobis_score(net, ood_loader, num_classes, sample_mean, precision, count - 1,
                                                  args.noise, num_batches)
        else:
            out_score, t_pred_score = get_ood_scores(ood_loader)
            certs = []

            for i in t_pred_score:
                idx, val, o = i
                idx = int(idx)
                to = torch.tensor(o)
                sto = torch.softmax(to, dim=0)
                val = torch.tensor(val)
                to_idx = (sto > 0).nonzero().squeeze(1)
                cdf_slice = []
                for j in to_idx:
                    nm = mnv_nm[idx][int(j)]
                    if j == idx:
                        #cdf_slice.append(1 - nm.cdf(sto[j]))
                        continue
                    cdf_slice.append(nm.cdf(sto[j]))
                if len(cdf_slice) == 0:
                    conf_scale = 1
                    sm_diff = 1
                else:
                    conf_scale = torch.stack(cdf_slice).mean()
                    sm_diff = torch.exp(mvn[idx].log_prob(sto) - mvn[idx].log_prob(mvn[idx].mean + mvn[idx].covariance_matrix[idx]))
                scale = (sm_diff + conf_scale) / 2
                tv = gd.cdf(val * scale)
                inp_val = tv
                certs.append(inp_val)
                ood_pred_certs[idx].append(inp_val)
                # id_gd_pred_certs[idx].append(tvv)
                # certs.append((gd.cdf(val * (norms[idx].stddev/norms[idx].mean) / (gd.stddev/gd.mean))).numpy())
            out_score = np.array(certs)

        if args.out_as_pos:  # OE's defines out samples as positive
            measures = dr.get_measures(out_score, in_score)
        else:
            if class_wise:
                class_measures = []
                cls_aurocs, cls_auprs, cls_fprs = [], [], []
                sorted_idd = {}
                sorted_ood = {}
                weight = []
                for x in np.unique(lbls):
                    idd = np.array(id_pred_certs[x])
                    g_idd = np.array(id_gd_pred_certs[x])
                    ood = np.array(ood_pred_certs[x])
                    g_ood = np.array(ood_gd_pred_certs[x])
                    if ood.shape[0] == 0:
                        ood = np.ones(1)
                    measures = dr.get_measures(-idd, -ood)
                    class_measures.append(measures)
                    cls_aurocs.append(measures[0])
                    cls_auprs.append(measures[1])
                    cls_fprs.append(measures[2])
                    weight.append(ood.shape[0] + idd.shape[0])
                w = weight / np.array(weight).sum()
                cls_aurocs_mean = np.sum(w * np.array(cls_aurocs))
                cls_auprs_mean = np.sum(w * np.array(cls_auprs))
                cls_fprs_mean = np.sum(w * np.array(cls_fprs))
                measures = [cls_aurocs_mean, cls_auprs_mean, cls_fprs_mean]
            else:
                measures = dr.get_measures(-in_score, -out_score)

        aurocs.append(measures[0])
        auprs.append(measures[1])
        fprs.append(measures[2])

    auroc = np.mean(aurocs)
    aupr = np.mean(auprs)
    fpr = np.mean(fprs)
    auroc_list.append(auroc)
    aupr_list.append(aupr)
    fpr_list.append(fpr)

    if num_to_avg >= 5:
        dr.print_measures_with_std(aurocs, auprs, fprs, args.method_name)
    else:
        dr.print_measures_fprs(auroc, aupr, fpr, args.method_name)


#
# if 'cifar10_' not in args.method_name:
#     # /////////////// CIFAR-10 ///////////////
#     ood_data = dset.CIFAR10('nobackup-slow/dataset/my_xfdu/cifarpy', train=False,
#                             transform=trn.Compose(
#                                 [trn.Resize((inp_size, inp_size)), trn.ToTensor(), trn.Normalize(mean, std)]))
#     ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True)
#
#     print('\n\nCIFAR-10 Detection')
#     get_and_print_results(ood_loader)

# /////////////// iSUN ///////////////
ood_data = dset.ImageFolder(root="nobackup-slow/dataset/iSUN",
                            transform=trn.Compose(
                                [trn.Resize((inp_size, inp_size)), trn.ToTensor(), trn.Normalize(mean, std)]))
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=1, pin_memory=True)
print('\n\niSUN Detection')
get_and_print_results(ood_loader)

# /////////////// LSUN-C ///////////////
ood_data = dset.ImageFolder(root="nobackup-slow/dataset/LSUN_C",
                            transform=trn.Compose(
                                [trn.Resize((inp_size, inp_size)), trn.ToTensor(), trn.Normalize(mean, std)]))
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=1, pin_memory=True)
print('\n\nLSUN_C Detection')
get_and_print_results(ood_loader)

# /////////////// SVHN /////////////// # cropped and no sampling of the test set
ood_data = svhn.SVHN(root='nobackup-slow/dataset/svhn/', split="test",
                     transform=trn.Compose(
                         [trn.Resize((inp_size, inp_size)),
                          trn.ToTensor(), trn.Normalize(mean, std)]), download=True)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=2, pin_memory=True)
print('\n\nSVHN Detection')
get_and_print_results(ood_loader)

# /////////////// Textures ///////////////
ood_data = dset.ImageFolder(root="nobackup-slow/dataset/dtd/images",
                            transform=trn.Compose([trn.Resize(inp_size), trn.CenterCrop(inp_size),
                                                   trn.ToTensor(), trn.Normalize(mean, std)]))
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=4, pin_memory=True)
print('\n\nTexture Detection')
get_and_print_results(ood_loader)

# /////////////// Places365 ///////////////
ood_data = dset.ImageFolder(root="nobackup-slow/dataset/places365/",
                            transform=trn.Compose([trn.Resize(inp_size), trn.CenterCrop(inp_size),
                                                   trn.ToTensor(), trn.Normalize(mean, std)]))
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=2, pin_memory=True)
print('\n\nPlaces365 Detection')
get_and_print_results(ood_loader)

# /////////////// LSUN-R ///////////////
ood_data = dset.ImageFolder(root="nobackup-slow/dataset/LSUN_resize",
                            transform=trn.Compose(
                                [trn.Resize((inp_size, inp_size)), trn.ToTensor(), trn.Normalize(mean, std)]))
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=1, pin_memory=True)
print('\n\nLSUN_Resize Detection')
get_and_print_results(ood_loader)

# /////////////// Mean Results ///////////////

print('\n\nMean Test Results!!!!!')
dr.print_measures(np.mean(auroc_list), np.mean(aupr_list), np.mean(fpr_list), method_name=args.method_name)

# /////////////// CIFAR-100 ///////////////

ood_data = dset.CIFAR100('nobackup-slow/dataset/cifar100', train=False, download=True,
                         transform=trn.Compose(
                             [trn.Resize((inp_size, inp_size)), trn.ToTensor(), trn.Normalize(mean, std)]))
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True)

print('\n\nCIFAR-100 Detection')
get_and_print_results(ood_loader)

#
# # /////////////// celeba ///////////////
# ood_data = dset.ImageFolder(root="/nobackup-slow/dataset/celeba",
#                             transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
#                                 trn.ToTensor(), trn.Normalize((.5, .5, .5), (.5, .5, .5))]))
# ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
#                                          num_workers=1, pin_memory=True)
# print('\n\nceleba Detection')
# get_and_print_results(ood_loader)


# /////////////// OOD Detection of Validation Distributions ///////////////

if args.validate is False:
    exit()

auroc_list, aupr_list, fpr_list = [], [], []

# /////////////// Uniform Noise ///////////////

dummy_targets = torch.ones(ood_num_examples * args.num_to_avg)
ood_data = torch.from_numpy(
    np.random.uniform(size=(ood_num_examples * args.num_to_avg, 3, 32, 32),
                      low=-1.0, high=1.0).astype(np.float32))
ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True)

print('\n\nUniform[-1,1] Noise Detection')
get_and_print_results(ood_loader)

# /////////////// Arithmetic Mean of Images ///////////////

if 'cifar10_' in args.method_name:
    ood_data = dset.CIFAR100('nobackup-slow/dataset/cifar100', train=False, transform=test_transform)
else:
    ood_data = dset.CIFAR10('nobackup-slow/dataset/my_xfdu/cifarpy', train=False, transform=test_transform)


class AvgOfPair(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.shuffle_indices = np.arange(len(dataset))
        np.random.shuffle(self.shuffle_indices)

    def __getitem__(self, i):
        random_idx = np.random.choice(len(self.dataset))
        while random_idx == i:
            random_idx = np.random.choice(len(self.dataset))

        return self.dataset[i][0] / 2. + self.dataset[random_idx][0] / 2., 0

    def __len__(self):
        return len(self.dataset)


ood_loader = torch.utils.data.DataLoader(AvgOfPair(ood_data),
                                         batch_size=args.test_bs, shuffle=True,
                                         num_workers=args.prefetch, pin_memory=True)

print('\n\nArithmetic Mean of Random Image Pair Detection')
get_and_print_results(ood_loader)

# /////////////// Geometric Mean of Images ///////////////

if 'cifar10_' in args.method_name:
    ood_data = dset.CIFAR100('nobackup-slow/dataset/cifar100', train=False, transform=test_transform)
else:
    ood_data = dset.CIFAR10('nobackup-slow/dataset/my_xfdu/cifarpy', train=False, transform=test_transform)


class GeomMeanOfPair(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.shuffle_indices = np.arange(len(dataset))
        np.random.shuffle(self.shuffle_indices)

    def __getitem__(self, i):
        random_idx = np.random.choice(len(self.dataset))
        while random_idx == i:
            random_idx = np.random.choice(len(self.dataset))

        return trn.Normalize(mean, std)(torch.sqrt(self.dataset[i][0] * self.dataset[random_idx][0])), 0

    def __len__(self):
        return len(self.dataset)


ood_loader = torch.utils.data.DataLoader(
    GeomMeanOfPair(ood_data), batch_size=args.test_bs, shuffle=True,
    num_workers=args.prefetch, pin_memory=True)

print('\n\nGeometric Mean of Random Image Pair Detection')
get_and_print_results(ood_loader)

# /////////////// Jigsaw Images ///////////////

ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=args.prefetch, pin_memory=True)

jigsaw = lambda x: torch.cat((
    torch.cat((torch.cat((x[:, 8:16, :16], x[:, :8, :16]), 1),
               x[:, 16:, :16]), 2),
    torch.cat((x[:, 16:, 16:],
               torch.cat((x[:, :16, 24:], x[:, :16, 16:24]), 2)), 2),
), 1)

ood_loader.dataset.transform = trn.Compose([trn.ToTensor(), jigsaw, trn.Normalize(mean, std)])

print('\n\nJigsawed Images Detection')
get_and_print_results(ood_loader)

# /////////////// Speckled Images ///////////////

speckle = lambda x: torch.clamp(x + x * torch.randn_like(x), 0, 1)
ood_loader.dataset.transform = trn.Compose([trn.ToTensor(), speckle, trn.Normalize(mean, std)])

print('\n\nSpeckle Noised Images Detection')
get_and_print_results(ood_loader)

# /////////////// Pixelated Images ///////////////

pixelate = lambda x: x.resize((int(32 * 0.2), int(32 * 0.2)), PILImage.BOX).resize((32, 32), PILImage.BOX)
ood_loader.dataset.transform = trn.Compose([pixelate, trn.ToTensor(), trn.Normalize(mean, std)])

print('\n\nPixelate Detection')
get_and_print_results(ood_loader)

# /////////////// RGB Ghosted/Shifted Images ///////////////

rgb_shift = lambda x: torch.cat((x[1:2].index_select(2, torch.LongTensor([i for i in range(32 - 1, -1, -1)])),
                                 x[2:, :, :], x[0:1, :, :]), 0)
ood_loader.dataset.transform = trn.Compose([trn.ToTensor(), rgb_shift, trn.Normalize(mean, std)])

print('\n\nRGB Ghosted/Shifted Image Detection')
get_and_print_results(ood_loader)

# /////////////// Inverted Images ///////////////

# not done on all channels to make image ood with higher probability
invert = lambda x: torch.cat((x[0:1, :, :], 1 - x[1:2, :, ], 1 - x[2:, :, :],), 0)
ood_loader.dataset.transform = trn.Compose([trn.ToTensor(), invert, trn.Normalize(mean, std)])

print('\n\nInverted Image Detection')
get_and_print_results(ood_loader)

# /////////////// Mean Results ///////////////

print('\n\nMean Validation Results')
dr.print_measures(np.mean(auroc_list), np.mean(aupr_list), np.mean(fpr_list), method_name=args.method_name)
