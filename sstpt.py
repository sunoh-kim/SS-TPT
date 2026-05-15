import argparse
import time
from copy import deepcopy
from PIL import Image
import numpy as np

import os

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torch.nn.functional as F

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import torchvision.models as models

from clip.custom_clip import get_coop
from data.imagnet_prompts import imagenet_classes
from data.datautils import AugMixAugmenter, build_dataset
from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, load_model_weight, set_random_seed
from data.cls_to_names import *
from data.fewshot_datasets import fewshot_datasets
from data.imagenet_variants import thousand_k_to_200, imagenet_a_mask, imagenet_r_mask, imagenet_v_mask
import random

import torchattacks
import matplotlib.pyplot as plt
import pandas as pd


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s
        
def visualize_all_metrics(suit_n, stab_n, combined_score, idx, save_path=None):
    import matplotlib.pyplot as plt

    def to_np(x):
        try:
            return x.detach().float().cpu().numpy()
        except Exception:
            return np.asarray(x, dtype=float)

    s = to_np(suit_n)
    t = to_np(stab_n)
    c = to_np(combined_score)

    style = {
        "font.family": "Times New Roman",
        "font.size": 20,
    }

    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(12.5, 8))
        x = np.arange(1, len(s) + 1)

        ax.plot(x, s, label='Suitability (z)', linewidth=5, marker='o', markersize=20, linestyle='-')
        ax.plot(x, t, label='Stability (z)',   linewidth=5, marker='x', markersize=20, linestyle='--')
        ax.plot(x, c, label='SS Score',        linewidth=5, marker='s', markersize=20, linestyle='-.')

        ax.set_title(f"Suitability, Stability, and SS Scores (iter {idx})", pad=12, fontweight='bold')
        ax.set_xlabel("Sample Index", fontsize=40, fontweight='bold', family="Times New Roman")
        ax.set_ylabel("Score", fontsize=45, fontweight='bold', family="Times New Roman")

        ax.set_xticks(x)
        all_vals = np.concatenate([s, t, c])
        vr = max(all_vals.max() - all_vals.min(), 1e-8)
        pad = max(0.6, vr * 0.05)
        ax.set_ylim([float(all_vals.min()) - pad, float(all_vals.max()) + pad])

        ax.tick_params(axis='both', labelsize=40)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

        leg = ax.legend(loc='best', frameon=True)
        leg.get_frame().set_linewidth(1.0)

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

class AddGaussianNoise(torch.nn.Module):
    def __init__(self, sigma_min=0.0, sigma_max=0.02):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
    def forward(self, x):
        sigma = torch.empty(1, device=x.device).uniform_(self.sigma_min,
                                                         self.sigma_max)
        noise = torch.randn_like(x) * sigma
        return (x + noise).clamp(0, 1)

def build_tiny_aug(
    rotation=5.0,
    translate=0.04,
    scale_var=0.08, 
    brightness=0.08,
    contrast=0.08,
    saturation=0.08,
    hue=0.02,
    blur_sigma=(0.5, 1.2),
    noise_sigma=0.02,
):
    affine = transforms.RandomAffine(
        degrees=rotation,
        translate=(translate, translate),
        scale=(1.0 - scale_var, 1.0 + scale_var),
        shear=0.0,
        interpolation=InterpolationMode.BILINEAR
    )

    color_jitter = transforms.ColorJitter(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue
    )

    tiny_aug = transforms.Compose([
        transforms.RandomApply([affine], p=1.0), 
        transforms.RandomApply([color_jitter], p=0.5),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=blur_sigma)], p=0.5),
        transforms.RandomApply([AddGaussianNoise(0.0, noise_sigma)], p=0.5),
    ])
    return tiny_aug

def select_confident_views(logits, k):
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    # We select at least two confident samples for the consistency loss.    
    # k = min(batch_entropy.size(0), max(int(batch_entropy.size(0) * p), 3))
    k = min(batch_entropy.size()[0], max(2, k))
    idx = batch_entropy.topk(k, largest=False).indices
    return logits[idx], idx

def js_divergence(p, q, eps=1e-8):
    m = 0.5 * (p + q)
    return 0.5 * (F.kl_div((p+eps).log(), (m+eps), reduction='none', log_target=False).sum(-1) +
                  F.kl_div((q+eps).log(), (m+eps), reduction='none', log_target=False).sum(-1))

def symmetric_kl(p, q, eps=1e-8):
    return 0.5 * (F.kl_div((p+eps).log(), (q+eps), reduction='none', log_target=False).sum(-1) +
            F.kl_div((q+eps).log(), (p+eps), reduction='none', log_target=False).sum(-1))
    
@torch.no_grad()
def compute_stability(
    model,
    views, 
    base_logits,
    aug_builder,
    K=1,
    divergence='js',
):

    model.eval()
    p = base_logits.softmax(dim=-1)

    divergences = []
    for _ in range(K):
        aug_views = aug_builder(views)
        q_logits = model(aug_views)
        q = q_logits.softmax(dim=-1)

        if divergence == 'js':
            d = js_divergence(p, q)
        elif divergence == 'sym':
            d = symmetric_kl(p, q)
        else:
            raise ValueError(f"Unsupported divergence: {divergence}")
        divergences.append(d)

    div = torch.stack(divergences, dim=0).mean(0)  # [B]
    stab_score = 1 / (div + 1e-8)  

    return stab_score

def entropy_avg(outputs):
    batch_entropy = -(outputs.softmax(1) * outputs.log_softmax(1)).sum(1)
    return batch_entropy.mean()

def self_consistency_loss_with_rs_view(outputs, weight, ref_mode):
    eps = 1e-8
    probs = outputs.softmax(1).clamp_min(eps) # [B,C]

    if ref_mode == "weighted":
        weight = weight.unsqueeze(1)
        weighted_probs = weight * probs # [B,C]
        weighted_probs_sum = torch.sum(weighted_probs, dim=0, keepdim=True) # [1,C]
        # Leave one out
        loo_num = weighted_probs_sum - weighted_probs # [B,C]
        loo_den = (weight.sum(dim=0, keepdim=True) - weight).clamp_min(eps)  # [B,1]
        ref_probs = (loo_num / loo_den).detach()
        kl = F.kl_div(probs.log(), ref_probs, reduction='batchmean', log_target=False)

    elif ref_mode == "weighted_self":
        weight = weight.unsqueeze(1)
        ref_prob = ((weight * probs).sum(dim=0, keepdim=True) / (weight.sum() + eps)).detach() 
        kl = F.kl_div(probs.log(), ref_prob.expand_as(probs),
            reduction='batchmean', log_target=False)

    elif ref_mode == "best_score":
        ref_idx = torch.argmax(weight)
        ref_prob = probs[ref_idx].detach()  
        kl = F.kl_div(probs.log(), ref_prob.expand_as(probs), 
            reduction='batchmean', log_target=False)
        
    elif ref_mode == "min_ent":
        ref_prob = probs[0].detach()
        kl = F.kl_div(probs.log(), ref_prob.expand_as(probs), 
            reduction='batchmean', log_target=False)
        
    elif ref_mode == "average":
        ref_prob = probs.mean(dim=0, keepdim=False).detach()
        kl = F.kl_div(probs.log(), ref_prob.expand_as(probs),
                      reduction='batchmean', log_target=False)

    elif ref_mode == "random":
        ref_idx = torch.randint(probs.size(0), (1,), device=probs.device).item()
        ref_prob = probs[ref_idx].detach()
        kl = F.kl_div(probs.log(), ref_prob.expand_as(probs),
                      reduction='batchmean', log_target=False)
    else:
        raise ValueError(f"Unsupported ref_mode: {ref_mode}")
    
    return kl

def test_time_tuning(model, inputs, suit_score, stab_score, optimizer, scaler, args, idx):
    selected_idx = None

    for j in range(args.tta_steps):

        output_all = model(inputs) 

        if selected_idx is not None:
            output = output_all[selected_idx]
        else:
            output, selected_idx = select_confident_views(output_all, args.num_select)
            
        sel_suit_score = suit_score[selected_idx]
        sel_stab_score = stab_score[selected_idx]
        
        suit_n  = (sel_suit_score  - sel_suit_score.min())  / (sel_suit_score.max()  - sel_suit_score.min()  + 1e-8)
        stab_n = (sel_stab_score - sel_stab_score.min()) / (sel_stab_score.max() - sel_stab_score.min() + 1e-8)
        score = args.ss_ratio * suit_n + (1-args.ss_ratio) * stab_n
        weight_cons = torch.nn.functional.softmax(score/args.temp, dim=-1)
        
        ent_loss = entropy_avg(output)
        
        if args.cons_weight > 0:
            assert output.size(0) > 1
            kl_loss = self_consistency_loss_with_rs_view(output, weight_cons, args.ref_mode)
            loss = ent_loss + args.cons_weight  * kl_loss
        else:
            loss = ent_loss
                
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return


def main():
    t0 = time.time()
    
    args = parser.parse_args()
    set_random_seed(args.seed)
    if hasattr(torchattacks, "config"):
        torchattacks.config.set_seed(args.seed)
    
    if 'alpha' not in vars(args) or args.alpha == 0.0:
        args.alpha = args.eps / 4.0
    safe_arch = args.arch.replace('/', '_')
    args.output_dir = os.path.join(args.output_dir, safe_arch, args.test_sets, 'eps_'+str(args.eps)+'_alpha_'+str(args.alpha)+'_step_'+str(args.steps)
                                   +'_batchsize_'+str(args.batch_size))

    os.makedirs(args.output_dir, exist_ok=True)

    args.out_file = open(os.path.join(args.output_dir, 'log.txt'), 'w')
    args.out_file.write(print_args(args)+'\n')
    args.out_file.flush()

    assert args.gpu is not None

    print("Use GPU: {} for training".format(args.gpu))

    # model
    dset = args.test_sets
    if len(dset) > 1: 
        classnames = eval("{}_classes".format(dset.lower()))
    else:
        assert dset in ['A', 'R', 'K', 'V', 'I']
        classnames_all = imagenet_classes
        classnames = []
        if dset in ['A', 'R', 'V']:
            label_mask = eval("imagenet_{}_mask".format(dset.lower()))
            if dset == 'R':
                for i, m in enumerate(label_mask):
                    if m:
                        classnames.append(classnames_all[i])
            else:
                classnames = [classnames_all[i] for i in label_mask]
        else:
            classnames = classnames_all
    args.classnames = classnames

    model = get_coop(args.arch, classnames, args.gpu, args.n_ctx, args.ctx_init)
    model_state = None

    ###### load robust vision encoder (TeCoA) ######
    if len(args.load_tecoa) > 0:
        args.robust_pretrain_path = {
            'RN50-eps1': '../pretrain/tecoa/weights/backbone/rn50_eps1.pth.tar',
            'ViT-B/32-eps4': '../pretrain/tecoa/weights/backbone/vitb32_eps4.pth.tar',
        }[args.load_tecoa]
        robust_state_dict = torch.load(args.robust_pretrain_path, map_location='cpu')
        model.image_encoder.load_state_dict(robust_state_dict['vision_encoder_state_dict'])
        print('load robust vision encoder')

    for name, param in model.named_parameters():
        if "prompt_learner" not in name:
            param.requires_grad_(False)

    print("=> Model created: visual backbone {}".format(args.arch))
    
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    else:
        assert args.gpu is not None
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    trainable_param = model.prompt_learner.parameters()
    optimizer = torch.optim.AdamW(trainable_param, args.lr)
    optim_state = deepcopy(optimizer.state_dict())

    scaler = None
    # cudnn.benchmark = True
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])

    # iterating through eval datasets
    
    if True:
        base_transform = transforms.Compose([
            transforms.Resize(args.resolution, interpolation=BICUBIC),
            transforms.CenterCrop(args.resolution)])
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            # normalize
            ])
        data_transform = AugMixAugmenter(base_transform, preprocess, n_views=args.batch_size-1, 
                                        augmix=len(dset)>1)
        batchsize = 1

        val_dataset = build_dataset(dset, data_transform, args.data, mode=args.dataset_mode)
        print("number of test samples: {}".format(len(val_dataset)))
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batchsize, shuffle=False,
                    num_workers=args.workers, pin_memory=True)

        print("evaluating: {}".format(dset))
        
        t1 = time.time()
        elapsed_ds = t1 - t0
        log_msg = f"[{dset}] Loading time: {elapsed_ds:.2f} seconds\n"
        print(log_msg.strip())
        args.out_file.write(log_msg)
        args.out_file.flush()
        
        result = test_time_adapt_eval(val_loader, model, model_state, optimizer, optim_state, scaler, args, data_transform)
        del val_dataset, val_loader
        if args.eps <= 0:
            print_log = "=> Acc. on testset [{}]: SSTPT Clean Acc @1 {}".format(dset, result)
            save_log = {'sstpt_clean_acc': result}
        else:
            print_log = "=> Acc. on testset [{}]: SSTPT Adv Acc @1 {} ".format(dset, result)
            save_log = {'sstpt_adv_acc': result}
      
        args.out_file.write(print_log + '\n')
        args.out_file.flush()
        print(print_log+'\n')

        elapsed_ds = time.time() - t1
        log_msg = f"[{dset}] Evaluation time: {elapsed_ds:.2f} seconds\n"
        print(log_msg.strip())
        args.out_file.write(log_msg)
        args.out_file.flush()

        # torch.save(save_log, os.path.join(args.output_dir, 'results_log.pt'))
        

def test_time_adapt_eval(val_loader, model, model_state, optimizer, optim_state, scaler, args, data_transform):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    tpt1 = AverageMeter('SSTPTAcc@1', ':6.2f', Summary.AVERAGE)

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, tpt1],
        prefix='Test: ')

    # reset model and switch to evaluate mode
    model.eval()

    if args.eps > 0.0:
        assert args.steps > 0
        atk = torchattacks.PGD(model, eps=args.eps/255, alpha=args.alpha/255, steps=args.steps)
        
    end = time.time()

    tiny_aug = build_tiny_aug()
    
    with torch.no_grad():
        text_features = model.get_text_features()
        text_features = F.normalize(text_features, dim=1)
        logit_scale = model.logit_scale.exp()
    
    for i, (images, target) in enumerate(val_loader):
        assert args.gpu is not None
        target = target.cuda(args.gpu, non_blocking=True)

        if args.eps > 0.0:
            image = images[0].cuda(args.gpu, non_blocking=True)
            adv_image = atk(image, target)        
            img_adv = transforms.ToPILImage()(adv_image.squeeze(0))
            images = data_transform(img_adv)
            images = [_.unsqueeze(0) for _ in images]

        if isinstance(images, list):
            for k in range(len(images)):
                images[k] = images[k].cuda(args.gpu, non_blocking=True)
            image = images[0]
        else:
            if len(images.size()) > 4:
                # when using ImageNet Sampler as the dataset
                assert images.size()[0] == 1
                images = images.squeeze(0)
            images = images.cuda(args.gpu, non_blocking=True)
            image = images
        
        images = torch.cat(images, dim=0)

        # reset model
        with torch.no_grad():
            model.reset()
        optimizer.load_state_dict(optim_state)

        # with torch.no_grad():
        #     clip_features, _, _ = model.forward_features(images)
        #     clip_outputs = model(images)

        with torch.no_grad():
            clip_features = model.image_encoder(model.normalize(images.type(model.dtype)))
            clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True)
            clip_outputs = logit_scale * clip_features @ text_features.t()
            
        suit_k = images.shape[0] - 1
        sim_matrix = torch.bmm(clip_features.unsqueeze(0), clip_features.unsqueeze(0).permute(0, 2, 1)) # [1, B, B]
        if args.suit_mode == "sim":
            sorted_vals, _ = sim_matrix.sort(dim=-1, descending=True)
            top_k_mean = sorted_vals[..., 1:suit_k+1].mean(dim=-1) 
            suit_score = top_k_mean.squeeze(0)
        elif args.suit_mode == "density":
            dist = 2 - 2 * sim_matrix  # [1, B, B]
            knn_dists, _ = dist.topk(suit_k+1, dim=-1, largest=False)
            suit_score = 1.0 / (knn_dists[:, :, 1:].mean(-1) + 1e-8) # [1, B]
            suit_score = suit_score.squeeze(0) # [B]
        else: 
            raise ValueError(f"Unsupported suit_mode: {args.suit_mode}")
        
        stab_score = compute_stability(
            model, images, base_logits=clip_outputs,
            aug_builder=tiny_aug
        )

        assert args.tta_steps > 0
        test_time_tuning(model, images, suit_score.detach(), stab_score.detach(), optimizer, scaler, args, i)
        with torch.no_grad():
            tuned_outputs = model(images)
            
        suit_n  = (suit_score  - suit_score.min())  / (suit_score.max()  - suit_score.min()  + 1e-8)
        stab_n = (stab_score - stab_score.min()) / (stab_score.max() - stab_score.min() + 1e-8)
        score = args.ss_ratio * suit_n + (1-args.ss_ratio) * stab_n
        weight_pred = torch.nn.functional.softmax(score/args.temp, dim=-1)
        tta_output = torch.bmm(weight_pred.unsqueeze(0).unsqueeze(-1).transpose(1, 2), tuned_outputs.unsqueeze(0)).squeeze(1)

        # measure accuracy and record loss
        tpt_acc1, _ = accuracy(tta_output, target, topk=(1, 5))
       
        tpt1.update(tpt_acc1[0], images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % args.print_freq == 0 or (i+1) == len(val_loader):
            if args.eps <= 0:
                print_log = 'iter:{}/{}, sstpt_acc1={}'.format(i, len(val_loader), tpt1.avg)
            else:
                print_log = 'iter:{}/{}, sstpt_adv1={}'.format(i, len(val_loader), tpt1.avg)
            args.out_file.write(print_log + '\n')
            args.out_file.flush()
            print(print_log+'\n')
            progress.display(i)

    progress.display_summary()

    return tpt1.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test-time Prompt Tuning')
    parser.add_argument('data', metavar='DIR', help='path to dataset root')
    parser.add_argument('--test_sets', type=str, default='Caltech101')
    parser.add_argument('--dataset_mode', type=str, default='test')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='RN50')
    parser.add_argument('--resolution', default=224, type=int, help='CLIP image resolution')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N')
    parser.add_argument('-p', '--print-freq', default=200, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    
    parser.add_argument('--n_ctx', default=4, type=int, help='number of tunable tokens')
    parser.add_argument('--ctx_init', default=None, type=str, help='init tunable prompts')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='output_results/ckps/sstpt')

    parser.add_argument('--eps', default=0.0, type=float)
    parser.add_argument('--alpha', default=0.0, type=float)
    parser.add_argument('--steps', type=int, default=0)
    parser.add_argument('--temp', type=float, default=0.25,
                        help='softmax temperature')
    parser.add_argument('--lr', '--learning-rate', default=5e-3, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--num_select', default=3, type=int, help='number of selected views')
    parser.add_argument('--tta_steps', default=1, type=int, help='test-time-adapt steps')
    parser.add_argument('--load_tecoa', type=str, default='', choices=['', 'RN50-eps1', 'ViT-B/32-eps4'])
    
    parser.add_argument('--cons_weight', type=float, default=1.0,
                        help='weight for consistency loss (λ)')
    parser.add_argument('--ref_mode', type=str, default='weighted',
                    choices=['weighted', 'weighted_self', 'best_score', 'min_ent', 'average', 'random'],
                    help='strategy to select reference predictions for consistency loss computation')
    parser.add_argument('--suit_mode', type=str, default='density', choices=['sim', 'density'],
                        help="method for computing suitability weights: 'sim' (cosine similarity) or 'density' (inverse mean kNN distance)")
    parser.add_argument('--ss_ratio', type=float, default=0.4,
                        help='weighting ratio to combine suitability and stability scores')
    main()
