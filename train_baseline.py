import argparse
from functools import partial
import numpy as np
import sys
import os
import os.path as osp
import scipy.misc
import random
import timeit
import pickle
import logging
import tqdm

import torch
import torch.nn as nn
from torch.utils import data, model_zoo
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transform
from torchvision.utils import save_image
from torch.nn.modules.loss import CrossEntropyLoss

from model.deeplabv2 import Res_Deeplab

# from model.deeplabv3p import Res_Deeplab
from tensorboardX import SummaryWriter
from utils.loss import CrossEntropy2d, DiceLoss
from utils.metric import get_iou
from utils import ramps
from data.voc_dataset import VOCDataSet, VOCGTDataSet, VOCCTADataSet, TwoStreamBatchSampler
from data import get_loader, get_data_path
from data.augmentations import *
from data.cta.ctaugment import CTAugment

start = timeit.default_timer()

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

# dataset params
NUM_CLASSES = 21  # 21 for PASCAL-VOC / 60 for PASCAL-Context

DATASET = "pascal_voc"  # pascal_voc or pascal_context

DATA_DIRECTORY = "./data/voc_dataset/"
DATA_LIST_PATH = "./data/voc_list/train_aug.txt"
CHECKPOINT_DIR = "./checkpoints/voc_full/"

MODEL = "DeepLab"
BATCH_SIZE = 10
NUM_STEPS = 40000
SAVE_PRED_EVERY = 5000

INPUT_SIZE = "321, 321"
IGNORE_LABEL = 255  # 255 for PASCAL-VOC / -1 for PASCAL-Context

LEARNING_RATE = 2.5e-4
WEIGHT_DECAY = 0.0005
POWER = 0.9
MOMENTUM = 0.9
NUM_WORKERS = 4
RANDOM_SEED = 1234

RESTORE_FROM = "./pretrained_models/resnet101-5d3b4d8f.pth"  # ImageNet pretrained encoder

SPLIT_ID = "./splits/voc/split_0.pkl"
LABELED_RATIO = 0.1


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL, help="available options : DeepLab/DRN")
    parser.add_argument("--dataset", type=str, default=DATASET, help="dataset to be used")
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE, help="Number of images sent to the network in one step."
    )
    parser.add_argument(
        "--num-workers", type=int, default=NUM_WORKERS, help="number of workers for multithread dataloading."
    )
    parser.add_argument(
        "--data-dir", type=str, default=DATA_DIRECTORY, help="Path to the directory containing the PASCAL VOC dataset."
    )
    parser.add_argument(
        "--data-list", type=str, default=DATA_LIST_PATH, help="Path to the file listing the images in the dataset."
    )
    parser.add_argument("--split-id", type=str, default=SPLIT_ID, help="name of split pickle file")
    parser.add_argument(
        "--input-size", type=str, default=INPUT_SIZE, help="Comma-separated string with height and width of images."
    )
    parser.add_argument(
        "--ignore-label", type=float, default=IGNORE_LABEL, help="label value to ignored for loss calculation"
    )
    parser.add_argument(
        "--labeled-ratio", type=float, default=LABELED_RATIO, help="ratio of labeled samples/total samples"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=LEARNING_RATE,
        help="Base learning rate for training with polynomial decay.",
    )
    parser.add_argument("--momentum", type=float, default=MOMENTUM, help="Momentum component of the optimiser.")
    parser.add_argument(
        "--num-classes", type=int, default=NUM_CLASSES, help="Number of classes to predict (including background)."
    )
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS, help="Number of iterations")
    parser.add_argument("--power", type=float, default=POWER, help="Decay parameter to compute the learning rate.")
    parser.add_argument(
        "--random-mirror", action="store_true", help="Whether to randomly mirror the inputs during the training."
    )
    parser.add_argument(
        "--random-scale", action="store_true", help="Whether to randomly scale the inputs during the training."
    )
    parser.add_argument(
        "--random-seed", type=int, default=RANDOM_SEED, help="Random seed to have reproducible results."
    )
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM, help="Where restore model parameters from.")
    parser.add_argument(
        "--save-pred-every", type=int, default=SAVE_PRED_EVERY, help="Save summaries and checkpoint every often."
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default=CHECKPOINT_DIR, help="Where to save checkpoints of the model."
    )
    parser.add_argument(
        "--weight-decay", type=float, default=WEIGHT_DECAY, help="Regularisation parameter for L2-loss."
    )
    parser.add_argument("--gpu", type=int, default=0, help="choose gpu device.")
    parser.add_argument("--out", default="result", help="directory to output the result")
    parser.add_argument("--consistency", type=float, default=0.1, help="consistency")
    parser.add_argument("--consistency_rampup", type=float, default=200.0, help="consistency_rampup")
    return parser.parse_args()


args = get_arguments()


def loss_calc(pred, label, gpu):
    label = Variable(label.long()).cuda(gpu)
    criterion = CrossEntropy2d(ignore_label=args.ignore_label).cuda(gpu)
    return criterion(pred, label)


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]["lr"] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]["lr"] = lr * 10


def worker_init_fn(worker_id):
    random.seed(args.random_seed + worker_id)


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def main():
    logpath = args.out + "/log_baseline"
    writer = SummaryWriter(logpath)

    h, w = map(int, args.input_size.split(","))
    input_size = (h, w)
    cudnn.enabled = True
    gpu = args.gpu

    # create network
    model = Res_Deeplab(num_classes=args.num_classes)
    model.cuda()

    # load pretrained parameters
    saved_state_dict = torch.load(args.restore_from)

    # only copy the params that exist in current model (caffe-like)
    new_params = model.state_dict().copy()
    for name, param in new_params.items():
        if name in saved_state_dict and param.size() == saved_state_dict[name].size():
            new_params[name].copy_(saved_state_dict[name])
    model.load_state_dict(new_params)

    model.train()
    model.cuda(args.gpu)

    cudnn.benchmark = True

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    if args.dataset == "pascal_voc":
        train_dataset = VOCDataSet(
            args.data_dir,
            args.data_list,
            crop_size=input_size,
            scale=args.random_scale,
            mirror=args.random_mirror,
            mean=IMG_MEAN,
        )
        valloader = data.DataLoader(
            VOCDataSet(args.data_dir, args.data_list, crop_size=(505, 505), mean=IMG_MEAN, scale=False,),
            batch_size=1,
            shuffle=False,
            pin_memory=True,
        )
        interp_val = nn.Upsample(size=(505, 505), mode="bilinear", align_corners=True)

    elif args.dataset == "pascal_context":
        input_transform = transform.Compose(
            [transform.ToTensor(), transform.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )
        data_kwargs = {"transform": input_transform, "base_size": 505, "crop_size": 321}
        # train_dataset = get_segmentation_dataset('pcontext', split='train', mode='train', **data_kwargs)
        data_loader = get_loader("pascal_context")
        data_path = get_data_path("pascal_context")
        train_dataset = data_loader(data_path, split="train", mode="train", **data_kwargs)

    elif args.dataset == "cityscapes":
        data_loader = get_loader("cityscapes")
        data_path = get_data_path("cityscapes")
        data_aug = Compose([RandomCrop_city((256, 512)), RandomHorizontallyFlip()])
        train_dataset = data_loader(data_path, is_transform=True, augmentations=data_aug)

    train_dataset_size = len(train_dataset)
    logging.info(f"dataset size: {train_dataset_size}")

    # fully supervised
    if args.labeled_ratio is None:
        trainloader = data.DataLoader(
            train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=4, pin_memory=True
        )
    # semi-supervised
    else:
        partial_size = int(args.labeled_ratio * train_dataset_size)
        labeled_bs = args.batch_size // 2
        if args.split_id is not None:
            train_ids = pickle.load(open(args.split_id, "rb"))
            logging.info("loading train ids from {}".format(args.split_id))
        else:
            train_ids = np.arange(train_dataset_size)
            np.random.shuffle(train_ids)

        pickle.dump(train_ids, open(os.path.join(args.checkpoint_dir, "split.pkl"), "wb"))

        # train_sampler = data.sampler.SubsetRandomSampler(train_ids[:partial_size])
        labeled_idxs = list(range(0, partial_size))
        unlabeled_idxs = list(range(partial_size, train_dataset_size))
        batch_sampler = TwoStreamBatchSampler(
            labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size - labeled_bs
        )
        # trainloader = data.DataLoader(
        #     train_dataset, batch_size=args.batch_size, sampler=batch_sampler, num_workers=4, pin_memory=True
        # )
        trainloader = data.DataLoader(
            train_dataset, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn,
        )

    trainloader_iter = iter(trainloader)

    # optimizer for segmentation network
    optimizer = optim.SGD(
        model.optim_parameters(args), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay
    )
    optimizer.zero_grad()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(args.num_classes)

    # loss/ bilinear upsampling
    interp = nn.Upsample(size=(input_size[0], input_size[1]), mode="bilinear", align_corners=True)

    for i_iter in range(args.num_steps):
        model.train()
        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)
        try:
            batch_lab = next(trainloader_iter)
        except:
            trainloader_iter = iter(trainloader)
            batch_lab = next(trainloader_iter)
        images, labels, _, _, index = batch_lab
        images = Variable(images).cuda(args.gpu)

        pred = interp(model(images))
        loss = loss_calc(pred, labels, args.gpu)

        loss.backward()

        optimizer.step()

        logging.info("iter = {0:8d}/{1:8d}, loss_seg = {2:.3f}".format(i_iter, args.num_steps, loss.item()))

        if i_iter % 10 == 0:
            writer.add_scalar("loss/loss", loss.item(), i_iter)
        if i_iter % 200 == 0 and i_iter > 0:
            miou_val, loss_val = validate(valloader, interp_val, model, writer, i_iter)
            logging.info(f"miou: {miou_val}, loss: {loss_val}")
            writer.add_scalar("val/miou", miou_val, i_iter)
        if i_iter >= args.num_steps - 1:
            print("save model ...")
            torch.save(model.state_dict(), osp.join(args.checkpoint_dir, "VOC_" + str(args.num_steps) + ".pth"))
            break

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print("saving checkpoint ...")
            torch.save(model.state_dict(), osp.join(args.checkpoint_dir, "VOC_" + str(i_iter) + ".pth"))

    end = timeit.default_timer()
    print(end - start, "seconds")


def validate(valloader, interp_val, model, writer, i_iter):
    print("validating...")
    loss_val = 0
    data_list = []
    model.eval()
    for index, batch in enumerate(valloader):
        if index == 50:
            break
        image, label, size, name, _ = batch
        size = size[0]
        with torch.no_grad():
            output = model(image.cuda())
            output_display = torch.argmax(output, dim=1)
        output = interp_val(output)
        # loss_ce = loss_calc(output, label, args.gpu)
        output = output.cpu().data[0].numpy()

        if index == 0:
            writer.add_image("test/image", image[0], i_iter)
            writer.add_image("test/prediction", output_display, i_iter)
            writer.add_image("test/label", label, i_iter)
        # loss_val += loss_ce.item()

        if args.dataset == "pascal_voc":
            output = output[:, : size[0], : size[1]]
            gt = np.asarray(label[0].numpy()[: size[0], : size[1]], dtype=int)
        elif args.dataset == "ade20k":
            output = output[:, : size[0], : size[1]]
            gt = np.asarray(label[0].numpy()[: size[0], : size[1]], dtype=int)
        elif args.dataset == "pascal_context":
            gt = np.asarray(label[0].numpy(), dtype=int)
        elif args.dataset == "cityscapes":
            gt = np.asarray(label[0].numpy(), dtype=int)

        output = output.transpose(1, 2, 0)
        output = np.asarray(np.argmax(output, axis=2), dtype=int)

        # writer.add_image("test/prediction", output, i_iter)
        data_list.append([gt.flatten(), output.flatten()])

    torch.cuda.empty_cache()

    filename = os.path.join(args.out, "result.txt")
    miou_val = get_iou(data_list, args.num_classes, filename)
    return miou_val, loss_val / 50


if __name__ == "__main__":
    torch.cuda.set_device(args.gpu)
    logging.basicConfig(
        filename=args.out + "/log.txt",
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    main()
