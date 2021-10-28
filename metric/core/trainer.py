#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tools for training and testing a model."""
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
import sklearn
import matplotlib.pyplot as plt
import tqdm

plt.switch_backend('agg')
import io
import cv2
import os
import metric.datasets.transforms as transforms
import numpy as np
import metric.core.benchmark as benchmark
import metric.core.builders as builders
import metric.core.checkpoint as checkpoint
import metric.core.config as config
import metric.core.distributed as dist
import metric.core.logging as logging
import metric.core.meters as meters
import metric.core.net as net
import metric.core.optimizer as optim
import metric.datasets.loader as loader
import torch
from metric.core.config import cfg
from torch.utils.data import Dataset
from PIL import Image, ImageFile
from metric.datasets.loader import _DATA_DIR

ImageFile.LOAD_TRUNCATED_IMAGES = True
import random

logger = logging.get_logger(__name__)
_MEAN = [0.406, 0.456, 0.485]
_SD = [0.225, 0.224, 0.229]

# Eig vals and vecs of the cov mat
_EIG_VALS = np.array([[0.2175, 0.0188, 0.0045]])
_EIG_VECS = np.array(
    [[-0.5675, 0.7192, 0.4009], [-0.5808, -0.0045, -0.8140], [-0.5836, -0.6948, 0.4203]]
)


def setup_env():
    """Sets up environment for training or testing."""
    if dist.is_master_proc():
        # Ensure that the output dir exists
        os.makedirs(cfg.OUT_DIR, exist_ok=True)
        # Save the config
        config.dump_cfg()
    # Setup logging
    logging.setup_logging()
    # Log the config as both human readable and as a json
    logger.info("Config:\n{}".format(cfg))
    logger.info(logging.dump_log_data(cfg, "cfg"))
    # Fix the RNG seeds (see RNG comment in core/config.py for discussion)
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    # Configure the CUDNN backend
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK


def setup_model():
    """Sets up a model for training or testing and log the results."""
    # Build the model
    model = builders.build_arch()
    logger.info("Model:\n{}".format(model))
    # Log model complexity
    # logger.info(logging.dump_log_data(net.complexity(model), "complexity"))
    # Transfer the model to the current GPU device
    err_str = "Cannot use more GPU devices than available"
    assert cfg.NUM_GPUS <= torch.cuda.device_count(), err_str
    cur_device = torch.cuda.current_device()
    model = model.cuda(device=cur_device)
    # Use multi-process data parallel model in the multi-gpu setting
    if cfg.NUM_GPUS > 1:
        # Make model replica operate on the current device
        model = torch.nn.parallel.DistributedDataParallel(
            module=model, device_ids=[cur_device], output_device=cur_device, find_unused_parameters=True
        )
        # Set complexity function to be module's complexity function
        # model.complexity = model.module.complexity
    return model


def train_epoch(train_loader, model, loss_fun, optimizer, train_meter, cur_epoch):
    """Performs one epoch of training."""
    # Shuffle the data
    loader.shuffle(train_loader, cur_epoch)
    # Update the learning rate
    lr = optim.get_epoch_lr(cur_epoch)
    optim.set_lr(optimizer, lr)
    # Enable training mode
    model.train()
    train_meter.iter_tic()
    for cur_iter, (inputs, labels) in enumerate(train_loader):
        # Transfer the data to the current GPU device
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        # Perform the forward pass
        logits, preds, targets = model(inputs, labels)
        # Compute the loss
        loss = loss_fun(logits, labels)
        # Perform the backward pass
        optimizer.zero_grad()
        loss.backward()
        # Update the parameters
        optimizer.step()
        # Compute the errors
        top1_err, top5_err = meters.topk_errors(logits, labels, [1, 5])
        # Combine the stats across the GPUs (no reduction if 1 GPU used)
        loss, top1_err, top5_err = dist.scaled_all_reduce([loss, top1_err, top5_err])
        # Copy the stats from GPU to CPU (sync point)
        loss, top1_err, top5_err = loss.item(), top1_err.item(), top5_err.item()
        train_meter.iter_toc()
        # Update and log stats
        mb_size = inputs.size(0) * cfg.NUM_GPUS
        train_meter.update_stats(top1_err, top5_err, loss, lr, mb_size)
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()
    # Log epoch stats
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()


@torch.no_grad()
def test_epoch(test_loader, model, test_meter, cur_epoch):
    """Evaluates the model on the test set."""
    # Enable eval mode
    model.eval()
    test_meter.iter_tic()
    for cur_iter, (inputs, labels) in enumerate(test_loader):
        # Transfer the data to the current GPU device
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        # Compute the predictions
        logits, preds, targets = model(inputs, labels)
        # Compute the errors
        top1_err, top5_err = meters.topk_errors(logits, labels, [1, 5])
        # Combine the errors across the GPUs  (no reduction if 1 GPU used)
        top1_err, top5_err = dist.scaled_all_reduce([top1_err, top5_err])
        # Copy the errors from GPU to CPU (sync point)
        top1_err, top5_err = top1_err.item(), top5_err.item()
        test_meter.iter_toc()
        # Update and log stats
        test_meter.update_stats(top1_err, top5_err, inputs.size(0) * cfg.NUM_GPUS)
        test_meter.log_iter_stats(cur_epoch, cur_iter)
        test_meter.iter_tic()
    # Log epoch stats
    test_meter.log_epoch_stats(cur_epoch)
    test_meter.reset()


def gen_plot(fpr, tpr):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.xlabel("FPR", fontsize=14)
    plt.ylabel("TPR", fontsize=14)
    plt.title("ROC Curve", fontsize=14)
    plot = plt.plot(fpr, tpr, linewidth=2)
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    plt.close()
    return buf


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def validate(model, val_dataloader, val_issame):
    model.eval()
    idx = 0
    embeddings = np.zeros([len(val_dataloader.dataset), cfg.MODEL.HEADS.REDUCTION_DIM])
    print("extracting embedding")
    with torch.no_grad():
        for batch in iter(val_dataloader):
            imgs = batch
            # tmp_embed, _ = self.model(imgs.to(conf.device))
            # tmp_embed = model(imgs.to('cuda'))
            fea = model(imgs.to('cuda'), targets=None)
            # fea = pool_layer(fea)
            fea = fea.squeeze()
            # if use_norm:
            #     fea = F.normalize(fea, p=2, dim=1)

            embeddings[idx:idx + len(imgs)] = to_numpy(fea.squeeze())

            idx += len(imgs)
    evaluate(embeddings, val_issame)


def evaluate(embeddings, actual_issame):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy, best_thresholds = calculate_roc(thresholds, embeddings1, embeddings2, np.asarray(actual_issame))


def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_thresholds = len(thresholds)
    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)

    acc_total = np.zeros((nrof_thresholds))
    recall_total = np.zeros((nrof_thresholds))
    fpr_total = np.zeros((nrof_thresholds))
    tnfp_total = np.zeros((nrof_thresholds))
    precision_total = np.zeros((nrof_thresholds))

    for threshold_idx, threshold in enumerate(thresholds):
        recall_total[threshold_idx], fpr_total[threshold_idx], acc_total[threshold_idx], tnfp_total[
            threshold_idx], precision_total[threshold_idx] = calculate_accuracy(threshold,
                                                                                dist,
                                                                                actual_issame,
                                                                                use_tnfp=True)
    best_thr_total = thresholds[np.argmax(acc_total)]
    print("total best acc: {}, recall: {}, tnfp:{}, fpr: {}, precision: {}, threshold: {}".format(np.max(acc_total),
                                                                                                  recall_total[
                                                                                                      np.argmax(
                                                                                                          acc_total)],
                                                                                                  tnfp_total[np.argmax(
                                                                                                      acc_total)],
                                                                                                  fpr_total[np.argmax(
                                                                                                      acc_total)],
                                                                                                  precision_total[
                                                                                                      np.argmax(
                                                                                                          acc_total)],
                                                                                                  best_thr_total))


def calculate_accuracy(threshold, dist, actual_issame, use_tnfp=False):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
    recall = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)

    precision = 0 if (tp + fp == 0) else float(tp) / float(tp + fp)

    acc = float(tp + tn) / dist.size
    if use_tnfp:
        tnfp = 0 if (tn + fp == 0) else float(tn) / float(tn + fp)
        return recall, fpr, acc, precision, tnfp
    else:
        return recall, fpr, acc, precision


def train_model():
    """Trains the model."""
    # Setup training/testing environment
    setup_env()
    # Construct the model, loss_fun, and optimizer
    model = setup_model()
    loss_fun = builders.build_loss_fun().cuda()
    optimizer = optim.construct_optimizer(model)
    # Load checkpoint or initial weights
    start_epoch = 0
    if cfg.TRAIN.AUTO_RESUME and checkpoint.has_checkpoint():
        last_checkpoint = checkpoint.get_last_checkpoint()
        checkpoint_epoch = checkpoint.load_checkpoint(last_checkpoint, model, optimizer)
        logger.info("Loaded checkpoint from: {}".format(last_checkpoint))
        start_epoch = checkpoint_epoch + 1
    elif cfg.TRAIN.WEIGHTS:
        checkpoint.load_checkpoint(cfg.TRAIN.WEIGHTS, model)
        logger.info("Loaded initial weights from: {}".format(cfg.TRAIN.WEIGHTS))
    # Create data loaders and meters
    train_loader = loader.construct_train_loader()
    # test_loader = loader.construct_test_loader()

    val_img_dir = os.path.join(_DATA_DIR, cfg.TEST.DATASET, cfg.TEST.SPLIT)
    val_dataloader, common_val_issame = get_val(val_img_dir,
                                                cfg.TEST.MAX_POSITIVE_CNT,
                                                cfg.TEST.BATCH_SIZE,
                                                cfg.DATA_LOADER.PIN_MEMORY,
                                                cfg.DATA_LOADER.NUM_WORKERS)

    train_meter = meters.TrainMeter(len(train_loader))
    # test_meter = meters.TestMeter(len(test_loader))
    # Compute model and loader timings
    # if start_epoch == 0 and cfg.PREC_TIME.NUM_ITER > 0:
    # benchmark.compute_time_full(model, loss_fun, train_loader, test_loader)
    # Perform the training loop
    logger.info("Start epoch: {}".format(start_epoch + 1))
    for cur_epoch in range(start_epoch, cfg.OPTIM.MAX_EPOCH):
        # Train for one epoch
        train_epoch(train_loader, model, loss_fun, optimizer, train_meter, cur_epoch)
        # Compute precise BN stats
        if cfg.BN.USE_PRECISE_STATS:
            net.compute_precise_bn_stats(model, train_loader)
        # Save a checkpoint
        if (cur_epoch + 1) % cfg.TRAIN.CHECKPOINT_PERIOD == 0:
            checkpoint_file = checkpoint.save_checkpoint(model, optimizer, cur_epoch)
            logger.info("Wrote checkpoint to: {}".format(checkpoint_file))
        # Evaluate the model
        next_epoch = cur_epoch + 1
        if next_epoch % cfg.TRAIN.EVAL_PERIOD == 0 or next_epoch == cfg.OPTIM.MAX_EPOCH:
            validate(model, val_dataloader, common_val_issame)
            # test_epoch(test_loader, model, test_meter, cur_epoch)


def get_val(data_path, max_positive_cnt, batch_size, pin_memory, num_workers):
    class ValDataset(Dataset):
        """__init__ and __len__ functions are the same as in TorchvisionDataset"""

        def __init__(self, files, transform=None):
            self.files = files
            self.transform = transform

        def _prepare_im(self, im):
            """Prepares the image for network input."""
            # Train and test setups differ
            train_size = cfg.TRAIN.IM_SIZE
            im = transforms.scale(cfg.TEST.IM_SIZE, im)
            im = transforms.center_crop(train_size, im)
            # HWC -> CHW
            im = im.transpose([2, 0, 1])
            # [0, 255] -> [0, 1]
            im = im / 255.0
            # PCA jitter
            im = transforms.lighting(im, 0.1, _EIG_VALS, _EIG_VECS)
            # Color normalization
            im = transforms.color_norm(im, _MEAN, _SD)
            return im

        def __getitem__(self, index):
            """
            Args:
                index (int): Index

            Returns:
                tuple: (sample, target) where target is class_index of the target class.
            """
            file = self.files[index]
            im = cv2.imread(file)
            im = im.astype(np.float32, copy=False)

            return self._prepare_im(im)

        def __len__(self):
            return len(self.files)

    import glob
    import os
    print("val data path", data_path)
    label_dirs = glob.glob(os.path.join(data_path, "*"))
    total_positive_cnt = 0
    label_files_list = []
    for label_dir in label_dirs:
        file_cnt = len(glob.glob(os.path.join(label_dir, "*")))
        total_positive_cnt += file_cnt * (file_cnt - 1) / 2
        if total_positive_cnt < 1:
            continue
        label_files_list.append(glob.glob(os.path.join(label_dir, "*")))
    print("total_positive_cnt", total_positive_cnt)
    if not max_positive_cnt or max_positive_cnt > total_positive_cnt:
        max_positive_cnt = total_positive_cnt

    if max_positive_cnt < 1:
        raise Exception("max_positive_cnt is 0")

    positive_files = []
    issame = []

    each_cnt = max_positive_cnt / len(label_dirs)
    for label_idx, label_files in enumerate(label_files_list):
        cur_cnt = 0
        try:
            for i in range(0, len(label_files) - 1):
                for j in range(i + 1, len(label_files)):
                    positive_files.append(label_files[i])
                    positive_files.append(label_files[j])
                    cur_cnt += 1
                    if cur_cnt >= each_cnt:
                        raise
        except:
            print("val positive label cnt", cur_cnt, os.path.basename(label_dirs[label_idx]))
            pass
    max_positive_cnt = len(positive_files) // 2
    issame += [True] * int(len(positive_files) / 2)
    print("val positive cnt", len(positive_files))
    negative_files = []
    total_negative_cnt = 0
    idx_map = {}
    while max_positive_cnt > total_negative_cnt:
        target_label_idx = random.randint(0, len(label_files_list) - 1)
        if len(label_files_list[target_label_idx]) < 1:
            continue
        target_item_idx = random.randint(0, len(label_files_list[target_label_idx]) - 1)
        neg_label_idx = random.randint(0, len(label_files_list) - 1)
        while target_label_idx == neg_label_idx:
            neg_label_idx = random.randint(0, len(label_files_list) - 1)

        if len(label_files_list[neg_label_idx]) < 1:
            continue
        neg_item_idx = random.randint(0, len(label_files_list[neg_label_idx]) - 1)

        if "{}_{}_{}_{}".format(target_label_idx, target_item_idx, neg_label_idx, neg_item_idx) in idx_map:
            continue
        idx_map["{}_{}_{}_{}".format(target_label_idx, target_item_idx, neg_label_idx, neg_item_idx)] = True
        negative_files.append(label_files_list[target_label_idx][target_item_idx])
        negative_files.append(label_files_list[neg_label_idx][neg_item_idx])
        total_negative_cnt += 1
    issame += [False] * int(len(negative_files) / 2)
    print("val negative cnt", len(negative_files))

    from torch.utils.data.distributed import DistributedSampler
    print("val len", len(positive_files + negative_files))
    sampler = DistributedSampler(ValDataset(positive_files + negative_files)) if cfg.NUM_GPUS > 1 else None
    # Create a loader
    loader = torch.utils.data.DataLoader(
        ValDataset(positive_files + negative_files),
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    return loader, issame


def test_model():
    """Evaluates a trained model."""
    # Setup training/testing environment
    setup_env()
    # Construct the model
    model = setup_model()
    # Load model weights
    checkpoint.load_checkpoint(cfg.TEST.WEIGHTS, model)
    logger.info("Loaded model weights from: {}".format(cfg.TEST.WEIGHTS))
    # Create data loaders and meters
    test_loader = loader.construct_test_loader()
    test_meter = meters.TestMeter(len(test_loader))
    # Evaluate the model
    test_epoch(test_loader, model, test_meter, 0)


def time_model():
    """Times model and data loader."""
    # Setup training/testing environment
    setup_env()
    # Construct the model and loss_fun
    model = setup_model()
    loss_fun = builders.build_loss_fun().cuda()
    # Create data loaders
    train_loader = loader.construct_train_loader()
    test_loader = loader.construct_test_loader()
    # Compute model and loader timings
    # benchmark.compute_time_full(model, loss_fun, train_loader, test_loader)
