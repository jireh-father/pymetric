import os
import sys
import cv2
import numpy as np

import torch

import metric.core.config as config
import metric.datasets.transforms as transforms
import metric.core.builders as builders
from metric.core.config import cfg
from linear_head import LinearHead
import glob
from metric.modeling.layers import GeneralizedMeanPoolingP
from sklearn.cluster import KMeans
import shutil
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

_MEAN = [0.406, 0.456, 0.485]
_SD = [0.225, 0.224, 0.229]
pool_layer = GeneralizedMeanPoolingP()
pool_layer.cuda()

class MetricModel(torch.nn.Module):
    def __init__(self):
        super(MetricModel, self).__init__()
        self.backbone = builders.build_model()
        self.head = LinearHead()

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


def preprocess(im):
    im = transforms.scale(cfg.TEST.IM_SIZE, im)
    im = transforms.center_crop(cfg.TRAIN.IM_SIZE, im)
    im = im.transpose([2, 0, 1])
    im = im / 255.0
    im = transforms.color_norm(im, _MEAN, _SD)
    return [im]


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def extract(imgpath, model):

    im = cv2.imread(imgpath)
    im = im.astype(np.float32, copy=False)
    im = preprocess(im)
    im_array = np.asarray(im, dtype=np.float32)
    input_data = torch.from_numpy(im_array)
    if torch.cuda.is_available():
        input_data = input_data.cuda()
    fea = model(input_data, targets=None)
    fea = pool_layer(fea)
    embedding = to_numpy(fea)
    # print("fea_shape: ", embedding.shape)
    return embedding


def main(model_path, output_dir, image_root):
    model = builders.MetricModel()
    print(model)
    load_checkpoint(model_path, model)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    class_dirs = glob.glob(os.path.join(image_root, "*"))

    for i, class_dir in enumerate(class_dirs):
        image_files = glob.glob(os.path.join(class_dir, '*'))
        embeddings = []
        for j, image_file in enumerate(image_files):
            print(i, len(class_dirs), j, len(image_files), class_dir, image_file)
            embedding = extract(image_file, model)
            embeddings.append(embedding)
        embeddings = np.array(embeddings)
        print(embeddings.shape)
        kmeans = KMeans(n_clusters=2, random_state=0).fit_predict(embeddings)

        for j, label in enumerate(kmeans.labels_):
            cur_output_dir = os.path.join(output_dir, os.path.basename(class_dir), "{}".format(label))
            os.makedirs(cur_output_dir, exist_ok=True)
            shutil.copy(image_files[j], cur_output_dir)

        pca = PCA(n_components=2)
        pca_embeddings = pca.fit_transform(embeddings)

        plt.scatter(pca_embeddings[:, 0], pca_embeddings[:, 1],
                    # c=digits.target,
                    edgecolor='none', alpha=0.5,
                    cmap=plt.cm.get_cmap('spectral', 10))
        plt.xlabel('component 1')
        plt.ylabel('component 2')
        plt.colorbar()

        output_file = os.path.join(output_dir, "{}.png".format(os.path.basename(class_dir)))
        plt.savefig(output_file)
    print("done")


def load_checkpoint(checkpoint_file, model, optimizer=None):
    """Loads the checkpoint from the given file."""
    err_str = "Checkpoint '{}' not found"
    assert os.path.exists(checkpoint_file), err_str.format(checkpoint_file)
    # Load the checkpoint on CPU to avoid GPU mem spike
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    try:
        state_dict = checkpoint["model_state"]
    except KeyError:
        state_dict = checkpoint
    # Account for the DDP wrapper in the multi-gpu setting
    ms = model
    model_dict = ms.state_dict()

    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    if len(pretrained_dict) == len(state_dict):
        print('All params loaded')
    else:
        print('construct model total {} keys and pretrin model total {} keys.'.format(len(model_dict), len(state_dict)))
        print('{} pretrain keys load successfully.'.format(len(pretrained_dict)))
        not_loaded_keys = [k for k in state_dict.keys() if k not in pretrained_dict.keys()]
        print(('%s, ' * (len(not_loaded_keys) - 1) + '%s') % tuple(not_loaded_keys))
    model_dict.update(pretrained_dict)
    ms.load_state_dict(model_dict)
    # ms.load_state_dict(checkpoint["model_state"])
    # Load the optimizer state (commonly not done when fine-tuning)
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    # return checkpoint["epoch"]
    return checkpoint


if __name__ == '__main__':
    print(sys.argv)
    args = config.load_cfg_and_args("Extract feature.")
    config.assert_and_infer_cfg()
    cfg.freeze()
    main(cfg.INFER.MODEL_WEIGHTS, cfg.INFER.OUTPUT_DIR, args.image_root)
