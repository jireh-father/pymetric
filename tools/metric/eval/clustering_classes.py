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
from sklearn import cluster
from sklearn import mixture
# .cluster import KMeans, Gaussian mixtures
import shutil
from sklearn.decomposition import PCA
from torch import nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd
from scipy.spatial import distance

_MEAN = [0.406, 0.456, 0.485]
_SD = [0.225, 0.224, 0.229]


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


def extract(imgpaths, model, pool_layer, use_norm):
    imgs = []
    for imgpath in imgpaths:
        im = cv2.imread(imgpath)
        im = im.astype(np.float32, copy=False)
        im = preprocess(im)
        imgs += im
    im_array = np.asarray(imgs, dtype=np.float32)
    input_data = torch.from_numpy(im_array)
    if torch.cuda.is_available():
        input_data = input_data.cuda()
    fea = model(input_data, targets=None)
    # fea = pool_layer(fea)
    fea = fea.squeeze()
    if use_norm:
        fea = F.normalize(fea, p=2, dim=1)
    embeddings = to_numpy(fea.squeeze())
    # print("fea_shape: ", embedding.shape)

    return embeddings


#     im = cv2.imread(imgpath)
#     im = im.astype(np.float32, copy=False)
#     im = preprocess(im)
#     im_array = np.asarray(im, dtype=np.float32)
#     input_data = torch.from_numpy(im_array)
#     if torch.cuda.is_available():
#         input_data = input_data.cuda()
#     fea = model(input_data, targets=None)
#     fea = pool_layer(fea)
#     embedding = to_numpy(fea.squeeze())
#     # print("fea_shape: ", embedding.shape)
#     return embedding
cluster_algos = {
    "KMeans": cluster.KMeans,
    "SpectralClustering": cluster.SpectralClustering,
    "MeanShift": cluster.MeanShift,
    "AffinityPropagation": cluster.AffinityPropagation,
    "AgglomerativeClustering": cluster.AgglomerativeClustering,
    "FeatureAgglomeration": cluster.FeatureAgglomeration,
    "MiniBatchKMeans": cluster.MiniBatchKMeans,
    "DBSCAN": cluster.DBSCAN,
    "OPTICS": cluster.OPTICS,
    "SpectralBiclustering": cluster.SpectralBiclustering,
    "SpectralCoclustering": cluster.SpectralCoclustering,
    "Birch": cluster.Birch,
    "GaussianMixture": mixture.GaussianMixture,
    "BayesianGaussianMixture": mixture.BayesianGaussianMixture
}


def main(model_path, output_dir, image_root, use_pca, pool_layer, use_norm, num_clusters, num_pca_comps, random_state):
    model = builders.MetricModel()
    print(model)
    load_checkpoint(model_path, model)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    class_dirs = glob.glob(os.path.join(image_root, "*"))
    os.makedirs(output_dir, exist_ok=True)
    for i, class_dir in enumerate(class_dirs):
        print(i, class_dir)
        image_files = glob.glob(os.path.join(class_dir, '*'))
        image_files.sort()
        # embeddings = []
        # for j, image_file in enumerate(image_files):
        #     print(i, len(class_dirs), j, len(image_files), class_dir, image_file)
        #     embedding = extract(image_file, model, pool_layer)
        #     embeddings.append(embedding)
        # embeddings = np.array(embeddings)
        embeddings = extract(image_files, model, pool_layer, use_norm)

        # print(embeddings.shape)
        # if use_norm:
        #     embeddings = np.linalg.norm(embeddings, ord=2)
        cdist = distance.cdist(embeddings, embeddings, 'euclidean')
        cdist_exlfile = os.path.join(output_dir, "{}.xlsx".format(os.path.basename(class_dir)))
        df = pd.DataFrame(cdist, columns=[os.path.basename(f).split("_")[-1].split(".")[0] for f in image_files])
        df = df.set_index(df.columns)
        df.to_excel(cdist_exlfile)

        print(embeddings.shape)

        for j, key in enumerate(cluster_algos):

            print(j, len(cluster_algos), key)
            try:
                if key == "AffinityPropagation":
                    clustered = cluster_algos[key](random_state=random_state)
                elif key in ["MeanShift", "DBSCAN", "OPTICS"]:
                    clustered = cluster_algos[key]()
                elif key in ["AgglomerativeClustering", "FeatureAgglomeration", "Birch"]:
                    clustered = cluster_algos[key](n_clusters=num_clusters)
                else:
                    clustered = cluster_algos[key](n_clusters=num_clusters, random_state=random_state)

                if use_pca:
                    pca = PCA(n_components=num_pca_comps, random_state=random_state)
                    embeddings = pca.fit_transform(embeddings)

                if key in ["GaussianMixture", "BayesianGaussianMixture"]:
                    labels = clustered.fit_predict()
                else:
                    clustered.fit(embeddings)
                    labels = clustered.labels_

                # kmeans.fit(embeddings)
                for j, label in enumerate(labels):
                    cur_output_dir = os.path.join(output_dir,
                                                  "{}_{}".format(os.path.basename(class_dir), key),
                                                  "{}".format(label))
                    os.makedirs(cur_output_dir, exist_ok=True)
                    shutil.copy(image_files[j], cur_output_dir)

                if not use_pca:
                    pca = PCA(n_components=2, random_state=random_state)
                    embeddings = pca.fit_transform(embeddings)

                plt.figure()
                plt.scatter(embeddings[:, 0], embeddings[:, 1],
                            edgecolor='none', alpha=0.5)
                plt.xlabel('component 1')
                plt.ylabel('component 2')
                plt.colorbar()

                output_file = os.path.join(output_dir, "{}_{}.png".format(os.path.basename(class_dir), key))
                plt.savefig(output_file)
            except:
                import traceback
                traceback.print_exc()

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

    if args.pool == "maxpool":
        pool_layer = nn.AdaptiveMaxPool2d(1)
    else:
        pool_layer = GeneralizedMeanPoolingP()
    pool_layer.cuda()
    main(cfg.INFER.MODEL_WEIGHTS, cfg.INFER.OUTPUT_DIR, args.image_root, args.use_pca, pool_layer, args.use_norm,
         args.num_clusters,
         args.num_pca_comps,
         args.random_state)
