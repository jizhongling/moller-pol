from __future__ import print_function
import argparse
import sys
import os
import pandas as pd
import uproot as ur
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import HDBSCAN
import umap


class DataWF():
    def __init__(self, args, files):
        branch = ["event"]
        if args.type == 0:
            branch += [f"fwzm_ch{ich}_p{ip}" for ich in range(4) for ip in range(3)]
        branch += [f"area_ch{ich}_p{ip}" for ich in range(4) for ip in range(3)]
        branch += [f"diff_ch{ich}_p{ip}" for ich in range(6) for ip in range(3)]
        root_tree = ur.concatenate(files, branch, library='pd')
        
        if args.type == 0:
            self.input = root_tree.drop("event", axis=1)
            self.target = root_tree["event"]
        elif args.type == 1 and args.label is not None:
            print(f"\nReclustering: Using previous labels to filter data with label {args.label}\n")
            # Read labels file and filter for label arg.label
            labels_df = pd.read_csv("data/labels-type0-method2.txt", sep=' ', header=None, names=['event', 'label'])
            events_label = labels_df[labels_df['label'] == args.label]['event'].values

            # Filter root_tree to only include events with label args.label
            mask = root_tree['event'].isin(events_label)
            filtered_tree = root_tree[mask]

            self.input = filtered_tree.drop("event", axis=1)
            self.target = filtered_tree["event"]
        else:
            sys.exit("\nError: Wrong type number\n")

    def __len__(self):
        return len(self.target)


class Cluster():
    def __init__(self, args):
        if args.method == 0:
            print("\nUsing k-means clustering\n")
            nclus = 3 if args.type == 1 else 10
            self.model = KMeans(n_clusters=nclus, random_state=args.seed, n_init="auto")
        elif args.method == 1:
            print("\nUsing DBSCAN\n")
            self.model = DBSCAN(eps=args.eps, min_samples=args.dim+1, n_jobs=-1)
        elif args.method == 2:
            print("\nUsing HDBSCAN\n")
            self.model = HDBSCAN(min_cluster_size=2*args.dim, min_samples=5, cluster_selection_epsilon=args.eps, n_jobs=-1, copy=False)
        else:
            sys.exit("\nError: Wrong method number\n")


def train(args, model, X_train, y_train):
    model.fit(X_train)
    labels = model.labels_
    with open(f"data/labels-type{args.type}-method{args.method}.txt", "w") as f:
        for i in range(len(labels)):
            f.write(f"{y_train.iloc[i]} {labels[i]}\n")


def main():
    # training settings
    parser = argparse.ArgumentParser(description='Moller polarimeter waveform clustering')
    parser.add_argument('--type', type=int, default=0, metavar='N',
                        help='training type (waveform: 0, recluster: 1, default: 0)')
    parser.add_argument('--dir', type=str, default='data', metavar='DIR',
                        help='directory of data (default: data)')
    parser.add_argument('--nfiles', type=int, default=10000, metavar='N',
                        help='max number of files used for training (default: 10000)')
    parser.add_argument('--data-size', type=int, default=200, metavar='N',
                        help='number of files for each training (default: 200)')
    parser.add_argument('--method', type=int, default=2, metavar='N',
                        help='classifier (k-means: 0, DBSCAN: 1, HDBSCAN: 2, default: 2)')
    parser.add_argument('--label', type=int, default=None, metavar='L',
                        help='label for reclustering (default: None)')
    parser.add_argument('--seed', type=int, default=None, metavar='S',
                        help='random seed (default: None)')
    parser.add_argument('--dim', type=int, default=10, metavar='N',
                        help='number of dimentions (default: 10)')
    parser.add_argument('--eps', type=float, default=1.5, metavar='F',
                        help='selection epsilon (default: 1.5)')
    parser.add_argument('--norm', action='store_true', default=False,
                        help='normalize input data (default: False)')
    parser.add_argument('--pca', action='store_true', default=False,
                        help='use PCA for dimensionality reduction (default: False)')
    parser.add_argument('--pca-components', type=int, default=10, metavar='N',
                        help='number of PCA components (default: 10)')
    parser.add_argument('--umap', action='store_true', default=False,
                        help='use UMAP for dimensionality reduction (default: False)')
    parser.add_argument('--umap-components', type=int, default=10, metavar='N',
                        help='number of UMAP components (default: 10)')
    parser.add_argument('--umap-neighbors', type=int, default=15, metavar='N',
                        help='number of neighbors for UMAP (default: 15)')
    parser.add_argument('--umap-min-dist', type=float, default=0.1, metavar='F',
                        help='minimum distance for UMAP (default: 0.1)')
    args = parser.parse_args()

    prefix = "training"
    key = "T"
    files = []
    for file in os.scandir(args.dir):
        if (file.name.startswith(prefix) and
            file.name.endswith(".root") and
            file.is_file()):
            files.append(file.path + ":" + key)
    nfiles = min(args.nfiles, len(files))
    data_size = min(args.data_size, len(files))

    if args.pca:
        args.dim = args.pca_components
    if args.umap:
        args.dim = args.umap_components
    model = Cluster(args).model

    for iset in range(0, nfiles, data_size):
        ilast = min(iset + data_size, nfiles)
        print(f"\nDataset: {iset + 1} to {ilast}\n")
        dataset = DataWF(args, files[iset:ilast])
        X_train, y_train = dataset.input, dataset.target
        if args.norm:
            print("\nNormalizing input data\n")
            scaler = StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
        if args.pca:
            print(f"\nApplying PCA: {X_train.shape[1]} -> {args.pca_components} dimensions\n")
            pca = PCA(n_components=args.pca_components, random_state=args.seed)
            X_train = pca.fit_transform(X_train)
        if args.umap:
            print(f"\nApplying UMAP: {X_train.shape[1]} -> {args.umap_components} dimensions\n")
            reducer = umap.UMAP(
                n_components=args.umap_components,
                n_neighbors=args.umap_neighbors,
                min_dist=args.umap_min_dist,
                random_state=args.seed
            )
            X_train = reducer.fit_transform(X_train)
        train(args, model, X_train, y_train)


if __name__ == '__main__':
    main()