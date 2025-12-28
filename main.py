from __future__ import print_function
import argparse
import sys
import os
import pandas as pd
import uproot as ur
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import HDBSCAN


class DataWF():
    def __init__(self, args, files):
        branch = ["event"]
        #branch += [f"time_ch{ich}_p{ip}" for ich in range(4) for ip in range(3)]
        #branch += [f"peak_ch{ich}_p{ip}" for ich in range(4) for ip in range(3)]
        if args.type == 0:
            branch += [f"fwzm_ch{ich}_p{ip}" for ich in range(4) for ip in range(3)]
        branch += [f"area_ch{ich}_p{ip}" for ich in range(4) for ip in range(3)]
        branch += [f"diff_ch{ich}_p{ip}" for ich in range(6) for ip in range(3)]
        root_tree = ur.concatenate(files, branch, library='pd')
        
        if args.type == 0:
            self.input = root_tree.drop("event", axis=1)
            self.target = root_tree["event"]
        elif args.type == 1:
            print("\nReclustering: Using previous labels to filter data\n")
            # Read labels file and filter for label 13
            labels_df = pd.read_csv("data/labels-type0-method2.txt", sep=' ', header=None, names=['event', 'label'])
            events_label_13 = labels_df[labels_df['label'] == 13]['event'].values

            # Filter root_tree to only include events with label 13
            mask = root_tree['event'].isin(events_label_13)
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
            self.model = DBSCAN(eps=1.2, min_samples=15, n_jobs=-1)
        elif args.method == 2:
            print("\nUsing HDBSCAN\n")
            self.model = HDBSCAN(min_cluster_size=28, min_samples=5, cluster_selection_epsilon=1.5, n_jobs=-1, copy=False)
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
    parser.add_argument('--seed', type=int, default=None, metavar='S',
                        help='random seed (default: None)')
    parser.add_argument('--norm', action='store_true', default=False,
                        help='normalize input data (default: False)')
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

    model = Cluster(args).model

    for iset in range(0, nfiles, data_size):
        ilast = min(iset + data_size, nfiles)
        print(f"\nDataset: {iset + 1} to {ilast}\n")
        dataset = DataWF(args, files[iset:ilast])
        X_train, y_train = dataset.input, dataset.target
        if args.norm:
            scaler = StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
        train(args, model, X_train, y_train)


if __name__ == '__main__':
    main()