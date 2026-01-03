from __future__ import print_function
import argparse
import sys
import os
import numpy as np
import pandas as pd
import uproot as ur
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import HDBSCAN
import umap
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


class DataWF():
    def __init__(self, args, files):
        # Determine which branches to load based on whether using autoencoder/VAE
        use_waveforms = args.autoencoder or args.vae or args.load_autoencoder or args.load_vae
        
        if use_waveforms:
            # Load waveform sample data for autoencoder/VAE
            print("\nLoading waveform sample data for autoencoder/VAE\n")
            sample_branches = [f"sample_ch{ich}" for ich in range(4)]
            waveform_data = self._load_waveforms(files, sample_branches)
            
            # Load event IDs
            event_branch = ur.concatenate(files, ["event"], library='pd')
            
            if args.type == 0:
                # Flatten waveforms for input: (num_events, num_channels * num_samples)
                self.input = pd.DataFrame(waveform_data.numpy().reshape(waveform_data.shape[0], -1))
                self.target = event_branch["event"]
            elif args.type == 1 and args.label is not None:
                print(f"\nReclustering: Using previous labels to filter data with label {args.label}\n")
                # Read labels file and filter for label arg.label
                labels_df = pd.read_csv("data/labels-type0-method2.txt", sep=' ', header=None, names=['event', 'label'])
                events_label = labels_df[labels_df['label'] == args.label]['event'].values

                # Filter by label
                mask = event_branch['event'].isin(events_label)
                indices = np.where(mask.values)[0]
                
                # Filter waveform data
                filtered_waveforms = waveform_data[indices]
                self.input = pd.DataFrame(filtered_waveforms.numpy().reshape(filtered_waveforms.shape[0], -1))
                self.target = event_branch["event"][mask].reset_index(drop=True)
            else:
                sys.exit("\nError: Wrong type number\n")
        else:
            # Load feature data (area, diff) for clustering
            print("\nLoading feature data for clustering\n")
            branch = ["event"]
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

    def _load_waveforms(self, files, sample_branches):
        """Load waveform samples from ROOT files and return as torch tensor.
        
        Args:
            files: List of ROOT file paths with format 'path/file.root:TreeName'
            sample_branches: List of branch names to load (e.g., ['sample_ch0', 'sample_ch1', ...])
        
        Returns:
            torch.Tensor of shape (num_events, num_channels, num_samples) or None if branches not found
        """
        waveforms_list = []
        
        for file_path in files:
            # Split file path and tree name
            path, tree_name = file_path.split(':')
            try:
                file = ur.open(path)
                tree = file[tree_name]
                
                # Check if sample branches exist
                if not all(branch in tree.keys() for branch in sample_branches):
                    print(f"Warning: Not all sample branches found in {path}")
                    return None
                
                # Load waveform data for all channels
                channels_data = []
                for branch_name in sample_branches:
                    branch_data = tree[branch_name].array(library='np')  # Shape: (num_events, num_samples)
                    channels_data.append(branch_data)
                
                # Stack channels: (num_events, num_channels, num_samples)
                waveforms = np.stack(channels_data, axis=1)
                waveforms_list.append(waveforms)
            except Exception as e:
                print(f"Error loading waveforms from {path}: {e}")
                return None
        
        if waveforms_list:
            # Concatenate all files
            all_waveforms = np.concatenate(waveforms_list, axis=0)
            # Convert to torch tensor
            waveform_tensor = torch.FloatTensor(all_waveforms)
            print(f"\nLoaded waveforms: shape {waveform_tensor.shape} (events, channels, samples)")
            return waveform_tensor
        else:
            return None

    def __len__(self):
        return len(self.target)
    

class Autoencoder(nn.Module):
    """Standard Autoencoder for dimensionality reduction"""
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed
    
    def encode(self, x):
        return self.encoder(x)


class VAE(nn.Module):
    """Variational Autoencoder for dimensionality reduction"""
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        # Encoder
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim, 64)
        self.fc4 = nn.Linear(64, 128)
        self.fc5 = nn.Linear(128, input_dim)
    
    def encode(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = torch.relu(self.fc3(z))
        h = torch.relu(self.fc4(h))
        return self.fc5(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar
    
    def get_latent(self, x):
        """Get latent representation (using mean)"""
        mu, _ = self.encode(x)
        return mu


class Cluster():
    def __init__(self, args):
        if args.method == 0:
            print("\nUsing k-means clustering\n")
            self.model = KMeans(n_clusters=args.kclus, random_state=args.seed, n_init="auto")
        elif args.method == 1:
            print("\nUsing DBSCAN\n")
            self.model = DBSCAN(eps=args.eps, min_samples=args.dim+1, n_jobs=-1)
        elif args.method == 2:
            print("\nUsing HDBSCAN\n")
            self.model = HDBSCAN(min_cluster_size=2*args.dim, min_samples=5, cluster_selection_epsilon=args.eps, n_jobs=-1, copy=False)
        else:
            sys.exit("\nError: Wrong method number\n")


def save_autoencoder(autoencoder, path, input_dim, latent_dim):
    """Save autoencoder state dict along with metadata."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'state_dict': autoencoder.state_dict(),
        'input_dim': input_dim,
        'latent_dim': latent_dim,
        'model': 'autoencoder'
    }, path)


def load_autoencoder(path, device=None):
    """Load autoencoder from checkpoint with metadata."""
    ckpt = torch.load(path, map_location=device or 'cpu')
    model = Autoencoder(ckpt['input_dim'], ckpt['latent_dim'])
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model, ckpt['input_dim'], ckpt['latent_dim']


def save_vae(vae, path, input_dim, latent_dim):
    """Save VAE state dict along with metadata."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'state_dict': vae.state_dict(),
        'input_dim': input_dim,
        'latent_dim': latent_dim,
        'model': 'vae'
    }, path)


def load_vae(path, device=None):
    """Load VAE from checkpoint with metadata."""
    ckpt = torch.load(path, map_location=device or 'cpu')
    model = VAE(ckpt['input_dim'], ckpt['latent_dim'])
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model, ckpt['input_dim'], ckpt['latent_dim']


def latent_from_autoencoder(autoencoder, X):
    """Compute latent vectors from an autoencoder."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    autoencoder = autoencoder.to(device)
    X_tensor = torch.FloatTensor(X if isinstance(X, np.ndarray) else X.values)
    with torch.no_grad():
        return autoencoder.encode(X_tensor.to(device)).cpu().numpy()


def latent_from_vae(vae, X):
    """Compute latent vectors (mu) from a VAE."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae = vae.to(device)
    X_tensor = torch.FloatTensor(X if isinstance(X, np.ndarray) else X.values)
    with torch.no_grad():
        return vae.get_latent(X_tensor.to(device)).cpu().numpy()


def train_autoencoder(autoencoder, X_train, epochs=50, batch_size=64, learning_rate=1e-3):
    """Train autoencoder and return latent representations"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining Autoencoder on {device}\n")
    
    autoencoder = autoencoder.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)
    
    # Prepare data
    X_tensor = torch.FloatTensor(X_train if isinstance(X_train, np.ndarray) else X_train.values)
    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    autoencoder.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            data = batch[0].to(device)
            
            optimizer.zero_grad()
            reconstructed = autoencoder(data)
            loss = criterion(reconstructed, data)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")
    
    # Get latent representations
    autoencoder.eval()
    with torch.no_grad():
        X_latent = autoencoder.encode(X_tensor.to(device)).cpu().numpy()
    
    return X_latent


def train_vae(vae, X_train, epochs=50, batch_size=64, learning_rate=1e-3):
    """Train VAE and return latent representations"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining VAE on {device}\n")
    
    vae = vae.to(device)
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
    
    # Prepare data
    X_tensor = torch.FloatTensor(X_train if isinstance(X_train, np.ndarray) else X_train.values)
    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    vae.train()
    for epoch in range(epochs):
        total_loss = 0
        total_recon_loss = 0
        total_kld_loss = 0
        
        for batch in dataloader:
            data = batch[0].to(device)
            
            optimizer.zero_grad()
            reconstructed, mu, logvar = vae(data)
            
            # Reconstruction loss
            recon_loss = nn.functional.mse_loss(reconstructed, data, reduction='sum')
            # KL divergence loss
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            
            loss = recon_loss + kld_loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kld_loss += kld_loss.item()
        
        avg_loss = total_loss / len(X_tensor)
        avg_recon = total_recon_loss / len(X_tensor)
        avg_kld = total_kld_loss / len(X_tensor)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Recon: {avg_recon:.4f}, KLD: {avg_kld:.4f}")
    
    # Get latent representations
    vae.eval()
    with torch.no_grad():
        X_latent = vae.get_latent(X_tensor.to(device)).cpu().numpy()
    
    return X_latent


def train(args, model, X_train, y_train):
    model.fit(X_train)
    labels = model.labels_
    os.makedirs("data", exist_ok=True)
    labels_path = f"data/labels-type{args.type}-method{args.method}.txt"
    with open(labels_path, "w") as f:
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
    parser.add_argument('--kclus', type=int, default=10, metavar='N',
                        help='number of clusters for k-means (default: 10)')
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
    parser.add_argument('--autoencoder', action='store_true', default=False,
                        help='use Autoencoder for dimensionality reduction (default: False)')
    parser.add_argument('--vae', action='store_true', default=False,
                        help='use VAE for dimensionality reduction (default: False)')
    parser.add_argument('--latent-dim', type=int, default=10, metavar='N',
                        help='latent dimension for autoencoder/VAE (default: 10)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of training epochs for autoencoder/VAE (default: 50)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='batch size for autoencoder/VAE training (default: 64)')
    parser.add_argument('--learning-rate', type=float, default=1e-3, metavar='F',
                        help='learning rate for autoencoder/VAE (default: 0.001)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='save trained AE/VAE model to disk (default: False)')
    parser.add_argument('--model-dir', type=str, default='models', metavar='DIR',
                        help='directory to save/load models (default: models)')
    parser.add_argument('--load-autoencoder', type=str, default=None, metavar='PATH',
                        help='path to a saved autoencoder checkpoint to use')
    parser.add_argument('--load-vae', type=str, default=None, metavar='PATH',
                        help='path to a saved VAE checkpoint to use')
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
    if args.autoencoder:
        args.dim = args.latent_dim
    if args.vae:
        args.dim = args.latent_dim
    model = Cluster(args).model

    # Initialize autoencoder/VAE before the loop to train across all datasets
    autoencoder = None
    vae = None
    ae_input_dim = None
    vae_input_dim = None

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
        # Optionally load pre-trained models to get latent space
        if args.load_autoencoder:
            print(f"\nLoading Autoencoder from {args.load_autoencoder}\n")
            ae, in_dim, lat_dim = load_autoencoder(args.load_autoencoder)
            if in_dim != (X_train.shape[1] if isinstance(X_train, np.ndarray) else X_train.shape[1]):
                print(f"Warning: checkpoint input_dim {in_dim} != current {X_train.shape[1]}")
            X_train = latent_from_autoencoder(ae, X_train)
        if args.load_vae:
            print(f"\nLoading VAE from {args.load_vae}\n")
            vae_model, in_dim, lat_dim = load_vae(args.load_vae)
            if in_dim != (X_train.shape[1] if isinstance(X_train, np.ndarray) else X_train.shape[1]):
                print(f"Warning: checkpoint input_dim {in_dim} != current {X_train.shape[1]}")
            X_train = latent_from_vae(vae_model, X_train)
        if args.autoencoder:
            input_dim = X_train.shape[1]
            # Create model on first iteration
            if autoencoder is None:
                print(f"\nInitializing Autoencoder: {input_dim} -> {args.latent_dim} dimensions\n")
                autoencoder = Autoencoder(input_dim, args.latent_dim)
                ae_input_dim = input_dim
            print(f"\nTraining Autoencoder on dataset {iset+1}-{ilast}\n")
            X_train = train_autoencoder(
                autoencoder, X_train,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate
            )
        if args.vae:
            input_dim = X_train.shape[1]
            # Create model on first iteration
            if vae is None:
                print(f"\nInitializing VAE: {input_dim} -> {args.latent_dim} dimensions\n")
                vae = VAE(input_dim, args.latent_dim)
                vae_input_dim = input_dim
            print(f"\nTraining VAE on dataset {iset+1}-{ilast}\n")
            X_train = train_vae(
                vae, X_train,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate
            )
        train(args, model, X_train, y_train)

    # Save final trained models after all datasets
    if args.save_model:
        os.makedirs(args.model_dir, exist_ok=True)
        if args.autoencoder and autoencoder is not None:
            model_path = os.path.join(args.model_dir, f"autoencoder-type{args.type}-dim{args.latent_dim}.pt")
            print(f"\nSaving final Autoencoder to {model_path}\n")
            save_autoencoder(autoencoder, model_path, ae_input_dim, args.latent_dim)
        if args.vae and vae is not None:
            model_path = os.path.join(args.model_dir, f"vae-type{args.type}-dim{args.latent_dim}.pt")
            print(f"\nSaving final VAE to {model_path}\n")
            save_vae(vae, model_path, vae_input_dim, args.latent_dim)


if __name__ == '__main__':
    main()