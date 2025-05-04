import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torchvision.models as models
from tqdm import tqdm
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from scipy.ndimage import gaussian_filter
import random
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex, MulticlassPrecision, MulticlassRecall, MulticlassF1Score
from collections import Counter
import os
import rasterio
import cv2
from sklearn.metrics import confusion_matrix


class LandUseDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that loads, normalizes, and remaps the labels of a 
    land use semantic segmentation task using multiband satellite data.

    Args:
        path (str): Path to a `.npz` file containing the images and their respective labels.
            transform (callable, optional): Optional transformation to apply to the input images.

    Returns:
        tuple: (image, label) pair, where image is a normalized tensor of shape (C, H, W),
            and label is a tensor of shape (H, W) with remapped class indices.
    """

    def __init__(self, path, transform=None):
        data = np.load(path)
        self.X = np.clip(data['X'] / 10000.0, 0.0, 1.0)
        self.y = self._remap_labels(data['y'])
        self.transform = transform

    def _remap_labels(self, y):
        id2idx = {10: 0, 20: 1, 30: 2, 40: 3, 50: 4, 60: 5, 80: 6, 90: 7}
        y_remapped = np.copy(y)
        for old, new in id2idx.items():
            y_remapped[y == old] = new
        return y_remapped

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.long)

        if self.transform:
            X = self.transform(X)

        return X, y
    
class RandomApplyTransform:
    """
    Applies a given transform randomly with a probability `p`.

    Args:
        transform (callable): The transformation function to apply.
        p (float): Probability of applying the transform (default: 0.5).

    Returns:
        Tensor: Either the transformed or the original tensor.
    """

    def __init__(self, transform, p=0.5):
        self.transform = transform
        self.p = p

    def __call__(self, x):
        if random.random() < self.p:
            return self.transform(x)
        return x

class RandomRadiometricShift:
    """
    Applies a random radiometric shift to each pixel value in the input tensor.

    Args:
        scale (float): Maximum magnitude of uniform shift applied to each channel (default: 0.05).

    Returns:
        Tensor: The radiometrically shifted tensor.
    """

    def __init__(self, scale=0.05):
        self.scale = scale

    def __call__(self, x):
        shift = torch.empty_like(x).uniform_(-self.scale, self.scale)
        return x + shift
    
class TransformSubset(torch.utils.data.Dataset):
        """
        A Dataset wrapper for applying transformations to a subset of a dataset given the indices.

        Args:
            dataset (Dataset): Dataset to sample from.
            indices (list of int): Indices of samples to include.
            transform (callable, optional): Transformation function to apply to the samplesges.

        Returns:
            tuple: (transformed image, label) pair from the specified subset.
        """

        def __init__(self, dataset, indices, transform=None):
            self.dataset = dataset
            self.indices = indices
            self.transform = transform

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            X, y = self.dataset[self.indices[idx]]
            if self.transform:
                X = self.transform(X)
            return X, y
    
class UNetResNet50(nn.Module):
    """
    U-Net-like model with a ResNet-50 encoder and a transposed convolutional decoder.
    Utilizes skip connections for the decoder part.

    Args:
        num_classes (int): Number of output classes.
        input_channels (int): Number of channels in the input data.

    Returns:
        Tensor: Predicted segmentation map of shape (B, num_classes, H, W).
    """

    def __init__(self, num_classes, input_channels=13):
        super(UNetResNet50, self).__init__()
        resnet = models.resnet50(pretrained=True)

        self.encoder_conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder_bn1 = resnet.bn1
        self.encoder_relu = resnet.relu
        self.encoder_maxpool = resnet.maxpool

        self.encoder_layer1 = resnet.layer1
        self.encoder_layer2 = resnet.layer2
        self.encoder_layer3 = resnet.layer3
        self.encoder_layer4 = resnet.layer4

        self.upconv4 = self._upsample(2048, 1024)
        self.upconv3 = self._upsample(1024 + 1024, 512)
        self.upconv2 = self._upsample(512 + 512, 256)
        self.upconv1 = self._upsample(256 + 256, 64)

        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

    def _upsample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x1 = self.encoder_relu(self.encoder_bn1(self.encoder_conv1(x)))
        x2 = self.encoder_layer1(self.encoder_maxpool(x1))
        x3 = self.encoder_layer2(x2)
        x4 = self.encoder_layer3(x3)
        x5 = self.encoder_layer4(x4)

        d4 = self.upconv4(x5)
        d4 = torch.cat([d4, x4], dim=1)

        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, x3], dim=1)

        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, x2], dim=1)

        d1 = self.upconv1(d2)

        out = F.interpolate(d1, scale_factor=2, mode='bilinear', align_corners=False)
        out = self.classifier(out)

        return out
    
def read_pansharpened_tiff(tiff_path):
    """
    Reads a pansharpened TIFF image file into a NumPy array and prints its shape.

    Parameters:
        tiff_path (str): Path to the pansharpened .tiff file.

    Returns:
        numpy.ndarray: 3D array representing the image stack (bands, rows, columns).
    """
    with rasterio.open(tiff_path) as src:
        stack = src.read()
        print(f"Stack shape: {stack.shape}")
    return stack

def check_band_coverage(directory):
    """
    Prints a summary of Sentinel-2 band coverage for all .jp2 files in a given directory.

    Parameters:
        directory (str): Path to the directory containing Sentinel-2 .jp2 files.

    Outputs:
        For each valid band image file (excluding '_TCI' images), prints:
            - Band code
            - Spatial resolution (in meters)
            - Image shape (rows, columns)
            - Percentage of valid (non-zero) pixels
    """
    print(f"\n {directory.split('/')[-2]}")
    print(f"\n{'Band':<6} {'Resolution':<10} {'Shape':<15} {'Valid Pixels (%)':<18}")
    print("-" * 60)

    for fname in sorted(os.listdir(directory)):
        if not fname.endswith('.jp2') or '_TCI' in fname:
            continue

        band_code = fname.split('_')[-1].replace('.jp2', '')
        path = os.path.join(directory, fname)

        with rasterio.open(path) as src:
            img = src.read(1)
            total_pixels = img.size
            valid_pixels = np.count_nonzero(img)
            valid_pct = 100 * valid_pixels / total_pixels

            print(f"{band_code:<6} {src.res[0]:<10.1f} {img.shape!s:<15} {valid_pct:<18.2f}")
    
def pansharpen_to_10m_and_save(directory, output_tiff="pansharpened.tif"):
    """
    Reads all Sentinel-2 bands from a directory, resamples bands to 10m resolution, 
    and saves a single pansharpened multi-band GeoTIFF.

    Parameters:
        directory (str): Path to the folder containing Sentinel-2 .jp2 band files.
        output_tiff (str): Filename for the output pansharpened GeoTIFF. Defaults to "pansharpened.tif".

    Raises:
        ValueError: If any expected Sentinel-2 bands (B01 to B12, including B8A) are missing.

    Process:
        - Identifies 13 Sentinel-2 bands (10m, 20m, and 60m resolutions).
        - Uses a 10m band as reference for target spatial shape and metadata.
        - Resamples 20m and 60m bands to 10m resolution using bicubic interpolation.
        - Writes the ordered bands (B01–B12 including B8A) into a single stacked GeoTIFF file.
    
    Output:
        A pansharpened GeoTIFF image with all 13 bands resampled to 10m resolution.
    """
    BANDS_10M = ['B02', 'B03', 'B04', 'B08']
    BANDS_20M = ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12']
    BANDS_60M = ['B01', 'B09', 'B10']
    ALL_BANDS = BANDS_10M + BANDS_20M + BANDS_60M
    TARGET_SHAPE = (10980, 10980)
    band_paths = {}
    for fname in os.listdir(directory):
        if fname.endswith('.jp2') and '_TCI' not in fname:
            band_code = fname.split('_')[-1].replace('.jp2', '')
            if band_code in ALL_BANDS:
                band_paths[band_code] = os.path.join(directory, fname)

    missing = [b for b in ALL_BANDS if b not in band_paths]
    if missing:
        raise ValueError(f"Missing bands: {missing}")

    with rasterio.open(band_paths[BANDS_10M[0]]) as ref_src:
        reference_meta = ref_src.meta.copy()
        reference_meta.update({
            "count": len(ALL_BANDS),
            "height": TARGET_SHAPE[0],
            "width": TARGET_SHAPE[1],
            "driver": "GTiff"
        })

    ordered_bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']
    print(ordered_bands)

    with rasterio.open(output_tiff, "w", **reference_meta) as dst:
        for idx, band in enumerate(ordered_bands, start=1):
            print(idx, band)
            with rasterio.open(band_paths[band]) as src:
                img = src.read(1)

                if band in BANDS_10M:
                    sharpened = img  
                else:
                    sharpened = cv2.resize(
                        img,
                        (TARGET_SHAPE[1], TARGET_SHAPE[0]),
                        interpolation=cv2.INTER_CUBIC
                    )

                dst.write(sharpened, idx)

    print(f"\nSaved pansharpened image to: {output_tiff}")


def plot_per_class_ious(best_model_val_metrics, learning_rates, num_classes):
    """
    Plots per-class Intersection over Union (IoU) scores for the best model
    corresponding to each learning rate.

    Args:
        best_model_val_metrics (dict): A dictionary mapping each learning rate to its
            validation metrics, including per-class IoU scores.
        learning_rates (list): A list of learning rates to compare.
        num_classes (int): The number of classes in the dataset.
    """
    per_class_ious = {lr: best_model_val_metrics[lr]['per_class_iou'] for lr in learning_rates}
    
    bar_width = 0.2
    index = np.arange(num_classes)
    fig, ax = plt.subplots(figsize=(15, 5))

    for i, lr in enumerate(learning_rates):
        iou_values = per_class_ious[lr]
        bars = ax.bar(index + i * bar_width, iou_values, bar_width, label=f"lr = {lr}")
        
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}',
                ha='center', va='bottom', fontsize=10
            )

    ax.set_xlabel('Class')
    ax.set_ylabel('IoU')
    ax.set_title('Per-Class IoUs of Each Best Model')
    ax.set_xticks(index + bar_width * (len(learning_rates) - 1) / 2)
    ax.set_xticklabels([f'Class {i}' for i in range(num_classes)])
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("best_models_per_class_iou.png", dpi=400)
    plt.savefig("best_models_per_class_iou.svg", dpi=400)
    plt.show()

def plot_metrics(all_metrics, num_epochs, learning_rates):
    """
    Plots training and validation metrics over epochs for different learning rates.

    Args:
        all_metrics (dict): A dictionary mapping each learning rate to a sub-dictionary
            containing lists of metric values (train/val losses, pixel accuracy, IoU).
        num_epochs (int): Total number of training epochs.
        learning_rates (list): A list of learning rates used in training.
    """
    metrics_to_plot = ["train_losses", "val_losses", "pixel_accs", "mean_ious"]
    titles = ["Average Training Loss vs Epochs", "Average Validation Loss vs Epochs", "Pixel Accuracy vs Epochs", "Macro-Averaged IoU vs Epochs"]
    y_labels = ["Average Training Loss", "Average Validation Loss", "Pixel Accuracy", "Macro-Averaged IoU"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    for i, metric in enumerate(metrics_to_plot):
        row, col = divmod(i, 2) 
        ax = axes[row, col]
        ax.set_title(titles[i])
        ax.set_xlabel("Epochs")
        ax.set_ylabel(y_labels[i])

        for lr in learning_rates:
            metric_values = all_metrics[lr][metric]
            ax.plot(range(1, num_epochs + 1), metric_values, label=f"lr = {lr}", marker='o')

        ax.legend()

    plt.tight_layout()
    plt.savefig("training_validation_metrics.png", dpi=400)
    plt.savefig("training_validation_metrics.svg", dpi=400)
    plt.show()

def plot_class_distribution(train_loader, val_loader, num_classes):
    """
    Plots the class distribution of the training and validation datasets.

    Args:
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        num_classes (int): Number of classes in the dataset.
    """

    train_class_counter = Counter()
    val_class_counter = Counter()

    for X, y in train_loader:
        y_flat = y.flatten().cpu().numpy()
        train_class_counter.update(y_flat)

    for X, y in val_loader:
        y_flat = y.flatten().cpu().numpy()
        val_class_counter.update(y_flat)

    class_labels = list(range(num_classes))
    train_class_counts = [train_class_counter.get(label, 0) for label in class_labels]
    val_class_counts = [val_class_counter.get(label, 0) for label in class_labels]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    train_bars = axes[0].bar(class_labels, train_class_counts, color='lightblue')
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Class Distribution in the Training Set')
    axes[0].set_xticks(class_labels)
    axes[0].set_xticklabels([f'Class {i}' for i in class_labels])

    for bar in train_bars:
        height = bar.get_height()
        axes[0].text(
            bar.get_x() + bar.get_width() / 2, height, f'{height}', 
            ha='center', va='bottom', fontsize=10
        )

    val_bars = axes[1].bar(class_labels, val_class_counts, color='lightgreen')
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Class Distribution in the Validation Set')
    axes[1].set_xticks(class_labels)
    axes[1].set_xticklabels([f'Class {i}' for i in class_labels])

    for bar in val_bars:
        height = bar.get_height()
        axes[1].text(
            bar.get_x() + bar.get_width() / 2, height, f'{height}', 
            ha='center', va='bottom', fontsize=10
        )

    plt.tight_layout()
    plt.savefig("dataset_distribution.png", dpi=300)
    plt.savefig("dataset_distribution.svg", dpi=300)
    plt.show()

def get_dataloaders(path, batch_size, train_split, seed):
    """
    Creates and returns Dataset and DataLoaders objects for training and validation.

    Args:
        path (str): Path to the dataset.
        batch_size (int, optional): Batch size for the DataLoaders.
        train_split (float, optional): Proportion of the dataset to use for training.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        tuple: (dataset, train_loader, val_loader)
    """

    train_transform = T.Compose([
        T.RandomHorizontalFlip(p=0.4),
        RandomApplyTransform(RandomRadiometricShift(scale=0.05), p=0.4),
    ])

    dataset = LandUseDataset(path)

    total_size = len(dataset)
    train_size = int(train_split * total_size)

    indices = list(range(total_size))
    random.seed(seed)
    random.shuffle(indices)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_ds = TransformSubset(dataset, train_indices, transform=train_transform)
    val_ds = TransformSubset(dataset, val_indices, transform=None)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    return dataset, train_loader, val_loader

def train_one_epoch(model, optimizer, train_dl, device, criterion):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): A UNET-like model.
        optimizer (Optimizer): Optimizer for model parameters.
        train_dl (DataLoader): DataLoader for the training dataset.
        device (torch.device): Device to perform training on.
        criterion (Loss): Loss function.

    Returns:
        float: Average training loss over the epoch.
    """

    model.train()
    curr_loss = 0.
    for X, y in train_dl:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        curr_loss += loss.item()
    return curr_loss / len(train_dl)

def validate(model, val_dl, device, criterion, num_classes):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): A UNET-like model.
        val_dl (DataLoader): DataLoader for the validation dataset.
        device (torch.device): Device to perform validation on.
        criterion (Loss): Loss function.
        num_classes (int): Number of classes in the dataset.

    Returns:
        dict: Dictionary containing:
            - "Average Validation Loss": Average validation loss.
            - "Pixel-Level Accuracy": Accuracy across all pixels.
            - "Macro-Averaged IoU": Mean IoU across all classes.
            - "per_class_iou": List of IoU values for each class.
    """

    model.eval()
    model.to(device)

    curr_loss = 0.
    pixel_acc = MulticlassAccuracy(num_classes=num_classes, average='micro').to(device)
    iou = MulticlassJaccardIndex(num_classes=num_classes, average=None).to(device)  
    mean_iou = MulticlassJaccardIndex(num_classes=num_classes, average='macro').to(device)  

    pixel_acc.reset()
    iou.reset()
    mean_iou.reset()

    with torch.inference_mode():
        for X, y in val_dl:
            X, y = X.to(device), y.to(device)

            logits = model(X)
            loss = criterion(logits, y)
            curr_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            preds_flat = preds.flatten()
            y_flat = y.flatten()

            pixel_acc.update(preds_flat, y_flat)
            iou.update(preds_flat, y_flat)
            mean_iou.update(preds_flat, y_flat)

    avg_loss = curr_loss / len(val_dl)
    pixel_accuracy = pixel_acc.compute().item()
    per_class_iou = iou.compute().cpu().numpy()
    mean_iou_val = mean_iou.compute().item()

    return {
        "Average Validation Loss": avg_loss,
        "Pixel-Level Accuracy": pixel_accuracy,
        "Macro-Averaged IoU": mean_iou_val,
        "per_class_iou": per_class_iou
    }

def train_model(model, num_epochs, device, train_loader, val_loader, num_classes, learning_rate):
    """
    Trains and validates the model for a given number of epochs, saving the best model.

    Args:
        model (nn.Module): A UNET-like model.
        num_epochs (int): Total number of training epochs.
        device (torch.device): Device to train on.
        train_loader (DataLoader): DataLoader for the training sey.
        val_loader (DataLoader): DataLoader for the validation sey.
        num_classes (int): Number of classes in the dataset.
        learning_rate (float): Inital learning rate.

    Returns:
        tuple: 
            - metrics (dict): Dictionary with lists of metrics across epochs:
                "train_losses", "val_losses", "pixel_accs", "mean_ious"
            - best_score (float): Best validation score achieved, computed as:
                0.5 * Pixel Accuracy + 0.5 * Macro-Averaged IoU
    """

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=7, verbose=True)
    model.to(device)

    best_score = -1.0   
    metrics = {"train_losses": [], "val_losses": [], "pixel_accs": [], "mean_ious": []}
    for epoch in range(num_epochs):
        avg_train_loss = train_one_epoch(model, optimizer, train_loader, device, criterion)
        val_metrics = validate(model, val_loader, device, criterion, num_classes)
        scheduler.step(val_metrics["Average Validation Loss"])
        current_score = 0.5 * val_metrics["Pixel-Level Accuracy"] + 0.5 * val_metrics["Macro-Averaged IoU"]

        metrics["train_losses"].append(avg_train_loss)
        metrics["val_losses"].append(val_metrics["Average Validation Loss"])
        metrics["pixel_accs"].append(val_metrics["Pixel-Level Accuracy"])
        metrics["mean_ious"].append(val_metrics["Macro-Averaged IoU"])

        print(f"\nEpoch {epoch + 1}")
        print(f"\tAverage Training Loss: {avg_train_loss: .4f}")
        for metric, value in val_metrics.items():
            if metric == "per_class_iou":
                continue
            print(f"\t{metric}: {value: .4f}")

        if current_score > best_score:
            best_score = current_score
            torch.save(model.state_dict(), f"best_model_lr_{learning_rate}.pt")
            print("\tSaved new best model.")

    print(f"\nBest model recorded a score of: 0.5 * Pixel-Level Accuracy + 0.5 * Macro-Averaged IoU = {best_score:.4f}")
    return metrics, best_score


def predict_full_tile(model, tiff_path, device, patch_size=128, stride=64, num_classes=8, output_path="predicted_mask.npy"):
    """
    Perform sliding window inference on a large TIFF tile using disk streaming.

    Args:
        model: Trained model.
        tiff_path: Path to the input GeoTIFF.
        device: 'cuda' or 'cpu'.
        patch_size: Size of square patches.
        stride: Step size between patches.
        num_classes: Number of classes.
        output_path: Where to save the prediction.
    """
    model.eval()

    with rasterio.open(tiff_path) as src:
        H, W = src.height, src.width
        output_probs = np.zeros((num_classes, H, W), dtype=np.float32)
        count = np.zeros((H, W), dtype=np.float32)

        with torch.no_grad():
            for row in tqdm(range(0, H - patch_size + 1, stride)):
                for col in range(0, W - patch_size + 1, stride):
                    patch = src.read(window=rasterio.windows.Window(col, row, patch_size, patch_size))
                    if patch.shape[1] != patch_size or patch.shape[2] != patch_size:
                        continue

                    # normalize and convert to  tensor
                    patch = patch.astype(np.float32) / 10000.0
                    patch = np.clip(patch, 0.0, 1.0)
                    patch_tensor = torch.from_numpy(patch).unsqueeze(0).to(device)

                    logits = model(patch_tensor)
                    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

                    output_probs[:, row:row + patch_size, col:col + patch_size] += probs
                    count[row:row + patch_size, col:col + patch_size] += 1

        avg_probs = output_probs / np.clip(count, 1e-8, None)
        final_prediction = np.argmax(avg_probs, axis=0).astype(np.uint8)
        np.save(output_path, final_prediction)
        print(f"Prediction saved to {output_path}")
        return final_prediction


def save_prediction_geotiff(prediction, reference_tiff_path, output_path):
    """
    Save the predicted mask as a GeoTIFF using the georeferencing info from the original tile.

    Args:
        prediction (np.ndarray): 2D array of predicted class IDs (H, W).
        reference_tiff_path (str): Path to the original input GeoTIFF (e.g., Sentinel-2 tile).
        output_path (str): Path to save the output GeoTIFF.
    """
    with rasterio.open(reference_tiff_path) as src:
        profile = src.profile
        transform = src.transform
        crs = src.crs

    # ppdate profile for single-band uint8 mask
    profile.update({
        'driver': 'GTiff',
        'height': prediction.shape[0],
        'width': prediction.shape[1],
        'count': 1,
        'dtype': 'uint8',
        'transform': transform,
        'crs': crs
    })

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(prediction, 1)

    print(f"GeoTIFF saved to: {output_path}")

def compute_metrics(pred_mask, true_mask, class_ids, ignore_index=None):
    """
    Evaluate semantic segmentation results.
    
    Args:
        pred_mask: 2D numpy array of predicted class IDs.
        true_mask: 2D numpy array of true class IDs.
        class_ids: List of valid class IDs to evaluate (e.g., [10, 20, 30, ..., 90]).
        ignore_index: Optional value to ignore in evaluation (e.g., background).
        
    Returns:
        Dictionary with pixel accuracy, per-class IoU, mean IoU, and confusion matrix.
    """

    pred = pred_mask.flatten()
    true = true_mask.flatten()
    
    if ignore_index is not None:
        valid = true != ignore_index
        pred = pred[valid]
        true = true[valid]
    
    cm = confusion_matrix(true, pred, labels=class_ids)
    
    intersection = np.diag(cm)
    ground_truth_set = cm.sum(axis=1)
    predicted_set = cm.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    
    iou_per_class = intersection / np.maximum(union, 1e-10)
    mean_iou = np.mean(iou_per_class)
    pixel_accuracy = np.sum(intersection) / np.maximum(np.sum(cm), 1e-10)

    freq = ground_truth_set / np.maximum(np.sum(ground_truth_set), 1e-10)
    fw_iou = np.sum(freq * iou_per_class)
    
    return {
        "pixel_accuracy": pixel_accuracy,
        "iou_per_class": dict(zip(class_ids, iou_per_class)),
        "mean_iou": mean_iou,
        "freq_weighted_iou": fw_iou,
        "confusion_matrix": cm
    }