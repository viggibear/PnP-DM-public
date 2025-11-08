import torch
import os
import logging
import hydra
import numpy as np
import matplotlib.pyplot as plt

import tqdm as tqdm
from collections import defaultdict
from torchvision import transforms

from pnpdm.data import get_dataset, get_dataloader, ECMMDDataset
from pnpdm.tasks import get_operator, get_noise
from pnpdm.ecmmd import get_knn, get_ecmmd
from pnpdm.models import get_model
from pnpdm.optimizers import get_optimizer
from monai.metrics import PSNRMetric, SSIMMetric
from taming.modules.losses.lpips import LPIPS


@hydra.main(version_base="1.2", config_path="configs", config_name="train_ecmmd")
def train_ecmmd(cfg):
    data_config = cfg.data
    model_config = cfg.model
    task_config = cfg.task
    ecmmd_config = cfg.ecmmd
    train_ratio = cfg.train_ratio
    val_ratio = cfg.val_ratio
    n_epochs = cfg.epochs
    eta_dim = cfg.eta_dim
    optim_config = cfg.optimizer

    # device setting
    device_str = f"cuda:{cfg.gpu}" if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)

    # prepare task
    operator = get_operator(**task_config.operator, device='cpu')
    noiser = get_noise(**task_config.noise)

    # Prepare ECMMD operator and KNN
    knn = get_knn(**ecmmd_config.knn)
    ecmmd = get_ecmmd(**ecmmd_config.ecmmd, knn=knn)
    model = get_model(**model_config).to(device)
    optimizer = get_optimizer(model.parameters(), **optim_config)

    # prepare dataloader
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Normalize((0.5), (0.5))
    ])
    inv_transform = transforms.Compose([
        transforms.Normalize((-1), (2)),
        transforms.Lambda(lambda x: x.clamp(0, 1).detach())
    ])

    dataset = get_dataset(**data_config, transform=transform)
    dataset = ECMMDDataset(dataset, eta_dim=eta_dim, operator=operator, noiser=noiser)

    # Create results directory structure
    exp_name = f"ecmmd_{dataset.display_name}_{model_config.name}_epochs{n_epochs}"
    logger = logging.getLogger(exp_name)
    out_path = os.path.join("results", exp_name)
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['test_images', 'train_samples', 'checkpoints']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)
    print(f"Results will be saved to {out_path}")

    # Initialize metrics
    metrics = {
        'psnr': PSNRMetric(max_val=1),
        'ssim': SSIMMetric(spatial_dims=2),
        'lpips': LPIPS().to(device).eval(),
    }

    # Frequency for saving test images
    save_image_every = cfg.get('save_image_every', 10)  # default: save every 10 epochs
    num_test_samples = cfg.get('num_test_samples', 4)  # default: save 4 test images

    num_images = len(dataset)
    train_len = int(num_images * train_ratio)
    val_len = int(num_images * val_ratio)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_len, val_len, num_images - train_len - val_len])
    logger.info(f"Dataset split: {train_len} train, {val_len} val, {num_images - train_len - val_len} test samples.")
    train_dataloader = get_dataloader(train_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, train=True)
    val_dataloader = get_dataloader(val_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, train=False)
    test_dataloader = get_dataloader(test_dataset, batch_size=1, num_workers=0, train=False)  # batch_size=1 for easier visualization

    train_losses = []
    val_losses = []
    test_metrics_history = defaultdict(list)

    for epoch in tqdm.tqdm(range(n_epochs), desc="Training Epochs"):
        model.train()
        total_train_loss = 0.0
        num_train_batches = 0

        for train_images, dirty_train_images, train_eta in tqdm.tqdm(train_dataloader, desc="Training Batches"):
            optimizer.zero_grad()
            curr_batch_size = len(train_images)

            train_images = train_images.to(device)
            dirty_train_images = dirty_train_images.to(device)
            train_eta = train_eta.to(device)

            # Squeeze extra dimension if present
            if dirty_train_images.ndim == 5:
                dirty_train_images = dirty_train_images.squeeze(1)

            denoised_train_images = model(dirty_train_images, train_eta)

            loss = ecmmd(denoised_train_images.reshape(
                curr_batch_size, -1
            ), train_images.reshape(
                curr_batch_size, -1
            ), dirty_train_images.reshape(
                curr_batch_size, -1
            )) ** 2
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            num_train_batches += 1

        avg_train_loss = total_train_loss / num_train_batches
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0.0
        num_val_batches = 0
        with torch.inference_mode():
            for val_images, dirty_val_images, val_eta in val_dataloader:
                curr_batch_size = len(val_images)
                val_images = val_images.to(device)
                dirty_val_images = dirty_val_images.to(device)
                val_eta = val_eta.to(device)

                # Squeeze extra dimension if present
                if dirty_val_images.ndim == 5:
                    dirty_val_images = dirty_val_images.squeeze(1)

                denoised_val_images = model(dirty_val_images, val_eta)

                loss = ecmmd(denoised_val_images.reshape(
                    curr_batch_size, -1
                ), val_images.reshape(
                    curr_batch_size, -1
                ), dirty_val_images.reshape(
                    curr_batch_size, -1
                )) ** 2

                total_val_loss += loss.item()
                num_val_batches += 1
        avg_val_loss = total_val_loss / num_val_batches
        val_losses.append(avg_val_loss)

        # Log progress
        logger.info(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        # Save test images and compute metrics every few epochs
        if (epoch + 1) % save_image_every == 0 or epoch == 0 or epoch == n_epochs - 1:
            logger.info(f"Saving test images at epoch {epoch+1}...")
            print(f"Saving test images at epoch {epoch+1}...")

            epoch_metrics = defaultdict(list)
            all_gt_imgs = []
            all_noisy_imgs = []
            all_denoised_imgs = []

            with torch.inference_mode():
                for i, (test_images, dirty_test_images, test_eta) in enumerate(test_dataloader):
                    if i >= num_test_samples:
                        break

                    test_images = test_images.to(device)
                    dirty_test_images = dirty_test_images.to(device)
                    test_eta = test_eta.to(device)

                    if dirty_test_images.ndim == 5:
                        dirty_test_images = dirty_test_images.squeeze(1)

                    denoised_test_images = model(dirty_test_images, test_eta)

                    # Convert to display format
                    gt_display = inv_transform(test_images)
                    noisy_display = inv_transform(dirty_test_images)
                    denoised_display = inv_transform(denoised_test_images)

                    # Compute metrics
                    for name, metric in metrics.items():
                        metric_value = metric(denoised_display, gt_display).item()
                        epoch_metrics[name].append(metric_value)

                    # Store images for combined visualization
                    gt_img = gt_display.permute(0, 2, 3, 1).squeeze().cpu().numpy()
                    noisy_img = noisy_display.permute(0, 2, 3, 1).squeeze().cpu().numpy()
                    denoised_img = denoised_display.permute(0, 2, 3, 1).squeeze().cpu().numpy()

                    all_gt_imgs.append(gt_img)
                    all_noisy_imgs.append(noisy_img)
                    all_denoised_imgs.append(denoised_img)

            # Determine colormap
            cmap = 'gray' if test_images.shape[1] == 1 else None

            # Create combined figure with all test samples
            test_images_dir = os.path.join(out_path, 'test_images')
            os.makedirs(test_images_dir, exist_ok=True)

            fig, axes = plt.subplots(num_test_samples, 3, figsize=(15, 5 * num_test_samples))
            # Handle case where num_test_samples == 1
            if num_test_samples == 1:
                axes = axes.reshape(1, -1)

            for i in range(len(all_gt_imgs)):
                axes[i, 0].imshow(all_gt_imgs[i], cmap=cmap)
                axes[i, 0].set_title(f'Sample {i+1}: Ground Truth')
                axes[i, 0].axis('off')

                axes[i, 1].imshow(all_noisy_imgs[i], cmap=cmap)
                axes[i, 1].set_title(f'Sample {i+1}: Noisy Input')
                axes[i, 1].axis('off')

                axes[i, 2].imshow(all_denoised_imgs[i], cmap=cmap)
                axes[i, 2].set_title(f'Sample {i+1}: Denoised')
                axes[i, 2].axis('off')

            plt.suptitle(f'Test Results - Epoch {epoch+1}', fontsize=16, y=0.995)
            plt.tight_layout()
            plt.savefig(os.path.join(test_images_dir, f'epoch_{epoch+1:04d}_all_samples.png'),
                       bbox_inches='tight', dpi=100)
            plt.close()

            # Save average metrics for this epoch
            for name in metrics.keys():
                avg_metric = np.mean(epoch_metrics[name])
                test_metrics_history[name].append(avg_metric)
                logger.info(f"Epoch {epoch+1} - Test {name.upper()}: {avg_metric:.4f}")
                print(f"Epoch {epoch+1} - Test {name.upper()}: {avg_metric:.4f}")

        # Save checkpoint
        if (epoch + 1) % save_image_every == 0 or epoch == n_epochs - 1:
            checkpoint_path = os.path.join(out_path, 'checkpoints', f'model_epoch_{epoch+1:04d}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

    # Save final training summary
    logger.info("Training complete. Saving final summary...")
    print("Training complete. Saving final summary...")

    # Save loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, n_epochs + 1), val_losses, label='Val Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_path, 'loss_curves.png'), bbox_inches='tight', dpi=150)
    plt.close()

    # Save metrics curves if available
    if test_metrics_history:
        epochs_with_metrics = [i * save_image_every for i in range(1, len(test_metrics_history['psnr']) + 1)]
        if 0 not in epochs_with_metrics:
            epochs_with_metrics = [1] + epochs_with_metrics

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for idx, (name, values) in enumerate(test_metrics_history.items()):
            axes[idx].plot(epochs_with_metrics[:len(values)], values, marker='o', linewidth=2)
            axes[idx].set_xlabel('Epoch')
            axes[idx].set_ylabel(name.upper())
            axes[idx].set_title(f'Test {name.upper()} over Training')
            axes[idx].grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_path, 'test_metrics_curves.png'), bbox_inches='tight', dpi=150)
        plt.close()

    # Save metadata
    metadata = {
        'config': dict(cfg),
        'n_epochs': n_epochs,
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
        'eta_dim': eta_dim,
        'num_train_images': train_len,
        'num_val_images': val_len,
        'num_test_images': len(test_dataset),
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_metrics_history': dict(test_metrics_history),
    }
    np.save(os.path.join(out_path, 'training_metadata.npy'), metadata)

    # Save text summary
    with open(os.path.join(out_path, 'training_summary.txt'), 'w') as f:
        f.write('='*70 + '\n')
        f.write('ECMMD Training Summary\n')
        f.write('='*70 + '\n\n')
        f.write(f'Experiment Name: {exp_name}\n')
        f.write(f'Number of Epochs: {n_epochs}\n')
        f.write(f'Train/Val/Test Split: {train_ratio:.2f}/{val_ratio:.2f}/{1-train_ratio-val_ratio:.2f}\n')
        f.write(f'Number of Train Images: {train_len}\n')
        f.write(f'Number of Val Images: {val_len}\n')
        f.write(f'Number of Test Images: {len(test_dataset)}\n')
        f.write(f'Batch Size: {cfg.batch_size}\n')
        f.write(f'Eta Dimension: {eta_dim}\n\n')
        f.write('Final Losses:\n')
        f.write(f'  Train Loss: {train_losses[-1]:.6f}\n')
        f.write(f'  Val Loss: {val_losses[-1]:.6f}\n\n')
        f.write('Best Losses:\n')
        f.write(f'  Best Train Loss: {min(train_losses):.6f} (Epoch {train_losses.index(min(train_losses)) + 1})\n')
        f.write(f'  Best Val Loss: {min(val_losses):.6f} (Epoch {val_losses.index(min(val_losses)) + 1})\n\n')

        if test_metrics_history:
            f.write('='*70 + '\n')
            f.write('Test Metrics History:\n')
            f.write('='*70 + '\n\n')
            for name, values in test_metrics_history.items():
                f.write(f'{name.upper()}:\n')
                f.write(f'  Final: {values[-1]:.4f}\n')
                best_fn = np.amin if name == 'lpips' else np.amax
                best_value = best_fn(values)
                best_idx = values.index(best_value)
                best_epoch = (best_idx + 1) * save_image_every
                f.write(f'  Best: {best_value:.4f} (Epoch {best_epoch})\n\n')

        f.write('='*70 + '\n')
        f.write(f'Results saved to: {out_path}\n')
        f.write('='*70 + '\n')

    logger.info(f"Training summary saved to {out_path}")
    print(f"Training complete! Results saved to {out_path}")

if __name__ == '__main__':
    train_ecmmd()
