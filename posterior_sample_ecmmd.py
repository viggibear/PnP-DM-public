import torch, os, hydra, logging
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from torchvision import transforms
from pnpdm.data import get_dataset, get_dataloader
from pnpdm.tasks import get_operator, get_noise, MotionBlurCircular
from pnpdm.models import get_model
from pnpdm.samplers import get_sampler
from hydra.core.hydra_config import HydraConfig
from monai.metrics import PSNRMetric, SSIMMetric
from taming.modules.losses.lpips import LPIPS

def calculate_variance(ecmmd_model, y_n, num_samples=100, operator=None, measurement_space=True):
    """
    Estimate variance across samples from ecmmd_model.
    If measurement_space=True, push samples through operator.forward and return variance in measurement domain
    (this is what you want when comparing to measurement noise_sigma).
    """
    if measurement_space and operator is None:
        raise ValueError("operator must be provided when measurement_space=True")

    samples = []
    for _ in range(num_samples):
        # batch size assumed equal to y_n.batch (usually 1)
        batch_eta = torch.randn(y_n.size(0), ecmmd_model.eta_dim).to(y_n.device)
        sample = ecmmd_model.forward(y_n, batch_eta)  # returns in model/image space (same space as ref_img used earlier)
        samples.append(sample)
    samples = torch.cat(samples, dim=0)  # shape (num_samples * batch, C, H, W)

    if measurement_space:
        # compute variance in measurement domain so it's comparable to noise_sigma**2
        with torch.no_grad():
            meas = operator.forward(samples)  # shape (num_samples * batch, C, H, W)
            return meas
    else:
        return samples

@hydra.main(version_base="1.2", config_path="configs", config_name="default")
def posterior_sample(cfg):
    # load configurations
    data_config = cfg.data
    task_config = cfg.task
    model_config = cfg.model
    ecmmd_config = cfg.ecmmd
    sampler_config = cfg.sampler

    # device setting
    device_str = f"cuda:{cfg.gpu}" if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)

    # prepare task (forward model and noise)
    operator = get_operator(**task_config.operator, device=device)
    noiser = get_noise(**task_config.noise)
    noise_sigma = task_config.noise.sigma

    # prepare dataloader
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Normalize((0.5), (0.5))
    ])
    inv_transform = transforms.Compose([
        transforms.Normalize((-1), (2)),
        transforms.Lambda(lambda x: x.clamp(0, 1).detach())
    ])
    # inv_transform = transforms.Compose([
    #     transforms.Normalize((-1), (2)),
    #     transforms.Lambda(lambda x: (x.clamp(0, 1)**0.4).detach())
    # ])
    dataset = get_dataset(**data_config, transform=transform)
    num_test_images = len(dataset)
    # Truncate for quick testing
    dataset.fpaths = dataset.fpaths[:num_test_images]
    dataloader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)

    # load model
    model = get_model(**model_config)
    model = model.to(device)
    model.eval()

    # load edm model
    ecmmd_model = get_model(**ecmmd_config)
    ecmmd_model = ecmmd_model.to(device)
    ecmmd_model.eval()

    # load sampler
    sampler = get_sampler(sampler_config, model=model, operator=operator, noiser=noiser, device=device)

    # working directory
    exp_name = '_'.join([dataset.display_name, operator.display_name, noiser.display_name, sampler.display_name])
    exp_name += '' if len(cfg.add_exp_name) == 0 else '_' + cfg.add_exp_name
    logger = logging.getLogger(exp_name)
    out_path = os.path.join("results", exp_name)
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['gt', 'meas', 'recon', 'progress', 'ecmmd_test']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)
    print(f"Results are saved to {out_path}")

    # inference
    meta_log = defaultdict(list)
    meta_log["statistics_based_on_one_sample"] = defaultdict(list)
    meta_log["statistics_based_on_mean"] = defaultdict(list)
    metrics = {
        'psnr': PSNRMetric(max_val=1),
        'ssim': SSIMMetric(spatial_dims=2),
        'lpips': LPIPS().to(device).eval(),
    }
    for i, ref_img in enumerate(dataloader):
        logger.info(f"Inference for image {i} on device {device_str}")
        file_idx = f"{i:05d}"
        ref_img = ref_img.to(device)
        cmap = 'gray' if ref_img.shape[1] == 1 else None

        logger.info(f"The shape of the reference image: {ref_img.shape}")

        # regenerate kernel for motion blur
        if isinstance(operator, MotionBlurCircular):
            operator.generate_kernel_(seed=i)

        # forward measurement model (Ax + n)
        y_n = noiser(operator.forward(ref_img))

        # estimate variance of ECMMD model
        if cfg.ecmmd_estimate_variance:
            logger.info(f"Estimating variance of ECMMD model with {cfg.ecmmd_variance_num_samples} samples...")
            meas_samples = calculate_variance(ecmmd_model, y_n, num_samples=cfg.ecmmd_variance_num_samples, operator=operator, measurement_space=True)
            # Calculate pixel-wise variance for visualization
            variance = torch.var(meas_samples, dim=0, unbiased=True)
            # Calculate max eigenvalue of covariance matrix using Gram matrix trick
            N = meas_samples.shape[0]
            X = meas_samples.reshape(N, -1)
            # X_centered = X - X.mean(dim=0, keepdim=True)
            # assert X_centered.mean().abs() < 1e-6, "Data not centered properly"


            for i, random_clean_img in enumerate(dataset):
                if i == 0:
                    break

            random_clean_img = random_clean_img.to(device)
            X_centered = X - random_clean_img.reshape(1, -1)
            print("My X has shape:", X_centered.shape)

            # Take column-wise L2-norm of X_centered
            X_col_wise_norms = torch.norm(X_centered, dim=0, keepdim=True) ** 2 / (N - 1)
            print("My X_col_wise_norms has shape:", X_col_wise_norms.shape)
            print("The largest column-wise norm is:", X_col_wise_norms.max().item())
            print("The smallest column-wise norm is:", X_col_wise_norms.min().item())
            print("The median column-wise norm is:", X_col_wise_norms.median().item())

            rho_scale = X_col_wise_norms.median().sqrt().item() / noise_sigma
            logger.info(f"Estimated variance scale (rho_scale): {rho_scale}")

            plt.imsave(os.path.join(out_path, 'ecmmd_test', file_idx+"_ecmmd_variance.png"), variance.permute(1, 2, 0).squeeze().cpu().numpy(), cmap=cmap)

        # Do a 1 pass ECMMD to see if everything is working fine
        ecmmd_onepass_test = ecmmd_model.forward(y_n, torch.randn(1, ecmmd_model.eta_dim).to(y_n.device))
        plt.imsave(os.path.join(out_path, 'ecmmd_test', file_idx+"_ecmmd_test.png"), inv_transform(ecmmd_onepass_test).permute(0, 2, 3, 1).squeeze().cpu().numpy(), cmap=cmap)

        # logging
        log = defaultdict(list)
        log["consistency_gt"] = torch.norm(operator.forward(ref_img) - y_n).item()
        log["gt"] = inv_transform(ref_img).permute(0, 2, 3, 1).squeeze().cpu().numpy()
        plt.imsave(os.path.join(out_path, 'gt', file_idx+'.png'), log["gt"], cmap=cmap)
        try:
            log["meas"] = inv_transform(y_n.reshape(*ref_img.shape)).permute(0, 2, 3, 1).squeeze().cpu().numpy()
            # log["meas"] = inv_transform(operator.forward(ref_img)).permute(0, 2, 3, 1).squeeze().cpu().numpy()
            plt.imsave(os.path.join(out_path, 'meas', file_idx+'.png'), log["meas"], cmap=cmap)
            if hasattr(operator, 'kernel'):
                plt.imsave(os.path.join(out_path, 'meas', file_idx+'_kernel.png'), operator.kernel.detach().cpu())
        except:
            try:
                # in case where y_n is bigger than ref_img
                log["meas"] = inv_transform(y_n).permute(0, 2, 3, 1).squeeze().cpu().numpy()
                plt.imsave(os.path.join(out_path, 'meas', file_idx+'.png'), log["meas"], cmap=cmap)
            except:
                log["meas"] = inv_transform(operator.A_pinv(y_n).reshape(*ref_img.shape)).permute(0, 2, 3, 1).squeeze().cpu().numpy()
                plt.imsave(os.path.join(out_path, 'meas', file_idx+'_pinv.png'), log["meas"], cmap=cmap)

        # sampling
        for j in tqdm(range(cfg.num_runs)):
            samples = sampler(
                gt=ref_img,
                y_n=y_n,
                record=cfg.record,
                fname=file_idx+f'_run_{j}',
                save_root=out_path,
                inv_transform=inv_transform,
                rho_scale=rho_scale if cfg.ecmmd_estimate_variance else 1.0,
                metrics=metrics
            )
            samples = inv_transform(samples)
            sample = samples[[-1]] # take the last sample as the single sample for calculating metrics
            if len(samples) > 1:
                mean, std = torch.mean(samples, dim=0, keepdim=True), torch.std(samples, dim=0, keepdim=True)

            # logging
            log["samples"].append(sample.permute(0, 2, 3, 1).squeeze().cpu().numpy())
            for name, metric in metrics.items():
                log[name+"_sample"].append(metric(sample, inv_transform(ref_img)).item())
            log["consistency_sample"].append(torch.norm(operator.forward(transform(sample)) - y_n).item())
            plt.imsave(os.path.join(out_path, 'recon', file_idx+f'_run_{j}_sample.png'), log["samples"][-1], cmap=cmap)

            if len(samples) > 1:
                log["means"].append(mean.permute(0, 2, 3, 1).squeeze().cpu().numpy())
                log["stds"].append(std.permute(0, 2, 3, 1).squeeze().cpu().numpy())
                for name, metric in metrics.items():
                    log[name+"_mean"].append(metric(mean, inv_transform(ref_img)).item())
                log["consistency_mean"].append(torch.norm(operator.forward(transform(mean)) - y_n).item())
                plt.imsave(os.path.join(out_path, 'recon', file_idx+f'_run_{j}_mean.png'), log["means"][-1], cmap=cmap)
                # plt.imsave(os.path.join(out_path, 'recon', file_idx+f'_run_{j}_std.png'), log["stds"][-1], cmap=cmap)

        np.save(os.path.join(out_path, 'recon', file_idx+'_log.npy'), log)

        with open(os.path.join(out_path, 'recon', file_idx+'_metrics.txt'), "w") as f:
            f.write(f'Statistics based on ONE sample for each run ({cfg.num_runs} runs in total):\n')
            f.write('\n')
            for name, _ in metrics.items():
                f.write(f'{name} (avg over {cfg.num_runs} runs): {np.mean(log[name+"_sample"])}\n')
            f.write(f'consistency_sample (avg over {cfg.num_runs} runs): {np.mean(log["consistency_sample"])}\n')
            f.write('\n')
            for name, _ in metrics.items():
                best_fn = np.amin if name == 'lpips' else np.amax
                f.write(f'{name} (best among {cfg.num_runs} runs): {best_fn(log[name+"_sample"])}\n')
            f.write(f'consistency_sample (best among {cfg.num_runs} runs): {np.amin(log["consistency_sample"])}\n')
            if len(samples) > 1:
                f.write('\n')
                f.write('='*70+'\n')
                f.write('\n')
                f.write(f'Statistics based on the mean over {len(samples)} samples for each run ({cfg.num_runs} runs in total):\n')
                f.write('\n')
                for name, _ in metrics.items():
                    f.write(f'{name} (avg over {cfg.num_runs} runs): {np.mean(log[name+"_mean"])}\n')
                f.write(f'consistency_mean (avg over {cfg.num_runs} runs): {np.mean(log["consistency_mean"])}\n')
                f.write('\n')
                for name, _ in metrics.items():
                    best_fn = np.amin if name == 'lpips' else np.amax
                    f.write(f'{name} (best among {cfg.num_runs} runs): {best_fn(log[name+"_mean"])}\n')
                f.write(f'consistency_mean (best among {cfg.num_runs} runs): {np.amin(log["consistency_mean"])}\n')
            f.write('\n')
            f.write('='*70+'\n')
            f.write('\n')
            f.write(f'consistency (gt): {log["consistency_gt"]}\n')
            f.close()

        # meta logging
        meta_log["consistency_gt"].append(log["consistency_gt"])
        sample_recon_mean = torch.mean(torch.from_numpy(np.array(log["samples"])), dim=0)
        if len(sample_recon_mean.shape) == 2:
            sample_recon_mean = sample_recon_mean.unsqueeze(2) # add a channel dimension
        sample_recon_mean = sample_recon_mean.permute(2, 0, 1).unsqueeze(0).to(device)
        for name, metric in metrics.items():
            meta_log["statistics_based_on_one_sample"][name+"_mean_recon_of_all_runs"].append(metric(sample_recon_mean, inv_transform(ref_img)).item())
            meta_log["statistics_based_on_one_sample"][name+"_last_of_all_runs"].append(log[name+"_sample"][-1])
            best_fn = np.amin if name == 'lpips' else np.amax
            meta_log["statistics_based_on_one_sample"][name+"_best_of_all_runs"].append(best_fn(log[name+"_sample"]))
        meta_log["statistics_based_on_one_sample"]["consistency_mean_recon_of_all_runs"].append(torch.norm(operator.forward(transform(sample_recon_mean)) - y_n).item())
        meta_log["statistics_based_on_one_sample"]["consistency_last_of_all_runs"].append(log["consistency_sample"][-1])
        meta_log["statistics_based_on_one_sample"]["consistency_best_of_all_runs"].append(np.amin(log["consistency_sample"]))
        if len(samples) > 1:
            mean_recon_mean = torch.mean(torch.from_numpy(np.array(log["means"])), dim=0)
            if len(mean_recon_mean.shape) == 2:
                mean_recon_mean = mean_recon_mean.unsqueeze(2) # add a channel dimension
            mean_recon_mean = mean_recon_mean.permute(2, 0, 1).unsqueeze(0).to(device)
            for name, metric in metrics.items():
                meta_log["statistics_based_on_mean"][name+"_mean_recon_of_all_runs"].append(metric(mean_recon_mean, inv_transform(ref_img)).item())
                meta_log["statistics_based_on_mean"][name+"_last_of_all_runs"].append(log[name+"_mean"][-1])
                best_fn = np.amin if name == 'lpips' else np.amax
                meta_log["statistics_based_on_mean"][name+"_best_of_all_runs"].append(best_fn(log[name+"_mean"]))
            meta_log["statistics_based_on_mean"]["consistency_mean_recon_of_all_runs"].append(torch.norm(operator.forward(transform(mean_recon_mean)) - y_n).item())
            meta_log["statistics_based_on_mean"]["consistency_last_of_all_runs"].append(log["consistency_mean"][-1])
            meta_log["statistics_based_on_mean"]["consistency_best_of_all_runs"].append(np.amin(log["consistency_mean"]))

    # meta logging
    np.save(os.path.join(out_path, 'meta_log.npy'), meta_log)
    with open(os.path.join(out_path, 'meta_metrics.txt'), "w") as f:
        f.write(f'Statistics based on ONE sample for each run ({cfg.num_runs} runs in total) of each test image:\n')
        f.write('\n')
        for name, _ in metrics.items():
            f.write(f'{name}_mean_recon_of_{cfg.num_runs}_runs (avg over {num_test_images} test images): {np.mean(meta_log["statistics_based_on_one_sample"][name+"_mean_recon_of_all_runs"])}\n')
        f.write(f'consistency_mean_recon_of_{cfg.num_runs}_runs (avg over {num_test_images} test images): {np.mean(meta_log["statistics_based_on_one_sample"]["consistency_mean_recon_of_all_runs"])}\n')
        f.write('\n')
        for name, _ in metrics.items():
            f.write(f'{name}_last_of_{cfg.num_runs}_runs (avg over {num_test_images} test images): {np.mean(meta_log["statistics_based_on_one_sample"][name+"_last_of_all_runs"])}\n')
        f.write(f'consistency_last_of_all_runs (avg over {num_test_images} test images): {np.mean(meta_log["statistics_based_on_one_sample"]["consistency_last_of_all_runs"])}\n')
        f.write('\n')
        for name, _ in metrics.items():
            f.write(f'{name}_best_of_{cfg.num_runs}_runs (avg over {num_test_images} test images): {np.mean(meta_log["statistics_based_on_one_sample"][name+"_best_of_all_runs"])}\n')
        f.write(f'consistency_best_of_all_runs (avg over {num_test_images} test images): {np.mean(meta_log["statistics_based_on_one_sample"]["consistency_best_of_all_runs"])}\n')
        if len(samples) > 1:
            f.write('\n')
            f.write('='*70+'\n')
            f.write('\n')
            f.write(f'Statistics based on the mean over {len(samples)} samples for each run ({cfg.num_runs} runs in total) of each test image:\n')
            f.write('\n')
            for name, _ in metrics.items():
                f.write(f'{name}_mean_recon_of_{cfg.num_runs}_runs (avg over {num_test_images} test images): {np.mean(meta_log["statistics_based_on_mean"][name+"_mean_recon_of_all_runs"])}\n')
            f.write(f'consistency_mean_recon_of_{cfg.num_runs}_runs (avg over {num_test_images} test images): {np.mean(meta_log["statistics_based_on_mean"]["consistency_mean_recon_of_all_runs"])}\n')
            f.write('\n')
            for name, _ in metrics.items():
                f.write(f'{name}_last_of_{cfg.num_runs}_runs (avg over {num_test_images} test images): {np.mean(meta_log["statistics_based_on_mean"][name+"_last_of_all_runs"])}\n')
            f.write(f'consistency_last_of_all_runs (avg over {num_test_images} test images): {np.mean(meta_log["statistics_based_on_mean"]["consistency_last_of_all_runs"])}\n')
            f.write('\n')
            for name, _ in metrics.items():
                f.write(f'{name}_best_of_{cfg.num_runs}_runs (avg over {num_test_images} test images): {np.mean(meta_log["statistics_based_on_mean"][name+"_best_of_all_runs"])}\n')
            f.write(f'consistency_best_of_all_runs (avg over {num_test_images} test images): {np.mean(meta_log["statistics_based_on_mean"]["consistency_best_of_all_runs"])}\n')
        f.write('\n')
        f.write('='*70+'\n')
        f.write('\n')
        f.write(f'consistency (gt) (avg over {num_test_images} test images): {np.mean(meta_log["consistency_gt"])}\n')
        f.close()

    logger.info(f"Finished inference")

if __name__ == '__main__':
    posterior_sample()