import torch
import torch.nn.functional as F

from piq import ssim


class StatsMetrics:
    def __init__(self):
        self.metrics = {}
        self.metrics['ssim_dist'] = 0.0
        self.metrics['hamming'] = 0.0
        self.metrics['bhattacharyya'] = 0.0
        self.metrics['kl_div'] = 0.0
        self.metrics['chebyshev'] = 0.0
        self.metrics['manhattan'] = 0.0
        self.metrics['euclidian'] = 0.0
        self.metrics['cosine_dist'] = 0.0
        self.metrics['mu_original'] = 0.0
        self.metrics['mu_augment'] = 0.0
        self.metrics['sigma_original'] = 0.0
        self.metrics['sigma_augment'] = 0.0

    def get_metrics(self, img_tensor_original, img_tensor_augmented):
        self.set_mu_sigma(img_tensor_original, img_tensor_augmented)
        self.compute_all_metrics(img_tensor_original, img_tensor_augmented)
        return self.metrics

    def compute_all_metrics(self, img_tensor_original, img_tensor_augmented):
        # self.ssim_dist(img_tensor_original, img_tensor_augmented)
        self.hamming(img_tensor_original, img_tensor_augmented)
        self.bhattacharyya(img_tensor_original, img_tensor_augmented)
        self.kl_div(img_tensor_original, img_tensor_augmented)
        self.chebyshev(img_tensor_original, img_tensor_augmented)
        self.manhattan(img_tensor_original, img_tensor_augmented)
        self.euclidian(img_tensor_original, img_tensor_augmented)
        self.cosine_dist(img_tensor_original, img_tensor_augmented)

    def set_mu_sigma(self, img_tensor_original, img_tensor_augmented):
        self.metrics['mu_original'] = img_tensor_original.float().mean()
        self.metrics['mu_augment'] = img_tensor_augmented.mean()
        self.metrics['sigma_original'] = img_tensor_original.float().std()
        self.metrics['sigma_augment'] = img_tensor_augmented.std()

    def ssim_dist(self, img_tensor_original, img_tensor_augmented):
        # obs_tensor = img_tensor_original.unsqueeze(1)
        self.metrics['ssim_dist'] = ssim(img_tensor_original, img_tensor_augmented, data_range=255.0, reduction='mean').item()

    def hamming(self, img_tensor_original, img_tensor_augmented):
        different_elements = img_tensor_original != img_tensor_augmented
        self.metrics['hamming'] = torch.sum(different_elements).item()

    def bhattacharyya(self, img_tensor_original, img_tensor_augmented):
        self.metrics['bhattacharyya'] = -torch.log(torch.sum(torch.sqrt(img_tensor_original * img_tensor_augmented))).item()

    def kl_div(self, img_tensor_original, img_tensor_augmented):
        self.metrics['kl_div'] = torch.sum(img_tensor_original * torch.log(img_tensor_original / img_tensor_augmented)).item()

    def chebyshev(self, img_tensor_original, img_tensor_augmented):
        self.metrics['chebyshev'] = torch.max(torch.abs(img_tensor_original - img_tensor_augmented)).item()

    def manhattan(self, img_tensor_original, img_tensor_augmented):
        self.metrics['manhattan'] = torch.sum(torch.abs(img_tensor_original - img_tensor_augmented)).item()

    def euclidian(self, img_tensor_original, img_tensor_augmented):
        self.metrics['euclidian'] = torch.norm(img_tensor_original - img_tensor_augmented).item()

    def cosine_dist(self, img_tensor_original, img_tensor_augmented):
        # Flatten the tensors
        tensor_original_flat = img_tensor_original.view(1, -1)
        tensor_augmented_flat = img_tensor_augmented.view(1, -1)
        # Compute cosine similarity
        # cosine_sim = F.cosine_similarity(tensor_original_flat, tensor_augmented_flat, dim=1)
        self.metrics['cosine_dist'] = 1 - F.cosine_similarity(tensor_original_flat, tensor_augmented_flat, dim=1).item()
