import torch
from torch.utils.data import Dataset


class MultiOmicsDataset(Dataset):
    def __init__(self, features: torch.Tensor, covariates: torch.Tensor = None, labels: torch.Tensor = None):
        """
        Args:
            features (torch.Tensor): [num_donors, num_features] 형태의 멀티오믹스 행렬
                (scRNA-seq은 donor 단위 pseudobulk로 집계된 값)
            covariates (torch.Tensor, optional): [num_donors, k] nuisance 공변량
                (나이, 성별, PMI, 배치 등) — 적대적 판별자의 z_nuisance로 사용
            labels (torch.Tensor, optional): 질병 상태 등 타겟 라벨
        """
        self.features = features
        self.covariates = covariates
        self.labels = labels

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        item = [self.features[idx]]
        if self.covariates is not None:
            item.append(self.covariates[idx])
        if self.labels is not None:
            item.append(self.labels[idx])
        return item[0] if len(item) == 1 else tuple(item)
