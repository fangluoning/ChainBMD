import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from typing import Tuple, Optional, Dict, Any, List

from sklearn.model_selection import GroupKFold, KFold


class ChainBMDDataset(Dataset):
    """Dataset wrapper around processed HDF5 features."""

    def __init__(
        self,
        hdf5_path: str,
        transform=None,
        target_field: str = "skill_levels",
        filter_invalid: bool = True,
        feature_indices: Optional[List[int]] = None,
    ):
        super().__init__()
        self.hdf5_path = hdf5_path
        self.transform = transform
        self.target_field = target_field
        self.filter_invalid = filter_invalid
        self._h5: Optional[h5py.File] = None
        self.feature_indices = feature_indices
        with h5py.File(hdf5_path, "r") as f:
            all_targets = f[target_field][:]
            if filter_invalid:
                valid_mask = all_targets >= 0
                self.indices = np.nonzero(valid_mask)[0].tolist()
            else:
                self.indices = list(range(all_targets.shape[0]))
            self.length = len(self.indices)
            self.sequence_length = f["feature_matrices"].shape[1]
            self.feature_dim = (
                len(feature_indices) if feature_indices is not None else f["feature_matrices"].shape[2]
            )
            raw_subjects = f.get("subject_ids")
            if raw_subjects is None:
                subjects = [f"sample_{idx:04d}" for idx in range(f["feature_matrices"].shape[0])]
            else:
                subjects = [
                    sid.decode("utf-8") if isinstance(sid, (bytes, bytearray)) else str(sid)
                    for sid in raw_subjects[:]
                ]
            self.subjects = [subjects[i] for i in self.indices]

    def __len__(self) -> int:
        return self.length

    def _ensure_open(self) -> None:
        if self._h5 is None:
            self._h5 = h5py.File(self.hdf5_path, "r")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        self._ensure_open()
        real_idx = self.indices[idx]
        feat = self._h5["feature_matrices"][real_idx]  # (T, F)
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        if self.feature_indices is not None:
            feat = feat[:, self.feature_indices]
        label = self._h5[self.target_field][real_idx]
        sample = {
            "sequence": torch.from_numpy(feat.astype("float32")),
            "label": torch.tensor(int(label), dtype=torch.long),
            "subject": self.subjects[idx],
        }
        if self.transform:
            sample = self.transform(sample)
        return sample

    def close(self) -> None:
        if self._h5 is not None:
            try:
                self._h5.close()
            finally:
                self._h5 = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _split_dataset_by_subject(
    dataset: ChainBMDDataset,
    val_split: float,
    test_split: float,
    random_seed: int,
) -> Tuple[Subset, Subset, Subset]:
    subjects = np.array(dataset.subjects)
    unique_subjects = np.unique(subjects)
    if len(unique_subjects) < 3:
        raise ValueError("Subject-level split requires at least 3 unique subjects.")
    rng = np.random.default_rng(random_seed)
    rng.shuffle(unique_subjects)
    total_subjects = len(unique_subjects)
    test_count = max(1, int(round(total_subjects * test_split)))
    val_count = max(1, int(round(total_subjects * val_split)))
    if test_count + val_count >= total_subjects:
        test_count = max(1, min(test_count, total_subjects - 2))
        val_count = max(1, min(val_count, total_subjects - test_count - 1))
    train_count = total_subjects - test_count - val_count
    if train_count <= 0:
        raise ValueError("Not enough subjects for requested train/val/test split.")
    test_subjects = set(unique_subjects[:test_count])
    val_subjects = set(unique_subjects[test_count:test_count + val_count])
    train_subjects = set(unique_subjects[test_count + val_count:])

    subject_to_indices: Dict[str, List[int]] = {}
    for idx, subject in enumerate(subjects):
        subject_to_indices.setdefault(subject, []).append(idx)

    def gather(subj_set: set) -> List[int]:
        subset_idx: List[int] = []
        for subj in subj_set:
            subset_idx.extend(subject_to_indices[subj])
        return subset_idx

    train_subset = Subset(dataset, gather(train_subjects))
    val_subset = Subset(dataset, gather(val_subjects))
    test_subset = Subset(dataset, gather(test_subjects))
    return train_subset, val_subset, test_subset


def build_dataloaders(
    hdf5_path: str,
    batch_size: int = 16,
    val_split: float = 0.2,
    test_split: float = 0.1,
    shuffle: bool = True,
    target_field: str = "skill_levels",
    random_seed: int = 42,
    split_by_subject: bool = False,
    feature_indices: Optional[List[int]] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if val_split + test_split >= 1.0:
        raise ValueError("val_split + test_split must be < 1 when splitting by samples.")
    dataset = ChainBMDDataset(
        hdf5_path,
        target_field=target_field,
        feature_indices=feature_indices,
    )
    if split_by_subject:
        train_ds, val_ds, test_ds = _split_dataset_by_subject(
            dataset, val_split, test_split, random_seed
        )
    else:
        total = len(dataset)
        test_size = max(1, int(total * test_split))
        val_size = max(1, int(total * val_split))
        train_size = total - val_size - test_size
        if train_size <= 0:
            raise ValueError("Dataset too small for requested splits.")
        generator = torch.Generator().manual_seed(random_seed)
        train_ds, val_ds, test_ds = random_split(
            dataset, [train_size, val_size, test_size], generator=generator
        )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def build_kfold_dataloaders(
    hdf5_path: str,
    batch_size: int = 16,
    n_splits: int = 5,
    target_field: str = "skill_levels",
    random_seed: int = 42,
    split_by_subject: bool = False,
    feature_indices: Optional[List[int]] = None,
) -> List[Tuple[DataLoader, DataLoader]]:
    if n_splits < 2:
        raise ValueError("k-fold cross validation requires n_splits >= 2.")
    dataset = ChainBMDDataset(
        hdf5_path,
        target_field=target_field,
        feature_indices=feature_indices,
    )
    indices = np.arange(len(dataset))
    folds: List[Tuple[DataLoader, DataLoader]] = []
    if split_by_subject:
        groups = np.array(dataset.subjects)
        splitter = GroupKFold(n_splits=n_splits)
        split_iter = splitter.split(indices, groups=groups)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        split_iter = splitter.split(indices)

    for fold_idx, (train_idx, val_idx) in enumerate(split_iter):
        train_subset = Subset(dataset, train_idx.tolist())
        val_subset = Subset(dataset, val_idx.tolist())
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        folds.append((train_loader, val_loader))
    return folds
