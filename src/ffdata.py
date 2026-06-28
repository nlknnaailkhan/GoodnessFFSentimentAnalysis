import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from src.utils import seed_worker

class FF_Data(Dataset):
    def __init__(self, images_path, labels_path, num_classes=10):
        self.images = np.load(images_path)
        self.labels = np.load(labels_path)
        self.num_classes = num_classes
        self.uniform_label = torch.ones(self.num_classes) / self.num_classes

    def __getitem__(self, index):
        pos_sample, neg_sample, neutral_sample, class_label = self._generate_sample(index)
        inputs = {
            "pos_images": pos_sample,
            "neg_images": neg_sample,
            "neutral_sample": neutral_sample,
        }
        labels = {"class_labels": class_label}
        return inputs, labels

    def __len__(self):
        return len(self.images)

    def _get_pos_sample(self, sample, class_label):
        one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(class_label), num_classes=self.num_classes
        ).float()
        return torch.cat([sample, one_hot_label], dim=-1)

    def _get_neg_sample(self, sample, class_label):
        classes = list(range(self.num_classes))
        classes.remove(class_label)
        wrong_class_label = np.random.choice(classes)
        one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(wrong_class_label), num_classes=self.num_classes
        ).float()
        return torch.cat([sample, one_hot_label], dim=-1)

    def _get_neutral_sample(self, sample):
        return torch.cat([sample, self.uniform_label], dim=-1)

    def _generate_sample(self, index):
        sample = torch.tensor(self.images[index], dtype=torch.float32)
        if sample.max() > 1.0:
            sample = sample / 255.0
        class_label = int(self.labels[index])
        sample = sample.flatten()

        pos_sample = self._get_pos_sample(sample, class_label)
        neg_sample = self._get_neg_sample(sample, class_label)
        neutral_sample = self._get_neutral_sample(sample)

        return pos_sample, neg_sample, neutral_sample, class_label

def get_data(opt, partition):
    images_path = os.path.join(os.getcwd(), opt.input.path, f"{partition}_images.npy")
    labels_path = os.path.join(os.getcwd(), opt.input.path, f"{partition}_labels.npy")

    dataset = FF_Data(images_path=images_path, labels_path=labels_path, num_classes=opt.input.num_classes)

    if partition == "train":
        permutation = np.random.permutation(len(dataset.images))
        dataset.images = dataset.images[permutation]
        dataset.labels = dataset.labels[permutation]

    g = torch.Generator()
    g.manual_seed(opt.seed)

    return DataLoader(
        dataset,
        batch_size=opt.input.batch_size,
        drop_last=True,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
        num_workers=2,
        persistent_workers=True,
    )
