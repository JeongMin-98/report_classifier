from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import torch
import os, re
import numpy as np
import random

from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
""" Check Device and Path for saving and loading """


def check_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def find_latest_ckpt(folder):
    """ find latest checkpoint """
    files = []
    for fname in os.listdir(folder):
        s = re.findall(r'\d+', fname)
        if len(s) == 1:
            files.append((int(s[0]), fname))
    if files:
        file = max(files)[1]
        file_name = os.path.splitext(file)[0]
        previous_iter = int(file_name.split("_")[1])
        return file, previous_iter
    else:
        return None, 0


""" Training Tool for model """


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def apply_gradients(loss, optim):
    optim.zero_grad()
    loss.backward()
    optim.step()


def infinite_iterator(loader):
    while True:
        for batch in loader:
            yield batch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def cross_entroy_loss(logit, label):
    loss = torch.nn.CrossEntropyLoss()(logit, label)
    return loss


def accuracy(outputs, label):
    """ if you want to make custom accuracy for your model, you need to implement this function."""
    y = torch.argmax(outputs, dim=1)
    return (y.eq(label).sum())


def reduce_loss(tmp):
    """ will implement reduce_loss func """
    loss = tmp
    return loss


# def reduce_loss_dict(loss_dict):
#     world_size = get_world_size()
#
#     if world_size < 2:
#         return loss_dict
#
#     with torch.no_grad():
#         keys = []
#         losses = []
#
#         for k in sorted(loss_dict.keys()):
#             keys.append(k)
#             losses.append(loss_dict[k])
#
#         losses = torch.stack(losses, 0)
#         dist.reduce(losses, dst=0)
#
#         if dist.get_rank() == 0:
#             losses /= world_size
#
#         reduced_losses = {k: v.mean().item() for k, v in zip(keys, losses)}
#
#     return reduced_losses

""" Tool to set for model by loading config files """


def read_config(config_path):
    """ read config file """
    file = open(config_path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]

    return lines


def parse_model_config(config_path):
    """ Parse your model of configuration files and set module defines"""
    lines = read_config(config_path)
    module_configs = []

    for line in lines:
        if line.startswith('['):
            layer_name = line[1:-1].rstrip()
            if layer_name == "net":
                continue
            module_configs.append({})
            module_configs[-1]['type'] = layer_name

            if module_configs[-1]['type'] == 'convolutional':
                module_configs[-1]['batch_normalize'] = 0
        else:
            if layer_name == "net":
                continue
            key, value = line.split("=")
            value = value.strip()
            if value.startswith('['):
                module_configs[-1][key.rstrip()] = list(map(int, value[1:-1].rstrip().split(',')))
            else:
                module_configs[-1][key.rstrip()] = value.strip()

    return module_configs


def show_img(img):
    """ Display an img"""
    img = np.array(img, dtype=np.uint8)
    img = Image.fromarray(img)
    img.show()


def data_transform(img_size):
    transform_list = [
        transforms.Resize(size=[img_size, img_size]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),  # [0, 255] -> [0, 1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),  # [0, 1] -> [-1, 1]
    ]
    return transforms.Compose(transform_list)


def mnist_transform():
    transforms_list = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ]
    return transforms.Compose(transforms_list)

def create_sampler(subset, dataset):
    """
    주어진 Subset에 대해 WeightedRandomSampler를 생성합니다.

    Args:
        subset (Subset): 데이터 Subset
        dataset (Dataset): 전체 Dataset

    Returns:
        WeightedRandomSampler: 클래스 균등성을 위한 샘플러
    """
    # Subset의 클래스 분포 계산
    class_counts = dataset.class_counts
    total_samples = sum(class_counts.values())

    # 클래스별 가중치 계산 (최대 가중치 제한)
    max_weight = 10.0
    class_weights = {cls: min(total_samples / count, max_weight) for cls, count in class_counts.items()}

    # 샘플별 가중치 생성
    sample_weights = []
    for idx in subset.indices:
        label = torch.argmax(torch.from_numpy(dataset[idx]['label'])).item()
        sample_weights.append(class_weights[dataset.idx_to_label[label]])

    # WeightedRandomSampler 생성
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float),
        num_samples=len(subset),
        replacement=True
    )
    return sampler

def create_sampler_parallel(subset, dataset, num_workers=4):
    """
    병렬 처리를 사용하여 WeightedRandomSampler를 생성합니다.

    Args:
        subset (Subset): 데이터 Subset.
        dataset (Dataset): 전체 Dataset.
        num_workers (int): 병렬 처리에 사용할 워커 수.

    Returns:
        WeightedRandomSampler: 클래스 균등성을 위한 샘플러.
    """
    # 클래스별 가중치 계산
    total_samples = sum(dataset.class_counts.values())
    max_weight = 10.0
    class_weights = {cls.lower(): min(total_samples / count, max_weight) for cls, count in dataset.class_counts.items()}

    def compute_sample_weight(idx):
        # Multi-label의 첫 번째 True 클래스를 기반으로 가중치를 계산
        label = torch.argmax(dataset[idx][1]).item()
        class_name = dataset.inverse_class_mapping[label]
        return class_weights[class_name]

    # 병렬로 샘플별 가중치 생성
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        sample_weights = list(executor.map(compute_sample_weight, subset.indices))

    # WeightedRandomSampler 생성
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float),
        num_samples=len(subset),
        replacement=True
    )
    return sampler

def show_class_distribution(subset, dataset):
    """
    Subset 내 클래스 분포를 출력합니다.

    Args:
        subset (Subset): 데이터 Subset
        dataset (Dataset): 전체 Dataset
    """
    labels = [
        dataset.idx_to_label[torch.argmax(torch.from_numpy(dataset[idx]["label"])).item()]
        for idx in subset.indices
    ]
    class_distribution = Counter(labels)
    print(f"Class Distribution in Subset: {class_distribution}")
    return class_distribution


def split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, batch_size=8, num_workers=4):
    """
    데이터셋을 train/val/test 비율로 분할하고, 각각의 DataLoader를 반환합니다.

    Args:
        dataset (Dataset): 전체 데이터셋.
        train_ratio (float): 훈련 데이터 비율 (0~1).
        val_ratio (float): 검증 데이터 비율 (0~1).
        test_ratio (float): 테스트 데이터 비율 (0~1).
        batch_size (int): 각 DataLoader의 배치 크기.
        num_workers (int): DataLoader에서 사용할 워커 수.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    assert train_ratio + val_ratio + test_ratio == 1.0, "train/val/test 비율의 합은 1이어야 합니다."

    # 전체 인덱스 생성 및 섞기
    indices = list(range(len(dataset)))
    train_idx, temp_idx = train_test_split(indices, test_size=(val_ratio + test_ratio), random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42)

    # Subset으로 데이터셋 분리
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)

    # 각 Subset의 클래스 분포 출력
    print("Train Set Distribution:")
    show_class_distribution(train_set, dataset)
    print("Validation Set Distribution:")
    show_class_distribution(val_set, dataset)
    print("Test Set Distribution:")
    show_class_distribution(test_set, dataset)

    # Sampler 생성
    train_sampler = create_sampler(train_set, dataset)
    val_sampler = create_sampler(val_set, dataset)
    test_sampler = create_sampler(test_set, dataset)

    # DataLoader 생성
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, sampler=test_sampler, num_workers=num_workers)

    return train_loader, val_loader, test_loader

def split_dataset_parallel(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, batch_size=8, num_workers=4, logger=None):
    """
    병렬 처리를 사용하여 데이터셋을 분할합니다.

    Args:
        dataset (Dataset): 전체 데이터셋.
        train_ratio (float): 훈련 데이터 비율 (0~1).
        val_ratio (float): 검증 데이터 비율 (0~1).
        test_ratio (float): 테스트 데이터 비율 (0~1).
        batch_size (int): 각 DataLoader의 배치 크기.
        num_workers (int): DataLoader에서 사용할 워커 수.
        logger (Logger): 로깅 객체.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    assert train_ratio + val_ratio + test_ratio == 1.0, "train/val/test 비율의 합은 1이어야 합니다."

    # 전체 인덱스 생성 및 섞기
    indices = list(range(len(dataset)))
    train_idx, temp_idx = train_test_split(indices, test_size=(val_ratio + test_ratio), random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42)

    # Subset으로 데이터셋 분리
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)

    # 각 Subset의 클래스 분포 출력
    if logger:
        print("Train Set Distribution:")
        show_class_distribution(train_set, dataset)
        print("Validation Set Distribution:")
        show_class_distribution(val_set, dataset)
        print("Test Set Distribution:")
        show_class_distribution(test_set, dataset)

    # Sampler 병렬 생성
    train_sampler = create_sampler_parallel(train_set, dataset, num_workers=num_workers)
    val_sampler = create_sampler_parallel(val_set, dataset, num_workers=num_workers)
    test_sampler = create_sampler_parallel(test_set, dataset, num_workers=num_workers)

    # DataLoader 생성
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, sampler=test_sampler, num_workers=num_workers)

    return train_loader, val_loader, test_loader

def balance_dataset(dataset, min_ratio=1.1, max_ratio=1.2):
    """
    Balance the dataset by ensuring all classes have data within a certain ratio of the smallest class.

    Args:
        dataset (DiagonsisTextDataset): The dataset to balance.
        min_ratio (float): Minimum scaling factor for class balancing.
        max_ratio (float): Maximum scaling factor for class balancing.

    Returns:
        list: A balanced list of dataset indices.
    """
    class_counts = Counter([record['class'] for record in dataset.data])
    min_class_count = min(class_counts.values())

    balanced_data = []
    for class_name, count in class_counts.items():
        class_data = [record for record in dataset.data if record['class'] == class_name]
        scale_factor = random.uniform(min_ratio, max_ratio) if count > min_class_count else 1
        target_count = int(min_class_count * scale_factor)
        balanced_data.extend(random.choices(class_data, k=target_count))

    dataset.data = balanced_data
    dataset.class_counts = Counter([record['class'] for record in dataset.data])
    return dataset