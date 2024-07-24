import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from scipy.stats import entropy
from torchvision.models import inception_v3
from tqdm import tqdm


class ImageDataset_IQ_IV_new:
    def __init__(self, path, n_particles, num_img):
        self.path = path
        self.iter = 0
        self.num_img = num_img
        self.n_particles = n_particles
        self.transform = torchvision.transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((299, 299)),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        self.image_path = glob.glob(f"{self.path}/*.png")

    def __len__(self):
        return self.num_img * self.n_particles

    def __getitem__(self, idx):
        get_img = torch.stack([self.transform(Image.open(self.image_path[idx]))])
        return get_img


class ImageDataset_IQ_IV_old:
    def __init__(self, path, n_particles, num_img):
        self.path = path
        self.iter = 0
        self.num_img = num_img
        self.n_particles = n_particles
        self.transform = torchvision.transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((299, 299)),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def __len__(self):
        return self.num_img

    def __getitem__(self, idx):
        get_img = torch.stack(
            [
                self.transform(Image.open(os.path.join(self.path, f"{self.iter}-particle-{i}.png")))
                for i in range(self.n_particles)
            ]
        )
        self.iter += 1
        return get_img


def inception_score_IQ(imgs, device="cuda:0", batch_size=1, splits=1):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)
    assert N >= batch_size

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device=device)
    inception_model.eval()

    def get_pred(x):
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions

    IQ_score = 0
    IV_score = 0
    n_samples = 0

    for _, batch in enumerate(tqdm(dataloader)):
        batch = batch.to(device=device).squeeze(0)
        predicts = get_pred(batch)
        score = entropy(predicts, axis=1)
        IQ_score += score
        IV_score += predicts
        n_samples += 1
    IV_score = entropy(np.mean(IV_score, axis=0))
    IQ_score = IQ_score / n_samples
    # return IQ_score
    return IQ_score, IV_score


def inception_score_IV(imgs, device="cuda:0", batch_size=1, splits=1):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)
    assert N >= batch_size

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device=device)
    inception_model.eval()

    def get_pred(x):
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions

    IQ_score = 0
    IV_score = 0
    n_samples = 0

    for _, batch in enumerate(tqdm(dataloader)):
        batch = batch.to(device=device).squeeze(0)
        predicts = get_pred(batch)
        score = entropy(predicts, axis=1)
        IQ_score += np.mean(score)
        IV_score += entropy(np.mean(predicts, axis=0))
        n_samples += 1
    return IQ_score / n_samples, IV_score / n_samples


def forward_modality(model, data, flag):
    device = next(model.parameters()).device
    if flag == "img":
        features = model.encode_image(data.to(device))
    elif flag == "txt":
        features = model.encode_text(data.to(device))
    else:
        raise TypeError
    return features


@torch.no_grad()
def calculate_clip_score(dataloader, model):
    score_acc = 0.0
    sample_num = 0.0
    for batch_data in tqdm(dataloader):
        real = batch_data["real"]
        real_features = forward_modality(model, real, "img")
        fake = batch_data["fake"]
        fake_features = forward_modality(model, fake, "txt")

        # normalize features
        real_features = real_features / real_features.norm(dim=1, keepdim=True).to(torch.float32)
        fake_features = fake_features / fake_features.norm(dim=1, keepdim=True).to(torch.float32)

        # calculate scores
        score = 1 - 1 * (fake_features * real_features).sum()
        score_acc += score
        sample_num += real.shape[0]
    return score_acc / sample_num


class DummyDataset:
    FLAGS = ["img", "txt"]

    def __init__(self, real_path, prompt="", transform=None, tokenizer=None) -> None:

        super().__init__()
        self.prompt = prompt
        self.real_folder = self._combine_without_prefix(real_path)
        self.transform = transform
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.real_folder)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError
        real_path = self.real_folder[index]
        real_data = self._load_img(real_path)
        if self.tokenizer is not None:
            data = self.tokenizer(self.prompt).squeeze()
        fake_data = data
        sample = dict(real=real_data, fake=fake_data)
        return sample

    def _load_img(self, path):
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def _combine_without_prefix(self, folder_path, prefix="."):
        folder = []
        for name in os.listdir(folder_path):
            if name[0] == prefix:
                continue
            folder.append(os.path.join(folder_path, name))
        folder.sort()
        return folder


@torch.no_grad()
def extract_dino_features(dataset, batch_size):
    # ============ preparing data ... ============
    vitb8 = torch.hub.load("facebookresearch/dino:main", "dino_vitb8").cuda().eval()
    data_loader_train = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
    )

    mean_cosine = []
    n_samples = 0

    for _, batch in enumerate(tqdm(data_loader_train)):
        batch = batch.cuda(non_blocking=True).squeeze(0)
        feats = vitb8(batch).clone()  # 8, 768
        mean_cosine_view = []
        for i in range(feats.shape[0]):
            epsilon = 0.000001
            feat1 = feats[i] / (torch.norm(feats[i]) + epsilon)
            for j in range(i + 1, feats.shape[0]):
                feat2 = feats[j] / (torch.norm(feats[j]) + epsilon)
                cosine_sim = torch.sum(feat1 * feat2)
                mean_cosine_view.append(cosine_sim.detach().cpu().numpy())

        mean_cosine.append(np.mean(mean_cosine_view))
        n_samples += 1

    return np.mean(mean_cosine)
