import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"

image_transform = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

mask_transform = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(128),
])


class PetsDataset(Dataset):
    def __init__(self, dataset_path):
        self.image_paths = sorted(list(Path(dataset_path / "images").iterdir()))
        self.map_paths = sorted(list(Path(dataset_path / "trimaps").iterdir()))

        self.species_names = ["dog", "cat"]
        self.species_classes = [int(image_path.name[0].isupper()) for image_path in self.image_paths]

        self.breed_names = [p.name.rsplit("_", 1)[0] for p in self.image_paths]
        self.cat_breed_names = [
            'Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair', 'Egyptian_Mau',
            'Maine_Coon', 'Persian', 'Ragdoll', 'Russian_Blue', 'Siamese', 'Sphynx'
        ]
        self.dog_breed_names = [
            'american_bulldog', 'american_pit_bull_terrier', 'basset_hound', 'beagle', 'boxer',
            'chihuahua', 'english_cocker_spaniel', 'english_setter', 'german_shorthaired',
            'great_pyrenees', 'havanese', 'japanese_chin', 'keeshond', 'leonberger',
            'miniature_pinscher', 'newfoundland', 'pomeranian', 'pug', 'saint_bernard',
            'samoyed', 'scottish_terrier', 'shiba_inu', 'staffordshire_bull_terrier',
            'wheaten_terrier', 'yorkshire_terrier'
        ]
        self.breed_list = self.cat_breed_names + self.dog_breed_names
        self.breed_idx2name = {
            0: 'american_bulldog', 1: 'basset_hound', 2: 'beagle', 3: 'boxer', 4: 'chihuahua',
            5: 'english_cocker_spaniel', 6: 'german_shorthaired', 7: 'great_pyrenees', 8: 'havanese',
            9: 'japanese_chin', 10: 'keeshond', 11: 'leonberger', 12: 'miniature_pinscher',
            13: 'newfoundland', 14: 'pomeranian', 15: 'pug', 16: 'saint_bernard', 17: 'samoyed',
            18: 'scottish_terrier', 19: 'shiba_inu', 20: 'staffordshire_bull_terrier',
            21: 'wheaten_terrier', 22: 'yorkshire_terrier', 23: 'american_pit_bull_terrier',
            24: 'english_setter', 25: 'Abyssinian', 26: 'Bengal', 27: 'Birman', 28: 'Bombay',
            29: 'British_Shorthair', 30: 'Egyptian_Mau', 31: 'Maine_Coon', 32: 'Persian',
            33: 'Ragdoll', 34: 'Russian_Blue', 35: 'Siamese', 36: 'Sphynx'
        }
        self.breed_name2idx = {n: i for i, n in self.breed_idx2name.items()}
        self.cat_breed_indices = [self.breed_name2idx[b] for b in self.cat_breed_names]
        self.dog_breed_indices = [self.breed_name2idx[b] for b in self.dog_breed_names]
        self.breed_classes = [self.breed_name2idx[b] for b in self.breed_names]

        assert len(self.image_paths) == len(self.species_classes), \
            f"Number of images and species_classes do not match: {len(self.image_paths)} != {len(self.species_classes)}"
        assert len(self.image_paths) == len(self.breed_classes), \
            f"Number of images and breeds do not match: {len(self.image_paths)} != {len(self.breed_classes)}"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = image_transform(image)
        mask = Image.open(self.map_paths[idx])
        mask = mask_transform(mask)
        mask = torch.tensor(np.array(mask)).long() - 1
        species_tensor = torch.tensor(self.species_classes[idx])
        breed_tensor = torch.tensor(self.breed_classes[idx])
        return image, species_tensor, breed_tensor, mask


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.channels = [32, 64, 128, 256, 512, 1024]
        self.num_convs = [1, 2, 3, 4, 5]

        # Encoder Layers
        self.encoder_convs = nn.ModuleList([
            DoubleConv(3 if i == 0 else self.channels[i - 1], self.channels[i]) for i in range(6)
        ])
        self.pools = nn.ModuleList([nn.MaxPool2d(2) for _ in range(5)])

        # Decoder Layers
        self.ups = nn.ModuleList([
            nn.ConvTranspose2d(self.channels[i + 1], self.channels[i], kernel_size=2, stride=2)
            for i in range(4, -1, -1)
        ])
        self.decoder_convs = nn.ModuleList([
            nn.ModuleList([
                DoubleConv((k + 2) * self.channels[4 - level], self.channels[4 - level])
                for k in range(self.num_convs[level])
            ])
            for level in range(5) 
        ])

        # Segmentation Output
        self.seg_final = nn.Conv2d(32, 3, kernel_size=1)

        # Classifiers
        self.species_classifier = nn.Sequential(
            nn.Linear(1024 * 4 * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )
        self.breed_classifier = nn.Sequential(
            nn.Linear(1024 * 4 * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 37)
        )

        # Breed and species metadata (unchanged)
        self.species_names = ['dog', 'cat']
        self.cat_breed_names = [
            'Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair', 'Egyptian_Mau',
            'Maine_Coon', 'Persian', 'Ragdoll', 'Russian_Blue', 'Siamese', 'Sphynx'
        ]
        self.dog_breed_names = [
            'american_bulldog', 'american_pit_bull_terrier', 'basset_hound', 'beagle', 'boxer',
            'chihuahua', 'english_cocker_spaniel', 'english_setter', 'german_shorthaired',
            'great_pyrenees', 'havanese', 'japanese_chin', 'keeshond', 'leonberger',
            'miniature_pinscher', 'newfoundland', 'pomeranian', 'pug', 'saint_bernard',
            'samoyed', 'scottish_terrier', 'shiba_inu', 'staffordshire_bull_terrier',
            'wheaten_terrier', 'yorkshire_terrier'
        ]
        self.breed_list = self.cat_breed_names + self.dog_breed_names
        self.breed_idx2name = {
            0: 'american_bulldog', 1: 'basset_hound', 2: 'beagle', 3: 'boxer', 4: 'chihuahua',
            5: 'english_cocker_spaniel', 6: 'german_shorthaired', 7: 'great_pyrenees', 8: 'havanese',
            9: 'japanese_chin', 10: 'keeshond', 11: 'leonberger', 12: 'miniature_pinscher',
            13: 'newfoundland', 14: 'pomeranian', 15: 'pug', 16: 'saint_bernard', 17: 'samoyed',
            18: 'scottish_terrier', 19: 'shiba_inu', 20: 'staffordshire_bull_terrier',
            21: 'wheaten_terrier', 22: 'yorkshire_terrier', 23: 'american_pit_bull_terrier',
            24: 'english_setter', 25: 'Abyssinian', 26: 'Bengal', 27: 'Birman', 28: 'Bombay',
            29: 'British_Shorthair', 30: 'Egyptian_Mau', 31: 'Maine_Coon', 32: 'Persian',
            33: 'Ragdoll', 34: 'Russian_Blue', 35: 'Siamese', 36: 'Sphynx'
        }
        self.breed_name2idx = {n: i for i, n in self.breed_idx2name.items()}
        self.cat_breed_indices = [self.breed_name2idx[b] for b in self.cat_breed_names]
        self.dog_breed_indices = [self.breed_name2idx[b] for b in self.dog_breed_names]

    def forward(self, x):
        # Encoder Pass
        encoder_features = []
        for i in range(6):
            x = self.encoder_convs[i](x)
            encoder_features.append(x)
            if i < 5:
                x = self.pools[i](x)

        # Classification
        flat = torch.flatten(encoder_features[5], start_dim=1)  # Flatten x5_0
        species_pred = self.species_classifier(flat)
        breed_pred = self.breed_classifier(flat)

        # Decoder Pass
        decoder_features = [[] for _ in range(5)] 

        # Level 4
        up = self.ups[0](encoder_features[5])  # up1_0(x5_0)
        decoder_features[0].append(self.decoder_convs[0][0](torch.cat([up, encoder_features[4]], dim=1)))  # x4_1

        # Level 3
        up = self.ups[1](decoder_features[0][0])  # up2_0(x4_1)
        decoder_features[1].append(self.decoder_convs[1][0](torch.cat([up, encoder_features[3]], dim=1)))  # x3_1
        decoder_features[1].append(self.decoder_convs[1][1](torch.cat([up, encoder_features[3], decoder_features[1][0]], dim=1)))  # x3_2

        # Level 2
        up1 = self.ups[2](decoder_features[1][0])  # up3_0(x3_1)
        decoder_features[2].append(self.decoder_convs[2][0](torch.cat([up1, encoder_features[2]], dim=1)))  # x2_1
        decoder_features[2].append(self.decoder_convs[2][1](torch.cat([up1, encoder_features[2], decoder_features[2][0]], dim=1)))  # x2_2
        up2 = self.ups[2](decoder_features[1][1])  # up3_0(x3_2)
        decoder_features[2].append(self.decoder_convs[2][2](torch.cat([up2, encoder_features[2], decoder_features[2][0], decoder_features[2][1]], dim=1)))  # x2_3

        # Level 1
        up1 = self.ups[3](decoder_features[2][0])  # up4_0(x2_1)
        decoder_features[3].append(self.decoder_convs[3][0](torch.cat([up1, encoder_features[1]], dim=1)))  # x1_1
        decoder_features[3].append(self.decoder_convs[3][1](torch.cat([up1, encoder_features[1], decoder_features[3][0]], dim=1)))  # x1_2
        up2 = self.ups[3](decoder_features[2][1])  # up4_0(x2_2)
        decoder_features[3].append(self.decoder_convs[3][2](torch.cat([up2, encoder_features[1], decoder_features[3][0], decoder_features[3][1]], dim=1)))  # x1_3
        up3 = self.ups[3](decoder_features[2][2])  # up4_0(x2_3)
        decoder_features[3].append(self.decoder_convs[3][3](torch.cat([up3, encoder_features[1], decoder_features[3][0], decoder_features[3][1], decoder_features[3][2]], dim=1)))  # x1_4

        # Level 0
        up1 = self.ups[4](decoder_features[3][0])  # up5_0(x1_1)
        decoder_features[4].append(self.decoder_convs[4][0](torch.cat([up1, encoder_features[0]], dim=1)))  # x0_1
        decoder_features[4].append(self.decoder_convs[4][1](torch.cat([up1, encoder_features[0], decoder_features[4][0]], dim=1)))  # x0_2
        up2 = self.ups[4](decoder_features[3][1])  # up5_0(x1_2)
        decoder_features[4].append(self.decoder_convs[4][2](torch.cat([up2, encoder_features[0], decoder_features[4][0], decoder_features[4][1]], dim=1)))  # x0_3
        up3 = self.ups[4](decoder_features[3][2])  # up5_0(x1_3)
        decoder_features[4].append(self.decoder_convs[4][3](torch.cat([up3, encoder_features[0], decoder_features[4][0], decoder_features[4][1], decoder_features[4][2]], dim=1)))  # x0_4
        up4 = self.ups[4](decoder_features[3][3])  # up5_0(x1_4)
        decoder_features[4].append(self.decoder_convs[4][4](torch.cat([up4, encoder_features[0], decoder_features[4][0], decoder_features[4][1], decoder_features[4][2], decoder_features[4][3]], dim=1)))  # x0_5

        # Segmentation output
        seg_pred = self.seg_final(decoder_features[4][-1])

        return species_pred, breed_pred, seg_pred

    def predict(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.to(device)
        self.eval()
        with torch.no_grad():
            species_pred, breed_pred, seg_pred = self.forward(x)
            species_idx = torch.argmax(species_pred, dim=1).item()
            species_pred = self.species_names[species_idx]
            breed_indices = self.dog_breed_indices if species_pred == "dog" else self.cat_breed_indices
            filtered_breed_logits = breed_pred[:, breed_indices]
            top3_indices = torch.topk(filtered_breed_logits, 3, dim=1).indices.squeeze(0).tolist()
            top3_breeds = [self.breed_idx2name[breed_indices[idx]] for idx in top3_indices]
            seg_pred = torch.argmax(seg_pred, dim=1).cpu()
            return species_pred, tuple(top3_breeds), seg_pred
