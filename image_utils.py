import os
import json
import numpy as np
import torch
from torchvision import datasets, transforms, models
from PIL import Image
from model_utils import Phases


def categories_map(cat_file):
    with open(cat_file, 'r') as f:
        cat_to_name = json.load(f)

    return cat_to_name


def process_image(image):
    """
    Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    """
    img = Image.open(image)
    image_transforms = ImageTransforms()

    transform = image_transforms.get_validation_test_transform()

    return np.array(transform(img))


class ImageTransforms:
    def __init__(self):
        self.rotation_angle = 30
        self.crop_box_size = 224
        self.image_size = 256
        self.norm_means = [0.485, 0.456, 0.406]
        self.norm_std = [0.229, 0.224, 0.225]

        self.train_transform = None
        self.validation_test_transform = None
        self.normalize_transform = None

    def get_train_transforms(self, force_create=False):
        if self.train_transform and not force_create:
            return self.train_transform

        self.train_transform = transforms.Compose([transforms.RandomRotation(self.rotation_angle),
                                                   transforms.RandomResizedCrop(self.crop_box_size),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor(),
                                                   self._normalize_transform(force_create)])
        return self.train_transform

    def get_validation_test_transform(self, force_create=False):
        if self.validation_test_transform and not force_create:
            return self.validation_test_transform

        self.validation_test_transform = transforms.Compose([transforms.Resize(self.image_size),
                                                             transforms.CenterCrop(self.crop_box_size),
                                                             transforms.ToTensor(),
                                                             self._normalize_transform(force_create)])
        return self.validation_test_transform

    def _normalize_transform(self, force_create=False):
        if self.normalize_transform and not force_create:
            return self.normalize_transform

        self.normalize_transform = transforms.Normalize(self.norm_means, self.norm_std)

        return self.normalize_transform


class ImageUtils:
    def __init__(self, data_directory):
        self.data_directory = data_directory
        self.data_sets = {}
        self.image_transforms = ImageTransforms()
        self.phases = Phases()

    def _guard_against_invalid_phase(self, phase):
        if phase not in self.phases():
            raise ValueError('{} is not a valid phase'.format(phase))

    def data_transforms(self, phase=None):
        if phase:
            self._guard_against_invalid_phase(phase)

        if phase == Phases.TRAIN_PHASE:
            return self.image_transforms.get_train_transforms()

        if phase == Phases.TEST_PHASE or phase == Phases.VALIDATION_PHASE:
            return self.image_transforms.get_validation_test_transform()

        return {
            Phases.TRAIN_PHASE: self.image_transforms.get_train_transforms(),
            Phases.VALIDATION_PHASE: self.image_transforms.get_validation_test_transform(),
            Phases.TEST_PHASE: self.image_transforms.get_validation_test_transform()
        }

    def phase_data_dir(self, phase):
        self._guard_against_invalid_phase(phase)

        phase_data_dir = 'valid' if phase == Phases.VALIDATION_PHASE else phase
        directory = '{}/{}'.format(self.data_directory, phase_data_dir)

        if not os.path.isdir(directory):
            raise NotADirectoryError('{} is not a directory'.format(directory))

        return directory

    def create_data_loaders(self, batch_size=32):
        data_sets = self.create_image_datasets()

        return {phase: torch.utils.data.DataLoader(data_sets[phase],
                                                   batch_size=batch_size,
                                                   shuffle=phase == Phases.TRAIN_PHASE)
                for phase in self.phases()}

    def train_dataset(self):
        return self.create_image_datasets(Phases.TRAIN_PHASE)

    def create_image_datasets(self, phase=None):
        transforms = self.data_transforms()

        if self.data_sets:
            return self.data_sets if not phase else self.data_sets[phase]

        self.data_sets = {phase: datasets.ImageFolder(self.phase_data_dir(phase), transform=transforms[phase])
                          for phase in self.phases()}

        if not phase:
            return self.data_sets

        self._guard_against_invalid_phase(phase)

        return self.data_sets[phase]
