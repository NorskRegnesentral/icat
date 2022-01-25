import os

from PIL import Image
from torchvision.transforms import transforms
from SimCLR.data_aug.gaussian_blur import GaussianBlur
from torchvision.datasets import VisionDataset

#################### INSTRUCTION #######################
# 1: Change path to a folder with your images

PATH = '/lokal-uten-backup/pro/iari/tmp_dataset_storage/ssl/sleeper'
DATASET_NAME = 'fasteners'

# 2: Set up a transform that changes your images as much at possible without changing the properties of them you are interested in
transform_at_training = transforms.Compose([
    transforms.CenterCrop(size=350),
    transforms.Resize(32*4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomApply([
        transforms.ColorJitter(0.8, 0.8, 0.8, 0.2 )],
        p=0.8),
    GaussianBlur(kernel_size=int(0.1 * 32)),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor()])

# 3: Ttransform for prediction, typically the deterministic parts of transform_at_training
transform_at_prediction = transforms.Compose([
    transforms.CenterCrop(size=350),
    transforms.Resize(32*4),
    transforms.ToTensor()])

########################################################

class SSLDataset():

    def __init__(self, transform):
        self.transform = transform
        self.files = []
        self.name = DATASET_NAME

        for file in os.listdir(PATH):
            if file.endswith('.jpeg') and not file.startswith('.'):
                self.files.append(os.path.join(PATH, file))
        self.files = sorted(self.files)


    def __getitem__(self, index):

        self.last_index = index
        file = self.files[index]
        img = Image.open(file)
        img = self.transform(img)

        return img, index

    def __len__(self):
        return len(self.files)