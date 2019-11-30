import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import pandas as pd
import torch
from PIL import Image
from torch.autograd import Variable
import torchvision.transforms as transforms
import train_model

NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
            'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
ALPHABET_ = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
             'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

ALL_CHAR_SET = NUMBER + ALPHABET + ALPHABET_
ALL_CHAR_SET_LEN = len(ALL_CHAR_SET)
MAX_CAPTCHA = 4

IMAGE_HEIGHT = 40
IMAGE_WIDTH = 120


def encode(text):
    vector = np.zeros(ALL_CHAR_SET_LEN * MAX_CAPTCHA, dtype=float)

    def char2pos(c):
        if c == '_':
            k = 62
            return k
        k = ord(c)-48
        if k > 9:
            k = ord(c) - 65 + 10
            if k > 35:
                k = ord(c) - 97 + 26 + 10
                if k > 61:
                    raise ValueError('error')
        return k

    for i, c in enumerate(text):
        idx = i * ALL_CHAR_SET_LEN + char2pos(c)
        vector[idx] = 1
    return vector


def decode(vec):
    char_pos = vec.nonzero()[0]
    text = []
    for i, c in enumerate(char_pos):
        char_at_pos = i  # c/63
        char_idx = c % ALL_CHAR_SET_LEN
        if char_idx < 10:
            char_code = char_idx + ord('0')
        elif char_idx < 36:
            char_code = char_idx - 10 + ord('A')
        elif char_idx < 62:
            char_code = char_idx - 36 + ord('a')
        elif char_idx == 62:
            char_code = ord('_')
        else:
            raise ValueError('error')
        text.append(chr(char_code))
    return "".join(text)


class mydataset(Dataset):

    def __init__(self, paths, folder, transform=None):
        self.train_image_file_paths = [folder+image_P for image_P in paths]
        self.transform = transform

    def __len__(self):
        return len(self.train_image_file_paths)

    def __getitem__(self, idx):
        image_root = self.train_image_file_paths[idx]
        image = Image.open(image_root)

        if self.transform is not None:
            image = self.transform(image)
        return image


class mytraindataset(Dataset):

    def __init__(self, path, transform=None):
        data = pd.read_csv(path)
        image_Paths = np.array(data['ID'])
        self.labels = np.array(data['label'])
        folder = path.split('/')[0]
        self.train_image_file_paths = [os.path.join(
            folder, image_P) for image_P in image_Paths]
        self.transform = transform

    def __len__(self):
        return len(self.train_image_file_paths)

    def __getitem__(self, idx):
        image_root = self.train_image_file_paths[idx]
        image = Image.open(image_root)
        # 之前的尝试
        #image = cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR)
        #image = cv2.medianBlur(image,3)
        #image = cv2.GaussianBlur(image, (13,13), 0)
        #image = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        if self.transform is not None:
            image = self.transform(image)
        label = encode(self.labels[idx])
        return image, label


def train_data_loader(batch_size, path):

    dataset = mytraindataset(path, transform=transform_train)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)


def test_data_loader(path):

    dataset = mytraindataset(path, transform=transform_test)
    return DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=True)


def predict_data_loader(paths, folder):

    dataset = mydataset(paths, folder, transform=transform_test)
    return DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)


transform_train = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
])


def test_data(model):
    model.eval()
    test_dataloader = test_data_loader()

    correct = 0
    total = 0

    for i, (images, labels) in enumerate(test_dataloader):

        image = images
        vimage = Variable(image).cuda()
        predict_label = model(vimage).cuda()

        c0 = ALL_CHAR_SET[np.argmax(
            predict_label[0, 0:ALL_CHAR_SET_LEN].data.cpu().numpy())]
        c1 = ALL_CHAR_SET[np.argmax(
            predict_label[0, ALL_CHAR_SET_LEN:2 * ALL_CHAR_SET_LEN].data.cpu().numpy())]
        c2 = ALL_CHAR_SET[np.argmax(
            predict_label[0, 2 * ALL_CHAR_SET_LEN:3 * ALL_CHAR_SET_LEN].data.cpu().numpy())]
        c3 = ALL_CHAR_SET[np.argmax(
            predict_label[0, 3 * ALL_CHAR_SET_LEN:4 * ALL_CHAR_SET_LEN].data.cpu().numpy())]
        predict_label = '%s%s%s%s' % (c0, c1, c2, c3)
        true_label = decode(labels.numpy()[0])
        total += labels.size(0)
        if(predict_label == true_label):
            correct += 1

    return str(100 * correct / total)


def test(model, predictD):
    predict_labels = []
    for i, images in enumerate(predictD):
        image = images
        vimage = Variable(image)
        predict_label = model(vimage)

        c0 = ALL_CHAR_SET[np.argmax(
            predict_label[0, 0:ALL_CHAR_SET_LEN].data.numpy())]
        c1 = ALL_CHAR_SET[np.argmax(
            predict_label[0, ALL_CHAR_SET_LEN:2 * ALL_CHAR_SET_LEN].data.numpy())]
        c2 = ALL_CHAR_SET[np.argmax(
            predict_label[0, 2 * ALL_CHAR_SET_LEN:3 * ALL_CHAR_SET_LEN].data.numpy())]
        c3 = ALL_CHAR_SET[np.argmax(
            predict_label[0, 3 * ALL_CHAR_SET_LEN:4 * ALL_CHAR_SET_LEN].data.numpy())]
        c = '%s%s%s%s' % (c0, c1, c2, c3)
        predict_labels.append(c)
    return predict_labels


def testlist(testpath):
    return os.listdir(testpath)


def model(testpath):

    loadfile = "CNN_14_layers_epoch2_loss0.000731_acc_99.9089068825911.pkl"
    model = train_model.CNN(num_classes=MAX_CAPTCHA*ALL_CHAR_SET_LEN)
    model.eval()

    model.load_state_dict(torch.load(
        './model/{}'.format(loadfile), map_location='cpu'))

    ids = testlist(testpath)
    predict_dataloader = predict_data_loader(ids, testpath)
    predict_labels = test(model, predict_dataloader)
    df = pd.DataFrame([ids, predict_labels]).T
    df.columns = ['ID', 'label']
    return df
