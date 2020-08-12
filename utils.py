import cv2
from cv2 import CascadeClassifier

import numpy as np

import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import Dataset, random_split


class FaceMaskDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.data = ImageFolder(root=root, transform=transforms)

    def __getitem__(self, index):
        sample = self.data.samples[index]
        try:
            image = cv2.imread(sample[0])
        except (IOError, SyntaxError) as e:
            print('Bad file:', sample[0])

        if self.transform:
            image = self.transform(image)

        return (image, sample[1])

    def __len__(self):
        return len(self.data)


def generate_datasets(train_path, test_path, train_val_ratio=0.2, transformations=None):
    train_val_data = FaceMaskDataset(root=train_path, transform=transformations)
    num_samples = train_val_data.__len__()
    num_samples_val = int(train_val_ratio * num_samples)
    num_samples_train = num_samples - num_samples_val
    train_data, val_data = random_split(train_val_data, [num_samples_train, num_samples_val])

    test_data = FaceMaskDataset(root=test_path, transform=transformations)

    return train_data, val_data, test_data


def train(model, train_loader, val_loader, loss, optimizer, epochs=5, verbose=1):
    print("=========== TRAINING ===========")
    model.train()
    train_losses = []
    train_acc = []
    val_losses = []
    val_acc = []
    for epoch in range(epochs + 1):
        model.train()
        total_loss_train = 0
        correct = 0
        for i, (x_train, y_train) in enumerate(train_loader):
            if i % 20 == 0:
                print('Train: [', i, '/', len(train_loader), ']')
            # clearing the Gradients of the model parameters
            optimizer.zero_grad()

            # prediction for training and validation set
            output_train = model(x_train)

            # computing the training and validation loss
            loss_train = loss(output_train, y_train)

            # computing the updated weights of all the model parameters
            loss_train.backward()
            optimizer.step()
            total_loss_train += loss_train.item()
            correct += (torch.argmax(output_train, axis=1) == y_train).float().sum()
        train_losses.append(total_loss_train/len(train_loader))
        train_acc.append(correct / (len(train_loader) * 8))

        model.eval()
        total_loss_val = 0
        correct = 0
        for i, (x_val, y_val) in enumerate(val_loader):
            if i % 5 == 0:
                print('Val: [', i, '/', len(val_loader), ']')
            # prediction for training and validation set
            output_val = model(x_val)

            # computing the validation loss
            loss_val = loss(output_val, y_val)
            total_loss_val += loss_val.item()
            correct += (torch.argmax(output_val, axis=1) == y_val).float().sum()
        val_losses.append(total_loss_val / len(val_loader))
        val_acc.append(correct / (len(val_loader) * 8))
        if verbose and epoch % 1 == 0:
            # printing the validation loss
            print('Epoch : ', epoch + 1, '\t', 'Train Loss :', train_losses[-1], '\t', 'Val Loss: ', val_losses[-1],
                  '\t', 'Train Accuracy: ', train_acc[-1].item()*100, '%\t', 'Val Accuracy: ', val_acc[-1].item()*100, '%')

    torch.save(model.state_dict(), "weights")

    return train_losses, val_losses, train_acc, val_acc


def evaluate(model, test_loader):
    print("=========== TESTING ===========")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            images = np.transpose(images[0].numpy(), (1, 2, 0))
            cv2.imshow('image', images)
            cv2.waitKey(1000)
    accuracy = round(100 * correct / total, 3)
    print('Accuracy of the network on the 194 test images: ' +
          str(accuracy))

    return accuracy


def classify(model, cascade_classifier_path_xml, transformations=None):
    cap = cv2.VideoCapture(0)
    face_cascade = CascadeClassifier(cascade_classifier_path_xml)

    while True:
        ret, img_bgr = cap.read()
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 2)

        for (x, y, w, h) in faces:
            w += 20
            h += 20
            r = max(w, h) / 2
            centerx = x + w / 2
            centery = y + h / 2
            nx = int(centerx - r)
            ny = int(centery - r)
            nr = int(r * 2)

            faceimg_bgr = img_bgr[ny:ny + nr, nx:nx + nr]
            predimg = transformations(faceimg_bgr)
            predimg = np.expand_dims(predimg.numpy(), axis=0)
            outputs = model(torch.Tensor(predimg))
            _, prediction = torch.max(outputs.data, 1)
            prediction = prediction.item()
            if prediction == 1:
                img_bgr = cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(img_bgr, 'No Mask', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            elif prediction == 0:
                img_bgr = cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img_bgr, 'Mask', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.namedWindow("img_1")
        cv2.imshow('img_1', img_bgr)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()