import matplotlib.pyplot as plt

import torch
from torchvision.models import resnet18
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

from utils import classify, evaluate, generate_datasets, train


if __name__ == '__main__':
    transformations = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    """
    train_data, val_data, test_data = generate_datasets(
        train_path="P:/face-mask-dataset/face-mask-dataset/train",
        test_path="P:/face-mask-dataset/face-mask-dataset/test",
        train_val_ratio=0.2,
        transformations=transformations
    )
    train_loader = DataLoader(dataset=train_data, batch_size=8, shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset=val_data, batch_size=8, shuffle=True, drop_last=True)

    model = resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = torch.nn.Linear(512, 2)

    loss = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.0005)

    train_losses, val_losses, train_acc, val_acc = train(
        model=model, train_loader=train_loader, val_loader=val_loader, loss=loss, optimizer=optimizer, epochs=5
    )

    plt.figure()
    plt.subplot(121)
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.subplot(122)
    plt.plot(train_acc)
    plt.plot(val_acc)
    plt.show()
    """

    model = resnet18(pretrained=False)
    model.fc = torch.nn.Linear(512, 2)
    model.load_state_dict(torch.load("weights"))
    model.eval()

    """
    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=True, drop_last=True)
    test_acc = evaluate(model, test_loader)
    """

    classify(
        model=model,
        cascade_classifier_path_xml='C:/Users/msure/Anaconda3/pkgs/libopencv-4.4.0-py37_2/Library/etc/haarcascades/haarcascade_frontalface_default.xml',
        transformations=transformations
    )



