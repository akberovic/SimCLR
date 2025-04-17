import os
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, classification_report

from simclr import SimCLR
from simclr.modules import LogisticRegression, get_resnet
from simclr.modules.transformations import TransformsSimCLR

from utils import yaml_config_hook

def get_test_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize(256),             # Resize so that the shorter side is 256.
        transforms.CenterCrop(image_size),  # Center crop to image_size x image_size.
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def inference(loader, simclr_model, device):
    feature_vector = []
    labels_vector = []
    for step, (x, y) in enumerate(loader):
        x = x.to(device)
        # Get encoding from the pretrained model.
        # We pass the same image twice (x, x) since the model expects a pair,
        # but only one view is needed for feature extraction.
        with torch.no_grad():
            h, _, z, _ = simclr_model(x, x)
        h = h.detach()
        feature_vector.extend(h.cpu().numpy())
        labels_vector.extend(y.numpy())
        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")
    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector

def get_features(simclr_model, train_loader, test_loader, device):
    train_X, train_y = inference(train_loader, simclr_model, device)
    test_X, test_y = inference(test_loader, simclr_model, device)
    return train_X, train_y, test_X, test_y

def create_data_loaders_from_arrays(X_train, y_train, X_test, y_test, batch_size):
    train = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    test = torch.utils.data.TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def train_epoch(args, loader, model, criterion, optimizer):
    model.train()
    loss_epoch = 0
    accuracy_epoch = 0
    for step, (x, y) in enumerate(loader):
        optimizer.zero_grad()
        x = x.to(args.device)
        y = y.to(args.device)
        output = model(x)
        loss = criterion(output, y)
        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc
        loss.backward()
        optimizer.step()
        loss_epoch += loss.item()
    return loss_epoch, accuracy_epoch

def test_epoch(args, loader, model, criterion):
    model.eval()
    loss_epoch = 0
    accuracy_epoch = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for step, (x, y) in enumerate(loader):
            x = x.to(args.device)
            y = y.to(args.device)
            output = model(x)
            loss = criterion(output, y)
            predicted = output.argmax(1)
            acc = (predicted == y).sum().item() / y.size(0)
            accuracy_epoch += acc
            loss_epoch += loss.item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    return loss_epoch, accuracy_epoch, all_preds, all_labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimCLR Linear Evaluation")
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize TensorBoard writer.
    writer = SummaryWriter(log_dir="./logs/linear_eval")

    # Data loading: set up dataset branches.
    if args.dataset == "STL10":
        train_dataset = torchvision.datasets.STL10(
            args.dataset_dir,
            split="train",
            download=True,
            transform=TransformsSimCLR(size=args.image_size).test_transform,
        )
        test_dataset = torchvision.datasets.STL10(
            args.dataset_dir,
            split="test",
            download=True,
            transform=TransformsSimCLR(size=args.image_size).test_transform,
        )
    elif args.dataset == "CIFAR10":
        train_dataset = torchvision.datasets.CIFAR10(
            args.dataset_dir,
            train=True,
            download=True,
            transform=TransformsSimCLR(size=args.image_size).test_transform,
        )
        test_dataset = torchvision.datasets.CIFAR10(
            args.dataset_dir,
            train=False,
            download=True,
            transform=TransformsSimCLR(size=args.image_size).test_transform,
        )
    elif args.dataset == "custom":
        # Example for a custom dataset (e.g., dogs vs. cats).
        train_dataset = torchvision.datasets.ImageFolder(
            os.path.join(args.dataset_dir, "dog_and_cat/dataset/training_set/"),
            transform=get_test_transform(args.image_size)
        )
        test_dataset = torchvision.datasets.ImageFolder(
            os.path.join(args.dataset_dir, "dog_and_cat/dataset/test_set/"),
            transform=get_test_transform(args.image_size)
        )
    else:
        raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.logistic_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.logistic_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
    )

    # Build the backbone and load pretrained SimCLR model.
    encoder = get_resnet(args.resnet, pretrained=False)
    n_features = encoder.fc.in_features  # Get dimensions of final FC layer.
    simclr_model = SimCLR(encoder, args.projection_dim, n_features)
    model_fp = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.epoch_num))
    simclr_model.load_state_dict(torch.load(model_fp, map_location=args.device.type))
    simclr_model = simclr_model.to(args.device)
    simclr_model.eval()  # Freeze the backbone.

    ## Logistic Regression on frozen features.
    n_classes = 10  # For CIFAR-10/STL10 (change if necessary)
    model = LogisticRegression(simclr_model.n_features, n_classes)
    model = model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    # Learning rate scheduler: you can experiment with different schedulers.
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    criterion = torch.nn.CrossEntropyLoss()

    print("### Creating features from pre-trained context model ###")
    (train_X, train_y, test_X, test_y) = get_features(simclr_model, train_loader, test_loader, args.device)
    arr_train_loader, arr_test_loader = create_data_loaders_from_arrays(train_X, train_y, test_X, test_y, args.logistic_batch_size)

    # Training loop for logistic regression.
    for epoch in range(args.logistic_epochs):
        train_loss, train_acc = train_epoch(args, arr_train_loader, model, criterion, optimizer)
        scheduler.step()  # Adjust learning rate.
        avg_train_loss = train_loss / len(arr_train_loader)
        avg_train_acc = train_acc / len(arr_train_loader)
        print(f"Epoch [{epoch}/{args.logistic_epochs}]\t Loss: {avg_train_loss:.4f}\t Accuracy: {avg_train_acc:.4f}")
        writer.add_scalar("LinearEval/Train_Loss", avg_train_loss, epoch)
        writer.add_scalar("LinearEval/Train_Accuracy", avg_train_acc, epoch)

    # Final testing.
    test_loss, test_acc, all_preds, all_labels = test_epoch(args, arr_test_loader, model, criterion)
    avg_test_loss = test_loss / len(arr_test_loader)
    avg_test_acc = test_acc / len(arr_test_loader)
    print(f"[FINAL]\t Loss: {avg_test_loss:.4f}\t Accuracy: {avg_test_acc:.4f}")
    writer.add_scalar("LinearEval/Test_Loss", avg_test_loss, args.logistic_epochs)
    writer.add_scalar("LinearEval/Test_Accuracy", avg_test_acc, args.logistic_epochs)

    # Compute and display the confusion matrix and classification report.
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", report)
    writer.add_text("LinearEval/Confusion_Matrix", str(cm), args.logistic_epochs)
    writer.add_text("LinearEval/Classification_Report", report, args.logistic_epochs)

    writer.close()
