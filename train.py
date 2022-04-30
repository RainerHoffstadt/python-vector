
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
transfer_model = models.resnet50(pretrained=True)
from torchvision.datasets import ImageFolder

for name, param in transfer_model.named_parameters():
    param.requires_grad = False

transfer_model.fc = nn.Sequential(nn.Linear(transfer_model.fc.in_features, 556),
                                  nn.ReLU(), nn.Dropout(), nn.Linear(556, 6))
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

transfer_model.to(device)
train_losses = []
val_losses = []
def train(model, optimizer, loss_fn, train_loader, val_loader, epochs=20, device="cpu"):
    for epoch in range(1, epochs + 1):
        training_loss = 0.0
        valid_loss = 0.0
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item() * inputs.size(0)
        training_loss /= len(train_loader.dataset)
        train_losses.append(training_loss)
        model.eval()
        num_correct = 0
        num_examples = 0
        for batch in val_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            output = model(inputs)
            targets = targets.to(device)
            loss = loss_fn(output, targets)
            valid_loss += loss.data.item() * inputs.size(0)
            correct = torch.eq(torch.max(F.softmax(output), dim=1)[1], targets).view(-1)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]
        valid_loss /= len(val_loader.dataset)
        val_losses.append(valid_loss)
        print(
            'Epoch: {}, Training Loss: {:.2f}, Validation Loss: {:.2f}, accuracy = {:.2f}'.format(epoch, training_loss,
                                                                                                  valid_loss,
                                                                                                  num_correct / num_examples))


def check_image(path):
    try:
        im = Image.open(path)
        return True
    except:
        return False


img_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

dirname = '/home/oem/Chessman-image-dataset/Chess'

train_data_path = dirname + '/train/'

img_transfors = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])

train_data = ImageFolder(root=train_data_path,
                         transform=img_transfors,
                         is_valid_file=check_image)

val_data_path = dirname + '/val/'
val_data = ImageFolder(root=val_data_path,
                         transform=img_transfors,
                         is_valid_file=check_image)

test_data_path = dirname + '/test/'
test_data = ImageFolder(root=test_data_path,
                         transform=img_transfors,
                         is_valid_file=check_image)
batch_size = 64
train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_data_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(len(val_data_loader.dataset))

transfer_model.to(device)
optimizer = optim.Adam(transfer_model.parameters(), lr=0.001)

train(transfer_model, optimizer, torch.nn.CrossEntropyLoss(), train_data_loader, val_data_loader, epochs=20,
      device=device)


# Plot training & validation loss values
plt.plot(train_losses)
plt.plot(val_losses)
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'val'], loc='upper left')
plt.show()

