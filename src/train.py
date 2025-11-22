import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import time
import os
import copy

# --- CONFIGURATION ---
# We resize to 256x256 to catch small defects (scratches)
# If it's too slow, change to 224
IMG_SIZE = 256
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10


# ---------------------

def train_model():
    # 1. Detect Hardware (Supports Mac M1/M2 'mps' acceleration)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on: {device}")

    # 2. Data Augmentation & Normalization
    # We add rotation and flipping to make the model robust
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Robustness to lighting
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = '../dataset'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                                                  shuffle=True, num_workers=0)
                   for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes  # ['defective', 'good']

    print(f"Classes found: {class_names}")

    # 3. Load Pre-trained ResNet50
    # Note: 'pretrained=True' downloads weights from the internet
    model = models.resnet50(pretrained=True)

    # Freeze early layers so we don't destroy the pre-trained patterns
    for param in model.parameters():
        param.requires_grad = False

    # 4. Modify the Head (Final Layer)
    # ResNet50 input features for the last layer is 2048
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # We have 2 classes: Good, Defective

    model = model.to(device)

    # 5. Loss Function with Class Weights
    # Your data is imbalanced (~4 Good : 1 Defective).
    # We weight the 'Defective' class higher so the model doesn't ignore it.
    # Assuming class_names order is ['defective', 'good'] (Alphabetical)
    # Defective = index 0, Good = index 1
    class_weights = torch.tensor([4.0, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

    # 6. Training Loop
    since = time.time()
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.float() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model if it's the best validation accuracy
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # Save the best model
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), "../models/defect_model.pth")
    print("Model saved as 'defect_model.pth'")


if __name__ == "__main__":
    train_model()