import torch
import torch.nn as nn
from torch.optim import SGD
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from PIL import Image

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define data transformations for training and testing
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

ct = f"F:/nsclc/Test3.v1/ct"
pet = f"F:/nsclc/Test3.v1/pet"

# Load custom dataset (replace 'data_dir' with your dataset directory)
data_dir = pet
image_datasets = {x: datasets.ImageFolder(root=f'{data_dir}/{x}', transform=data_transforms[x])
                  for x in ['train', 'test']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4)
               for x in ['train', 'test']}

# Use a pre-trained VGG16 model and modify the final fully connected layer for the number of classes in your dataset
# weights=vgg16.VGG16_Weights.DEFAULT
model = models.vgg16(pretrained=True)
num_classes = len(image_datasets['train'].classes)
model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)  # Modify the last FC layer

# Send the model to device
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)

def train_model(model, criterion, optimizer, dataloaders, num_epochs=10):
# Training loop
    # num_epochs = 10
    for epoch in range(num_epochs):
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            all_preds = []
            all_labels = []

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # Backpropagation and optimization only in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_accuracy = accuracy_score(all_labels, all_preds)

            print(f'Epoch {epoch + 1}/{num_epochs} [{phase}] Loss: {epoch_loss:.4f} Acc: {epoch_accuracy:.4f}')

    # Save the trained model
    torch.save(model.state_dict(), 'vgg16_custom_dataset.pth')

# # Load the trained model for inference
# model.load_state_dict(torch.load('vgg16_custom_dataset.pth'))
# model.eval()

# Function to classify a new image using the trained model
def classify_image(image_path, model, transform):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
    probabilities = torch.softmax(output, dim=1)[0] * 100
    predicted_class_idx = torch.argmax(output).item()
    predicted_class = image_datasets['train'].classes[predicted_class_idx]
    confidence = probabilities[predicted_class_idx].item()
    return predicted_class, confidence

# # Example usage of the classify_image function
# image_path = 'path/to/your/test/image.jpg'  # Replace with the path to the test image
# predicted_class, confidence = classify_image(image_path, model, data_transforms['test'])
# print(f'Predicted Class: {predicted_class}, Confidence: {confidence:.2f}%')


if __name__ == '__main__':
    # Define your data transformations, model, and other necessary components here

    # Check if CUDA (GPU) is available and set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Rest of your code (data loading, model training, etc.)

    # Training loop
    train_model(model, criterion, optimizer, dataloaders, num_epochs=10)

    # Load the trained model for inference
    model.load_state_dict(torch.load('vgg16_custom_dataset.pth'))
    model.eval()

    # # Example usage of the classify_image function
    # image_path = 'path/to/your/test/image.jpg'  # Replace with the path to the test image
    # predicted_class, confidence = classify_image(image_path, model, data_transforms['test'])
    # print(f'Predicted Class: {predicted_class}, Confidence: {confidence:.2f}%')
