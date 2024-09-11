import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Verileri normalize etmek ve boyutlandırmak için dönüşümler
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resimleri 224x224 boyutuna getir
    transforms.ToTensor(),          # Tensöre çevir
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalizasyon
])

# Eğitim veri seti
train_dataset = datasets.ImageFolder('veri_dizini/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Doğrulama veri seti
val_dataset = datasets.ImageFolder('veri_dizini/val', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Önceden eğitilmiş bir ResNet18 modeli yükleyin
model = models.resnet18(pretrained=True)

# Son katmanı değiştirmek (Çıkış katmanını 4 sınıfa göre ayarlamak)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 4)  # 4 sınıf için

# Modeli cihazda çalıştırmak için
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()  # Sınıflandırma için çapraz entropi kaybı
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizörü

# Eğitim fonksiyonu
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()  # Modeli eğitim moduna alın
        running_loss = 0.0
        correct = 0
        total = 0

        # Eğitim döngüsü
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Doğrulama döngüsü
        model.eval()  # Modeli değerlendirme moduna alın
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {running_loss/len(train_loader):.4f}, "
              f"Train Accuracy: {100 * correct/total:.2f}%, "
              f"Val Loss: {val_loss/len(val_loader):.4f}, "
              f"Val Accuracy: {100 * val_correct/val_total:.2f}%")

    return model

# Modeli eğit
model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)

torch.save(model, 'model_egitim.pth')
