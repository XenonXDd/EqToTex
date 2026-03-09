import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader
from PIL import Image,ImageOps
from torchvision import transforms
import torch.nn.functional as F

image_dir = "formula_images_processed/formula_images_processed/"

df_train = pd.read_csv("C:\Users\uzivatel\Documents\STC\projekt_ai\im2latex_train.csv",nrows=1000)

df_test = pd.read_csv("C:\Users\uzivatel\Documents\STC\projekt_ai\im2latex_test.csv",nrows=10)

df_valid = pd.read_csv("C:\Users\uzivatel\Documents\STC\projekt_ai\im2latex_validate.csv",nrows=10)

print(df_train.head())

df_train["image"] = df_train["image"].apply(lambda x: image_dir + x)
df_test["image"] = df_test["image"].apply(lambda x: image_dir + x)
df_valid["image"] = df_valid["image"].apply(lambda x: image_dir + x)

print(df_train['image'].astype(str))

#df_train["image"] = df_train.apply(lambda x: x if x.endswith(".png") else np.nan)
df_train["image"] = df_train["image"].astype(str).apply(lambda x: x if x.endswith(".png") else None)
df_test["image"] = df_test["image"].astype(str).apply(lambda x: x if x.endswith(".png") else None)
df_valid["image"] = df_valid["image"].astype(str).apply(lambda x: x if x.endswith(".png") else None)

df_train.dropna().reset_index(drop=True)
df_test.dropna().reset_index(drop=True)
df_valid.dropna().reset_index(drop=True)

train_characters = set()
test_characters = set()
valid_characters = set()

for formula in df_train["formula"]:
    for char in formula:
        train_characters.add(char)

for formula in df_test["formula"]:
    for char in formula:
        test_characters.add(char)

for formula in df_valid["formula"]:
    for char in formula:
        valid_characters.add(char)


print(train_characters)
print("\n")
print(test_characters)
print("\n")
print(valid_characters)
print("\n")

train_characters.update(test_characters)
train_characters.update(valid_characters)

characters = train_characters
'''
print("\n")
print(characters)
print("\n")
print(sorted(train_characters))
'''
sorted_characters = sorted(characters)

char_to_idx = {}
idx_to_char = {}

index = 0
for char in sorted_characters:
    char_to_idx[char] = index
    index += 1
print(char_to_idx)

for key,value in char_to_idx.items():
    idx_to_char[value] = key  

class FormulaDataset(Dataset):
    def __init__(self,image_paths,formulas,transform=None):
        self.image_paths = image_paths
        self.formulas = formulas
        self.transform = transform
    
    def __len__(self): # vraci delku celeho datasetu
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        formula = self.formulas[idx]
        
        image = Image.open(image_path)
        imageG = ImageOps.grayscale(image)
        image = imageG
        if self.transform:
            image = self.transform(image)
            
        return image,formula

transform = transforms.Compose([
    transforms.Resize((75, 300)),  # Resize to fixed size (height x width)
    transforms.Grayscale(num_output_channels=1),  # Convert image to grayscale
    transforms.ToTensor()  # Convert image to tensor (scales pixel values to [0, 1])
])

train_dataset = FormulaDataset(
    image_paths=df_train["image"].tolist(),
    formulas=df_train["formula"].tolist(),
    transform=transform
)
print(train_dataset.image_paths)
test_dataset = FormulaDataset(
    image_paths=df_test["image"].tolist(),
    formulas=df_test["formula"].tolist(),
    transform=transform
)
valid_dataset = FormulaDataset(
    image_paths=df_valid["image"].tolist(),
    formulas=df_valid["formula"].tolist(),
    transform=transform
)

#ted loadery pro datasety
train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=32,shuffle=False)
valid_loader = DataLoader(valid_dataset,batch_size=32,shuffle=False)

# Checking the dataset size
print(f"Number of training samples: {len(train_loader.dataset)}")
print(f"Number of test samples: {len(test_loader.dataset)}")
print(f"Number of validation samples: {len(valid_loader.dataset)}")

# Example: Getting one batch of data



class CRNN(torch.nn.Module):
    def __init__(self, vocab_size):
        super(CRNN, self).__init__()

        self.conv_1 = torch.nn.Conv2d(1, 128, kernel_size=(3, 3), padding=(1, 1))
        self.pool_1 = torch.nn.MaxPool2d(kernel_size=(2, 2))
        self.conv_2 = torch.nn.Conv2d(128, 64, kernel_size=(3, 3), padding=(1, 1))
        self.pool_2 = torch.nn.MaxPool2d(kernel_size=(2, 2))
        self.linear1 = torch.nn.Linear(3200, 64)
        self.drop_1 = torch.nn.Dropout(0.2)
        self.lstm = torch.nn.GRU(64, 32, bidirectional=True, num_layers=2, dropout=0.25, batch_first=True)
        self.output = torch.nn.Linear(64,vocab_size + 1)

    def forward(self, images, targets=None):
        bs, c, h, w = images.size()
        print(bs, c, h, w)
        x = F.relu(self.conv_1(images))
        print(x.size())
        x = self.pool_1(x)
        print(x.size())
        x = F.relu(x)
        print(x.size())
        x = self.pool_2(x)
        print(x.size())
        x = x.permute(0, 3, 1, 2)
        x = x.view(bs, x.size(1), -1) # potreba pro nasledny vstup do RNN
        x = F.relu(self.linear1(x))
        print(x.size())
        x = self.drop_1(x)
        print(x.size())
        x, _ = self.lstm(x)
        print(x.size())
        x = self.output(x)
        x = x.permute(1,0,2)

        if targets is not None:
            log_probs = F.log_softmax(x,2)
            input_lengths = torch.full(
                size=(bs,), fill_value=log_probs.size(0), dtype=torch.int32
            )
            target_lengths = torch.full(
                size=(bs,), fill_value=targets.size(1), dtype=torch.int32
            )
            loss = torch.nn.CTCLoss(blank=0)(
                log_probs, targets, input_lengths, target_lengths
            )
            print(loss)
            return x, loss
        return x, None
    
def calculate_accuracy(predictions, ground_truths):
    correct = 0
    total = len(predictions)
    for pred, true in zip(predictions, ground_truths):
        if pred == true:  # Compare entire predicted formula to the true formula
            correct += 1
    return correct / total  # Return the proportion that were correct

# Define vocab_size based on char_to_idx
vocab_size = len(char_to_idx)

# Instantiate model
model = CRNN(vocab_size)

# Load model weights if you have them
# model.load_state_dict(torch.load("model.pth"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
# Helper function to encode formulas to tensor of indices
def encode_formulas(formulas, char_to_idx):
    max_len = max(len(f) for f in formulas)
    targets = []
    for formula in formulas:
        indices = [char_to_idx[char] for char in formula]
        indices += [0] * (max_len - len(indices))  # Padding if needed
        targets.append(indices)
    return torch.tensor(targets, dtype=torch.long)

# Training parameters
num_epochs = 10
learning_rate = 1e-3

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)

model.train()

for epoch in range(num_epochs):
    total_loss = 0
    for images, formulas in train_loader:
        images = images.to(device)

        # Convert formulas to tensor of indices
        targets = encode_formulas(formulas, char_to_idx).to(device)

        # Forward pass
        outputs, loss = model(images, targets)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # (Optional) Evaluation on validation set at each epoch
    model.eval()
val_predictions = []
val_formulas = []

model.eval()
with torch.no_grad():
    for i, (images, formulas) in enumerate(valid_loader):
        if i >= 10:
            break
        images = images.to(device)
        outputs, _ = model(images)

        _, predicted_indices = torch.max(outputs, dim=2)
        for j in range(predicted_indices.size(1)):
            pred_indices = predicted_indices[:, j].cpu().numpy()
            pred_formula = []
            prev_idx = -1
            for idx in pred_indices:
                if idx != vocab_size and idx != prev_idx:
                    if idx in idx_to_char:
                        pred_formula.append(idx_to_char[idx])
                prev_idx = idx
            val_predictions.append(''.join(pred_formula))
            val_formulas.append(formulas[j])

val_acc = calculate_accuracy(val_predictions, val_formulas)
print(f"Validation Accuracy (10 batches): {val_acc*100:.2f}%")
model.train()
