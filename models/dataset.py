import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader
from PIL import Image,ImageOps
from torchvision import transforms
import torch.nn.functional as F

class CRNN(torch.nn.Module):
    def __init__(self, vocab_size):
        super(CRNN, self).__init__()

        self.conv_1 = torch.nn.Conv2d(1, 128, kernel_size=(3, 3), padding=(1, 1))
        self.pool_1 = torch.nn.MaxPool2d(kernel_size=(2, 2))
        self.conv_2 = torch.nn.Conv2d(128, 64, kernel_size=(3, 3), padding=(1, 1))
        self.pool_2 = torch.nn.MaxPool2d(kernel_size=(2, 2))
        self.linear1 = torch.nn.Linear(2304,64)
        self.drop_1 = torch.nn.Dropout(0.2)
        self.lstm = torch.nn.GRU(64, 32, bidirectional=True, num_layers=2, dropout=0.25, batch_first=True)
        self.output = torch.nn.Linear(64,vocab_size + 1)

    def forward(self, images, targets=None):
        print("forward done")
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
            print("target is there")
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
            return x, loss
        return x, None

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



image_dir = "formula_images_processed/formula_images_processed/"

df_train = pd.read_csv("C:\\Users\\uzivatel\\Documents\\STC\\projekt_ai\\im2latex_train.csv", nrows=1000)

df_test = pd.read_csv("C:\\Users\\uzivatel\\Documents\\STC\\projekt_ai\\im2latex_test.csv", nrows=10)

df_valid = pd.read_csv("C:\\Users\\uzivatel\\Documents\\STC\\projekt_ai\\im2latex_validate.csv", nrows=10)

print(df_train.head())

df_train["image"] = df_train["image"].apply(lambda x: image_dir + x)
df_test["image"] = df_test["image"].apply(lambda x: image_dir + x)
df_valid["image"] = df_valid["image"].apply(lambda x: image_dir + x)

print(df_train["image"].head())
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

data_iter = iter(train_loader)
images, formulas = next(data_iter)
print(formulas)
print(images)

# Initialize the model
vocab_size = len(char_to_idx)
model = CRNN(vocab_size)
print(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Move model to device
model = CRNN(len(char_to_idx)).to(device)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Number of epochs
num_epochs = 10

def encode_formula(formula, char_to_idx):
    return [char_to_idx[char] + 1 for char in formula if char in char_to_idx]  # shift by 1 due to blank=0

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch_idx, (images, formulas) in enumerate(train_loader):
        images = images.to(device)

        # Encode formulas
        targets = [torch.tensor(encode_formula(f, char_to_idx), dtype=torch.long) for f in formulas]
        targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0).to(device)


        # Forward pass
        outputs, loss = model(images, targets)
        print(f"outputs:\n",outputs)
        print(f"loss:\n",loss)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}] completed. Avg Loss: {avg_loss:.4f}")

    # Validation loop (optional)
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for val_images, val_formulas in valid_loader:
            val_images = val_images.to(device)
            val_targets = [torch.tensor(encode_formula(f, char_to_idx), dtype=torch.long) for f in val_formulas]
            val_targets = torch.nn.utils.rnn.pad_sequence(val_targets, batch_first=True, padding_value=0).to(device)

            _, loss = model(val_images, val_targets)
            val_loss += loss.item()

        val_avg_loss = val_loss / len(valid_loader)
        print(f"Validation Loss: {val_avg_loss:.4f}")