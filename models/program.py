import numpy as np
import pandas as pd

import tqdm
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
from tensorflow.keras.backend import ctc_batch_cost, ctc_decode
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split

image_dir = "formula_images_processed/formula_images_processed/"

df_train = pd.read_csv("C:/Users/slade/source/TUI/dataset/im2latex_train.csv",nrows=1000)

df_test = pd.read_csv("C:/Users/slade/source/TUI/dataset/im2latex_test.csv",nrows=10)

df_valid = pd.read_csv("C:/Users/slade/source/TUI/dataset/im2latex_validate.csv",nrows=10)
print(df_train.head())
print(df_train.columns)
print(df_train.isnull().sum())



df_train["image"] = df_train["image"].apply(lambda x: image_dir + x)
df_test["image"] = df_test["image"].apply(lambda x: image_dir + x)
df_valid["image"] = df_valid["image"].apply(lambda x: image_dir + x)

df_train["image"] = df_train["image"].apply(lambda x: x if x.endswith(".png") else np.nan)
df_test["image"] = df_test["image"].apply(lambda x: x if x.endswith(".png") else np.nan)
df_valid["image"] = df_valid["image"].apply(lambda x: x if x.endswith(".png") else np.nan)
 
df_train = df_train.dropna().reset_index(drop=True)
df_valid = df_valid.dropna().reset_index(drop=True)
df_test = df_test.dropna().reset_index(drop=True)


train_characters = set()  # Create an empty set
test_characters = set()
valid_characters = set()

for formula in df_train['formula']:  # Iterate through each formula
    for char in formula:  # Iterate through each character in the formula
        train_characters.add(char)  # Add the character to the set

for formula in df_test['formula']:  # Iterate through each formula
    for char in formula:  # Iterate through each character in the formula
        test_characters.add(char)  # Add the character to the set

for formula in df_valid['formula']:  # Iterate through each formula
    for char in formula:  # Iterate through each character in the formula
        valid_characters.add(char)  # Add the character to the set

print(train_characters)
print(test_characters)
print(valid_characters)

train_characters.update(test_characters)
train_characters.update(valid_characters)

characters = train_characters #vsechny charactery
'''
df_train_with_backslash = df_train[df_train['formula'].str.contains(r"\\", regex=True)]
df_test_with_backslash = df_test[df_test['formula'].str.contains(r"\\", regex=True)]
df_valid_with_backslash = df_valid[df_valid['formula'].str.contains(r"\\", regex=True)]
print("train")
print(df_train_with_backslash)
print("test")
print(df_test_with_backslash)
print("validation")
print(df_valid_with_backslash)
'''

char_to_num = layers.StringLookup(
    vocabulary = list(characters),
    num_oov_indices = 0,
    mask_token = None
)
print("char to num")
print(char_to_num.get_vocabulary())

arrangement = np.arange(1,len(char_to_num.get_vocabulary())+1)

print(arrangement)
print(len(char_to_num.get_vocabulary()))


pd.DataFrame({"char": char_to_num.get_vocabulary(),
              "num": np.arange(1, len(char_to_num.get_vocabulary())+1)})

num_to_char = layers.StringLookup(
    vocabulary = char_to_num.get_vocabulary(),
    mask_token = None,
    invert = True
)

Xtrain = df_train["image"].values.tolist()
Ytrain = df_train["formula"].values.tolist()

#print(f"Xtrain:{Xtrain}")
#print(f"Ytrain:{Ytrain}")

Xtest = df_test["image"].values.tolist()
Ytest = df_test["formula"].values.tolist()

#print(f"Xtest:{Xtest}")
#print(f"Ytest:{Ytest}")

Xvalidate = df_valid["image"].values.tolist()
Yvalidate = df_valid["formula"].values.tolist()

#print(f"Xtrain:{Xvalidate}")
#print(f"Ytrain:{Yvalidate}")

print("longest formula in Ytrain")

print("Max formula lengths:")
print("Train:", max(map(len, Ytrain)))
print("Test:", max(map(len, Ytest)))
print("Valid:", max(map(len, Yvalidate)))



print(max([len(_) for _ in Ytrain]))
print(max([len(_) for _ in Ytest]))
print(max([len(_) for _ in Yvalidate]))

label = tf.constant(["hello"])  # Shape: (1,)
split_label = tf.strings.unicode_split(label, input_encoding="UTF-8")
print(split_label)

@tf.function
def encode_single_sample(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.io.decode_png(image, channels=1)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [50, 200])
    image = tf.transpose(image, perm=[1, 0, 2])
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    label = tf.pad(label, paddings=[[0, 600 - tf.shape(label)[0]]], constant_values=0)
    return {"Input": image, "Label": label}, label


class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = ctc_batch_cost

    def call(self, y_true, y_pred):
        
        batch_length = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_length, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_length, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        return y_pred

@tf.function
def process_dataset(X, y):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.map(encode_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(1024).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

def visualize_df(df: pd.DataFrame):
    fig, axes = plt.subplots(4, 1, figsize=(10, 5))

    for i, ax in enumerate(axes.ravel()):
        if i < len(df):
            a = np.random.randint(1, len(df), 1)[0]
            img_path = df.loc[a][['image']].values[0]
            label = df.loc[a][["formula"]].values[0]
            
            image = Image.open(img_path).convert('RGB')
            
            ax.imshow(image)
            ax.set_title(f"LateX: {label}")
            ax.axis('off')
            
        else:
            ax.axis('off')
            
    plt.tight_layout()
    plt.show()

train_dataset = process_dataset(Xtrain, Ytrain)
valid_dataset = process_dataset(Xvalidate, Yvalidate)
test_dataset = process_dataset(Xtest, Ytest)

#visualize_df(df_train)


input_shape = (200, 50, 1)   #shape je kvuli tomu, ze image je rozmeru 200x50
num_classes = len(characters) + 1  # +1 for the CTC blank label

input_layer = layers.Input(shape=input_shape, name="Input", dtype="float32")
label_layer = layers.Input(shape=(None,), name="Label", dtype="float32")

conv2_1 = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(input_layer)
print(f"conv2_1:{conv2_1.shape}")
max2_1 = layers.MaxPooling2D(pool_size=(2, 1))(conv2_1)

conv2_2 = layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(max2_1)
print(f"conv2_1:{conv2_2.shape}")
max2_2 = layers.MaxPooling2D(pool_size=(2, 1))(conv2_2)

reshape_shape = (input_shape[0] // 2, (input_shape[1] // 2) * 64)
reshape_layer = layers.Reshape(target_shape=reshape_shape)(max2_2)
print(f"reshape layer: {reshape_layer.shape}")

#dense_1 = layers.Dense(units=64, activation="relu")(reshape_layer)
drop_1 = layers.Dropout(0.2)(reshape_layer)

bilstm_1 = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(drop_1)
bilstm_2 = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(bilstm_1)

output_layer = layers.Dense(num_classes, activation="softmax", name="Output")(bilstm_2)

output = CTCLayer(name="ctc_loss" )(label_layer, output_layer)

model = models.Model(inputs=[input_layer, label_layer], outputs=output, name="OCR")

model.compile(optimizer=optimizers.Adam())


model.summary()

early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "Version1.h5", save_best_only=True, monitor="val_loss", mode="min"
)


history = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=50,
    callbacks=[early_stopping, checkpoint]
)

test_loss = model.evaluate(test_dataset)
print(f"Test Loss: {test_loss}")

model.save("ocr_modelFINALversion1.h5")
