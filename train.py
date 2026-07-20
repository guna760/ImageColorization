from model import build_model
from data_loader import load_images

print("Loading Training Dataset...")
X_train, Y_train = load_images("dataset/train")

print("Loading Validation Dataset...")
X_val, Y_val = load_images("dataset/validation")

print("Training Shape :", X_train.shape)
print("Validation Shape :", X_val.shape)

model = build_model()

model.summary()

history = model.fit(
    X_train,
    Y_train,
    validation_data=(X_val, Y_val),
    epochs= 50,
    batch_size=4
)

model.save("models/colorization_model.h5")

print("Training Completed Successfully!")