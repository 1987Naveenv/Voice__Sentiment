import torch
from torchvision import transforms
from torchaudio.transforms import Resample
import torchaudio
from PIL import Image
import numpy as np
import joblib  # For saving/loading LabelEncoder

# ----------------------------
# Set up device
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Load Model
# ----------------------------
num_classes = 5  # 🔁 Change based on your dataset
model = EnhancedAVClassifier(num_classes=num_classes)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

# ----------------------------
# Load Label Encoder
# ----------------------------
le = joblib.load("label_encoder.pkl")  # Make sure you saved this during training

# ----------------------------
# Prediction Function
# ----------------------------
def predict(audio_path, video_path):
    # --- Preprocess Audio ---
    waveform, sr = torchaudio.load(audio_path)
    if sr != AUDIO_SAMPLE_RATE:
        waveform = Resample(orig_freq=sr, new_freq=AUDIO_SAMPLE_RATE)(waveform)
    waveform = waveform.mean(dim=0, keepdim=True)
    if waveform.size(1) < AUDIO_LENGTH:
        pad_size = AUDIO_LENGTH - waveform.size(1)
        waveform = torch.nn.functional.pad(waveform, (0, pad_size))
    else:
        waveform = waveform[:, :AUDIO_LENGTH]
    waveform = waveform.unsqueeze(0).to(device)

    # --- Preprocess Video ---
    try:
        image = Image.open(video_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        image = transform(image).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error loading image {video_path}: {e}")
        image = torch.zeros((1, 3, 224, 224)).to(device)

    # --- Inference ---
    with torch.no_grad():
        output = model(waveform, image)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        pred_idx = np.argmax(probs)
        pred_label = le.inverse_transform([pred_idx])[0]

    # --- Print Output ---
    print("Prediction:", pred_label)
    print("Probabilities:")
    for idx, prob in enumerate(probs):
        print(f"{le.inverse_transform([idx])[0]}: {prob:.4f}")

    return pred_label, probs

# ----------------------------
# Example Usage
# ----------------------------
if __name__ == "__main__":
    audio_path = "audio\neutral\YAF_bar_neutral.wav"
    video_path = "path/to/sample.jpg"
    predict(audio_path, video_path)
