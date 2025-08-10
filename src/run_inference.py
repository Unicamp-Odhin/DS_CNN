import torch
import numpy as np
from train_binary_dscnn import DS_CNN_Binary  # Reuse model definition

def load_mfcc(csv_path):
    mfcc = np.loadtxt(csv_path, delimiter=',')          # [399, 12]
    mfcc = mfcc.astype(np.float32) / 32768.0            # Normalize fixed-point Q15
    mfcc = torch.tensor(mfcc, dtype=torch.float32)      # To tensor
    mfcc = mfcc.unsqueeze(0).unsqueeze(0)               # Shape: [1, 1, 399, 12]
    return mfcc

def predict(model_path, mfcc_path):
    model = DS_CNN_Binary()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    mfcc = load_mfcc(mfcc_path)

    with torch.no_grad():
        logit = model(mfcc)
        prob = torch.sigmoid(logit).item()
        print(f"Prediction probability: {prob:.4f}")
        if prob > 0.5:
            print("🟢 Keyword DETECTED")
        else:
            print("⚪️ Background (no keyword)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to .pt file')
    parser.add_argument('--csv', type=str, required=True, help='Path to input .csv MFCC')
    args = parser.parse_args()

    predict(args.model, args.csv)
