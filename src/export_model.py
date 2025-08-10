import torch
import numpy as np
from train_binary_dscnn import DS_CNN_Binary

# Load model
model = DS_CNN_Binary()
state = torch.load("dscnn_fixed_q15.pt", map_location="cpu")
model.load_state_dict(state)
model.eval()

header_file = "model_q15.h"

# Q15 conversion helper
def float_to_q15(x):
    # clip to [-1, 0.99997]
    if x >= 1.0:
        return 32767
    if x < -1.0:
        return -32768
    return int(np.round(x * 32768))

with open(header_file, "w") as f:
    f.write("// Auto-generated DS-CNN Q15 weights header\n")
    f.write("#ifndef MODEL_Q15_H\n#define MODEL_Q15_H\n\n")

    for name, param in model.named_parameters():
        arr = param.detach().cpu().numpy()
        flat = arr.flatten()
        cname = name.replace('.', '_')

        # Determine type: biases to int32_t, weights to int16_t
        dtype = 'int32_t' if 'bias' in name else 'int16_t'
        f.write(f"// {name}, shape={arr.shape}\n")
        f.write(f"static const {dtype} {cname}[{flat.size}] = {{\n")

        # Convert and write elements
        line_vals = []
        for v in flat:
            q15 = float_to_q15(float(v))
            line_vals.append(str(q15))
        # chunk lines
        for i in range(0, len(line_vals), 16):
            f.write('    ' + ', '.join(line_vals[i:i+16]))
            f.write(',\n')
        f.write("};\n\n")

    f.write("#endif // MODEL_Q15_H\n")

