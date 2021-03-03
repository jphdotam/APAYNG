import torch.onnx
from lib.models import *
from lib.data import LABELS

WEIGHTS_PATH = r"G:\APAYN\models\005\80_0.80656.pt"
DEVICE = "cuda"
INPUT_DIM = (1, 1, 576, 576)

model = UneXt50(n_inputs=1, n_outputs=len(LABELS)).to(DEVICE)
model.load_state_dict(torch.load(WEIGHTS_PATH)['model'])
model.eval()

dummy_input = torch.rand(*INPUT_DIM).to(DEVICE)

torch.onnx.export(model,
                  dummy_input,
                  f"{WEIGHTS_PATH}.onnx",
                  export_params=True,
                  input_names=['input'],
                  output_names=['output'],
                  opset_version=11,
                  dynamic_axes={'input': {0: 'batch_size',
                                          2: 'height',
                                          3: 'width'},
                                'output': {0: 'batch_size',
                                           2: 'height',
                                           3: 'width'}})

traced_model = torch.jit.trace(model, dummy_input)
y = traced_model(dummy_input)
print(traced_model.graph)
traced_model.save(f"{WEIGHTS_PATH}_{DEVICE}.ts")

DEVICE = "cpu"
model = model.to(DEVICE)
dummy_input = dummy_input.to(DEVICE)
traced_model = torch.jit.trace(model, dummy_input)
y = traced_model(dummy_input)
print(traced_model.graph)
traced_model.save(f"{WEIGHTS_PATH}_{DEVICE}.ts")