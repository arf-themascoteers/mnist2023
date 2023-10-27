from train import train
from test import test
import constants
import os
import torch
from analyze import analyze


model = None
if os.path.exists(constants.DEFAULT_MODEL_PATH):
    model = torch.load(constants.DEFAULT_MODEL_PATH)
else:
   model = train()

test()
analyze(model)