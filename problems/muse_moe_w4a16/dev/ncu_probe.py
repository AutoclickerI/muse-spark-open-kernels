"""One eager launch for Nsight Compute; outside the submitted source."""
import importlib.util
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import reference

spec = importlib.util.spec_from_file_location("candidate", sys.argv[1])
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)
weights = reference.generate_weights(reference.WEIGHTS_SEED)
state = candidate.prepare(weights)
data = reference.generate_input(int(sys.argv[2]), "ep8", 1266)
candidate.kernel_fn(data, state)
torch.cuda.synchronize()
