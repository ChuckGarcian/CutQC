"""
Title: dist_driver.py
Description: Example of how CutQC can be used to efficiently reconstruct subcircuits
"""

import os
from cutqc import CircuitReconstructor
from cutqc.cutqc_model import CutQCModel

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


if __name__ == "__main__":
    # Load CutQC Instance from Pickle
    print("--- Running ---")
    filename = "adder.cutqc_model"
    cutqc_model = CutQCModel.load_cutqc_model(filename)

    # Initiate Reconstruct
    cqc = CircuitReconstructor(cutqc_model, 32, 1)
    compute_time = cqc.build()

    approximation_error = cutqc_model.verify()
