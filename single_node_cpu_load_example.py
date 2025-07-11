"""
Title: single_node_cpu_load_example.py
Description: Example of loading a cutqc_model file, using a single node, and 
reconstructing. Notice how cutqc distributed is not initialized
"""

import os
import cutqc
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


if __name__ == "__main__":    
    print("--- Load CutQC Instance from Pickle ---")
    filename = "adder.cutqc_model"
    cutqc_model = cutqc.CutQCModel.load_cutqc_model(filename)
  
    print("--- Reconstructing ---")
    reconstructor = cutqc.CircuitReconstructor(cutqc_model, 32, 1)
    reconstructor.build()
        
    print("--- Verify ---")
    cutqc_model.verify()
