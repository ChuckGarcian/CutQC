"""
Title: cut_and_eval.py
Description: Example of how to cut and evaluate for the purposes of
distributed reconstruction
"""

import cutqc
import cutqc.helper_functions

if __name__ == "__main__":
    # Generate AdderCircuit and Initialize CutQC
    circuit, cutter_constraints = cutqc.helper_functions.simple_adder()
    cutter = cutqc.CircuitCutter(circuit=circuit, cutter_constraints=cutter_constraints)
    
    print("Finding cuts and creating subcircuits...")
    cutter.cut()

    print("--- Evaluating subcircuits ---")
    cutqc_model = cutter.evaluate(eval_mode="sv", num_shots_fn=None)

    # Initiate Reconstruct
    print("--- Saving model as 'adder_cutqc_model' file --- ")
    cutqc_model.save_cutqc_model(filename="adder_supremacy.cutqc_model")
    print("Completed")
