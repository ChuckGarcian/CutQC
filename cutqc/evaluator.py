import itertools, copy, pickle, subprocess, psutil, os
import numpy as np
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.circuit.library.standard_gates import HGate, SGate, SdgGate, XGate

from helper_functions.non_ibmq_functions import find_process_jobs, scrambled


def get_num_workers(num_jobs, ram_required_per_worker):
    ram_avail = psutil.virtual_memory().available / 1024**3
    ram_avail = ram_avail / 4 * 3
    num_cpus = int(os.cpu_count() / 4 * 3)
    num_workers = int(min(ram_avail / ram_required_per_worker, num_jobs, num_cpus))
    return num_workers


def run_subcircuit_instances(
    subcircuits, subcircuit_instances, eval_mode, num_shots_fn, data_folder
):
    """
    subcircuit_instance_probs[subcircuit_idx][(instance_init,instance_meas)] = measured probability

    eval_mode:
    sv: statevector simulation
    qasm: noiseless qasm simulation
    runtime: for benchmarking, pseudo QPU backend generates uniform distribution
    """
    
    print(f"subcircuits: {subcircuits}")
    print(f"subcircuit_instances: {type(subcircuit_instances)}")
    print(f"eval_mode: {eval_mode}")
    print(f"num_shots_fn: {num_shots_fn}")
    print(f"data_folder: {data_folder}")
    
    # subicrcuit_instances: Combination of different measurment bases of the same subcircuit    
    #
    subcircuits__instances_init_meas = {}
    instance_init_meas_ids = {}
    measured_prob_dict = {} # Dictionary with key: unique subcircuit, value: dictionary containing instance and measured probability

    for subcircuit_idx in subcircuit_instances:
        jobs = subcircuit_instances[subcircuit_idx]
        instance_init_meas_ids[subcircuit_idx] = {jobs[i]: i for i in range(len(jobs))}                
        
        print ("measids keys: {}".format (instance_init_meas_ids.keys()))
        
        # print (type(subcircuit_instances[subcircuit_idx]))
        # print (subcircuit_instances[subcircuit_idx])

        # subcircuits__instances_init_meas[subcircuit_idx] = {jobs[i]: i for i in range(len(jobs))} # Assign unique idenifier to each job                                
        subcircuit = subcircuits[subcircuit_idx]
        uniform_p = 1 / 2**subcircuit.num_qubits
        num_shots = num_shots_fn(subcircuit) if num_shots_fn is not None else None
        from helper_functions.non_ibmq_functions import evaluate_circ
        
        for instance_init_meas in jobs:
            print ("in main")
            print(instance_init_meas[0])
            

            if 'Z' in instance_init_meas[1]:
                continue
        
            subcircuit_instance = modify_subcircuit_instance(
                subcircuit=subcircuit,
                init=instance_init_meas[0],
                meas=instance_init_meas[1],
            )
        
            if eval_mode == "runtime":
                subcircuit_inst_prob = uniform_p
            elif eval_mode == "sv":
                subcircuit_inst_prob = evaluate_circ(
                    circuit=subcircuit_instance, backend="statevector_simulator"
                )
            elif eval_mode == "qasm":
                subcircuit_inst_prob = evaluate_circ(
                    circuit=subcircuit_instance,
                    backend="noiseless_qasm_simulator",
                    options={"num_shots": num_shots},
                )
            else:
                raise NotImplementedError
            
            mutated_meas = mutate_measurement_basis(meas=instance_init_meas[1])
        
            for meas in mutated_meas:
                measured_prob = measure_prob(
                    unmeasured_prob=subcircuit_inst_prob, meas=meas
                )
                
                # print ("main: instanmea[0]: {}".format(instance_init_meas[0]))
                # print ("meas{}".format(meas))
                # print ("measids keys: {}".format (instance_init_meas_ids.keys()))
                
                instance_init_meas_id = instance_init_meas_ids[subcircuit_idx][(instance_init_meas[0], meas)]                
                
                # print (measured_prob_dict[instance_init_meas_id])
                # print ("chuckInstance_id: {}\ncircui_idx: {}".format(instance_init_meas_id, subcircuit_idx))
                measured_prob_dict[(subcircuit_idx, instance_init_meas_id)] = measured_prob               
    
    print (measured_prob_dict)
    return measured_prob_dict, instance_init_meas_ids

def run_subcircuit_instances2(
    subcircuits, subcircuit_instances, eval_mode, num_shots_fn, data_folder
):
    """
    subcircuit_instance_probs[subcircuit_idx][(instance_init,instance_meas)] = measured probability

    eval_mode:
    sv: statevector simulation
    qasm: noiseless qasm simulation
    runtime: for benchmarking, pseudo QPU backend generates uniform distribution
    """
    instance_init_meas_ids = {}
    measured_prob_dict = {}
    for subcircuit_idx in subcircuit_instances:
        jobs = subcircuit_instances[subcircuit_idx]
        instance_init_meas_ids[subcircuit_idx] = {jobs[i]: i for i in range(len(jobs))}                
        
        import pickle
        pickle.dump(
            {
                "subcircuits": subcircuits,
                "eval_mode": eval_mode,
                "num_shots_fn": num_shots_fn,
                "instance_init_meas_ids": instance_init_meas_ids,
            },
            open("%s/meta_info.pckl" % data_folder, "wb"),
        )
        num_workers = get_num_workers(
            num_jobs=len(jobs),
            ram_required_per_worker=2 ** subcircuits[subcircuit_idx].num_qubits
            * 4
            / 1e9,
        )
        procs = []
        for rank in range(num_workers):
            rank_jobs = find_process_jobs(jobs=jobs, rank=rank, num_workers=num_workers)
            if len(rank_jobs) > 0:
                pickle.dump(
                    rank_jobs, open("%s/rank_%d.pckl" % (data_folder, rank), "wb")
                )
                python_command = (
                    "python -m cutqc.parallel_run_subcircuits --data_folder %s --subcircuit_idx %d --rank %d"
                    % (data_folder, subcircuit_idx, rank)
                )
                proc = subprocess.Popen(python_command.split(" "))
                procs.append(proc)
        [proc.wait() for proc in procs]
        
        
        import glob
        import pickle
        
        # Get all pickle files for this subcircuit
        pickle_files = glob.glob(f"{data_folder}/subcircuit_{subcircuit_idx}_instance_*.pckl")
        
        # Load all the pickle files into a dictionary
        measured_prob_dict = {}
        for pickle_file in pickle_files:
                        
            # Extract instance ID from filename
            instance_id = int(pickle_file.split('_instance_')[1].split('.pckl')[0])
            with open(pickle_file, 'rb') as f:                
                measured_prob_dict[(subcircuit_idx, instance_id)] = pickle.load(f)               
                                    
    
    print (measured_prob_dict)
    print ("Done!")
    return measured_prob_dict
   




def mutate_measurement_basis(meas):
    """
    I and Z measurement basis correspond to the same logical circuit
    """
    if all(x != "I" for x in meas):
        return [meas]
    else:
        mutated_meas = []
        for x in meas:
            if x != "I":
                mutated_meas.append([x])
            else:
                mutated_meas.append(["I", "Z"])
        mutated_meas = list(itertools.product(*mutated_meas))
        return mutated_meas


def modify_subcircuit_instance(subcircuit, init, meas):
    """
    Modify the different init, meas for a given subcircuit
    Returns:
    Modified subcircuit_instance
    List of mutated measurements
    """
    subcircuit_dag = circuit_to_dag(subcircuit)
    subcircuit_instance_dag = copy.deepcopy(subcircuit_dag)
    for i, x in enumerate(init):
        q = subcircuit.qubits[i]
        if x == "zero":
            continue
        elif x == "one":
            subcircuit_instance_dag.apply_operation_front(
                op=XGate(), qargs=[q], cargs=[]
            )
        elif x == "plus":
            subcircuit_instance_dag.apply_operation_front(
                op=HGate(), qargs=[q], cargs=[]
            )
        elif x == "minus":
            subcircuit_instance_dag.apply_operation_front(
                op=HGate(), qargs=[q], cargs=[]
            )
            subcircuit_instance_dag.apply_operation_front(
                op=XGate(), qargs=[q], cargs=[]
            )
        elif x == "plusI":
            subcircuit_instance_dag.apply_operation_front(
                op=SGate(), qargs=[q], cargs=[]
            )
            subcircuit_instance_dag.apply_operation_front(
                op=HGate(), qargs=[q], cargs=[]
            )
        elif x == "minusI":
            subcircuit_instance_dag.apply_operation_front(
                op=SGate(), qargs=[q], cargs=[]
            )
            subcircuit_instance_dag.apply_operation_front(
                op=HGate(), qargs=[q], cargs=[]
            )
            subcircuit_instance_dag.apply_operation_front(
                op=XGate(), qargs=[q], cargs=[]
            )
        else:
            raise Exception("Illegal initialization :", x)
    for i, x in enumerate(meas):
        q = subcircuit.qubits[i]
        if x == "I" or x == "comp":
            continue
        elif x == "X":
            subcircuit_instance_dag.apply_operation_back(
                op=HGate(), qargs=[q], cargs=[]
            )
        elif x == "Y":
            subcircuit_instance_dag.apply_operation_back(
                op=SdgGate(), qargs=[q], cargs=[]
            )
            subcircuit_instance_dag.apply_operation_back(
                op=HGate(), qargs=[q], cargs=[]
            )
        else:
            raise Exception("Illegal measurement basis:", x)
    subcircuit_instance_circuit = dag_to_circuit(subcircuit_instance_dag)
    return subcircuit_instance_circuit


def measure_prob(unmeasured_prob, meas):
    if meas.count("comp") == len(meas) or type(unmeasured_prob) is float:
        return unmeasured_prob
    else:
        measured_prob = np.zeros(int(2 ** meas.count("comp")))
        # print('Measuring in',meas)
        for full_state, p in enumerate(unmeasured_prob):
            sigma, effective_state = measure_state(full_state=full_state, meas=meas)
            measured_prob[effective_state] += sigma * p
        return measured_prob


def measure_state(full_state, meas):
    """
    Compute the corresponding effective_state for the given full_state
    Measured in basis `meas`
    Returns sigma (int), effective_state (int)
    where sigma = +-1
    """
    bin_full_state = bin(full_state)[2:].zfill(len(meas))
    sigma = 1
    bin_effective_state = ""
    for meas_bit, meas_basis in zip(bin_full_state, meas[::-1]):
        if meas_bit == "1" and meas_basis != "I" and meas_basis != "comp":
            sigma *= -1
        if meas_basis == "comp":
            bin_effective_state += meas_bit
    effective_state = int(bin_effective_state, 2) if bin_effective_state != "" else 0
    # print('bin_full_state = %s --> %d * %s (%d)'%(bin_full_state,sigma,bin_effective_state,effective_state))
    return sigma, effective_state


def attribute_shots(subcircuit_entries, subcircuits, eval_mode, instance_init_meas_ids, prob_dict, num_workers = 20):
    # meta_info = pickle.load(open("%s/meta_info.pckl" % data_folder, "rb"))      
    
    
    entry_probabilities = {}
    entry_init_meas_ids = {}
    
    for subcircuit_idx in subcircuit_entries:
        entry_init_meas_ids[subcircuit_idx] = {}
        i = 0
        for key in subcircuit_entries[subcircuit_idx]:
            entry_init_meas_ids[subcircuit_idx][key] = i
            i += 1
        jobs = scrambled(list(subcircuit_entries[subcircuit_idx].keys()))
        
  
        subcircuit = subcircuits[subcircuit_idx]
        entry_init_meas_ids = entry_init_meas_ids[subcircuit_idx]
        
        # rank_jobs = pickle.load(
        #     open("%s/rank_%d.pckl" % (args.data_folder, args.rank), "rb")
        # )

        uniform_p = 1 / 2**subcircuit.num_qubits
        jobs = {
                key: subcircuit_entries[subcircuit_idx][key] for key in jobs
        }
        for subcircuit_entry_init_meas in jobs:
            if eval_mode != "runtime":
                subcircuit_entry_term = jobs[subcircuit_entry_init_meas]
                subcircuit_entry_prob = None
                for term in subcircuit_entry_term:
                    coefficient, subcircuit_instance_init_meas = term
                    subcircuit_instance_init_meas_id = instance_init_meas_ids[
                        subcircuit_idx
                    ][subcircuit_instance_init_meas]
                    subcircuit_instance_prob = prob_dict[(subcircuit_idx, subcircuit_instance_init_meas_id)]
                    
                    if subcircuit_entry_prob is None:
                        subcircuit_entry_prob = coefficient * subcircuit_instance_prob
                    else:
                        subcircuit_entry_prob += coefficient * subcircuit_instance_prob
            else:
                subcircuit_entry_prob = uniform_p
            entry_init_meas_id = entry_init_meas_ids[subcircuit_entry_init_meas]
            # print('%s --> rank %d writing subcircuit_%d_entry_%d'%(args.data_folder,args.rank,subcircuit_idx,entry_init_meas_id))
            entry_probabilities[(subcircuit_idx, entry_init_meas_id)] = subcircuit_entry_prob
    print ("Printing single threaded version")
    print (entry_probabilities)
    return entry_probabilities
            


def attribute_shots2(subcircuit_entries, subcircuits, eval_mode, data_folder):
    import pickle
    meta_info = pickle.load(open("%s/meta_info.pckl" % data_folder, "rb"))
    instance_init_meas_ids = meta_info["instance_init_meas_ids"]
    num_workers = 20
    entry_init_meas_ids = {}
    for subcircuit_idx in subcircuit_entries:
        entry_init_meas_ids[subcircuit_idx] = {}
        i = 0
        for key in subcircuit_entries[subcircuit_idx]:
            entry_init_meas_ids[subcircuit_idx][key] = i
            i += 1
        jobs = scrambled(list(subcircuit_entries[subcircuit_idx].keys()))
        pickle.dump(
            {
                "subcircuits": subcircuits,
                "eval_mode": eval_mode,
                "instance_init_meas_ids": instance_init_meas_ids,
                "entry_init_meas_ids": entry_init_meas_ids,
            },
            open("%s/meta_info.pckl" % data_folder, "wb"),
        )
        num_workers = get_num_workers(
            num_jobs=len(jobs),
            ram_required_per_worker=2 ** subcircuits[subcircuit_idx].num_qubits
            * 4
            / 1e9,
        )
        procs = []
        for rank in range(num_workers):
            rank_jobs = find_process_jobs(jobs=jobs, rank=rank, num_workers=num_workers)
            rank_jobs = {
                key: subcircuit_entries[subcircuit_idx][key] for key in rank_jobs
            }
            if len(rank_jobs) > 0:
                pickle.dump(
                    rank_jobs, open("%s/rank_%d.pckl" % (data_folder, rank), "wb")
                )
                python_command = (
                    "python -m cutqc.parallel_attribute_shots --data_folder %s --subcircuit_idx %d --rank %d"
                    % (data_folder, subcircuit_idx, rank)
                )
                proc = subprocess.Popen(python_command.split(" "))
                procs.append(proc)
        [proc.wait() for proc in procs]
                      
    import glob
    import pickle
    
    measured_prob_dict = {}
    for subcircuit_idx in subcircuit_entries:
        # Get all pickle files for this subcircuit
      pickle_files = glob.glob(f"{data_folder}/subcircuit_{subcircuit_idx}_entry*.pckl")
          
          # Load all the pickle files into a dictionary
      
      for pickle_file in pickle_files:
                      
          # Extract instance ID from filename
          instance_id = int(pickle_file.split('_entry_')[1].split('.pckl')[0])
          with open(pickle_file, 'rb') as f:                
              measured_prob_dict[(subcircuit_idx, instance_id)] = pickle.load(f) 

    return measured_prob_dict
