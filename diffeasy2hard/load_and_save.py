import pickle
import os

import inspect
import hashlib

def function_hash(func):
    """
    Returns a short hash (8 chars) from the source code of a function.
    If no source is available (e.g., builtins), returns the function name.
    """
    try:
        src = inspect.getsource(func)
    except (OSError, TypeError):
        # fallback: just use the name
        return func.__name__

    h = hashlib.sha256(src.encode()).hexdigest()
    return h[:8]

def make_filename_from_args(directory, prefix="exp", **kwargs):
    def fmt(v):
        if isinstance(v, bool):
            return int(v)
        if callable(v):
            return function_hash(v)     # ⬅ hash used here
        if isinstance(v, (list, tuple)):
            return "-".join(map(str, v))
        return str(v)

    parts = [prefix]
    for k, v in kwargs.items():
        parts.append(f"{k}{fmt(v)}")

    return directory + "_".join(parts) 

def save_experiment_results(filename, results_all, **kwargs):
    """
    Save experiment results + all arguments to one file using pickle.
    
    Args:
        filename (str): Path to output file.
        results_all (dict/list): Simulation results to save.
        **kwargs: All arguments passed to learning_experiment
    """
    data = {
        "args": kwargs,
        "results": results_all
    }
    with open(filename, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved results to: {filename}")



def load_experiment_by_args(directory="experiments/", prefix="exp", **kwargs):
    """
    Load a saved experiment by reconstructing its filename from hyperparameters.

    Args:
        directory (str): Where experiment_*.pkl files are stored.
        prefix (str): Prefix used in the filename constructor.
        **kwargs: Hyperparameters used in the experiment.

    Returns:
        (args_dict, results) from the .pkl file.

    Raises:
        FileNotFoundError if no matching file exists.
    """
    filename = make_filename_from_args(prefix=prefix, **kwargs)
    path = os.path.join(directory, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No experiment file found for parameters. Tried:\n  {path}"
        )

    with open(path, "rb") as f:
        data = pickle.load(f)

    return data["args"], data["results"]