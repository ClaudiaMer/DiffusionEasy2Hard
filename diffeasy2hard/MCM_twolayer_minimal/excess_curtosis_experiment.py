import argparse
import gc
import torch
import torch.nn.functional as F
import numpy as np
from data import three_mixture, students_t
from train import learning_experiment




# ----------------------
# Activation functions
# ----------------------
def scaled_tanh(x):
    return 0.1 * F.tanh(10.0 * x)


def quad_relu(x):
    return F.relu(x) ** 2


def quad_tanh(x):
    return F.tanh(x) ** 2


def quad(x):
    return x ** 2


ACTIVATIONS = {
    "relu": F.relu,
    "tanh": F.tanh,
    "scaled_tanh": scaled_tanh,
    "quad": quad,
    "quad_relu": quad_relu,
    "quad_tanh": quad_tanh,
    "sigmoid": F.sigmoid
}


SAMPLE_FNS = {
    "sign": np.sign,
    "three_mixture": three_mixture,
    "students_t": students_t,
}


# ----------------------
# Main
# ----------------------
def main(args):
    act = ACTIVATIONS[args.activation]
    sample_fn = SAMPLE_FNS[args.sample_fn]

    print(
        f"Running experiment with:\n"
        f"  activation = {args.activation}\n"
        f"  m          = {args.m}\n"
        f"  nits       = {args.nits}\n"
        f"  d          = {args.d}\n"
        f"  ninits     = {args.ninits}\n"
        f"  eta        = {args.eta}\n"
        f"  bs         = {args.bs}\n"
        f"  t          = {args.t_val}\n"
        f"  adam       = {args.adam}\n"
        f"  add_cov_spike = {args.add_cov_spike}\n"
        f"  correlated_latents = {args.correlated_latents}\n"
    )

    results = learning_experiment(
        nits=args.nits,
        d=args.d,
        m=args.m,
        ninits=args.ninits,
        eta=args.eta,
        bs=args.bs,
        wnorm=args.w_norm,
        sample_first_dim=sample_fn,
        Adam=args.adam,
        act=act,
        t=args.t_val,
        save=True,
        N=args.N, 
        skip=args.skip, 
        add_cov_spike = args.add_cov_spike, 
        correlated_latents = args.correlated_latents
    )

    # Explicit cleanup (important on clusters)
    del results
    gc.collect()
    torch.cuda.empty_cache()

    print("Experiment finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--activation", type=str, required=True,
                        choices=ACTIVATIONS.keys())
    parser.add_argument("--m", type=int, required=True)

    parser.add_argument("--nits", type=int, default=500_000)
    parser.add_argument("--d", type=int, default=100)
    parser.add_argument("--ninits", type=int, default=5)
    parser.add_argument("--eta", type=float, default=1e-3)
    parser.add_argument("--bs", type=int, default=100)
    parser.add_argument("--w-norm", type=float, default=1.0)
    parser.add_argument("--t-val", type=float, default=0.1)
    parser.add_argument("--adam", action="store_true")
    parser.add_argument("--sample-fn", type=str, default="sign",
                        choices=SAMPLE_FNS.keys())
    
    parser.add_argument("--N", type=int, default=2)
    parser.add_argument("--skip", type=float, default=0)
    parser.add_argument("--add-cov-spike", action="store_true")
    parser.add_argument("--correlated-latents", action="store_true")

    args = parser.parse_args()
    main(args)
