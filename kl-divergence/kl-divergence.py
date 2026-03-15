import numpy as np

def kl_divergence(p, q, eps=1e-12):
    """
    Compute KL Divergence D_KL(P || Q).
    """
    # Write code here
    p = np.asarray(p, dtype=np.float32) + eps 
    q = np.asarray(q, dtype=np.float32) + eps

    kl_div = np.sum(p * np.log(p / q))

    return kl_div
    