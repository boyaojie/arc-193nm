# dataset_gen.py（方案B：n_r=4 + 多进程）
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from config import (WLS, R_APERTURE, R_MIN, R_MAX, SEED)
from forward import sphere_average_reflectance

np.random.seed(SEED)

N = 50000
N_CORES = 8

d_base = np.array([53.72, 30.15, 49.98, 23.23, 47.87, 58.44, 124.48])
d_ranges = [(d * 0.8, d * 1.2) for d in d_base]

d_samples = np.zeros((N, 7))
for i, (lo, hi) in enumerate(d_ranges):
    d_samples[:, i] = np.random.uniform(lo, hi, N)

n_la = np.random.uniform(1.58, 1.60, N)
n_mg = np.random.uniform(1.38, 1.40, N)
R_samples = np.random.uniform(R_MIN, R_MAX, N)

params = np.column_stack([d_samples, n_la, n_mg, R_samples])


def compute_one(args):
    i, d, n1, n2, R = args
    layers = [(n1, d[0]), (n2, d[1]), (n1, d[2]), (n2, d[3]),
              (n1, d[4]), (n2, d[5]), (n1, d[6])]
    spec, _ = sphere_average_reflectance(
        WLS, layers, R, r_max=R_APERTURE, n_r=4, pol='avg')  # n_r=4
    return i, spec


if __name__ == '__main__':
    args_list = [(i, d_samples[i], n_la[i], n_mg[i], R_samples[i])
                 for i in range(N)]

    print(f"Generating {N} samples with {N_CORES} cores...")
    with Pool(N_CORES) as p:
        results = list(tqdm(p.imap(compute_one, args_list), total=N))

    spectra = np.zeros((N, len(WLS)), dtype=np.float32)
    for i, spec in results:
        spectra[i] = spec

    SIGMA_NOISE = 0.005
    spectra_noisy = spectra + np.random.normal(0, SIGMA_NOISE, spectra.shape)
    spectra_noisy = np.clip(spectra_noisy, 0, 1)

    np.savez('dataset.npz',
             spectra=spectra_noisy.astype(np.float32),
             spectra_clean=spectra.astype(np.float32),
             params=params.astype(np.float32),
             wls=WLS)
    print("Saved: dataset.npz")