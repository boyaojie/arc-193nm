# Reproduction Guide
# Identifiability-Aware Inverse Design of 193 nm Anti-Reflective
# Coatings on Strongly Curved Substrates

This document explains how to reproduce all figures and tables of
the paper using a single script, `reproduce.py`.

--------------------------------------------------------------------------
1. SYSTEM REQUIREMENTS
--------------------------------------------------------------------------

- Operating system : Linux / macOS / Windows
- Python           : 3.9 or newer (3.10 recommended)
- RAM              : ≥ 8 GB (16 GB recommended)
- Disk             : ≥ 2 GB free (for dataset + figures + models)
- GPU              : optional but strongly recommended for Steps 6–9
                     (NVIDIA GPU with CUDA 11+; CPU works but is slow)

--------------------------------------------------------------------------
2. INSTALLATION
--------------------------------------------------------------------------

Create a fresh virtual environment and install the dependencies:

    # (a) Create & activate a virtual environment
    python -m venv venv
    source venv/bin/activate          # Linux / macOS
    venv\Scripts\activate             # Windows

    # (b) Install required packages
    pip install --upgrade pip
    pip install numpy scipy matplotlib scikit-learn
    pip install torch                 # CPU version by default
    pip install SALib                 # for Sobol sensitivity analysis

Optional, only if you want GPU-accelerated training:

    # NVIDIA GPU (CUDA 11.8)
    pip install torch --index-url https://download.pytorch.org/whl/cu118

If you already have PyTorch installed and it works, skip the torch step.

--------------------------------------------------------------------------
3. DIRECTORY LAYOUT
--------------------------------------------------------------------------

Place `reproduce.py` in an empty folder, e.g.:

    arc-193nm/
    ├── reproduce.py
    ├── README.md
    ├── fig/          (created automatically)
    ├── data/         (created automatically)
    └── paper/        (created automatically)

Run all commands from inside `arc-193nm/`.

--------------------------------------------------------------------------
4. QUICK START
--------------------------------------------------------------------------

To run the entire pipeline (all 10 steps) from scratch:

    python reproduce.py

This will:
  - generate the training dataset (Step 5),
  - run all 10 steps in order,
  - save figures to `fig/`, intermediate data to `data/`,
    and Markdown tables to `paper/`.

Total runtime (with GPU, pre-generated dataset):
  ≈ 30–50 minutes.

Total runtime (from scratch, CPU only):
  ≈ 8–16 hours, dominated by dataset generation and PINN training.

--------------------------------------------------------------------------
5. RUNNING SELECTED STEPS
--------------------------------------------------------------------------

Each step can be run independently. For example, to run only the
forward-model validation, the PINN, and the ablation study:

    python reproduce.py 1 6 7

Available steps:

    Step  1  Forward model validation           -> fig1_forward
    Step  2  FIM / Jacobian analysis            -> fig4_jacobian, fig5_identifiability
    Step  3  Sobol sensitivity analysis         -> data/sobol_results.npz
    Step  4  Curvature dependence               -> fig6_curvature
    Step  5  Dataset + baseline MLP             -> fig7_baseline, data/dataset.npz
    Step  6  Curvature-conditioned PINN         -> data/pinn.pt, data/pinn_results.npz
    Step  7  Ablation study                     -> fig10_ablation
    Step  8  Robustness analysis                -> fig14_robustness
    Step  9  NSGA-II benchmark                  -> fig15_nsga_time, fig16_nsga_perf
    Step 10  Optimal design + Table 7           -> fig12_schematic, fig13a_radial_valid,
                                                   fig13b_radial_full, paper/table7.md

Note: some steps depend on earlier outputs. See Section 7 below.

--------------------------------------------------------------------------
6. OUTPUT FILES
--------------------------------------------------------------------------

After a successful run, the following files will be present:

  fig/
    fig1_forward.pdf / .png          Forward model validation (3 panels)
    fig4_jacobian.pdf / .png         Jacobian magnitude + max sensitivity
    fig5_identifiability.pdf / .png  FIM, CRLB, weakest direction, Sobol
    fig6_curvature.pdf / .png        Curvature dependence (3 panels)
    fig7_baseline.pdf / .png         Dataset + baseline MLP (3 panels)
    fig10_ablation.pdf / .png        Ablation study (2 panels)
    fig12_schematic.pdf / .png       Layer structure of the optimal design
    fig13a_radial_valid.pdf / .png   Radial reflectance, valid aperture
    fig13b_radial_full.pdf / .png    Radial reflectance, full aperture
    fig14_robustness.pdf / .png      Monte-Carlo + thermal robustness
    fig15_nsga_time.pdf / .png       Time & evaluation count comparison
    fig16_nsga_perf.pdf / .png       Performance comparison (3 panels)

  data/
    dataset.npz                     50,000 training spectra + params
    fim_results.npz                 Jacobian, FIM, eigenvalues, CRLB
    sobol_results.npz               Sobol S1 / ST indices
    curvature_results.npz           Condition numbers vs R and theta
    baseline_results.npz            Baseline R² per parameter
    baseline.pt                     Trained baseline MLP weights
    pinn_results.npz                PINN R² per parameter
    pinn.pt                         Trained PINN weights
    ablation_results.npz            Ablation R² and loss curves
    robust_results.npz              Monte-Carlo robustness statistics
    benchmark_results.npz           NSGA-II vs PINN timings & accuracy

  paper/
    table7.md                       Optimal layer thicknesses (Markdown)

--------------------------------------------------------------------------
7. STEP DEPENDENCIES
--------------------------------------------------------------------------

The following dependencies must be respected:

  Step 5  ->  independent (generates dataset.npz)
  Step 6  ->  requires data/dataset.npz           (from Step 5)
  Step 7  ->  requires data/dataset.npz           (from Step 5)
  Step 8  ->  requires data/dataset.npz and data/pinn.pt (from Steps 5, 6)
  Step 9  ->  requires data/dataset.npz and data/pinn.pt (from Steps 5, 6)
  Step 1, 2, 3, 4, 10  ->  independent

If you run only Step 8 without having run Steps 5 and 6, the script
will raise a FileNotFoundError and tell you which file is missing.

Recommended full order:

    python reproduce.py 1 2 3 4 5 6 7 8 9 10

or simply:

    python reproduce.py

--------------------------------------------------------------------------
8. REPRODUCING ONLY THE FIGURES (WITH PRE-COMPUTED DATA)
--------------------------------------------------------------------------

If you already have:

    data/dataset.npz
    data/pinn.pt

you can skip dataset generation and training, and reproduce most
figures in a few minutes:

    python reproduce.py 1 2 3 4 7 8 9 10

Steps 5 and 6 will be skipped, and the remaining steps will use the
existing data.

--------------------------------------------------------------------------
9. TROUBLESHOOTING
--------------------------------------------------------------------------

(1) "ModuleNotFoundError: No module named 'tmm'"
    The script uses a minimal built-in TMM implementation and does not
    require the external `tmm` package. If you still see this error, it
    means you are running an older version of the script; download the
    latest `reproduce.py` and try again.

(2) "CUDA out of memory"
    Reduce the batch size in Step 6 / 7 (search for `batch_size=256` in
    `reproduce.py` and change to 128 or 64).

(3) Sobol analysis fails with "SALib not installed"
    Install it: `pip install SALib`. Step 3 is optional; if SALib is
    missing, the script skips it and Step 2 still uses the published
    Sobol values from the paper.

(4) Dataset generation is too slow on CPU
    If you already have `data/dataset.npz` from a previous run, move it
    into the `data/` folder before running Step 5. The script will load
    it instead of regenerating.

(5) Figures look different from the paper
    Check that you are using the same matplotlib version (≥ 3.5).
    Font differences (Arial vs. Helvetica) can also shift text layout;
    install Arial if necessary.

(6) "FileNotFoundError: data/pinn.pt"
    You must run Step 6 before Step 8 or Step 9. See Section 7.

--------------------------------------------------------------------------
10. CITATION
--------------------------------------------------------------------------

If you use this code in your own work, please cite:

    Bo, Y.; Hong, L. Identifiability-Aware Inverse Design of 193 nm
    Anti-Reflective Coatings on Strongly Curved Substrates.
    Coatings 2026, 1, 0. https://doi.org/10.3390/coatings1010000

--------------------------------------------------------------------------
11. CONTACT
--------------------------------------------------------------------------

For questions about the code, please open an issue on the GitHub
repository listed in the Data Availability Statement of the paper,
or contact the corresponding author.
