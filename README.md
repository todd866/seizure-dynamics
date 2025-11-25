# Seizure as Dimensional Collapse

**Paper**: "Seizure Risk as Dimensional Profile Collapse: An EEG Complexity Framework"

**Author**: Ian Todd (Sydney Medical School, University of Sydney)

---

## Core Thesis

> **Seizure is dimensional collapse.** The healthy brain operates with a fat-tailed distribution of effective dimensions across scales and timescales. Seizure risk increases when this dimensional profile compresses—fewer modes, steeper eigenvalue spectra, lower participation ratio. This is the brain's "HRV": a continuously trackable complexity signature that predicts state transitions.

This extends the coherence compute framework (see `15_code_formation`) to clinical application:
- **D_eff is not a single number**—it's a profile D_eff(ε, τ) over error tolerance and timescale
- **Fat-tailed eigenvalue spectra** indicate rich, multi-scale dynamics (healthy)
- **Spectrum collapse** (steep, few-mode dominated) indicates seizure vulnerability
- **Stress and inflammation** may accelerate collapse by reducing the effective dimension of neural oscillators

---

## Key Hypotheses

1. **Pre-ictal dimensional collapse**: D_eff profile compresses 30-60 min before seizure onset
2. **Participation ratio drops**: Fewer effective modes explain variance as seizure approaches
3. **Tail steepening**: K_95/K_99 (components to reach 95%/99% variance) decrease pre-ictally
4. **Model-based D_eff**: Minimal latent dimension for reliable EEG prediction drops before seizures

---

## Planned Experiments

### Experiment 1: CHB-MIT Dimensional Profile Analysis
- Compute eigenvalue spectra per sliding window
- Extract: participation ratio, K_80/K_90/K_95/K_99
- Compare interictal vs pre-ictal vs ictal distributions
- Train classifier on dimensional features

### Experiment 2: Model-Based D_eff Estimation
- Fit linear state-space models with varying latent dimension k
- Define D_eff = minimal k for 95% of asymptotic prediction performance
- Track D_eff(t) around seizures

### Experiment 3: TUH Generalization
- Repeat analysis on Temple University Hospital corpus
- Test robustness across heterogeneous clinical data

---

## Repository Structure

```
16_seizure_dimensionality/
├── paper/
│   ├── seizure_dimensionality.tex
│   ├── seizure_dimensionality.pdf
│   ├── references.bib
│   └── figures/
│
├── src/                          # Flat Python files for review
│   ├── dimensional_profile.py    # Core analysis functions
│   ├── chb_mit_analysis.py       # CHB-MIT experiment
│   └── visualize_collapse.py     # Figure generation
│
├── experiments/
│   ├── 01_chb_mit_profile/
│   ├── 02_model_based_deff/
│   └── 03_tuh_generalization/
│
├── data/
│   └── chb-mit/                  # Downloaded dataset
│
└── archive/
    └── latex/
```

---

## Data Sources

1. **CHB-MIT Scalp EEG Database** (primary)
   - 24 pediatric patients, 163 seizures
   - 23-channel EEG, 256 Hz
   - https://physionet.org/content/chbmit/1.0.0/

2. **TUH EEG Seizure Corpus** (validation)
   - 500+ patients, 1000+ seizures
   - Heterogeneous clinical data

---

## Running the Code

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download CHB-MIT data
python src/download_chb_mit.py

# Run dimensional profile analysis
python experiments/01_chb_mit_profile/dimensional_profile.py

# Generate figures
python src/visualize_collapse.py
```

---

## Theoretical Framework

### EEG as Neural Oscillators
- Scalp EEG = linear mixture of many oscillatory modes
- Eigenvalue spectrum of covariance matrix reveals mode distribution
- **Fat tail** = many weak modes active (rich dynamics)
- **Steep spectrum** = few dominant modes (compressed dynamics)

### D_eff(ε, τ) - Effective Dimensionality as a Profile
- ε = error tolerance (how reliable must tracking be?)
- τ = timescale / bandwidth of interest
- D_eff = minimal model dimension for target reliability

### The Collapse Signature
```
Interictal:     [λ₁, λ₂, λ₃, λ₄, λ₅, λ₆, ...]  ← fat tail
Pre-ictal:      [λ₁, λ₂, λ₃, ...]               ← tail compresses
Ictal:          [λ₁, λ₂]                        ← collapse
```

---

## Key Citations

- Lundqvist et al. (2023) - Spatial computing in prefrontal cortex
- Gao et al. (2015) - Multiscale entropy in EEG
- Jeong (1998) - Embedding dimension in EEG
- Silva et al. (1999) - Correlation dimension in absence seizures

---

## Related Work

This extends:
> Todd, I. (2025) "Codes at Critical Capacity" - coherence compute framework

---

## License

MIT License. See LICENSE for details.

## Contact

Ian Todd
Sydney Medical School, University of Sydney
itod2305@uni.sydney.edu.au
