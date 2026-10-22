# Multi-Omic Causal Discovery in Retinal Degeneration (VB-GNN)

Code for the dependency-aware, multi-omic causal discovery framework based on Variational Bayesian Graph Neural Networks (VB-GNN), accompanying the manuscript *Multi-omic causal discovery in age-related retinal degeneration* and its Supplementary Information (SI; Supplementary Notes 1–3 and Supplementary Algorithm 1).

The framework treats acyclicity as a **soft prior** rather than a hard constraint, so that molecular feedback loops (e.g. the candidate mitochondrial–epigenetic loop) can be represented. Edge directions inside inferred cycles are hypotheses: the identifiability results for additive noise models apply to acyclic systems.

---

## Repository structure

```text
OmicsDepth-VB-GNN-RETINOPATHY/
├── data/
│   ├── dataset.py            # Multi-omics dataset (features + nuisance covariates)
│   ├── preprocess.py         # Donor-level pseudobulk, log, residualisation, standardisation, Ledoit-Wolf Sigma
│   └── synthetic.py          # Cyclic non-linear SEM with batch confounder (simulation study)
├── models/
│   ├── vbgnn.py              # Bernoulli edge posterior, Gumbel-Softmax relaxation, graph-masked SEM decoder, KL with soft acyclicity prior
│   └── discriminator.py      # Spectrally normalised, output-clipped adversarial discriminator + adversarial loss
├── utils/
│   ├── metrics.py            # EGS (Gini-Simpson) regulariser, Shannon baseline, acyclicity penalty h(Pi)
│   └── fdr_control.py        # Bayesian FDR for edges; Benjamini-Hochberg for p-values (Table 2)
├── scripts/
│   ├── train.py              # SI Algorithm 1 (minimax training, projection onto M_eps, Bayesian FDR)
│   ├── evaluate.py           # AUROC, PR-AUC, FDR, SHD
│   └── simulate_cf.py        # Counterfactual do()-intervention (abduction-action-prediction, fixed point for cycles)
├── tests/
│   └── test_smoke.py         # Smoke tests
├── requirements.txt
└── README.md
```

## Correspondence with the Supplementary Information

| SI element | Code |
|---|---|
| Bernoulli posterior q(Z), edge probabilities Pi, projection onto M_eps = [eps, 1-eps] | `VBGNN.pi`, `VBGNN.project_()` |
| Prior p(Z) with rho_high / rho_low and soft acyclicity gamma·h(Z), h = Tr(exp(Z∘Z)) − d | `VBGNN.kl_divergence`, `utils.metrics.acyclicity_penalty` |
| Gumbel-Softmax relaxation (temperature tau) | `VBGNN.sample_gumbel_softmax` |
| Structural equations x_j = f_j(x_Pa_j) + eps_j, eps_j ~ N(0, sigma_j²) | `VBGNN.decode`, `VBGNN.neg_log_likelihood` |
| EGS structural loss lambda·D·beta·Tr(Piᵀ Sigma Pi), D ∈ [1 − d², 1] | `utils.metrics.EGSRegularizer` |
| Adversarial objective, T = product of marginals (batch shuffling), D_psi ∈ [delta, 1 − delta] | `models.discriminator` |
| Algorithm 1 (encoder descent, discriminator ascent, projection) | `scripts.train.train_vbgnn` |
| Bayesian FDR on posterior edge probabilities | `utils.fdr_control.bayesian_fdr_edges` |
| Donor-level pseudobulk; residualisation for age, sex, PMI, batch | `data.preprocess` |
| Counterfactual in silico perturbation | `scripts.simulate_cf.do_intervention`, `restoration_percent` |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run all commands from the repository root.

```bash
python -m tests.test_smoke        # smoke tests
python -m scripts.train           # demo on synthetic cyclic data
python -m scripts.simulate_cf     # demo counterfactual intervention
```

Using your own data:

```python
import numpy as np
from data.preprocess import preprocess_pipeline
from scripts.train import TrainConfig, train_vbgnn

# X_raw: [n_donors, d] donor-level features (scRNA-seq already aggregated to pseudobulk)
# covariates: [n_donors, k] (batch, age at death, sex, PMI)
# prior_adj: [d, d] {0,1} curated prior (e.g. KEGG), entry [i, j] = edge i -> j
X, Sigma = preprocess_pipeline(X_raw, covariates)
out = train_vbgnn(X, prior_adj, covariates=covariates, cov_matrix=Sigma, cfg=TrainConfig())
Pi, edges = out["Pi"], out["edges"]   # posterior edge probabilities, FDR-controlled edges
```

## Notes and limitations

* Memory scales as O(d² · hidden_dim) because each node has its own parent-masked first layer; pre-select features before fitting.
* The EGS diversity term D = 1 − Σ pi² becomes negative once Σ pi² > 1. In that regime the product D·Tr(PiᵀΣPi) decreases as edges become denser, so the structural term can favour dense graphs. The default follows the manuscript; `EGSRegularizer(normalized=True)` (D = 1 − mean pi²) is available for sensitivity analyses only and is not the formulation analysed in the SI.
* The clipping level `clip_eps` and temperature `tau` are part of the method: the gradient bounds of SI Proposition 1 hold on the clipped domain.
* Counterfactual results are model predictions and require experimental validation.

## Citation

See `CITATION.cff`. A Zenodo DOI for the archived release will be added on publication.

## Data availability

Raw and processed data will be available at [GEO/EGA accession XXXXXX] and [MetaboLights / Metabolomics Workbench accession XXXXXX] (to be completed by the authors).
