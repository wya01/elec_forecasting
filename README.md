# Elec_Forecasting: Privacy-Preserving Electricity Load Forecasting

This repository implements centralized (CL), federated (FL), and clustered federated learning (C-FL) approaches for household electricity load forecasting.  
The models are LSTM-based and extended with temporal (hour-of-day, day-of-week) and optional weather features.

---

## Repository Structure

elec_forecasting/
 ├── cluster/            # C-FL scripts (clustering, client definitions, postprocessing)
 ├── federated/          # FL scripts (client definitions, simulation, postprocessing)
 ├── models/             # LSTM model
 ├── utils/              # Preprocessing, training, visualization utilities
 ├── notebooks/          # Example notebooks (30min→30min horizon)
 │   ├── cl_30min_30min.ipynb
 │   ├── fl_30min_30min.ipynb
 │   ├── cfl_30min_30min.ipynb
 │   ├── cfl_t_30min_30min.ipynb
 │   └── cfl_tw_30min_30min.ipynb
 └── README.md

## Getting Started

### 1. Install dependencies
```bash
pip install -r requirements.txt**
```

### 2. Prepare data

Download datasets manually (not included in repo):

- Community 2024 dataset
- UK-DALE dataset

Place them under:

data/community2024/
data/UK-DALE/

### 3. Run experiments

- CL: centralized training
- FL: global federated model
- C-FL: clustered federated models
- C-FL+T: C-FL + time features
- C-FL+T+W: C-FL + time + weather features

Adjust `window_size` and `prediction_horizon` to try other horizons.

## Key Results

- C-FL improves over FL, especially with temporal features.
- Example (30min→30min):
  - MAE ↓ from 230.09 (FL) → 126.73 (C-FL+T) (~45% improvement)
  - R² ↑ from 0.48 → 0.64