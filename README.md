# Brain Connectivity Project

This project contains Python notebooks, plots, data files, and code for analyzing brain connectivity metrics. It uses scientific Python libraries along with the Brain Connectivity Toolbox (`bctpy`) and `nltools`.

## What's Included

- `.ipynb` — Jupyter notebooks with analysis  
- `.csv`, `.txt` — input datasets and output values  
- `.png`, `.jpg` — generated plots  
- `utils.py` — helper functions (if any)  
- `environment.yml` — conda environment file to install everything  

---

## How to Set It Up (Step-by-Step)

This guide assumes you are **starting from scratch** with no setup.

### 1. Install Miniconda (once)

Go to: https://docs.conda.io/en/latest/miniconda.html  
Download and install the version for your system.

---

### 2. Unzip the Project Folder

You likely received this project as a `.zip` file.  
- Right-click and extract it  
- Open a terminal and `cd` into the folder

Example:
```bash
cd ~/Downloads/brain-connectivity-project
```

---

### 3. Create the Environment

This creates a Python environment with all the required libraries:

```bash
conda env create -f environment.yml
```

---

### 4. Activate the Environment

```bash
conda activate brain-env
```

If you see an error, make sure the previous step succeeded and you're in the correct folder.

Run this the first time when you first open the project folder
```bash
python -m ipykernel install --user --name=brain-env --display-name "Python (brain-env)"
```

---

### 5. Launch JupyterLab

```bash
jupyter lab
```

A browser tab will open. You can now open and run the `.ipynb` files.

---

## Troubleshooting

- If conda gives a "file not found" error, make sure you're in the same folder as `environment.yml`
- If the environment breaks, delete and recreate it:
  ```bash
  conda env remove -n brain-env
  conda env create -f environment.yml
  ```

---

## Included Libraries

- `numpy`, `pandas`, `matplotlib`, `seaborn`
- `networkx`, `scikit-learn`
- `bctpy`, `nltools` (installed via pip)

All of these will install automatically when you create the environment.
