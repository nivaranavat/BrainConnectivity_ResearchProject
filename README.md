# Brain Connectivity Project

This project contains Python notebooks, plots, data files, and code for analyzing brain connectivity metrics. It uses scientific Python libraries along with the Brain Connectivity Toolbox (`bctpy`) and `nltools`.

## What's Included

- `.ipynb` — Jupyter notebooks with analysis  
- `.csv`, `.txt` — input datasets and output values  
- `.png`, `.jpg` — generated plots  
- `source/` — all main Python analysis code  
- `environment.yml` — conda environment file to install everything  

---

## What is Conda?

**Conda** is a tool for managing Python environments and packages. It makes it easy to install all the scientific libraries you need, without worrying about system conflicts. You do not need to use `pip` for this project—**just follow the steps below!**

---

## Step-by-Step Setup Guide (for Beginners)

### 1. Install Miniconda (one-time)

- Go to: https://docs.conda.io/en/latest/miniconda.html  
- Download and install the version for your system (Windows, Mac, or Linux).
- Follow the instructions on the website. (On Mac, you may need to allow the installer in System Preferences > Security.)

### 2. Unzip the Project Folder

- If you received a `.zip` file, right-click and extract it.
- Open a **terminal** (on Mac: open the Terminal app; on Windows: open Anaconda Prompt or Command Prompt).
- Use `cd` to change into the project folder. Example:

```bash
cd ~/Downloads/BrainConnectivity_ResearchProject
```

### 3. Create the Conda Environment

This will install Python and all required libraries:

```bash
conda env create -f environment.yml
```

- This may take a few minutes. It will create an environment called **brain-env**.

### 4. Activate the Environment

```bash
conda activate brain-env
```

- You should see `(brain-env)` at the start of your terminal prompt.
- If you see an error, make sure you are in the folder with `environment.yml` and that the previous step finished without errors.

### 5. (Optional) Register the Environment with Jupyter

This step lets you select the environment in JupyterLab:

```bash
python -m ipykernel install --user --name=brain-env --display-name "Python (brain-env)"
```

### 6. Launch JupyterLab

```bash
jupyter lab
```

- A browser tab will open. You can now open and run the `.ipynb` files.

---

## Troubleshooting Conda

- **Command not found?** Make sure you installed Miniconda and restarted your terminal.
- **File not found error?** Make sure you are in the folder with `environment.yml`.
- **Environment already exists?** Remove it and try again:
  ```bash
  conda env remove -n brain-env
  conda env create -f environment.yml
  ```
- **Can't activate?** Try closing and reopening your terminal, or run `conda init` and restart the terminal.
- **Jupyter not found?** Make sure you activated the environment before running `jupyter lab`.

---

## How to Deactivate or Remove the Environment

- To **deactivate** the environment:
  ```bash
  conda deactivate
  ```
- To **remove** the environment completely:
  ```bash
  conda env remove -n brain-env
  ```

---

## Included Libraries

- `numpy`, `pandas`, `matplotlib`, `seaborn`
- `networkx`, `scikit-learn`
- `bctpy`, `nltools` (installed via pip)

All of these will install automatically when you create the environment.

---

## Need More Help?
- See the [Conda Getting Started Guide](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html)
- Or ask your project maintainer for help!
