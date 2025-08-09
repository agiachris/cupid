## 🛠️ Installation: Miniconda + Mamba for Linux

### Step 1: Install Miniconda
```bash
# Install conda. Optionally, see instructions: https://www.anaconda.com/docs/getting-started/miniconda/install
curl -sL "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" > "Miniconda3.sh"
bash Miniconda3.sh -b -p ~/miniconda3
rm Miniconda3.sh

# Activate conda.
source ~/miniconda3/bin/activate
~/miniconda3/bin/conda init bash
conda config --set auto_activate_base true

# Update and clean.
conda update -n base -c defaults conda
conda clean --all --yes
rm -rf ~/.conda/locks/*
```

### Step 2: Disable Classic Solver
Manually edit the `~/.condarc` file to disable classic solver and include the following:
```bash
auto_activate_base: true
channels:
  - conda-forge
  - defaults
# solver: classic
```

### Step 3: Install Mamba (Version 2.0.4)
```bash
source ~/.bashrc
conda install -n base -c conda-forge mamba=2.0.4
```