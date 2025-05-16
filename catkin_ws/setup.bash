source ~/miniconda3/etc/profile.d/conda.sh
conda activate DAP

export PYTHONPATH=$CONDA_PREFIX/lib/python3.8/site-packages:$PYTHONPATH
export PATH=$CONDA_PREFIX/bin:$PATH