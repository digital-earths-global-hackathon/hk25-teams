# Getting started on JASMIN

1. Go to https://notebooks.jasmin.ac.uk/ and log in
2. Open a terminal
3. Run:

```bash
mkdir Downloads
cd Downloads
wget https://raw.githubusercontent.com/digital-earths-global-hackathon/tools/refs/heads/main/python_envs/environment.yaml
conda env create -f environment.yaml -n hackathon_env
conda activate hackathon_env
python -m ipykernel install --user --name hackathon_env  # is this step necessary?
```
4. <wait a couple of minutes>
5. Create a new notebook, you *should* be able to select `hackathon_env` 