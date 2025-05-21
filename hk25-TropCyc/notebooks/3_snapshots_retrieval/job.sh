#!/bin/bash
#SBATCH --job-name="NodeFileCompose"
#SBATCH --time=04:00:00
#SBATCH -n 1
#SBATCH --mem=10GB
#SBATCH --account=huracan
#SBATCH --partition=standard #standard or debug
#SBATCH --qos=short
#SBATCH -o NFC.out
#SBATCH -e NFC.err

conda run -n hackathon python Pre-processing.py

./NodeFileCompose.sh