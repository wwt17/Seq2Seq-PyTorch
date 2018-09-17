#!/bin/bash

#SBATCH --job-name=tb
#SBATCH --mem=12g
#SBATCH -c 4
#SBATCH --time=0
#SBATCH -o .log.tb

#singularity exec --nv /containers/images/ubuntu-16.04_tensforflow-1.8.0.img \
singularity_run "export LC_ALL='C' ; tensorboard --logdir log --port 20000"
