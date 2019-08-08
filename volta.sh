#!/bin/bash
# Configure the resources required
#SBATCH -M volta      
#SBATCH -p batch                                                # partition (this is the queue your job will be added to)
#SBATCH -n 1              	                                # number of tasks (sequential job starts 1 task) (check this if your job unexpectedly uses 2 nodes)
#SBATCH -c 1              	                                # number of cores (sequential job calls a multi-thread program that uses 8 cores)
#SBATCH --time=23:30:0                                        # time allocation, which has the format (D-HH:MM), here set to 1 hour
#SBATCH --gres=gpu:2                                      	# generic resource required (here requires 4 GPUs)
#SBATCH --mem=64GB                                              # specify memory required per node (here set to 16 GB)
# Configure notifications 
#SBATCH --mail-type=END                                         # Type of email notifications will be sent (here set to END, which means an email will be sent when the job is done)
#SBATCH --mail-type=FAIL                                        # Type of email notifications will be sent (here set to FAIL, which means an email will be sent when the job is fail to complete)
#SBATCH --mail-user=sunlibocs@gmail.com@gmail.com                    		# Email to which notification will be sent

source activate /fast/users/a1746546/envs/myenv
module load GCC/5.4.0-2.26
module load CUDA/9.0.176
module load cuDNN/7.3.1-CUDA-9.0.176

cd /fast/users/a1746546/code/DeepLab-v3-plus-cityscapes-Res50

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py



