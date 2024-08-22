#!/bin/bash

#PBS 
#PBS -l ncpus=18
#PBS -l mem=16GB
#PBS -l walltime=48:00:00
#PBS -l storage=gdata/dk92+gdata/gh47
#PBS -q normal
#PBS -r y
#PBS -l wd

module use /g/data/dk92/apps/Modules/modulefiles
module avail NCI-data-analysis
module load NCI-data-analysis/2021.06
source /g/data/dk92/apps/anaconda3/2020.12/etc/profile.d/conda.sh
conda activate /home/587/pc3426/envs/pmbp_env

BASE="/g/data/gh47/pc3426/jobs/E2E_hybrid_reg_mean_nobasereg"

for i in 0
do
    FILE="${BASE}/hypertuning/2D_T${i}.p"
    FINAL="${BASE}/hypertuning/2D_T${i}_h.p"
    if [ -f "$FILE" ] 
    then
        echo "$FILE is processing / already processed."
    else
        touch $FILE

        SCRATCH="/scratch/gh47/pc3426/E2E_hybrid_reg_mean_nobasereg_$i"
        
        if [ ! -d "${SCRATCH}" ]
        then
            mkdir -p "${SCRATCH}"
            mkdir "${SCRATCH}/samples"
            mkdir "${SCRATCH}/hypertuning"
            mkdir "${SCRATCH}/log"
        fi
        
        cd ${PBS_O_WORKDIR}
        cp samples/us_and_au_reducedopinionset_hourly_farright.p "${SCRATCH}/samples"
        cp samples/googletrends.p "${SCRATCH}/samples"
        cp samples/RELIABLE_UNRELIABLE_K12.p "${SCRATCH}/samples"
        cp samples/RELIABLE_UNRELIABLE_K12_d.p "${SCRATCH}/samples"
        
        cd ${SCRATCH}
        python ${PBS_O_WORKDIR}/opinion_main_varyT_2D.py $i

        cp "${SCRATCH}/hypertuning/"* "${BASE}/hypertuning/"
        cp "${SCRATCH}/log/"* "${BASE}/log/"
        
        if [ -f "$FINAL" ]
        then
            rm -rf ${SCRATCH}
        fi

        cd ${PBS_O_WORKDIR}
    fi
done
