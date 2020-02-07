#!/bin/bash -l

/isg/shared/apps/singularity/3.5.2/bin/singularity build ~/copasi_latest.img docker://crbm/biosimulations_copasi_api:latest <<<y  1> ~/cron_copasi.log 2>&1 &