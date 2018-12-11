#This script submits jobs to the sunnyvale cluster (and resubmits jobs that stopped because of the 48hour walltime limit).

import os
import os.path
import numpy as np

def submit_job(path, job_name):
    os.system('mv %s %s'%(path, job_name))
    os.system('qsub %s'%job_name)
    os.system('mv %s %s'%(job_name,path))

###############################

idrange = [0,1]        #[first, last] job to be submitted

Njobs_counter = 0
for i in np.arange(idrange[0],idrange[1]):
    job_name = 'soap_job_id%d.pbs'%i
    path = 'jobs/%s'%job_name
    submit_job(path, job_name)     #submitting job for the first time
    Njobs_counter += 1

print('submitted %d jobs over idrange %d-%d'%(Njobs_counter,idrange[0],idrange[1]))
