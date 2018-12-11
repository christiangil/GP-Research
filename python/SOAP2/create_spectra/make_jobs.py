# This script generates jobs.
import numpy as np

########### Generate Jobs Singles ###########
def generate_jobs(jobs_dir, ids):
    for id in ids:
        job_name = "soap_job_id%d.pbs"%(id)
        sh_script_name = "%s%s"%(jobs_dir, job_name)
        with open(sh_script_name, 'w') as f:
            # f_head = open('job_header_cyberlamp.txt','r')
            f_head = open('job_header.txt','r')
            f.write(f_head.read())
            f_head.close()
            f.write('\npython soap_run.py %d >& outputs/soap_job_id%d_out.txt\n'%(id, id))
            f.close()

if __name__ == '__main__':
    jobs_dir = 'jobs/'
    ids = np.arange(10,20)
    print('made %d soap jobs'%len(ids))
    generate_jobs(jobs_dir, ids)
