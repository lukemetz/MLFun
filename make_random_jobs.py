import pickle

job_names = ["jobs/job%d.pkl"%x for x in range(10)]

with open("jobs.txt", "w+") as jobs:
    jobs.write("\n".join(job_names)+"\n")

for job_name in job_names:
    pickle.dump({}, open(job_name, "w+"))
