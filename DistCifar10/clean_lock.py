from distribute import worker_from_url
github_remote = "git@github.com:lukemetz/cifar10_sync.git"
worker = worker_from_url(github_remote, path="cifar10_sync_clean")
worker.release_lock(force_release=True)

f = open(worker.path + "running.txt", "wb+")
f.write("")
f.close()

worker._commit_changes("clear running")
