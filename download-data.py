import os
import shutil
import tarfile

os.system("curl -O https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz")
with tarfile.open('aclImdb_v1.tar.gz', 'r:gz') as tar:
    tar.extractall()
os.rename('aclImdb', 'data')
os.remove('aclImdb_v1.tar.gz')
