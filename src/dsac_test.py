import os

print("------------------------------------------")
print("From inside python script")
print("Conda Env:"+os.environ['CONDA_DEFAULT_ENV'])
print("Conda Path:"+os.environ['CONDA_PREFIX'])
print("------------------------------------------")