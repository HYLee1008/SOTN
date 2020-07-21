# SOTN
Paper and codes of my SOTN algoritm submitted to ICPR2020.

Because of the file size limit, I could't upload the weights of the backbone netowrk and SOTN network, .npy and .tfrecord of the input embedding vectors.

Please create ./plt directory before run main.py. It will produce code histogram of each iteration.
Every wegiths file should be cotained at ./model directory. Please create ./model directory on your own.
Every .npy files should be contained at ./npy directory. Please creatd ./npy directory on your own.
Every .tfrecord files should be contaied at ./src directory.

# Python file description.
main.py - It runs the SOTN network.
backbone_eval.py - It evaluates the backbone network (in here, arcface network) based on the cosine distance.
original_IOM.py - It evaluates the Index-of-Max (IoM) hashing.
permutation.py - It evaluates the permutated SOTN hash codes.

# Directory description.
backbones - The arcface backbone network. 
configs - Configuration of the arcface backbone network.
model - The weights of the arcface backbone network and the SOTN network.
npy - .npy files for evaluation.
plt - Code histogram of each iteration will be generated here when you run main.py.
src - Utility codes which are necessary for my codes.
unlinkability - Unlinkability evaluation tools of my SOTN network.
