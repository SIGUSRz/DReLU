'''
GPU_availability returns [least_occupied_GPU, ..., most_occupied_GPU].
Each element of the list is an GPU index (GPU index starts from 0).
It is ensured that the current performance of each GPU in the list is at most P2.
P0 is the maximum performance, indicating that one GPU is completely occupied.
P12 is the minimum performance.

Example:
  
  import mxnet as mx
  from GPU_availability import *

  l = GPU_availability()
  if len(l)>0:
    device = mx.gpu(l[0])
  else:
    raise Exception('No GPU available')
'''

def GPU_availability():
  import itertools
  from subprocess import Popen, PIPE
  output = Popen(['nvidia-smi'], stdout=PIPE).communicate()[0]
  lines = output.split('\n')
  performance = {}
  index = 0
  for i in range(len(lines)):
    if 'GTX' in lines[i]:
      p = int(lines[i+1].split(' '*4)[1][-1])
      if p>1:
        try:
          performance[p].append(index)
        except:
          performance.update({p : [index]})
      index += 1
  return list(itertools.chain(*[performance[key] for key in reversed(sorted(performance.keys()))]))

if __name__ == '__main__':
  print GPU_availability() 
