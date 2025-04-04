# HierarchicalPCGrad

This repository provides a reimplementation for [Gradient Surgery for Multi-Task Learning](https://arxiv.org/pdf/2001.06782.pdf) in PyTorch 1.6.0. 
The method has further been extended to handle hierarchical task projection and combinations of intra- and inter-priority task projections.

## Setup
Install the required packages via:
```
pip install -r requirements.txt
```

## Usage

```python
import torch
import torch.nn as nn
import torch.optim as optim
from pcgrad import PCGrad

# wrap your favorite optimizer
optimizer = PCGrad(optim.Adam(net.parameters())) 
losses = [...] # a list of per-task losses
priorities = [...] # a list of priorities of tasks, e.g. [1,1,3,2,...]
assert len(losses) == num_tasks
optimizer.pc_backward(losses) # calculate the gradient can apply gradient modification
optimizer.step()  # apply gradient step
```

## Training
- Mulit-MNIST 
  Please run the training script via the following command. Part of implementation is leveraged from https://github.com/intel-isl/MultiObjectiveOptimization
  ```
  python main_multi_mnist.py
  ```
  The result is shown below.
  | Method                  | left-digit | right-digit |
  | ----------------------- | ---------: | ----------: |
  | Jointly Training        |      90.30 |       90.01 |
  | **PCGrad (this repo.)** |  **95.00** |   **92.00** |
  | PCGrad (official)       |      96.58 |       95.50 |

- Cifar100-MTL
  coming soon 
## Reference

Please cite as:

```
@article{yu2020gradient,
  title={Gradient surgery for multi-task learning},
  author={Yu, Tianhe and Kumar, Saurabh and Gupta, Abhishek and Levine, Sergey and Hausman, Karol and Finn, Chelsea},
  journal={arXiv preprint arXiv:2001.06782},
  year={2020}
}

@misc{Pytorch-PCGrad,
  author = {Wei-Cheng Tseng},
  title = {WeiChengTseng/Pytorch-PCGrad},
  url = {https://github.com/WeiChengTseng/Pytorch-PCGrad.git},
  year = {2020}
}
```
