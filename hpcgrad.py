import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
import numpy as np
import copy
import random


class HPCGrad():
    def __init__(self, optimizer, reduction='sum'):
        self._optim, self._reduction = optimizer, reduction
        self.eps = np.finfo(float).eps
        return

    @property
    def optimizer(self):
        return self._optim

    def zero_grad(self):
        '''
        clear the gradient of the parameters
        '''

        return self._optim.zero_grad(set_to_none=True)

    def step(self):
        '''
        update the parameters with the gradient
        '''

        return self._optim.step()

    def pc_backward(self, objectives, priorities):
        '''
        calculate the gradient of the parameters

        input:
        - objectives: a list of objectives
        - priorities: a list of integers of objective priorities
        '''

        grads, shapes, has_grads = self._pack_grad(objectives)
        pc_grad = self._project(grads, has_grads, priorities)
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad)
        return
    
    def _project(self, grads, has_grads, priorities, shapes=None):
        prio_lb = np.min(np.array(priorities))
        prio_ub = np.max(np.array(priorities))
        grads_list = []
        has_grads_list = []
        # get all grads of this prio and perform standard PCGrad
        for prio in range(prio_lb, prio_ub+1):
            idx = [i for i, x in enumerate(priorities) if x == prio]
            grads_prio = []
            has_grads_prio = []
            for i in idx:
                grads_prio.append(grads[i])
                has_grads_prio.append(has_grads[i])
            #print('priorrity ', prio, ' has ', len(grads_prio), ' objectives')
            grad_prio = self._project_conflicting(grads_prio, has_grads_prio)
            grads_list.append(grad_prio)
            has_grad = torch.clip(torch.sum(torch.stack(has_grads_prio),dim=0), 0, 1)# sum all has_grads_prio tensors along the list dimension
            has_grads_list.append(has_grad) 

        # perform hierarchical PCgrad among the prio classes
        pc_grad = self._project_conflicting_hierarchical(grads_list, has_grads_list)
        return pc_grad
    
    def _project_conflicting(self, grads, has_grads, shapes=None):
        shared = torch.stack(has_grads).prod(0).bool()
        pc_grad, num_task = copy.deepcopy(grads), len(grads)
        for g_i in pc_grad:
            random.shuffle(grads)
            for g_j in grads:
                g_i_g_j = torch.dot(g_i, g_j)
                if g_i_g_j < 0 and g_j.norm() > self.eps:
                    g_i -= (g_i_g_j) * g_j / (g_j.norm()**2)

        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
        if self._reduction:
            merged_grad[shared] = torch.stack([g[shared]
                                           for g in pc_grad]).mean(dim=0)
        elif self._reduction == 'sum':
            merged_grad[shared] = torch.stack([g[shared]
                                           for g in pc_grad]).sum(dim=0)
        else: exit('invalid reduction method')

        merged_grad[~shared] = torch.stack([g[~shared]
                                            for g in pc_grad]).sum(dim=0)
        return merged_grad
    
    def _project_conflicting_hierarchical(self, grads, has_grads, shapes=None):
        shared = torch.stack(has_grads).prod(0).bool()
        pc_grad, num_task = copy.deepcopy(grads), len(grads)
        # project each gradient onto the nullspace of the higher-priority tasks
        for j, g_j in enumerate(grads):
            if j+1<num_task:
                secondary_grads = pc_grad[j+1:]
                for i, g_i in enumerate(secondary_grads):
                    g_i_g_j = torch.dot(g_i, g_j)
                    if g_i_g_j < 0 and g_j.norm() > self.eps:
                        g_i -= (g_i_g_j) * g_j / (g_j.norm()**2)
            else:
                pass

        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
        if self._reduction:
            merged_grad[shared] = torch.stack([g[shared]
                                           for g in pc_grad]).mean(dim=0)
        elif self._reduction == 'sum':
            merged_grad[shared] = torch.stack([g[shared]
                                           for g in pc_grad]).sum(dim=0)
        else: exit('invalid reduction method')

        merged_grad[~shared] = torch.stack([g[~shared]
                                            for g in pc_grad]).sum(dim=0)
        return merged_grad

    def _set_grad(self, grads):
        '''
        set the modified gradients to the network
        '''

        idx = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                p.grad = grads[idx]
                idx += 1
        return

    def _pack_grad(self, objectives):
        '''
        pack the gradient of the parameters of the network for each objective
        
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grads, shapes, has_grads = [], [], []
        for obj in objectives:
            self._optim.zero_grad(set_to_none=True)
            obj.backward(retain_graph=True)
            grad, shape, has_grad = self._retrieve_grad()
            grads.append(self._flatten_grad(grad, shape))
            has_grads.append(self._flatten_grad(has_grad, shape))
            shapes.append(shape)
        return grads, shapes, has_grads

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _flatten_grad(self, grads, shapes):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _retrieve_grad(self):
        '''
        get the gradient of the parameters of the network with specific 
        objective
        
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grad, shape, has_grad = [], [], []
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                # tackle the multi-head scenario
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    has_grad.append(torch.zeros_like(p).to(p.device))
                    continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
                has_grad.append(torch.ones_like(p).to(p.device))
        return grad, shape, has_grad


class TestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self._linear = nn.Linear(3, 4)

    def forward(self, x):
        return self._linear(x)


class MultiHeadTestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self._linear = nn.Linear(3, 2)
        self._head1 = nn.Linear(2, 4)
        self._head2 = nn.Linear(2, 4)

    def forward(self, x):
        feat = self._linear(x)
        return self._head1(feat), self._head2(feat)


if __name__ == '__main__':

    # fully shared network test
    torch.manual_seed(4)
    x, y = torch.randn(2, 3), torch.randn(2, 4)
    net = TestNet()
    y_pred = net(x)
    pc_adam = HPCGrad(optim.Adam(net.parameters()))
    pc_adam.zero_grad()
    loss1_fn, loss2_fn = nn.L1Loss(), nn.MSELoss()
    loss1, loss2 = loss1_fn(y_pred, y), loss2_fn(y_pred, y)

    pc_adam.pc_backward([loss1, loss2], [1,1])
    for p in net.parameters():
        print(p.grad)

    print('-' * 80)
    # seperated shared network test

    torch.manual_seed(4)
    x, y = torch.randn(2, 3), torch.randn(2, 4)
    net = MultiHeadTestNet()
    y_pred_1, y_pred_2 = net(x)
    pc_adam = HPCGrad(optim.Adam(net.parameters()))
    pc_adam.zero_grad()
    loss1_fn, loss2_fn = nn.MSELoss(), nn.MSELoss()
    loss1, loss2 = loss1_fn(y_pred_1, y), loss2_fn(y_pred_2, y)

    pc_adam.pc_backward([loss1, loss2], [1,1])
    for p in net.parameters():
        print(p.grad)
