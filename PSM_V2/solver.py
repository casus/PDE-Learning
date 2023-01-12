import numpy as np
import torch
import time

from benchmark import Benchmark


class Solver(Benchmark):

    def __init__(self, dmns, crits, model, test_axes, grads=None, gt=None, data=None):
        super().__init__(model, gt, test_axes, data)
        self._dmns = dmns
        self._crits = crits
        self._grads = grads

    def closure(self, optim):
        optim.zero_grad()
        outputs = [self.model(_).T[0] for _ in self._dmns]

        # backward propagation
        if self._grads is not None:
            with torch.no_grad():
                grads = [grad(outputs[_]) for _, grad in enumerate(self._grads)]
                loss = sum([crit(outputs[_]) for _, crit in enumerate(self._crits)])

            for param in self.model.parameters():
                param.grad = sum([torch.autograd.grad(grads[_]*outputs[_], param, retain_graph=True, grad_outputs=torch.ones_like(grads[_]))[0] for _ in range(len(self._grads))])
        else:
            loss = sum([crit(outputs[_]) for _, crit in enumerate(self._crits)])
            loss.backward(retain_graph=True)

        self._losses = np.append(self._losses, loss.item())

        return loss

    def train(self, epochs, print_rate, optim, do_print=True):
        start = time.time()

        for epoch in range(epochs):
            optim.step(
                lambda: self.closure(
                    optim,
                ))

            # print criterion losses
            if ((epoch + 1) % print_rate == 0) & do_print:
                print(f'epoch {epoch+1}: loss = {self._losses[-1]:.16f}')

        end = time.time()
        self._time += end-start
