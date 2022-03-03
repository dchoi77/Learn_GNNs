- Resource: https://github.com/dmlc/dgl/tree/master/examples/pytorch/grand

- Author's code: https://github.com/THUDM/GRAND

- Paper: https://arxiv.org/abs/2005.11079

- Illustration of GRAND in the paper

![illustration of GRAND](assets/papers_05.png)

- DropNode

![dropnode](assets/dropnode.png)

- Propagation: adopt the mixed-order propagation

![propagation](assets/propagation.png)

- Prediction

![prediction](assets/prediction.png)

We can apply softmax (or log_softmax followed by exp) to the output of of the MLP layer so that we can compute loss assuming positive prediction.

- Loss 

![loss](assets/loss.png)


Note the following:

* <img src="https://latex.codecogs.com/png.latex?\overline{\mathbf{Z}}^{\prime}"> is detached from the computation graph in the code. Thus its gradient is not computed in the loss function.
* The original code computes training accuracy over one output <img src="https://latex.codecogs.com/png.latex?\overline{\mathbf{Z}}^{(1)}">, but we computes average accuracy over the list of outputs.
* Each of  <img src="https://latex.codecogs.com/png.latex?\tilde{\mathbf{Z}}_{i}^{(s)},\overline{\mathbf{Z}}_{i},\overline{\mathbf{Z}}_{i}^{\prime}"> has nonnegative entries whose sum is one.
