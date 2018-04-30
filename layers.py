from neon.layers import Affine
from neon.transforms import Rectlin, Softmax
from neon.layers.layer import LookupTable
from neon.initializers import Xavier, GlorotUniform
from neon.backends.backend import Tensor
from neon.backends import gen_backend, Autodiff
from neon.layers.layer import Layer
from neon.models import Model
import numpy as np

be = gen_backend(backend='cpu')

def create_embedding_layer(vocab_size, embedding_dim, init=Xavier, update=True, pad_idx=None, name=None):
	return LookupTable(vocab_size, embedding_dim, init, update, pad_idx, name)

class LayerNorm(Layer):

    def __init__(self, shape, eps=1e-9, name=None):
        super(LayerNorm, self).__init__(eps, name)
        self.gamma = Tensor(be.ones(shape))
        self.beta = Tensor(be.zeros(shape))
        self.eps = eps

    def get_forward_optree(self):
        xvar = self.be.var(self.inputs, axis=0)
        xmean = self.be.mean(self.inputs, axis=0)
        print(xvar.shape,xmean.shape)
        xhat = (self.inputs - xmean) / self.be.sqrt(xvar + self.eps)
        return xhat * self.gamma + self.beta


    def fprop(self,inputs, inference=False):
        """
        Compute the actual fprop from op-tree, update the global estimations
        """
        self.inputs = inputs
        self.output = self.get_forward_optree()
        return self.output

    # def bprop(self, error):
    #     """
    #     Use Autodiff.back_prop_grad to back propagate gradients for the
    #     corresponding tensors.
    #     """
    #     if not self.deltas:
    #         self.deltas = error.reshape((self.nfm, -1))

    #     # autodiff will automatically cache and reuse the object
    #     # if we know the `error` buffer at init, we can also create the autodiff
    #     # object at layer's init
    #     ad = Autodiff(self.fprop_op_tree, self.be, next_error=self.deltas)

    #     # back propagate
    #     ad.back_prop_grad([self.x, self.gamma, self.beta],
    #                       [self.deltas, self.grad_gamma, self.grad_beta])

    #     return error

if __name__ == '__main__':
    n = 512
    a = np.random.randint(100,size=(1,100))
    a = be.array(a)
    print(a.shape)
    m = Model([LayerNorm(a.shape)])
    print(m.fprop(a))
