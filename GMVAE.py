import matplotlib
import matplotlib.pyplot as plt
import argparse
import random
import numpy as np
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.utils.data
import torch
from torch import nn
from torch.nn import functional as F
from scipy.io import loadmat
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import normalized_mutual_info_score
import torch.nn.init as init
# Sample from the Gumbel-Softmax distribution and optionally discretize.
# The Gumbel-Softmax distribution is a continuous relaxation of the categorical distribution
class GumbelSoftmax(nn.Module):

  def __init__(self, f_dim, c_dim):
    super(GumbelSoftmax, self).__init__()
    self.logits = nn.Linear(f_dim, c_dim)
    self.f_dim = f_dim
    self.c_dim = c_dim
    """
    f_dim := feature dimension
    c_dim := number of categories
    logits := takes input of size f_dim and outputs c_dim dimensional logits
    """

  def sample_gumbel(self, shape, is_cuda=False, eps=1e-20):
    U = torch.rand(shape)
    if is_cuda:
      U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)
  # campiona da una distribuzione di Gumbel

  def gumbel_softmax_sample(self, logits, temperature):
    y = logits + self.sample_gumbel(logits.size(), logits.is_cuda)
    return F.softmax(y / temperature, dim=-1)

  def gumbel_softmax(self, logits, temperature, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    #categorical_dim = 10
    y = self.gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard

  def forward(self, x, temperature=1.0, hard=False):
    logits = self.logits(x).view(-1, self.c_dim)
    prob = F.softmax(logits, dim=-1)
    y = self.gumbel_softmax(logits, temperature, hard)
    return logits, prob, y
  
  # Sample from a Gaussian distribution
class Gaussian(nn.Module):
  def __init__(self, in_dim, z_dim):
    super(Gaussian, self).__init__()
    self.mu = nn.Linear(in_dim, z_dim)
    self.var = nn.Linear(in_dim, z_dim)

  def reparameterize(self, mu, var):
    std = torch.sqrt(var + 1e-10)
    noise = torch.randn_like(std)
    z = mu + noise * std
    return z

  def forward(self, x):
    mu = self.mu(x)
    var = F.softplus(self.var(x))
    z = self.reparameterize(mu, var)
    return mu, var, z

# Inference Network
class InferenceNet(nn.Module):
  def __init__(self, x_dim, z_dim, y_dim):
    super(InferenceNet, self).__init__()
    # ci sono due reti neurali: una per q(y|x) e una per q(z|y,x)
    # q(y|x)
    #print("x_dim", x_dim, y_dim,z_dim)
    self.inference_qyx = torch.nn.ModuleList([
        nn.Linear(x_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        GumbelSoftmax(512, y_dim)
    ])

    # q(z|y,x)
    self.inference_qzyx = torch.nn.ModuleList([
        nn.Linear(x_dim + y_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        Gaussian(512, z_dim)
    ])

  # q(y|x)
  def qyx(self, x, temperature, hard):
    #print("Entra in qyx")
    num_layers = len(self.inference_qyx)
    for i, layer in enumerate(self.inference_qyx):
      if i == num_layers - 1:
        #print("entra in if")
        #last layer is gumbel softmax
        x = layer(x, temperature, hard)
      else:
        #print("entra in else")
        # print("x:", x)
        # print("x dimension", x.shape)
        #print("layer:", layer)
        x=layer(x) # dimension of x: torch.Size([1, 746496])
                    # layer is a torch linear object
    #print("Esce da qyx")
    return x
  # funzione per calcolare q(y|x)

  # q(z|x,y)
  def qzxy(self, x, y):
    concat = torch.cat((x, y), dim=1) # combina l'input di x e y
    for layer in self.inference_qzyx:
      concat = layer(concat)
    return concat

  def forward(self, x, temperature=1.0, hard=0):
    #x = Flatten(x)
    #print("Entra in forward infNet")
    # q(y|x)
    logits, prob, y = self.qyx(x, temperature, hard)

    # q(z|x,y)
    mu, var, z = self.qzxy(x, y)

    output = {'mean': mu, 'var': var, 'gaussian': z,
              'logits': logits, 'prob_cat': prob, 'categorical': y}
    #print("Esce da forward infNet")
    return output
# in input prende un immagine x
# la rete usa il metodo qyx  inferire la variabile latente discreta y data l'immagine di input x. Questo viene fatto approssimando la distribuzione categoriale con Gumbel-Softmax.
# La rete usa il metodo qzxy per inferire la variabile latente continua z data l'immagine x e la variabile latente discreta y.
# in output restituisce la media mu, la varianza var e il campione z della variabile latente continua z, i logit, la probabilità e il campione y della variabile latente discreta y.

# Generative Network
class GenerativeNet(nn.Module):
  def __init__(self, x_dim, z_dim, y_dim):
    super(GenerativeNet, self).__init__()

    # p(z|y)
    self.y_mu = nn.Linear(y_dim, z_dim)#y_dim, z_dim
    self.y_var = nn.Linear(y_dim, z_dim)#

    # p(x|z) genera x dato z
    self.generative_pxz = torch.nn.ModuleList([
        nn.Linear(z_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, x_dim),
        torch.nn.Sigmoid() # garantisce che l'output sia compreso tra 0 e 1
    ])

  # p(z|y)
  def pzy(self, y):
    #print("y",y.size())
    y_mu = self.y_mu(y)
    y_var = F.softplus(self.y_var(y)) # garantisce che la varianza sia sempre positiva
    return y_mu, y_var

  # p(x|z)
  def pxz(self, z):
    for layer in self.generative_pxz:
      z = layer(z)
    return z

  def forward(self, z, y):
    # p(z|y)
    y_mu, y_var = self.pzy(y)

    # p(x|z)
    x_rec = self.pxz(z)

    output = {'y_mean': y_mu, 'y_var': y_var, 'x_rec': x_rec}
    return output
  
  # in input la classe prende una variabile latente z e una variabile categorica y
  # la rete usa il metodo pzy per calcolare la media e la varianza della distribuzione gaussiana di z data y
  # e il metodo pxz per generare l'immagine x dato il campione z
  # in output la rete restituisce la media e la varianza delle variabili latenti y e l'immagine generata x
  
  ### LOSSES ###
  
class LossFunctions:
  eps = 1e-8

  def mean_squared_error(self, real, predictions):
      """Mean Squared Error between the true and predicted outputs
         loss = (1/n)*Σ(real - predicted)^2

      Args:
          real: (array) corresponding array containing the true labels
          predictions: (array) corresponding array containing the predicted labels

      Returns:
          output: (array/float) depending on average parameters the result will be the mean
                                of all the sample losses or an array with the losses per sample
      """
      loss = (real - predictions).pow(2)
      return loss.sum(-1).mean()


  def reconstruction_loss(self, real, predicted, rec_type='mse' ):
      """Reconstruction loss between the true and predicted outputs
         mse = (1/n)*Σ(real - predicted)^2
         bce = (1/n) * -Σ(real*log(predicted) + (1 - real)*log(1 - predicted))

      Args:
          real: (array) corresponding array containing the true labels
          predictions: (array) corresponding array containing the predicted labels

      Returns:
          output: (array/float) depending on average parameters the result will be the mean
                                of all the sample losses or an array with the losses per sample
      """
      # print(f"Type of 'real': {type(real)}")
      # print(f"Type of 'predicted': {type(predicted)}")
      # print("valore predicted",predicted)
      #print("real",real.size())
      predicted_flatten=predicted.view( 2,-1)
      #print("predicted_flatten",predicted_flatten.size())
      #print("predicted_flatten device:", predicted_flatten.device)
      #print("real device:", real.device)
      predicted_flatten = predicted_flatten.to(real.device)
      if rec_type == 'mse':
        loss = (real - predicted_flatten).pow(2)
      elif rec_type == 'bce':
        loss = F.binary_cross_entropy(predicted_flatten, real, reduction='none')
      else:
        raise "invalid loss function... try bce or mse..."
      return loss.sum(-1).mean()


  def log_normal(self, x, mu, var):
      """Logarithm of normal distribution with mean=mu and variance=var
         log(x|μ, σ^2) = loss = -0.5 * Σ log(2π) + log(σ^2) + ((x - μ)/σ)^2

      Args:
         x: (array) corresponding array containing the input
         mu: (array) corresponding array containing the mean
         var: (array) corresponding array containing the variance

      Returns:
         output: (array/float) depending on average parameters the result will be the mean
                                of all the sample losses or an array with the losses per sample
      """
      if self.eps > 0.0:
        var = var + self.eps
        var=F.softplus(var)
      return -0.5 * torch.sum(
        np.log(2.0 * np.pi) + torch.log(var) + torch.pow(x - mu, 2) / var, dim=-1)


  def gaussian_loss(self, z, z_mu, z_var, z_mu_prior, z_var_prior):
      """Variational loss when using labeled data without considering reconstruction loss
         loss = log q(z|x,y) - log p(z) - log p(y)

      Args:
         z: (array) array containing the gaussian latent variable
         z_mu: (array) array containing the mean of the inference model
         z_var: (array) array containing the variance of the inference model
         z_mu_prior: (array) array containing the prior mean of the generative model
         z_var_prior: (array) array containing the prior variance of the generative mode

      Returns:
         output: (array/float) depending on average parameters the result will be the mean
                                of all the sample losses or an array with the losses per sample
      """
      print("z",z)
      print("z_mu",z_mu)
      print("z_var",z_var)
      print("z_mu_prior",z_mu_prior)
      print("z_var_prior",z_var_prior)
      loss = self.log_normal(z, z_mu, z_var) - self.log_normal(z, z_mu_prior, z_var_prior)
      print("loss",loss.mean())
      return loss.mean()


  def entropy(self, logits, targets):
      """Entropy loss
          loss = (1/n) * -Σ targets*log(predicted)

      Args:
          logits: (array) corresponding array containing the logits of the categorical variable
          real: (array) corresponding array containing the true labels

      Returns:
          output: (array/float) depending on average parameters the result will be the mean
                                of all the sample losses or an array with the losses per sample
      """
      log_q = F.log_softmax(logits, dim=-1)
      return -torch.mean(torch.sum(targets * log_q, dim=-1))
    
  

class Metrics:

  # Code taken from the work
  # VaDE (Variational Deep Embedding:A Generative Approach to Clustering)
  def cluster_acc(self, Y_pred, Y):
    Y_pred, Y = np.array(Y_pred), np.array(Y)
    #print("Y_pred dimension",Y_pred.size)
    #print("Y dimension",Y.size)
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D,D), dtype=np.int64)
    for i in range(Y_pred.size):
      w[Y_pred[i], Y[i]] += 1
    row, col = linear_sum_assignment(w.max()-w)
    return sum([w[row[i],col[i]] for i in range(row.shape[0])]) * 1.0/Y_pred.size


  def nmi(self, Y_pred, Y):
    Y_pred, Y = np.array(Y_pred), np.array(Y)
    assert Y_pred.size == Y.size
    return normalized_mutual_info_score(Y_pred, Y, average_method='arithmetic')
  
  



# GMVAE Network
class GMVAENet(nn.Module):
  def __init__(self, x_dim, z_dim, y_dim):
    super(GMVAENet, self).__init__()

    self.inference = InferenceNet(x_dim, z_dim, y_dim)
    self.generative = GenerativeNet(x_dim, z_dim, y_dim)

    # weight initialization
    for m in self.modules():
      if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias.data is not None:
          init.constant_(m.bias, 0)

  def forward(self, x, temperature=1.0, hard=0):
    x = x.view(x.size(0), -1)
    out_inf = self.inference(x, temperature, hard)
    z, y = out_inf['gaussian'], out_inf['categorical']
    out_gen = self.generative(z, y)

    # merge output
    output = out_inf
    for key, value in out_gen.items():
      output[key] = value
    return output