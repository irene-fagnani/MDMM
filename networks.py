import numpy as np
import GMVAE
import torch
import torch.nn as nn
from torch.autograd import Variable
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F


####################################################################
#---------------------- For MultiDomain (MD) -----------------------
####################################################################
class MD_E_content(nn.Module):
  def __init__(self, input_dim):
    super(MD_E_content, self).__init__()
    enc_c = []
    tch = 64
    enc_c += [LeakyReLUConv2d(input_dim, tch, kernel_size=7, stride=1, padding=3)]
    for i in range(1, 3):
      enc_c += [ReLUINSConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1)]
      tch *= 2
    for i in range(0, 3):
      enc_c += [INSResBlock(tch, tch)]

    for i in range(0, 1):
      enc_c += [INSResBlock(tch, tch)]
      enc_c += [GaussianNoiseLayer()]
    self.conv = nn.Sequential(*enc_c)

  def forward(self, x):
    #print("dimensione x dim prima del problema ", x.size())
    return self.conv(x)

class MD_E_attr(nn.Module):
  def __init__(self, input_dim, output_nc=8, c_dim=3):
    super(MD_E_attr, self).__init__()
    dim = 64
    self.model = nn.Sequential(
        nn.ReflectionPad2d(3),
        nn.Conv2d(input_dim+c_dim , dim, 7, 1),
        nn.ReLU(inplace=True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(dim, dim*2, 4, 2),
        nn.ReLU(inplace=True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(dim*2, dim*4, 4, 2),
        nn.ReLU(inplace=True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(dim*4, dim*4, 4, 2),
        nn.ReLU(inplace=True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(dim*4, dim*4, 4, 2),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(1),
        nn.Conv2d(dim*4, output_nc, 1, 1, 0))

  def forward(self, x, c):
    c = c.view(c.size(0), c.size(1), 1, 1)
    c = c.repeat(1, 1, x.size(2), x.size(3))
    x_c = torch.cat([x, c], dim=1)
    output = self.model(x_c)
    return output.view(output.size(0), -1)

class old_MD_E_attr_concat(nn.Module):
  def __init__(self, input_dim,z_dim,y_dim, output_nc=8, c_dim=3, norm_layer=None, nl_layer=None):
    super(MD_E_attr_concat, self).__init__()

    ndf = 64
    n_blocks=4
    max_ndf = 4

    conv_layers = [nn.ReflectionPad2d(1)]
    conv_layers += [nn.Conv2d(input_dim+c_dim, ndf, kernel_size=4, stride=2, padding=0, bias=True)]
    for n in range(1, n_blocks):
      input_ndf = ndf * min(max_ndf, n)  # 2**(n-1)
      output_ndf = ndf * min(max_ndf, n+1)  # 2**n
      conv_layers += [BasicBlock(input_ndf, output_ndf, norm_layer, nl_layer)]
    conv_layers += [nl_layer(), nn.AdaptiveAvgPool2d(1)] # AvgPool2d(13)
    self.fc = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
    self.fcVar = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
    self.conv = nn.Sequential(*conv_layers)
    self.inference_net = GMVAE.InferenceNet(x_dim=output_nc, z_dim=z_dim, y_dim=y_dim)
    

  def forward(self, x, c,temperature=1.0, hard=0):
    c = c.view(c.size(0), c.size(1), 1, 1)
    c = c.repeat(1, 1, x.size(2), x.size(3))
    x_c = torch.cat([x, c], dim=1)
    x_conv = self.conv(x_c)
    conv_flat = x_conv.view(x.size(0), -1)
    output = F.softplus(self.fc(conv_flat))
    outputVar = F.softplus(self.fcVar(conv_flat))
    inference_output = self.inference_net(output, temperature, hard)
    inference_outputVar = self.inference_net(outputVar, temperature, hard)
    return inference_output, inference_outputVar
  
class MD_E_attr_concat(nn.Module):
  def __init__(self, input_dim,z_dim,y_dim, output_nc=8, c_dim=3, norm_layer=None, nl_layer=None):
    super(MD_E_attr_concat, self).__init__()

    ndf = 64
    n_blocks=4
    max_ndf = 4

    conv_layers = [nn.ReflectionPad2d(1)]
    conv_layers += [nn.Conv2d(input_dim+c_dim, ndf, kernel_size=4, stride=2, padding=0, bias=True)]
    for n in range(1, n_blocks):
      input_ndf = ndf * min(max_ndf, n)  # 2**(n-1)
      output_ndf = ndf * min(max_ndf, n+1)  # 2**n
      conv_layers += [BasicBlock(input_ndf, output_ndf, norm_layer, nl_layer)]
    conv_layers += [nl_layer(), nn.AdaptiveAvgPool2d(1)] # AvgPool2d(13)
    self.fc = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
    self.fcVar = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
    self.conv = nn.Sequential(*conv_layers)
    # ci sono due reti neurali: una per q(y|x) e una per q(z|y,x)
    # q(y|x)
    #print("x_dim", x_dim, y_dim,z_dim)
    self.inference_qyx = torch.nn.ModuleList([
        nn.Linear(output_nc, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        GMVAE.GumbelSoftmax(512, c_dim)
    ])

    # q(z|y,x)
    self.inference_qzyx = torch.nn.ModuleList([
        nn.Linear(output_nc + y_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        GMVAE.Gaussian(512, z_dim)
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

  def forward(self, x, c,temperature=1.0, hard=0):
    c = c.view(c.size(0), c.size(1), 1, 1)
    c = c.repeat(1, 1, x.size(2), x.size(3))
    x_c = torch.cat([x, c], dim=1)
    x_conv = self.conv(x_c)
    conv_flat = x_conv.view(x.size(0), -1)
    output = F.softplus(self.fc(conv_flat))
    #outputVar = F.softplus(self.fcVar(conv_flat))
    logits, prob, y = self.qyx(output, temperature, hard)
    mu, var, z = self.qzxy(output, y)
    output = {'mean': mu, 'var': var, 'gaussian': z,
              'logits': logits, 'prob_cat': prob, 'categorical': y}
    return output



class MD_G_uni(nn.Module):
  def __init__(self, output_dim, c_dim=3):
    super(MD_G_uni, self).__init__()
    self.c_dim = c_dim
    tch = 256
    dec_share = []
    dec_share += [INSResBlock(tch, tch)]
    self.dec_share = nn.Sequential(*dec_share)
    tch = 256+self.c_dim
    dec = []
    for i in range(0, 3):
      dec += [INSResBlock(tch, tch)]
    dec += [ReLUINSConvTranspose2d(tch, tch//2, kernel_size=3, stride=2, padding=1, output_padding=1)]
    tch = tch//2
    dec += [ReLUINSConvTranspose2d(tch, tch//2, kernel_size=3, stride=2, padding=1, output_padding=1)]
    tch = tch//2
    dec += [nn.ConvTranspose2d(tch, output_dim, kernel_size=1, stride=1, padding=0)]+[nn.Tanh()]
    self.dec = nn.Sequential(*dec)

  def forward(self, x, c):
    out0 = self.dec_share(x)
    c = c.view(c.size(0), c.size(1), 1, 1)
    c = c.repeat(1, 1, out0.size(2), out0.size(3))
    x_c = torch.cat([out0, c], dim=1)
    return self.dec(x_c)

class old_MD_G_multi_concat(nn.Module):
  def __init__(self, output_dim,x_dim, z_dim, crop_size, c_dim=3, nz=8):
    super(MD_G_multi_concat, self).__init__()
    self.nz = nz
    self.c_dim = c_dim
    tch = 256
    dec_share = []
    dec_share += [INSResBlock(tch, tch)]
    self.dec_share = nn.Sequential(*dec_share)
    tch = 256+self.nz+self.c_dim
    dec1 = []
    for i in range(0, 3):
      dec1 += [INSResBlock(tch, tch)]
    tch = tch + self.nz
    dec2 = [ReLUINSConvTranspose2d(tch, tch//2, kernel_size=3, stride=2, padding=1, output_padding=1)]
    tch = tch//2
    tch = tch + self.nz
    dec3 = [ReLUINSConvTranspose2d(tch, tch//2, kernel_size=3, stride=2, padding=1, output_padding=1)]
    tch = tch//2
    tch = tch + self.nz
    dec4 = [nn.ConvTranspose2d(tch, output_dim, kernel_size=1, stride=1, padding=0)]+[nn.Tanh()]
    self.dec1 = nn.Sequential(*dec1)
    self.dec2 = nn.Sequential(*dec2)
    self.dec3 = nn.Sequential(*dec3)
    self.dec4 = nn.Sequential(*dec4)
    self.generative_net = GMVAE.GenerativeNet(crop_size, z_dim, c_dim)#x_dim, z_dim, y_dim

  def sample_z(self, y):
      # Ottieni i parametri della distribuzione di z condizionata su y usando GenerativeNet
      y_mu, y_var = self.generative_net.pzy(y)
      # Campiona z da una distribuzione normale con media y_mu e varianza y_var
      z = y_mu + torch.sqrt(y_var) * torch.randn_like(y_var)
      return z

  def forward(self, x, z, c,y): #content,attr,c
    #z=self.sample_z(y)
    out0 = self.dec_share(x)
    z_img = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3))
    c0=c
    c = c.view(c.size(0), c.size(1), 1, 1)
    c = c.repeat(1, 1, out0.size(2), out0.size(3))
    x_c_z = torch.cat([out0, c, z_img], 1)
    #print("size xcz",x_c_z.size())
    out1 = self.dec1(x_c_z)
    z_img2 = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), out1.size(2), out1.size(3))
    x_and_z2 = torch.cat([out1, z_img2], 1)
    out2 = self.dec2(x_and_z2)
    z_img3 = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), out2.size(2), out2.size(3))
    x_and_z3 = torch.cat([out2, z_img3], 1)
    out3 = self.dec3(x_and_z3)
    z_img4 = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), out3.size(2), out3.size(3))
    x_and_z4 = torch.cat([out3, z_img4], 1)
    out4 = self.dec4(x_and_z4)
    out4=self.generative_net(out4,c0)
    return out4

class MD_G_multi_concat(nn.Module):
  def __init__(self, output_dim,x_dim, z_dim, crop_size, c_dim=3, nz=8):
    super(MD_G_multi_concat, self).__init__()
    self.nz = nz
    self.c_dim = c_dim
    tch = 256
    dec_share = []
    dec_share += [INSResBlock(tch, tch)]
    self.dec_share = nn.Sequential(*dec_share)
    tch = 256+self.nz+self.c_dim
    dec1 = []
    for i in range(0, 3):
      dec1 += [INSResBlock(tch, tch)]
    tch = tch + self.nz
    dec2 = [ReLUINSConvTranspose2d(tch, tch//2, kernel_size=3, stride=2, padding=1, output_padding=1)]
    tch = tch//2
    tch = tch + self.nz
    dec3 = [ReLUINSConvTranspose2d(tch, tch//2, kernel_size=3, stride=2, padding=1, output_padding=1)]
    tch = tch//2
    tch = tch + self.nz
    dec4 = [nn.ConvTranspose2d(tch, output_dim, kernel_size=1, stride=1, padding=0)]+[nn.Tanh()]
    self.dec1 = nn.Sequential(*dec1)
    self.dec2 = nn.Sequential(*dec2)
    self.dec3 = nn.Sequential(*dec3)
    self.dec4 = nn.Sequential(*dec4)
    #self.generative_net = GMVAE.GenerativeNet(crop_size, z_dim, c_dim)#x_dim, z_dim, y_dim
        # p(z|y)
    self.y_mu = nn.Linear(c_dim, z_dim)#y_dim, z_dim
    self.y_var = nn.Linear(c_dim, z_dim)#
    # p(x|z) genera x dato z
    self.generative_pxz = torch.nn.ModuleList([
        nn.Linear(z_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        #nn.Linear(512, x_dim),
        nn.Linear(512, 108),
        torch.nn.Sigmoid() # garantisce che l'output sia compreso tra 0 e 1
    ])
  def sample_z(self, y):
      # Ottieni i parametri della distribuzione di z condizionata su y usando GenerativeNet
      y_mu, y_var = self.generative_net.pzy(y)
      # Campiona z da una distribuzione normale con media y_mu e varianza y_var
      z = y_mu + torch.sqrt(y_var) * torch.randn_like(y_var)
      return z

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

  def forward(self, x, z, c, y): #content,attr,c
    #z=self.sample_z(y)
    c0 = c
    out0 = self.dec_share(x)
    z_img = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3))
    c = c.view(c.size(0), c.size(1), 1, 1)
    c = c.repeat(1, 1, out0.size(2), out0.size(3))
    # print("dim out0: ", out0.size())
    # print("dim c: ", c.size())
    # print("dim z_img: ", z_img.size())
    x_c_z = torch.cat([out0, c, z_img], 1)
    #print("size xcz",x_c_z.size())
    out1 = self.dec1(x_c_z)
    z_img2 = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), out1.size(2), out1.size(3))
    x_and_z2 = torch.cat([out1, z_img2], 1)
    out2 = self.dec2(x_and_z2)
    z_img3 = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), out2.size(2), out2.size(3))
    x_and_z3 = torch.cat([out2, z_img3], 1)
    out3 = self.dec3(x_and_z3)
    z_img4 = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), out3.size(2), out3.size(3))
    x_and_z4 = torch.cat([out3, z_img4], 1)
    out4 = self.dec4(x_and_z4)
    y_mu, y_var = self.pzy(c0)

    # p(x|z)
    x_rec = self.pxz(out4)

    output = {'y_mean': y_mu, 'y_var': y_var, 'x_rec': x_rec}
    return output

class MD_G_multi(nn.Module):
  def __init__(self, output_dim, c_dim=3, nz=8):
    super(MD_G_multi, self).__init__()
    self.nz = nz
    ini_tch = 256
    tch_add = ini_tch
    tch = ini_tch
    self.tch_add = tch_add
    self.dec1 = MisINSResBlock(tch, tch_add)
    self.dec2 = MisINSResBlock(tch, tch_add)
    self.dec3 = MisINSResBlock(tch, tch_add)
    self.dec4 = MisINSResBlock(tch, tch_add)

    dec5 = []
    dec5 += [ReLUINSConvTranspose2d(tch, tch//2, kernel_size=3, stride=2, padding=1, output_padding=1)]
    tch = tch//2
    dec5 += [ReLUINSConvTranspose2d(tch, tch//2, kernel_size=3, stride=2, padding=1, output_padding=1)]
    tch = tch//2
    dec5 += [nn.ConvTranspose2d(tch, output_dim, kernel_size=1, stride=1, padding=0)]
    dec5 += [nn.Tanh()]
    self.decA5 = nn.Sequential(*dec5)

    self.mlp = nn.Sequential(
        nn.Linear(nz+c_dim, 256),
        nn.ReLU(inplace=True),
        nn.Linear(256, 256),
        nn.ReLU(inplace=True),
        nn.Linear(256, tch_add*4))
    return

  def forward(self, x, z, c):
    z_c = torch.cat([c, z], 1)
    z_c = self.mlp(z_c)
    z1, z2, z3, z4 = torch.split(z_c, self.tch_add, dim=1)
    z1, z2, z3, z4 = z1.contiguous(), z2.contiguous(), z3.contiguous(), z4.contiguous()
    out1 = self.dec1(x, z1)
    out2 = self.dec2(out1, z2)
    out3 = self.dec3(out2, z3)
    out4 = self.dec4(out3, z4)
    out = self.decA5(out4)
    return out

class MD_Dis(nn.Module):
  def __init__(self, input_dim, norm='None', sn=False, c_dim=3, image_size=216):
    super(MD_Dis, self).__init__()
    ch = 64
    n_layer = 6
    self.model, curr_dim = self._make_net(ch, input_dim, n_layer, norm, sn)
    self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=1, stride=1, padding=1, bias=False)
    kernal_size = int(image_size/np.power(2, n_layer))
    self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernal_size, bias=False)
    self.pool = nn.AdaptiveAvgPool2d(1)

  def _make_net(self, ch, input_dim, n_layer, norm, sn):
    model = []
    model += [LeakyReLUConv2d(input_dim, ch, kernel_size=3, stride=2, padding=1, norm=norm, sn=sn)] #16
    tch = ch
    for i in range(1, n_layer-1):
      model += [LeakyReLUConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1, norm=norm, sn=sn)] # 8
      tch *= 2
    model += [LeakyReLUConv2d(tch, tch, kernel_size=3, stride=2, padding=1, norm='None', sn=sn)] # 2
    #model += [LeakyReLUConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1, norm='None', sn=sn)] # 2
    #tch *= 2
    return nn.Sequential(*model), tch
  
  # MODIFICHE NVIDIA
  def cuda(self,gpu):
  #def cuda(self):
    # MODIFICHE NVIDIA
    # self.model.cpu()
    # self.conv1.cpu()
    # self.conv2.cpu()
    self.model.cuda(gpu)
    self.conv1.cuda(gpu)
    self.conv2.cuda(gpu)
    

  def forward(self, x):
    h = self.model(x)
    out = self.conv1(h)
    out_cls = self.conv2(h)
    out_cls = self.pool(out_cls)
    return out, out_cls.view(out_cls.size(0), out_cls.size(1))

class MD_Dis_content(nn.Module):
  def __init__(self, c_dim=3):
    super(MD_Dis_content, self).__init__()
    model = []
    model += [LeakyReLUConv2d(256, 256, kernel_size=7, stride=2, padding=1, norm='Instance')]
    model += [LeakyReLUConv2d(256, 256, kernel_size=7, stride=2, padding=1, norm='Instance')]
    model += [LeakyReLUConv2d(256, 256, kernel_size=7, stride=2, padding=1, norm='Instance')]
    model += [LeakyReLUConv2d(256, 256, kernel_size=4, stride=1, padding=0)]
    model += [nn.Conv2d(256, c_dim, kernel_size=1, stride=1, padding=0)]
    self.model = nn.Sequential(*model)

  def forward(self, x):
    out = self.model(x)
    out = out.view(out.size(0), out.size(1))
    return out


####################################################################
#------------------------- Basic Functions -------------------------
####################################################################
def get_scheduler(optimizer, opts, cur_ep=-1):
  if opts.lr_policy == 'lambda':
    def lambda_rule(ep):
      lr_l = 1.0 - max(0, ep - opts.n_ep_decay) / float(opts.n_ep - opts.n_ep_decay + 1)
      return lr_l
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=cur_ep)
  elif opts.lr_policy == 'step':
    scheduler = lr_scheduler.StepLR(optimizer, step_size=opts.n_ep_decay, gamma=0.1, last_epoch=cur_ep)
  else:
    return NotImplementedError('no such learn rate policy')
  return scheduler

def meanpoolConv(inplanes, outplanes):
  sequence = []
  sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
  sequence += [nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=True)]
  return nn.Sequential(*sequence)

def convMeanpool(inplanes, outplanes):
  sequence = []
  sequence += conv3x3(inplanes, outplanes)
  sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
  return nn.Sequential(*sequence)

def get_norm_layer(layer_type='instance'):
  if layer_type == 'batch':
    norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
  elif layer_type == 'instance':
    norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
  elif layer_type == 'none':
    norm_layer = None
  else:
    raise NotImplementedError('normalization layer [%s] is not found' % layer_type)
  return norm_layer

def get_non_linearity(layer_type='relu'):
  if layer_type == 'relu':
    nl_layer = functools.partial(nn.ReLU, inplace=True)
  elif layer_type == 'lrelu':
    nl_layer = functools.partial(nn.LeakyReLU, negative_slope=0.2, inplace=False)
  elif layer_type == 'elu':
    nl_layer = functools.partial(nn.ELU, inplace=True)
  else:
    raise NotImplementedError('nonlinearity activitation [%s] is not found' % layer_type)
  return nl_layer
def conv3x3(in_planes, out_planes):
  return [nn.ReflectionPad2d(1), nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=0, bias=True)]

def gaussian_weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1 and classname.find('Conv') == 0:
    m.weight.data.normal_(0.0, 0.02)

####################################################################
#-------------------------- Basic Blocks --------------------------
####################################################################

## The code of LayerNorm is modified from MUNIT (https://github.com/NVlabs/MUNIT)
class LayerNorm(nn.Module):
  def __init__(self, n_out, eps=1e-5, affine=True):
    super(LayerNorm, self).__init__()
    self.n_out = n_out
    self.affine = affine
    if self.affine:
      self.weight = nn.Parameter(torch.ones(n_out, 1, 1))
      self.bias = nn.Parameter(torch.zeros(n_out, 1, 1))
    return
  def forward(self, x):
    normalized_shape = x.size()[1:]
    if self.affine:
      return F.layer_norm(x, normalized_shape, self.weight.expand(normalized_shape), self.bias.expand(normalized_shape))
    else:
      return F.layer_norm(x, normalized_shape)

class BasicBlock(nn.Module):
  def __init__(self, inplanes, outplanes, norm_layer=None, nl_layer=None):
    super(BasicBlock, self).__init__()
    layers = []
    if norm_layer is not None:
      layers += [norm_layer(inplanes)]
    layers += [nl_layer()]
    layers += conv3x3(inplanes, inplanes)
    if norm_layer is not None:
      layers += [norm_layer(inplanes)]
    layers += [nl_layer()]
    layers += [convMeanpool(inplanes, outplanes)]
    self.conv = nn.Sequential(*layers)
    self.shortcut = meanpoolConv(inplanes, outplanes)
  def forward(self, x):
    out = self.conv(x) + self.shortcut(x)
    return out

class LeakyReLUConv2d(nn.Module):
  def __init__(self, n_in, n_out, kernel_size, stride, padding=0, norm='None', sn=False):
    super(LeakyReLUConv2d, self).__init__()
    model = []
    model += [nn.ReflectionPad2d(padding)]
    if sn:
      model += [spectral_norm(nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=0, bias=True))]
    else:
      model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=0, bias=True)]
    if 'norm' == 'Instance':
      model += [nn.InstanceNorm2d(n_out, affine=False)]
    model += [nn.LeakyReLU(inplace=True)]
    self.model = nn.Sequential(*model)
    self.model.apply(gaussian_weights_init)
    #elif == 'Group'
  def forward(self, x):
    return self.model(x)

class ReLUINSConv2d(nn.Module):
  def __init__(self, n_in, n_out, kernel_size, stride, padding=0):
    super(ReLUINSConv2d, self).__init__()
    model = []
    model += [nn.ReflectionPad2d(padding)]
    model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=0, bias=True)]
    model += [nn.InstanceNorm2d(n_out, affine=False)]
    model += [nn.ReLU(inplace=True)]
    self.model = nn.Sequential(*model)
    self.model.apply(gaussian_weights_init)
  def forward(self, x):
    return self.model(x)

class INSResBlock(nn.Module):
  def conv3x3(self, inplanes, out_planes, stride=1):
    return [nn.ReflectionPad2d(1), nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride)]
  def __init__(self, inplanes, planes, stride=1, dropout=0.0):
    super(INSResBlock, self).__init__()
    model = []
    model += self.conv3x3(inplanes, planes, stride)
    model += [nn.InstanceNorm2d(planes)]
    model += [nn.ReLU(inplace=True)]
    model += self.conv3x3(planes, planes)
    model += [nn.InstanceNorm2d(planes)]
    if dropout > 0:
      model += [nn.Dropout(p=dropout)]
    self.model = nn.Sequential(*model)
    self.model.apply(gaussian_weights_init)
  def forward(self, x):
    residual = x
    out = self.model(x)
    out += residual
    return out

class MisINSResBlock(nn.Module):
  def conv3x3(self, dim_in, dim_out, stride=1):
    return nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=stride))
  def conv1x1(self, dim_in, dim_out):
    return nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0)
  def __init__(self, dim, dim_extra, stride=1, dropout=0.0):
    super(MisINSResBlock, self).__init__()
    self.conv1 = nn.Sequential(
        self.conv3x3(dim, dim, stride),
        nn.InstanceNorm2d(dim))
    self.conv2 = nn.Sequential(
        self.conv3x3(dim, dim, stride),
        nn.InstanceNorm2d(dim))
    self.blk1 = nn.Sequential(
        self.conv1x1(dim + dim_extra, dim + dim_extra),
        nn.ReLU(inplace=False),
        self.conv1x1(dim + dim_extra, dim),
        nn.ReLU(inplace=False))
    self.blk2 = nn.Sequential(
        self.conv1x1(dim + dim_extra, dim + dim_extra),
        nn.ReLU(inplace=False),
        self.conv1x1(dim + dim_extra, dim),
        nn.ReLU(inplace=False))
    model = []
    if dropout > 0:
      model += [nn.Dropout(p=dropout)]
    self.model = nn.Sequential(*model)
    self.model.apply(gaussian_weights_init)
    self.conv1.apply(gaussian_weights_init)
    self.conv2.apply(gaussian_weights_init)
    self.blk1.apply(gaussian_weights_init)
    self.blk2.apply(gaussian_weights_init)
  def forward(self, x, z):
    residual = x
    z_expand = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3))
    o1 = self.conv1(x)
    o2 = self.blk1(torch.cat([o1, z_expand], dim=1))
    o3 = self.conv2(o2)
    out = self.blk2(torch.cat([o3, z_expand], dim=1))
    out += residual
    return out

class GaussianNoiseLayer(nn.Module):
  def __init__(self,):
    super(GaussianNoiseLayer, self).__init__()
  def forward(self, x):
    if self.training == False:
      return x
    # MODIFICA NVIDIA
    noise = Variable(torch.randn(x.size()).cuda(x.get_device()))
    #device =x.get_device() if x.is_cuda else 'cpu' # commentra se si usa CUDA
    #noise = Variable(torch.randn(x.size()).to(device)) # commentra se si usa CUDA
    return x + noise

class ReLUINSConvTranspose2d(nn.Module):
  def __init__(self, n_in, n_out, kernel_size, stride, padding, output_padding):
    super(ReLUINSConvTranspose2d, self).__init__()
    model = []
    model += [nn.ConvTranspose2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=True)]
    model += [LayerNorm(n_out)]
    model += [nn.ReLU(inplace=True)]
    self.model = nn.Sequential(*model)
    self.model.apply(gaussian_weights_init)
  def forward(self, x):
    return self.model(x)


####################################################################
#--------------------- Spectral Normalization ---------------------
#  This part of code is copied from pytorch master branch (0.5.0)
####################################################################
class SpectralNorm(object):
  def __init__(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12):
    self.name = name
    self.dim = dim
    if n_power_iterations <= 0:
      raise ValueError('Expected n_power_iterations to be positive, but '
                       'got n_power_iterations={}'.format(n_power_iterations))
    self.n_power_iterations = n_power_iterations
    self.eps = eps
  def compute_weight(self, module):
    weight = getattr(module, self.name + '_orig')
    u = getattr(module, self.name + '_u')
    weight_mat = weight
    if self.dim != 0:
      # permute dim to front
      weight_mat = weight_mat.permute(self.dim,
                                            *[d for d in range(weight_mat.dim()) if d != self.dim])
    height = weight_mat.size(0)
    weight_mat = weight_mat.reshape(height, -1)
    with torch.no_grad():
      for _ in range(self.n_power_iterations):
        v = F.normalize(torch.matmul(weight_mat.t(), u), dim=0, eps=self.eps)
        u = F.normalize(torch.matmul(weight_mat, v), dim=0, eps=self.eps)
    sigma = torch.dot(u, torch.matmul(weight_mat, v))
    weight = weight / sigma
    return weight, u
  def remove(self, module):
    weight = getattr(module, self.name)
    delattr(module, self.name)
    delattr(module, self.name + '_u')
    delattr(module, self.name + '_orig')
    module.register_parameter(self.name, torch.nn.Parameter(weight))
  def __call__(self, module, inputs):
    if module.training:
      weight, u = self.compute_weight(module)
      setattr(module, self.name, weight)
      setattr(module, self.name + '_u', u)
    else:
      r_g = getattr(module, self.name + '_orig').requires_grad
      getattr(module, self.name).detach_().requires_grad_(r_g)

  @staticmethod
  def apply(module, name, n_power_iterations, dim, eps):
    fn = SpectralNorm(name, n_power_iterations, dim, eps)
    weight = module._parameters[name]
    height = weight.size(dim)
    u = F.normalize(weight.new_empty(height).normal_(0, 1), dim=0, eps=fn.eps)
    delattr(module, fn.name)
    module.register_parameter(fn.name + "_orig", weight)
    module.register_buffer(fn.name, weight.data)
    module.register_buffer(fn.name + "_u", u)
    module.register_forward_pre_hook(fn)
    return fn

def spectral_norm(module, name='weight', n_power_iterations=1, eps=1e-12, dim=None):
  if dim is None:
    if isinstance(module, (torch.nn.ConvTranspose1d,
                           torch.nn.ConvTranspose2d,
                           torch.nn.ConvTranspose3d)):
      dim = 1
    else:
      dim = 0
  SpectralNorm.apply(module, name, n_power_iterations, dim, eps)
  return module

def remove_spectral_norm(module, name='weight'):
  for k, hook in module._forward_pre_hooks.items():
    if isinstance(hook, SpectralNorm) and hook.name == name:
      hook.remove(module)
      del module._forward_pre_hooks[k]
      return module
  raise ValueError("spectral_norm of '{}' not found in {}".format(name, module))
