import networks
import torch
import torch.nn as nn
import numpy as np
import GMVAE
torch.autograd.set_detect_anomaly(True)


class MD_multi(nn.Module):
  def __init__(self, opts):
    super(MD_multi, self).__init__()
    self.opts = opts
    lr = 0.0001
    lr_dcontent = lr/2.5 
    self.nz =64#8

    self.isDcontent = opts.isDcontent
    if opts.concat == 1:
      self.concat = True
    else:
      self.concat = False

    self.dis1 = networks.MD_Dis(opts.input_dim, norm=opts.dis_norm, sn=opts.dis_spectral_norm, c_dim=opts.num_domains, image_size=opts.crop_size)
    self.dis2 = networks.MD_Dis(opts.input_dim, norm=opts.dis_norm, sn=opts.dis_spectral_norm, c_dim=opts.num_domains, image_size=opts.crop_size)
    self.enc_c = networks.MD_E_content(opts.input_dim)
    if self.concat:
      self.enc_a = networks.MD_E_attr_concat(opts.input_dim,opts.gaussian_size,opts.num_classes, output_nc=self.nz, c_dim=opts.num_domains, \
          norm_layer=None, nl_layer=networks.get_non_linearity(layer_type='lrelu'))
      self.gen = networks.MD_G_multi_concat(opts.input_dim,opts.x_dim,opts.gaussian_size,opts.crop_size, c_dim=opts.num_domains, nz=self.nz)
    else:
      self.enc_a = networks.MD_E_attr(opts.input_dim, output_nc=self.nz, c_dim=opts.num_domains)
      self.gen = networks.MD_G_multi(opts.input_dim, nz=self.nz, c_dim=opts.num_domains)

    self.dis1_opt = torch.optim.Adam(self.dis1.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.dis2_opt = torch.optim.Adam(self.dis2.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.enc_c_opt = torch.optim.Adam(self.enc_c.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.enc_a_opt = torch.optim.Adam(self.enc_a.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    #if self.isDcontent:
    self.disContent = networks.MD_Dis_content(c_dim=opts.num_domains) 
    self.disContent_opt = torch.optim.Adam(self.disContent.parameters(), lr=lr_dcontent, betas=(0.5, 0.999), weight_decay=0.0001)

    self.cls_loss = nn.BCEWithLogitsLoss()

  def initialize(self):
    self.dis1.apply(networks.gaussian_weights_init)
    self.dis2.apply(networks.gaussian_weights_init)
    self.disContent.apply(networks.gaussian_weights_init)
    self.gen.apply(networks.gaussian_weights_init)
    self.enc_c.apply(networks.gaussian_weights_init)
    self.enc_a.apply(networks.gaussian_weights_init)

  def set_scheduler(self, opts, last_ep=0):
    self.dis1_sch = networks.get_scheduler(self.dis1_opt, opts, last_ep)
    self.dis2_sch = networks.get_scheduler(self.dis2_opt, opts, last_ep)
    self.disContent_sch = networks.get_scheduler(self.disContent_opt, opts, last_ep)
    self.enc_c_sch = networks.get_scheduler(self.enc_c_opt, opts, last_ep)
    self.enc_a_sch = networks.get_scheduler(self.enc_a_opt, opts, last_ep)
    self.gen_sch = networks.get_scheduler(self.gen_opt, opts, last_ep)

  def update_lr(self):
    self.dis1_sch.step()
    self.dis2_sch.step()
    self.disContent_sch.step()
    self.enc_c_sch.step()
    self.enc_a_sch.step()
    self.gen_sch.step()

  def setgpu(self, gpu):
    self.gpu = gpu
    # MODIFICHE NVIDIA
    self.dis1.cuda(self.gpu)
    self.dis2.cuda(self.gpu)
    self.enc_c.cuda(self.gpu)
    self.enc_a.cuda(self.gpu)
    self.gen.cuda(self.gpu)
    if self.isDcontent:
      self.disContent.cuda(self.gpu)
    # self.dis1.cuda()
    # self.dis2.cuda()
    #self.enc_c.cuda()
    #self.enc_a.cuda()
    #self.gen.cuda()
    # self.enc_c.cpu()
    # self.enc_a.cpu()
    # self.gen.cpu()
    # if self.isDcontent:
    #   self.disContent.cpu()

  def get_z_random(self, batchSize, nz, random_type='gauss'):
    # NVIDIA
    z = torch.randn(batchSize, nz).cuda(self.gpu)
    #z = torch.randn(batchSize, nz).cpu()
    return z

  def test_forward_random(self, image):
    self.z_content = self.enc_c.forward(image)
    outputs = []
    for i in range(self.opts.num_domains):
      self.z_random = self.get_z_random(image.size(0), self.nz, 'gauss')
      c_trg = np.zeros((image.size(0),self.opts.num_domains))
      c_trg[:,i] = 1
      c_trg = torch.FloatTensor(c_trg).cuda()
      output = self.gen.forward(self.z_content, self.z_random, c_trg)
      outputs.append(output)
    return outputs

  def test_forward_transfer(self, image, image_trg, c_trg,temperature=1.0,hard=0):
    self.z_content = self.enc_c.forward(image)
    inf, infvar = self.enc_a.forward(self.image_trg, self.c_trg,temperature,hard)
    self.mu=inf["mean"]
    self.logvar=infvar['var'].log()
    #print("log",self.logvar)
    std = self.logvar.mul(0.5).exp_()
    eps = self.get_z_random(std.size(0), std.size(1), 'gauss')
    self.z_attr = eps.mul(std).add_(self.mu)
    output = self.gen.forward(self.z_content, self.z_attr, c_trg)
    return output

  def forward(self):
    # input images
    if not self.input.size(0)%2 == 0:
      #print("Need to be even QAQ")
      input()
    half_size = self.input.size(0)//2
    self.real_A = self.input[0:half_size]
    self.real_B = self.input[half_size:]
    c_org_A = self.c_org[0:half_size]
    c_org_B = self.c_org[half_size:]

    # get encoded z_c
    self.real_img = torch.cat((self.real_A, self.real_B),0)
    self.z_content = self.enc_c.forward(self.real_img)
    #print("z",self.z_content.size())
    self.z_content_a, self.z_content_b = torch.split(self.z_content, half_size, dim=0)

    # get encoded z_a
    if self.concat:
      inf, infvar = self.enc_a.forward(self.real_img, self.c_org)
      #print("inf",inf)
      self.mu=inf["mean"]
      self.logvar=infvar['var'].log()
      #print("log",self.logvar.size())
      std = self.logvar.mul(0.5).exp_()
      eps = self.get_z_random(std.size(0), std.size(1), 'gauss')
      #self.z_attr = eps.mul(std).add_(self.mu)
      self.z_attr=inf["gaussian"]
      y=inf["categorical"]
    else:
      self.z_attr = self.enc_a.forward(self.real_img, self.c_org)
    self.z_attr_a, self.z_attr_b = torch.split(self.z_attr, half_size, dim=0)
    # get random z_a
    self.z_random = self.get_z_random(half_size, self.nz, 'gauss')

    # first cross translation
    input_content_forA = torch.cat((self.z_content_b, self.z_content_a, self.z_content_b),0)
    input_content_forB = torch.cat((self.z_content_a, self.z_content_b, self.z_content_a),0)
    #print("z_attra",self.z_attr_a.size())
    #print("z_random",self.z_random.size())
    input_attr_forA = torch.cat((self.z_attr_a, self.z_attr_a, self.z_random),0)
    input_attr_forB = torch.cat((self.z_attr_b, self.z_attr_b, self.z_random),0)
    input_c_forA = torch.cat((c_org_A, c_org_A, c_org_A), 0)
    input_c_forB = torch.cat((c_org_B, c_org_B, c_org_B), 0)
    #print("content",input_content_forA.size())
    #print("attr",input_attr_forA.size())
    #print("c",input_c_forA.size())
    #print("y",y.size())
    #print("y",y.size())
    infA = self.gen.forward(input_content_forA, input_attr_forA, input_c_forA,y)
    infB = self.gen.forward(input_content_forB, input_attr_forB, input_c_forB,y)
    output_fakeA=infA['x_rec']
    output_fakeB=infB['x_rec']
    #print("dim",output_fakeA.size())
    self.fake_A_encoded, self.fake_AA_encoded, self.fake_A_random = torch.split(output_fakeA, self.z_content_a.size(0), dim=0)
    self.fake_B_encoded, self.fake_BB_encoded, self.fake_B_random = torch.split(output_fakeB, self.z_content_a.size(0), dim=0)

    # get reconstructed encoded z_c
    self.fake_encoded_img = torch.cat((self.fake_A_encoded, self.fake_B_encoded),0)
    self.z_content_recon = self.enc_c.forward(self.fake_encoded_img)
    self.z_content_recon_b, self.z_content_recon_a = torch.split(self.z_content_recon, half_size, dim=0)

    # get reconstructed encoded z_a
    if self.concat:
      inf, infvar = self.enc_a.forward(self.fake_encoded_img, self.c_org)
      self.mu_recon=inf["mean"]
      self.logvar_recon=infvar['var'].log()
      std_recon = self.logvar_recon.mul(0.5).exp_()
      eps_recon = self.get_z_random(std_recon.size(0), std_recon.size(1), 'gauss')
      #self.z_attr_recon = eps_recon.mul(std_recon).add_(self.mu_recon)
      self.z_attr_recon=inf["gaussian"]
      y=inf["categorical"]
    else:
      self.z_attr_recon = self.enc_a.forward(self.fake_encoded_img, self.c_org)
    self.z_attr_recon_a, self.z_attr_recon_b = torch.split(self.z_attr_recon, half_size, dim=0)

    # second cross translation
    infA = self.gen.forward(self.z_content_recon_a, self.z_attr_recon_a, c_org_A,y)
    self.fake_A_recon=infA['x_rec']
    infB= self.gen.forward(self.z_content_recon_b, self.z_attr_recon_b, c_org_B,y)
    self.fake_B_recon =infB['x_rec']

    # for display
    self.image_display = torch.cat((self.real_A[0:1].detach().cpu(), self.fake_B_encoded[0:1].detach().cpu(), \
                                    self.fake_B_random[0:1].detach().cpu(), self.fake_AA_encoded[0:1].detach().cpu(), self.fake_A_recon[0:1].detach().cpu(), \
                                    self.real_B[0:1].detach().cpu(), self.fake_A_encoded[0:1].detach().cpu(), \
                                    self.fake_A_random[0:1].detach().cpu(), self.fake_BB_encoded[0:1].detach().cpu(), self.fake_B_recon[0:1].detach().cpu()), dim=0)

    # for latent regression
    self.fake_random_img = torch.cat((self.fake_A_random, self.fake_B_random), 0)
    if self.concat:
      inf, _= self.enc_a.forward(self.fake_random_img, self.c_org)
      self.mu2=inf['mean']
      self.mu2_a, self.mu2_b = torch.split(self.mu2, half_size, 0)
    else:
      self.z_attr_random = self.enc_a.forward(self.fake_random_img, self.c_org)
      self.z_attr_random_a, self.z_attr_random_b = torch.split(self.z_attr_random, half_size, 0)


  def update_D_content(self, image, c_org):
    self.input = image
    self.z_content = self.enc_c.forward(self.input)
    self.disContent_opt.zero_grad()
    pred_cls = self.disContent.forward(self.z_content.detach())
    loss_D_content = self.cls_loss(pred_cls, c_org)
    loss_D_content.backward()
    self.disContent_loss = loss_D_content.item()
    nn.utils.clip_grad_norm_(self.disContent.parameters(), 5)
    self.disContent_opt.step()

  def update_D(self, image, c_org):
    self.input = image
    self.c_org = c_org
    self.forward()

    self.dis1_opt.zero_grad()
    self.D1_gan_loss, self.D1_cls_loss = self.backward_D(self.dis1, self.input, self.fake_encoded_img)
    self.dis1_opt.step()

    self.dis2_opt.zero_grad()
    self.D2_gan_loss, self.D2_cls_loss = self.backward_D(self.dis2, self.input, self.fake_random_img)
    self.dis2_opt.step()

  def backward_D(self, netD, real, fake):
    pred_fake, pred_fake_cls = netD.forward(fake.detach())
    pred_real, pred_real_cls = netD.forward(real)
    loss_D_gan = 0
    for it, (out_a, out_b) in enumerate(zip(pred_fake, pred_real)):
      out_fake = nn.functional.sigmoid(out_a)
      out_real = nn.functional.sigmoid(out_b)
      # NVIDIA
      all0 = torch.zeros_like(out_fake).cuda(self.gpu)
      all1 = torch.ones_like(out_real).cuda(self.gpu)
      #all0 = torch.zeros_like(out_fake).cpu()
      #all1 = torch.ones_like(out_real).cpu()
      ad_fake_loss = nn.functional.binary_cross_entropy(out_fake, all0)
      ad_true_loss = nn.functional.binary_cross_entropy(out_real, all1)
      loss_D_gan += ad_true_loss + ad_fake_loss

    loss_D_cls = self.cls_loss(pred_real_cls, self.c_org)
    loss_D = loss_D_gan + self.opts.lambda_cls * loss_D_cls 
    self.loss_D = loss_D.item()
    loss_D = + self.opts.lambda_cls * loss_D_cls 
    loss_D.backward()
    return loss_D_gan, loss_D_cls

  def update_EG(self):
    # update G, Ec, Ea
    self.enc_c_opt.zero_grad()
    self.enc_a_opt.zero_grad()
    self.gen_opt.zero_grad()
    self.backward_EG()
    self.enc_c_opt.step()
    self.enc_a_opt.step()
    self.gen_opt.step()

    # update G, Ec
    self.enc_c_opt.zero_grad()
    self.gen_opt.zero_grad()
    self.forward()
    self.backward_G_alone()
    self.enc_c_opt.step()
    self.gen_opt.step()

  def backward_EG(self):
    # content Ladv for generator
    if self.opts.isDcontent:
      loss_G_GAN_content = self.backward_G_GAN_content(self.z_content)

    # Ladv for generator
    pred_fake, pred_fake_cls = self.dis1.forward(self.fake_encoded_img)
    loss_G_GAN = 0
    for out_a in pred_fake:
      outputs_fake = nn.functional.sigmoid(out_a)
      #NVIDIA
      all_ones = torch.ones_like(outputs_fake).cuda(self.gpu)
      #all_ones = torch.ones_like(outputs_fake).cpu()
      loss_G_GAN += nn.functional.binary_cross_entropy(outputs_fake, all_ones)

    # classification
    loss_G_cls = self.cls_loss(pred_fake_cls, self.c_org) * self.opts.lambda_cls_G

    # self and cross-cycle recon
    loss_G_L1_self = torch.mean(torch.abs(self.input - torch.cat((self.fake_AA_encoded, self.fake_BB_encoded), 0))) * self.opts.lambda_rec
    loss_G_L1_cc = torch.mean(torch.abs(self.input - torch.cat((self.fake_A_recon, self.fake_B_recon), 0))) * self.opts.lambda_rec


    # KL loss - z_c
    loss_kl_zc = self._l2_regularize(self.z_content) * 0.01 

    # KL loss - z_a
    if self.concat:
      kl_element = self.mu.pow(2).add_(self.logvar.exp()).mul_(-1).add_(1).add_(self.logvar)
      loss_kl_za = torch.sum(kl_element).mul_(-0.5) * 0.01
    else:
      loss_kl_za = self._l2_regularize(self.z_attr) * 0.01


    loss_G = loss_G_GAN + loss_G_cls + loss_G_L1_self + loss_G_L1_cc + loss_kl_zc + loss_kl_za
    if self.opts.isDcontent:
      loss_G += loss_G_GAN_content
    loss_G.backward(retain_graph=True)

    self.gan_loss = loss_G_GAN.item()
    self.gan_cls_loss = loss_G_cls.item()
    if self.opts.isDcontent:
      self.gan_loss_content = loss_G_GAN_content.item()
    self.kl_loss_zc = loss_kl_zc.item()
    self.kl_loss_za = loss_kl_za.item()
    self.l1_self_rec_loss = loss_G_L1_self.item()
    self.l1_cc_rec_loss = loss_G_L1_cc.item()
    self.loss_G = loss_G.item()
    self.G_loss = loss_G.item()

  def backward_G_GAN_content(self, data):
    pred_cls = self.disContent.forward(data)
    loss_G_content = self.cls_loss(pred_cls, 1-self.c_org)
    return loss_G_content

  def backward_G_alone(self):
    # Ladv for generator
    pred_fake, pred_fake_cls = self.dis2.forward(self.fake_random_img)
    loss_G_GAN2 = 0
    for out_a in pred_fake:
      outputs_fake = nn.functional.sigmoid(out_a)
      #NVIDIA
      all_ones = torch.ones_like(outputs_fake).cuda(self.gpu)
      #all_ones = torch.ones_like(outputs_fake).cpu()
      loss_G_GAN2 += nn.functional.binary_cross_entropy(outputs_fake, all_ones)

    # classification
    loss_G_cls2 = self.cls_loss(pred_fake_cls, self.c_org) * self.opts.lambda_cls_G

    # latent regression loss
    if self.concat:
      loss_z_L1_a = torch.mean(torch.abs(self.mu2_a - self.z_random)) * 10
      loss_z_L1_b = torch.mean(torch.abs(self.mu2_b - self.z_random)) * 10
    else:
      loss_z_L1_a = torch.mean(torch.abs(self.z_attr_random_a - self.z_random)) * 10
      loss_z_L1_b = torch.mean(torch.abs(self.z_attr_random_b - self.z_random)) * 10

    loss_z_L1 = loss_z_L1_a + loss_z_L1_b + loss_G_GAN2 + loss_G_cls2
    loss_z_L1.backward()
    self.l1_recon_z_loss = loss_z_L1_a.item() + loss_z_L1_b.item()
    self.gan2_loss = loss_G_GAN2.item()
    self.gan2_cls_loss = loss_G_cls2.item()

  def _l2_regularize(self, mu):
    mu_2 = torch.pow(mu, 2)
    encoding_loss = torch.mean(mu_2)
    return encoding_loss

  def update_lr(self):
    self.dis1_sch.step()
    self.dis2_sch.step()
    self.enc_c_sch.step()
    self.enc_a_sch.step()
    self.gen_sch.step()

  def assemble_outputs(self):
    images_a = self.normalize_image(self.real_A).detach()
    images_b = self.normalize_image(self.real_B).detach()
    images_a1 = self.normalize_image(self.fake_A_encoded).detach()
    images_a2 = self.normalize_image(self.fake_A_random).detach()
    images_a3 = self.normalize_image(self.fake_A_recon).detach()
    images_a4 = self.normalize_image(self.fake_AA_encoded).detach()
    images_b1 = self.normalize_image(self.fake_B_encoded).detach()
    images_b2 = self.normalize_image(self.fake_B_random).detach()
    images_b3 = self.normalize_image(self.fake_B_recon).detach()
    images_b4 = self.normalize_image(self.fake_BB_encoded).detach()
    row1 = torch.cat((images_a[0:1, ::], images_b1[0:1, ::], images_b2[0:1, ::], images_a4[0:1, ::], images_a3[0:1, ::]),3)
    row2 = torch.cat((images_b[0:1, ::], images_a1[0:1, ::], images_a2[0:1, ::], images_b4[0:1, ::], images_b3[0:1, ::]),3)
    return torch.cat((row1,row2),2)

  def normalize_image(self, x):
    return x[:,0:3,:,:]


  def save(self, filename, ep, total_it):
    state = {
             'dis1': self.dis1.state_dict(),
             'dis2': self.dis2.state_dict(),
             'disContent': self.disContent.state_dict(),
             'enc_c': self.enc_c.state_dict(),
             'enc_a': self.enc_a.state_dict(),
             'gen': self.gen.state_dict(),
             'dis1_opt': self.dis1_opt.state_dict(),
             'dis2_opt': self.dis2_opt.state_dict(),
             'disContent_opt': self.disContent_opt.state_dict(),
             'enc_c_opt': self.enc_c_opt.state_dict(),
             'enc_a_opt': self.enc_a_opt.state_dict(),
             'gen_opt': self.gen_opt.state_dict(),
             'ep': ep,
             'total_it': total_it
              }
    torch.save(state, filename)
    return

  def resume(self, model_dir, train=True):
    checkpoint = torch.load(model_dir)
    # weight
    if train:
      self.dis1.load_state_dict(checkpoint['dis1'])
      self.dis2.load_state_dict(checkpoint['dis2'])
      if self.isDcontent:
        self.disContent.load_state_dict(checkpoint['disContent'])
    self.enc_c.load_state_dict(checkpoint['enc_c'])
    self.enc_a.load_state_dict(checkpoint['enc_a'])
    self.gen.load_state_dict(checkpoint['gen'])
    # optimizer
    if train:
      self.dis1_opt.load_state_dict(checkpoint['dis1_opt'])
      self.dis2_opt.load_state_dict(checkpoint['dis2_opt'])
      if self.isDcontent:
        self.disContent_opt.load_state_dict(checkpoint['disContent_opt'])
      self.enc_c_opt.load_state_dict(checkpoint['enc_c_opt'])
      self.enc_a_opt.load_state_dict(checkpoint['enc_a_opt'])
      self.gen_opt.load_state_dict(checkpoint['gen_opt'])
    return checkpoint['ep'], checkpoint['total_it']
  
  
  ### GMVAE LOSSES ###
  
  
  def train_epoch_GMVAE(self, optimizer, data_loader):
    """Train the model for one epoch
    Args:
        optimizer: (Optim) optimizer to use in backpropagation
        data_loader: (DataLoader) corresponding loader containing the training data
    Returns:
        average of all loss values, accuracy, nmi
    """
    self.network.train()
    total_loss = 0.
    recon_loss = 0.
    cat_loss = 0.
    gauss_loss = 0.
    accuracy = 0.
    nmi = 0.
    num_batches = 0.
    true_labels_list = []
    predicted_labels_list = []
    metrics=GMVAE.Metrics()
    
    # iterate over the dataset
    for (data, labels) in data_loader: # le dimensioni di questo data loader sono come quelle che abbiamo nel train??
      if self.cuda == 1:
        data = data.cuda()
      print("data",data.shape)
      print("labels",labels.shape)
      optimizer.zero_grad()
      # flatten data
      data = data.view(data.size(0), -1)
      # forward call
      out_net = self.network(data, self.gumbel_temp, self.opts.hard_gumbel) # GUARDA QUI_: network genera un out_net che ha in x_recon stringhe
      #print("on",out_net)
      unlab_loss_dic = self.unlabeled_loss(data, out_net)
      total = unlab_loss_dic['total']
      # accumulate values
      total_loss += total.item()
      recon_loss += unlab_loss_dic['reconstruction'].item()
      gauss_loss += unlab_loss_dic['gaussian'].item()
      cat_loss += unlab_loss_dic['categorical'].item()
      # perform backpropagation
      total.backward()
      optimizer.step()
      # save predicted and true labels
      predicted = unlab_loss_dic['predicted_labels']
      print(f"Predicted labels shape: {predicted.shape}")
      print(f"True labels shape: {labels.shape}")
      true_labels_list.append(labels)
      predicted_labels_list.append(predicted)
      num_batches += 1.
    # average per batch
    total_loss /= num_batches
    recon_loss /= num_batches
    gauss_loss /= num_batches
    cat_loss /= num_batches
    # concat all true and predicted labels
    print("true_lanels_list dimension: ",len(true_labels_list))
    print("predicted_lanels_list dimension: ",len(predicted_labels_list))
    true_labels = torch.cat(true_labels_list, dim=0).cpu().numpy()
    predicted_labels = torch.cat(predicted_labels_list, dim=0).cpu().numpy()
    # compute metrics
    accuracy = 100.0 * metrics.cluster_acc(predicted_labels, true_labels)
    nmi = 100.0 * metrics.nmi(predicted_labels, true_labels)
    return total_loss, recon_loss, gauss_loss, cat_loss, accuracy, nmi
  
  
  def unlabeled_loss(self, data, out_net):
    """Method defining the loss functions derived from the variational lower bound
    Args:
        data: (array) corresponding array containing the input data
        out_net: (dict) contains the graph operations or nodes of the network output. Output del GenerativeNet
    Returns:
        loss_dic: (dict) contains the values of each loss function and predictions
    """
    # obtain network variables
    loss_functions = GMVAE.LossFunctions()
    #print("prima",type(out_net['x_rec']))
    z, data_recon = out_net['gaussian'], out_net['x_rec']
    logits, prob_cat = out_net['logits'], out_net['prob_cat']
    print(f"logits shape: {logits.shape}, batch size: {data.shape[0]}")
    y_mu, y_var = out_net['y_mean'], out_net['y_var']
    mu, var = out_net['mean'], out_net['var']
    # reconstruction loss
    # print("dopo",type(data_recon))
    # print("data type",type(data))
    loss_rec = loss_functions.reconstruction_loss(data, data_recon) # data: tensore, data_recon: stringa
    # gaussian loss
    loss_gauss = loss_functions.gaussian_loss(z, mu, var, y_mu, y_var)
    # categorical loss
    loss_cat = -loss_functions.entropy(logits, prob_cat) - np.log(0.1)
    # total loss
    loss_total = self.opts.w_rec * loss_rec + self.opts.w_gauss * loss_gauss + self.opts.w_categ * loss_cat
    # obtain predictions
    _, predicted_labels = torch.max(logits, dim=1)
    loss_dic = {'total': loss_total,
                'predicted_labels': predicted_labels,
                'reconstruction': loss_rec,
                'gaussian': loss_gauss,
                'categorical': loss_cat}
    return loss_dic
