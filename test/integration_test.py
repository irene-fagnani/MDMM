import sys
import os
#from turtle import pd

# Add src folder to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Now import your local modules
from options import TrainOptions
from datasets import dataset_multi  # dataset_multi is a class/function inside src/datasets.py
from model import MD_multi
import GMVAE
import torch
from torch import nn, optim
from saver import Saver
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def integration_test():
    print("=== Running Integration Test ===")
    
    # Create dummy options
    class DummyOpts:
        # already provided ones
        batch_size = 2
        nThreads = 0
        use_cpu =False
        use_cuda=True
        gpu = 0
        resume = None
        max_it = 10
        n_ep = 10
        init_temp = 1.0
        decay_temp_rate = 0.01
        min_temp = 0.5
        x_dim = 139968
        gaussian_size = 108
        num_classes = 2
        isDcontent = False
        d_iter = 1
        n_ep_decay = -1
        no_display_img = True
        plot_losses = True
        dataroot = "../test/datasets/toy_dataset"
        num_domains = 2
        input_dim = 3
        phase = "train"
        resize_size = 128
        crop_size = 108
        no_flip=False


        # added ones from your options.py
        name = "integration_test"
        display_dir = "./logs"
        result_dir = "./results"
        display_freq = 10
        img_save_freq = 5
        model_save_freq = 10
        concat = 1
        dis_scale = 3
        dis_norm = "None"
        dis_spectral_norm = False
        lr_policy = "lambda"
        lambda_rec = 10.0
        lambda_cls = 1.0
        lambda_cls_G = 5.0

        # gumbel parameters
        decay_temp = 1
        hard_gumbel = 0

        # loss weights
        w_gauss = 1.0
        w_categ = 1.0
        w_rec = 1.0
        rec_type = "bce"

        # generator options
        use_adain = False
        double_layer_ReLUINSConvTranspose = False
        two_time_scale_update_rule = None

    opts = DummyOpts()
    
    # Initialize dummy dataset
    print("Creating dummy dataset...")
    dataset = dataset_multi(opts)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.nThreads)
    
        # losses dictionary
    losses_graph = {
        "loss_D": [],
        "loss_G": []
    }
    # model
    print('\n--- load model ---')
    model = MD_multi(opts,train_loader)
    # NVIDIA
    if opts.use_cuda:
        model.setgpu(opts.gpu)
    if opts.resume is None:
        model.initialize()
        ep0 = -1
        total_it = 0
    else:
        ep0, total_it = model.resume(opts.resume)
    model.set_scheduler(opts, last_ep=ep0)
    ep0 += 1
    print('start the training at epoch %d'%(ep0))

    # saver for display and output
    saver = Saver(opts)

  # train
    print('\n--- train ---')
    max_it = opts.max_it # 50000
    model.network=GMVAE.GMVAENet(opts.x_dim, opts.gaussian_size, opts.num_classes)
    optimizer = optim.Adam(model.network.parameters(), lr=0.0001)
    model.gumbel_temp = opts.init_temp
    for ep in range(ep0, opts.n_ep):
      for it, (images, c_org) in enumerate(train_loader):
        if images.size(0) != opts.batch_size:
          continue
        # input data
        # NVIDIA
        if opts.use_cuda:
          images = images.cuda(opts.gpu).detach()
          c_org = c_org.cuda(opts.gpu).detach()
        else:
          images = images.cpu().detach()
          c_org = c_org.cpu().detach()
        
        # update model
        if opts.isDcontent:
          if (it + 1) % opts.d_iter != 0 and it < len(train_loader) - 2:
            model.update_D_content(images, c_org)
            continue
          else:
            model.update_D(images, c_org)
            model.update_EG()
        else:
          model.update_D(images, c_org)
          model.update_EG()

        # save to display file
        if not opts.no_display_img:
          saver.write_display(total_it, model)
        print('total_it: %d (ep %d, it %d), lr %08f' % (total_it, ep, it, model.gen_opt.param_groups[0]['lr']))
        losses_graph["loss_D"].append(model.loss_D)
        losses_graph["loss_G"].append(model.loss_G)
        total_it += 1

        if total_it >= max_it:
          saver.write_img(-1, model)
          saver.write_model(-1, total_it, model)
          break

      if ep>=1:
        model.gumbel_temp = np.maximum(opts.init_temp * np.exp(-opts.decay_temp_rate * ep), opts.min_temp)
      # decay learning rate
      if opts.n_ep_decay > -1:
        model.update_lr()
      if opts.plot_losses:
        #if ep==ep0:
          for key, value in losses_graph.items():
              plt.figure(figsize=(10, 5))
              plt.plot(value, label=key)
              plt.title(f"Loss Curve: {key}")
              plt.xlabel("Iteration")
              plt.ylabel("Loss")
              plt.legend()
              plt.grid()
              plt.savefig(f"loss_{key}_integration_test.png")
              # Save values to CSV
              df = pd.DataFrame({key: value})
              df.to_csv(f"loss_csv_{key}_integration_test.csv", index_label="Iteration")

      # save result image
      saver.write_img(ep, model)

      # Save network weights
      saver.write_model(ep, total_it, model)
    
    return

if __name__ == '__main__':
    integration_test()