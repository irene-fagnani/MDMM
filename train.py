import torch
from options import TrainOptions
from datasets import dataset_multi
from model import MD_multi
from saver import Saver

def main():
  # parse options
  parser = TrainOptions()
  opts = parser.parse()

  # daita loader
  print('\n--- load dataset ---')
  dataset = dataset_multi(opts)
  train_loader = torch.utils.data.DataLoader(dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.nThreads)
  # losses dictionary
  losses = {
    "loss_D_content": [],
    "loss_D": [],
    "loss_G": []
}
  # model
  print('\n--- load model ---')
  model = MD_multi(opts)
  # NVIDIA
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
  max_it = 1000 # 50000
  for ep in range(ep0, opts.n_ep):
    for it, (images, c_org) in enumerate(train_loader):
      #print("x_dim",images.size())
      if images.size(0) != opts.batch_size:
        #print("in if")
        continue

      # input data
      # NVIDIA
      images = images.cuda(opts.gpu).detach()
      c_org = c_org.cuda(opts.gpu).detach()
      #print("c",c_org)
      #print("i",images.size())
      #images = images.cpu().detach()
      #c_org = c_org.cpu().detach()
      #c_trg = c_trg.cuda(opts.gpu).detach()
      #input()
      
      
      # update model
      if opts.isDcontent:
        if (it + 1) % opts.d_iter != 0 and it < len(train_loader) - 2:
          model.update_D_content(images, c_org)
          losses["loss_D_content"].append(model.disContent_loss)
          print("loss_D_content", model.disContent_loss)
          continue
        else:
          model.update_D(images, c_org)
          losses["loss_D"].append(model.loss_D)
          print("loss_D", model.loss_D)
          model.update_EG()
          losses["loss_G"].append(model.loss_G)
          print("loss_G", model.loss_G)
      else:
        model.update_D(images, c_org)
        losses["loss_D"].append(model.loss_D)
        print("loss_D", model.loss_D)
        model.update_EG()
        losses["loss_G"].append(model.loss_G)
        print("loss_G", model.loss_G)
      # save to display file
      if not opts.no_display_img:
        saver.write_display(total_it, model)
      print('total_it: %d (ep %d, it %d), lr %08f' % (total_it, ep, it, model.gen_opt.param_groups[0]['lr']))
      total_it += 1
      if total_it >= max_it:
        saver.write_img(-1, model)
        saver.write_model(-1, total_it, model)
        break
      
    # decay learning rate
    if opts.n_ep_decay > -1:
      model.update_lr()

    # save result image
    saver.write_img(ep, model)

    # Save network weights
    saver.write_model(ep, total_it, model)
    
    import matplotlib.pyplot as plt

  # Plot each loss
  for key, value in losses.items():
      plt.figure(figsize=(10, 5))
      plt.plot(value, label=key)
      plt.title(f"Loss Curve: {key}")
      plt.xlabel("Iteration")
      plt.ylabel("Loss")
      plt.legend()
      plt.grid()
      plt.show()
      plt.savefig(f"loss_{key}.png")

  return

if __name__ == '__main__':
  main()
