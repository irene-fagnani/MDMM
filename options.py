import argparse

class TrainOptions():
  def __init__(self):
    self.parser = argparse.ArgumentParser()

    # data loader related
    self.parser.add_argument('--dataroot', type=str, required=True, help='path of data')
    self.parser.add_argument('--num_domains', type=int, default=3)
    self.parser.add_argument('--phase', type=str, default='train', help='phase for dataloading')
    self.parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    self.parser.add_argument('--resize_size', type=int, default=128, help='resized image size for training')
    self.parser.add_argument('--crop_size', type=int, default=108, help='cropped image size for training')
    self.parser.add_argument('--input_dim', type=int, default=3, help='# of input channels for domain A')
    #self.parser.add_argument('--nThreads', type=int, default=8, help='# of threads for data loader')
    self.parser.add_argument('--nThreads', type=int, default=2, help='# of threads for data loader')
    self.parser.add_argument('--no_flip', action='store_true', help='specified if no flipping')

    # ouptput related
    self.parser.add_argument('--name', type=str, default='trial', help='folder name to save outputs')
    self.parser.add_argument('--display_dir', type=str, default='./logs', help='path for saving display results')
    self.parser.add_argument('--result_dir', type=str, default='./results', help='path for saving result images and models')
    self.parser.add_argument('--display_freq', type=int, default=10, help='freq (iteration) of display')
    self.parser.add_argument('--img_save_freq', type=int, default=5, help='freq (epoch) of saving images')
    self.parser.add_argument('--model_save_freq', type=int, default=10, help='freq (epoch) of saving models')
    self.parser.add_argument('--no_display_img', action='store_true', help='specified if no dispaly')

    # training related
    self.parser.add_argument('--concat', type=int, default=1, help='concatenate attribute features for translation, set 0 for using feature-wise transform')
    self.parser.add_argument('--dis_scale', type=int, default=3, help='scale of discriminator')
    self.parser.add_argument('--dis_norm', type=str, default='None', help='normalization layer in discriminator [None, Instance]')
    self.parser.add_argument('--dis_spectral_norm', action='store_true', help='use spectral normalization in discriminator')
    self.parser.add_argument('--lr_policy', type=str, default='lambda', help='type of learn rate decay')
    self.parser.add_argument('--n_ep', type=int, default=1200, help='number of epochs') # 400 * d_iter
    self.parser.add_argument('--n_ep_decay', type=int, default=600, help='epoch start decay learning rate, set -1 if no decay') # 200 * d_iter
    self.parser.add_argument('--resume', type=str, default=None, help='specified the dir of saved models for resume the training')
    self.parser.add_argument('--d_iter', type=int, default=3, help='# of iterations for updating content discriminator')
    self.parser.add_argument('--lambda_rec', type=float, default=10)
    self.parser.add_argument('--lambda_cls', type=float, default=1.0)
    self.parser.add_argument('--lambda_cls_G', type=float, default=5.0)
    self.parser.add_argument('--isDcontent', action='store_true')
    self.parser.add_argument('--gpu', type=int, default=0, help='gpu')

    #GMVAE parameters
    self.parser.add_argument('--num_classes', type=int, default=2, help='number of classes (default: 2)')
    self.parser.add_argument('--gaussian_size', default=108, type=int, help='gaussian size (default: 64) (z_dim)')
    self.parser.add_argument('--x_dim', default=139968, type=int, help='input size (default: 2916) (2916=3*216*3*3)')
    
    ## Gumbel parameters
    self.parser.add_argument('--init_temp', default=1.0, type=float,
                        help='Initial temperature used in gumbel-softmax (recommended 0.5-1.0, default:1.0)')
    self.parser.add_argument('--decay_temp', default=1, type=int,
                        help='Set 1 to decay gumbel temperature at every epoch (default: 1)')
    self.parser.add_argument('--hard_gumbel', default=0, type=int,
                        help='Set 1 to use the hard version of gumbel-softmax (default: 1)')
    self.parser.add_argument('--min_temp', default=0.5, type=float,
                        help='Minimum temperature of gumbel-softmax after annealing (default: 0.5)' )
    self.parser.add_argument('--decay_temp_rate', default=0.013862944, type=float,
                        help='Temperature decay rate at every epoch (default: 0.013862944)')
    
    ## Loss function parameters
    self.parser.add_argument('--w_gauss', default=1, type=float,
                        help='weight of gaussian loss (default: 1)')
    self.parser.add_argument('--w_categ', default=1, type=float,
                        help='weight of categorical loss (default: 1)')
    self.parser.add_argument('--w_rec', default=1, type=float,
                        help='weight of reconstruction loss (default: 1)')
    self.parser.add_argument('--rec_type', type=str, choices=['bce', 'mse'],
                        default='bce', help='desired reconstruction loss function (default: bce)')


  def parse(self):
    self.opt = self.parser.parse_args()
    args = vars(self.opt)
    print('\n--- load options ---')
    for name, value in sorted(args.items()):
      print('%s: %s' % (str(name), str(value)))
    return self.opt

class TestOptions():
  def __init__(self):
    self.parser = argparse.ArgumentParser()

    # data loader related
    self.parser.add_argument('--dataroot', type=str, required=True, help='path of data')
    self.parser.add_argument('--num_domains', type=int, default=3)
    self.parser.add_argument('--phase', type=str, default='test', help='phase for dataloading')
    self.parser.add_argument('--resize_size', type=int, default=256, help='resized image size for training')
    self.parser.add_argument('--crop_size', type=int, default=216, help='cropped image size for training')
    self.parser.add_argument('--nThreads', type=int, default=4, help='for data loader')
    self.parser.add_argument('--input_dim', type=int, default=3, help='# of input channels for domain A')
    self.parser.add_argument('--a2b', type=int, default=1, help='translation direction, 1 for a2b, 0 for b2a')

    # ouptput related
    self.parser.add_argument('--num', type=int, default=5, help='number of outputs per image')
    self.parser.add_argument('--name', type=str, default='trial', help='folder name to save outputs')
    self.parser.add_argument('--result_dir', type=str, default='./outputs', help='path for saving result images and models')

    # model related
    self.parser.add_argument('--concat', type=int, default=1, help='concatenate attribute features for translation, set 0 for using feature-wise transform')
    self.parser.add_argument('--resume', type=str, required=True, help='specified the dir of saved models for resume the training')
    self.parser.add_argument('--gpu', type=int, default=0, help='gpu')

    self.parser.add_argument('--isDcontent', action='store_true')

    #GMVAE parameters
    self.parser.add_argument('--num_classes', type=int, default=2, help='number of classes (default: 2)')
    self.parser.add_argument('--gaussian_size', default=216, type=int, help='gaussian size (default: 64) (z_dim)')#64
    self.parser.add_argument('--x_dim', default=2916, type=int, help='input size (default: 2916) (2916=3*216*3*3)')
    
    ## Gumbel parameters
    self.parser.add_argument('--init_temp', default=1.0, type=float,
                        help='Initial temperature used in gumbel-softmax (recommended 0.5-1.0, default:1.0)')
    self.parser.add_argument('--decay_temp', default=1, type=int,
                        help='Set 1 to decay gumbel temperature at every epoch (default: 1)')
    self.parser.add_argument('--hard_gumbel', default=0, type=int,
                        help='Set 1 to use the hard version of gumbel-softmax (default: 1)')
    self.parser.add_argument('--min_temp', default=0.5, type=float,
                        help='Minimum temperature of gumbel-softmax after annealing (default: 0.5)' )
    self.parser.add_argument('--decay_temp_rate', default=0.013862944, type=float,
                        help='Temperature decay rate at every epoch (default: 0.013862944)')
    
    ## Loss function parameters
    self.parser.add_argument('--w_gauss', default=1, type=float,
                        help='weight of gaussian loss (default: 1)')
    self.parser.add_argument('--w_categ', default=1, type=float,
                        help='weight of categorical loss (default: 1)')
    self.parser.add_argument('--w_rec', default=1, type=float,
                        help='weight of reconstruction loss (default: 1)')
    self.parser.add_argument('--rec_type', type=str, choices=['bce', 'mse'],
                        default='bce', help='desired reconstruction loss function (default: bce)')

  def parse(self):
    self.opt = self.parser.parse_args()
    args = vars(self.opt)
    print('\n--- load options ---')
    for name, value in sorted(args.items()):
      print('%s: %s' % (str(name), str(value)))
    # set irrelevant options
    self.opt.dis_scale = 3
    self.opt.dis_norm = 'None'
    self.opt.dis_spectral_norm = False
    return self.opt
