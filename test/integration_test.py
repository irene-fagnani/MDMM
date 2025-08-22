import sys
import os

# Add src folder to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Now import your local modules
from options import TrainOptions
from datasets import dataset_multi  # dataset_multi is a class/function inside src/datasets.py
from model import MD_multi
import GMVAE
import torch
import numpy as np


def integration_test():
    print("=== Running Integration Test ===")
    
    # Create dummy options
    class DummyOpts:
        batch_size = 2
        nThreads = 0
        use_cuda = False
        gpu = 0
        resume = None
        max_it = 5
        n_ep = 1
        init_temp = 1.0
        decay_temp_rate = 0.01
        min_temp = 0.5
        x_dim = 2916
        gaussian_size = 108
        num_classes = 2
        isDcontent = False
        d_iter = 1
        n_ep_decay = -1
        no_display_img = True
        plot_losses = False
        dataroot = "../datasets/apple2orange"
        num_domains = 2
        input_dim = 3
        phase = 'train'
        resize_size=128
        crop_size=108
    
    opts = DummyOpts()
    
    # Initialize dummy dataset
    print("Creating dummy dataset...")
    dataset = dataset_multi(opts)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.nThreads)
    
    # Initialize model
    print("Initializing model...")
    model = model.MD_multi(opts, train_loader)
    if opts.use_cuda:
        model.setgpu(opts.gpu)
    
    # Initialize GMVAE inside model
    model.network = GMVAE.GMVAENet(opts.x_dim, opts.gaussian_size, opts.num_classes)
    
    # Dummy training loop (single iteration)
    print("Running dummy forward/backward pass...")
    for images, c_org in train_loader:
        if images.size(0) != opts.batch_size:
            continue
        
        # Forward pass
        if opts.use_cuda:
            images = images.cuda(opts.gpu)
            c_org = c_org.cuda(opts.gpu)
        output = model.network(images)
        
        # Check output shape
        assert output['mean'].shape[0] == opts.batch_size
        assert output['gaussian'].shape[0] == opts.batch_size
        
        # Compute dummy loss and backward
        loss = ((output['mean'] - images.view(opts.batch_size, -1))**2).mean()
        loss.backward()
        print(f"Loss computed: {loss.item()}")
        break
    
    print("Integration test passed! Model, GMVAE, and forward/backward pass work.")

if __name__ == "__main__":
    integration_test()
