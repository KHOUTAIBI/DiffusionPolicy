import json
import yaml
import tqdm
import argparse
import torch
import torchvision
import torch.nn as nn
import numpy as np
from unet import Unet
from noise_scheduler import NoiseScheduler
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.optim import Adam

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------------
# Train
# -------------------------------
def train(args):
    """
    Training the network to predict the noise.
    """
    # Load config
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except Exception as e:
            print(f'Error loading config: {e}')
            return None
    
    # Configuration of model
    diffusion_config = config['diffusion_config']
    dataset_config = config['dataset_config']
    model_config = config['model_config']
    train_config = config['train_config']

    # Dataset and DataLoader
    mnist_trainset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transforms.ToTensor()
    )
    mnist_loader = DataLoader(
        mnist_trainset, batch_size=train_config['batch_size'], shuffle=True, num_workers=4
    )

    # Noise scheduler
    scheduler = NoiseScheduler(
        num_timesteps=diffusion_config['num_timesteps'],
        beta_init=diffusion_config['beta_init'],
        beta_end=diffusion_config['beta_end']
    )

    # Model
    model = Unet(model_config).to(device)
    model.train()

    # Loss & optimizer
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=train_config['lr'])
    num_epochs = train_config['num_epochs']

    # Training loop
    for epoch in range(num_epochs):
        losses = []
        for img, _ in tqdm.tqdm(mnist_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):

            # image sampling
            img = img.float().to(device)
            optimizer.zero_grad()

            # Noise to be added to image
            noise = torch.randn_like(img).to(device)
            t = torch.randint(0, diffusion_config['num_timesteps'], size=(img.shape[0],), device=device)

            # making the image noisy and predicting the noise
            image_noisy = scheduler.add_noise(img, noise, t)
            noise_prediction = model(image_noisy, t)  # if your Unet expects timestep input

            # Loss and backpropagate
            loss = criterion(noise_prediction, noise)
            losses.append(loss.item())

            loss.backward()
            optimizer.step()

        print(f"Finished epoch {epoch+1}/{num_epochs} | Loss: {np.mean(losses):.4f}")
        torch.save(model.state_dict(), f'./saves/ddpm_chkpt_{epoch}.pth')
    print("Finished training!")


# -------------------------------
# Sample
# -------------------------------
def sample(model, scheduler, training_config, model_config, diffusion_config):
    """
    Sample new image using random noise.
    """

    xt = torch.randn(
        size=(
            training_config['batch_size'],
            model_config['im_channels'],
            model_config['im_size'],
            model_config['im_size']
        ),  
    device=device
    )
    
    for i in tqdm.tqdm(reversed(range(diffusion_config['num_timesteps']))):
        # Noise prediciton 
        noise_prediciton = model(xt, torch.as_tensor(i).unsqueeze(0).to(device)) # I unsqueezed here to add new axis in dim = 0 (i.e the very first element)

        # scheduling
        xt, x0 = scheduler.reverse_process(xt, noise_prediciton, torch.as_tensor(i).to(device))
        
        # Image sampled
        ims = torch.clamp(xt, -1., 1.).detach().cpu()
        ims = (ims + 1) / 2

        grid = make_grid(ims, nrow=training_config['num_grid_rows'])
        img = torchvision.transforms.ToPILImage()(grid)

        if i % 100 == 0:
            img.save(f'./samples/sample_{i}.png')
            img.close()

# -------------------------------
# Inference
# -------------------------------

def infer(args):

    # Reading yaml
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    
    diffusion_config = config['diffusion_config']
    model_config = config['model_config']
    train_config = config['train_config']

    # Loading model
    model = Unet(model_config).to(device)
    model.load_state_dict(torch.load(model_config['save_path']))

    model.eval()

    scheduler = NoiseScheduler(num_timesteps=diffusion_config['num_timesteps'], beta_init=diffusion_config['beta_init'], beta_end=diffusion_config['beta_end'])

    with torch.no_grad():
        sample(model, scheduler, train_config, model_config, diffusion_config)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for DDPM training')
    parser.add_argument('--config', dest='config_path', default='./config.yaml', type=str)
    args = parser.parse_args()
    infer(args)
