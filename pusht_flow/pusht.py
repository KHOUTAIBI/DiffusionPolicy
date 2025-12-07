import yaml
import tqdm
import argparse
from torch.optim import AdamW
from dataset import *
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from building_blocks import *
from noise_scheduler import *

# Login using e.g. `huggingface-cli login` to access this dataset
def collate_fn(batch):
    # batch is a list of dicts
    obs = torch.tensor([b['observation_state'] for b in batch], dtype=torch.float32)
    act = torch.tensor([b['action'] for b in batch], dtype=torch.float32)
    return {'observation_state': obs, 'action': act}

# -------------------------------
# Train Using the push-t dataset
# -------------------------------
def train(args):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Horizons and variables
    observation_dim = 5
    observation_horizon = 2
    action_dim = 2
    pred_horizon = 16
    action_horizon = 8
    num_epochs = 100
    

    #|o|o|                             observations: 2
    #| |a|a|a|a|a|a|a|a|               actions executed: 8
    #|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16

    noise_prediction_model = ConditionalUnet1D(
        input_dim = action_dim,
        global_cond_dim = observation_dim * observation_horizon,
        n_groups = 8
    ).to(device)
    ema = EMAModel(parameters=noise_prediction_model.parameters(), power=0.75)

    noise_scheduler = NoiseScheduler(num_timesteps=100).to(device)    

    # Login using e.g. `huggingface-cli login` to access this dataset
    dataset_path = "pusht_cchi_v7_replay.zarr.zip"

    
    dataset = PushTStateDataset(
        dataset_path=dataset_path,
        pred_horizon=pred_horizon,
        obs_horizon=observation_horizon,
        action_horizon=action_horizon
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, #type: ignore  
        batch_size=256,
        num_workers=0,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True,
    )

    # optimizer
    optimizer = AdamW(noise_prediction_model.parameters(), lr=1e-4, weight_decay=1e-6) 
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * num_epochs
    )

    print("----------------------------------")
    print(f"Staring to train")
    print("----------------------------------")
    
    with tqdm.tqdm(range(num_epochs), desc='Epoch') as tglobal:
        
        for epoch_idx in tglobal:
        
            epoch_loss = list()
            # batch loop
            
            # ! THERE IS PROBLEM WITH LAST BATCH with dims wrong

            with tqdm.tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                
                for nbatch in tepoch:
                    # data normalized in dataset
                    # device transfer
                    
                    optimizer.zero_grad()
                    nobs = nbatch['obs'].to(device)
                    naction = nbatch['action'].to(device)

                    B = nobs.shape[0] # batch, size of trianing samples
                    
                    obs_cond = nobs[:, :observation_horizon, :]
                    obs_cond = nobs.flatten(start_dim = 1)
                    
                    
                    # This needs to get fixed
                    
                    # sample noise to add to actions
                    noise = torch.randn(naction.shape, device=device)

                    # sample a diffusion iteration for each data point
                    timesteps = torch.randint(
                        0, noise_scheduler.num_timesteps,
                        (B, ), device=device
                    )

                    # add noise to the clean images according to the noise magnitude at each diffusion iteration
                    # (this is the forward diffusion process)
                    
                    noisy_actions = noise_scheduler.add_noise(
                        naction, noise, timesteps)
                    # predict the noise residual
                    noise_pred = noise_prediction_model(
                        noisy_actions, timesteps, global_cond=obs_cond)

                    # L2 loss
                    loss = nn.functional.mse_loss(noise_pred, noise)

                    # optimize
                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()
                    ema.step(noise_prediction_model.parameters())

                    # logging
                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)
                    
            tglobal.set_postfix(loss=np.mean(epoch_loss))

            if (epoch_idx + 1) % 10 == 0:
                print(f"Finished epoch {epoch_idx+1}/{num_epochs} | Loss: {np.mean(epoch_loss):.4f}")
                torch.save(noise_prediction_model.state_dict(), f'./saves/pusht_chkpt_{epoch_idx + 1}.pth')
                torch.save(ema.state_dict(), f'./saves/ema_chkpt_{epoch_idx + 1}.pth')
        
        print("Finished training!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for DDPM training')
    parser.add_argument('--config', dest='config_path', default='./config.yaml', type=str)
    args = parser.parse_args()
    train(args)
    