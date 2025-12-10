import yaml
import tqdm
import argparse
from torch.optim import AdamW
from policy_diffusion_model import *
from dataset import *
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from transformer import *
import matplotlib.pyplot as plt

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

    noise_prediction_model = TransformerForDiffusion(
            input_dim=action_dim,
            output_dim=action_dim,          # predict noise with same shape
            horizon=pred_horizon,           # T = 16
            n_obs_steps=observation_horizon, # 2
            cond_dim=observation_dim,       # we pass obs as cond
            n_layer=4,
            n_head=4,
            n_emb=128,
            p_drop_emb=0.1,
            p_drop_attn=0.1,
            causal_attn=True,
            time_as_cond=True,
            n_cond_layers=1
    ).to(device)


    ema = EMAModel(parameters=noise_prediction_model.parameters(), power=0.75)

    noise_scheduler = NoiseScheduler(num_timesteps=10).to(device)    

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

        total_loss = list()
        
        for epoch_idx in tglobal:
        
            epoch_loss = list()
            # batch loop
            
            # ! THERE IS PROBLEM WITH LAST BATCH with dims wrong

            with tqdm.tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                
                for nbatch in tepoch:
                    # data normalized in dataset
                    # device transfer
                    
                    optimizer.zero_grad()

                    nobs = nbatch['obs'].to(device)      # (B, obs_horizon, obs_dim)
                    naction = nbatch['action'].to(device)  # (B, pred_horizon, act_dim)

                    B = nobs.shape[0]

                    # conditioning sequence: (B, obs_horizon, obs_dim)
                    obs_cond = nobs[:, :observation_horizon, :]

                    noise = torch.randn_like(naction, device=device)
                    timesteps = torch.randint(
                        0, noise_scheduler.num_timesteps,
                        (B,), device=device
                    )
                    noisy_actions = noise_scheduler.add_noise(naction, noise, timesteps)

                    noise_pred = noise_prediction_model(
                        sample=noisy_actions,    # (B, T, act_dim)
                        timestep=timesteps,      # (B,)
                        cond=obs_cond            # (B, To, obs_dim)
                    )
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
            total_loss.append(np.mean(epoch_loss))

            if (epoch_idx + 1) % 10 == 0:
                print(f"Finished epoch {epoch_idx+1}/{num_epochs} | Loss: {np.mean(epoch_loss):.4f}")
                torch.save(noise_prediction_model.state_dict(), f'./saves/pusht_chkpt_{epoch_idx + 1}.pth')
                torch.save(ema.state_dict(), f'./saves/ema_chkpt_{epoch_idx + 1}.pth')
        
        
        print("Finished training!")
        total_loss = np.array(total_loss)
        np.save("./loss_pusht_transformer_ddpm", total_loss)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for DDPM training')
    parser.add_argument('--config', dest='config_path', default='./config.yaml', type=str)
    args = parser.parse_args()
    train(args)
    