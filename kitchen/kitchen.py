from dataloader import *
from blocks import *
from noise_scheduler import *
from tqdm import tqdm
from torch.optim import Adam
from diffusers.training_utils import EMAModel 
from diffusers.optimization import get_scheduler

def train():
    """
    Training the kitchen dataset using Diffusion policy
    """ 

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Loading dataset
    dataset = minari.load_dataset("D4RL/kitchen/partial-v2", download=True)

    dataset_torch = MinariTransitionDataset(dataset)
    B = 256

    loader = DataLoader(
        dataset_torch,
        batch_size=B,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    for idx, batch in enumerate(loader):
        # batch is a dict of tensors:
        #   batch["observations"]      -> (B, obs_dim)
        #   batch["actions"]           -> (B, act_dim)
        #   batch["rewards"]           -> (B,)
        #   batch["next_observations"] -> (B, obs_dim)
        #   batch["done"]              -> (B,)
        print(idx, {k: v.shape for k, v in batch.items()})
        print(f"the observation space number of features and actions space number of features is: {batch['observations'][0].shape} and {batch['actions'][0].shape}")
        print(f"An example of the observation data is:\n {batch['observations'][idx]}")
        break

    observation_dim = dataset.observation_space['observation'].shape[0]
    observation_horizon = 2

    action_dim = dataset.action_space.shape[0]
    action_horizon = 8


    prediction_horizon = observation_horizon * action_horizon
    num_epochs = 100
    num_steps = 100
    num_warmup_steps = 500

    #|o|o|                             observations: 2
    #| |a|a|a|a|a|a|a|a|               actions executed: 8
    #|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16

    denoising_model = ConditionalUnet1D(input_dim=action_dim, global_cond_dim= observation_dim * observation_horizon,
                                        n_groups = 8
                                        ).to(device)
    
    ema = EMAModel(parameters=denoising_model.parameters(), power=0.75)
    noise_scheduler = NoiseScheduler(num_timesteps=100, device=device)

    # optimizer
    optimizer = Adam(params=denoising_model.parameters(), lr=1e-4, weight_decay=1e-6)
    lr_scheduler = get_scheduler('cosine', optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=len(loader) * num_epochs)
    loss_func = nn.functional.mse_loss

    print("----------------------------------")
    print(f"Staring to train")
    print("----------------------------------")

    tglobal = tqdm(range(num_epochs), desc='epoch', leave=False)

    for epoch_indx in tglobal:

        tepoch = tqdm(loader, desc='batch', leave=False)

        epoch_loss = list()
        episode_loss = list()

        for batch in tepoch:

            # Going through batches

            optimizer.zero_grad()
            normalized_observations = batch['observations']
            normalized_actions = batch['actions'] 

            normalized_observation_cond = normalized_observations[:, :observation_horizon, :]
            normalized_observation_cond = normalized_observations.flatten(start_dim = 1)
                

            t = torch.randint(0, num_steps, size=(B, ), device=device)
            noise = torch.rand_like(normalized_actions, device=device)
                
            noisy_action = noise_scheduler.add_noise(normalized_actions, noise, t)
            predicted_noise = denoising_model(noisy_action, normalized_observation_cond)

            loss = loss_func(predicted_noise, noise)
            loss.backward()
                
            loss_cpu = loss.item()
            
            epoch_loss.append(loss_cpu)

            optimizer.step()
            lr_scheduler.step()
            ema.step(denoising_model.parameters())
                
        tepoch.set_postfix(loss = np.mean(epoch_loss))
        
        if (epoch_indx + 1) % 10 == 0:
                print(f"Finished epoch {epoch_indx+1}/{num_epochs} | Loss: {np.mean(epoch_loss):.4f}")
                torch.save(denoising_model.state_dict(), f'./saves/pusht_chkpt_{epoch_indx + 1}.pth')
                torch.save(ema.state_dict(), f'./saves/ema_chkpt_{epoch_indx + 1}.pth')
        
        print("Finished training!")


if __name__ == "__main__":
    train()