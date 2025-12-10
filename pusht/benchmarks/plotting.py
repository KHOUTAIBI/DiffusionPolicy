import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import gym_pusht

loss_pusht_unet = np.load("./loss_pusht_ddpm.npy")
loss_pusht_transformer = np.load("./loss_pusht_transformer_ddpm.npy")
print(loss_pusht_unet, loss_pusht_transformer)

plt.plot(loss_pusht_unet, label = 'PushT Unet')
plt.plot(loss_pusht_transformer, label = 'PushT Transformer')
plt.xlabel("epoch")
plt.ylabel("loss mean")
plt.title("PushT")
plt.grid()
plt.legend()
plt.savefig("loss_pusht_unet_transformer.pdf")
plt.show()

# env = gym.make("gym_pusht/PushT-v0", render_mode="rgb_array")
# observation, _ = env.reset()

# img = env.render()
# img = np.array(img)

# from PIL import Image
# im = Image.fromarray(img)
# im.save("PushT.pdf", dpi = (2000, 2000))

