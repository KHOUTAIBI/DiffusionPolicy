import matplotlib.pyplot as plt
import numpy as np


loss_pusht_flow = np.load("./loss_pusht_flow.npy")
loss_pusht_ddpm = np.load("./loss_pusht_ddpm.npy")
print(loss_pusht_flow, loss_pusht_ddpm)

plt.plot(np.arange(loss_pusht_flow.shape[0]), loss_pusht_flow, label='Flow-matching')
plt.plot(np.arange(loss_pusht_ddpm.shape[0]), loss_pusht_ddpm, label='DDPM')
plt.xlabel("epoch")
plt.ylabel("loss mean")
plt.title("Mean loss for PushT task trained on Flow Matching")
plt.grid()
plt.legend()
plt.savefig("losses_pusht.svg")
plt.show()