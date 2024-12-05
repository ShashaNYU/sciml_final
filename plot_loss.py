import os, sys
import numpy as np
import matplotlib.pyplot as plt


layers="2 100 100 100 100 2"
N0=50
N_b=50
N_f=20000
num_epoch=50000
device=0
ft_max_iter=2000
ft_tolerance_grad=1e-6
ft_tolerance_change=1e-7
ft_chunks=5


def read_loss(path):
    iters = []
    train_losses = []
    test_losses = []
    terr = None
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if "Iter" in line:
                iter_chunk, loss_chunk = line.split(',')[1:]
                iid = int(iter_chunk.split(': ')[-1])
                if iid > 9999:
                    break
                loss = float(loss_chunk.split(': ')[-1])
                iters.append(iid)
                if "Training" in loss_chunk:
                    train_losses.append(loss)
                else:
                    test_losses.append(loss)
            elif "Test Error" in line:
                terr = float(line.split(': ')[-1])
    return iters, train_losses, test_losses, terr


train_loss_summary = []
test_loss_summary = []
for seed in [999, 536, 186, 197, 987, 406, 967, 204, 221, 73]:
    exp_path = "exp/pinn_wt"
    exp_path += "_ly-" + layers.replace(' ', '-') + '_'
    exp_path += f"_N0-{N0}_Nb-{N_b}_Nf-{N_f}_NE-{num_epoch}"
    exp_path += f"_ft-mi-{ft_max_iter}-tg-{ft_tolerance_grad}-tc-{ft_tolerance_change}-ch-{ft_chunks}"
    exp_path += f"_seed-{seed}"
    iters, train_losses, test_losses, terr = read_loss(os.path.join(exp_path, 'log_train.txt'))
    train_loss_summary.append(np.array(train_losses))
    test_loss_summary.append(np.array(test_losses))
all_train_losses = np.stack(train_loss_summary, axis=0)
all_test_losses = np.stack(test_loss_summary, axis=0)


# Convert lists to NumPy arrays for easier processing
#all_train_losses = np.array(all_train_losses)

# Compute mean and standard deviation
mean_train_losses = np.mean(all_train_losses, axis=0)
std_train_losses = np.std(all_train_losses, axis=0)
mean_test_losses = np.mean(all_test_losses, axis=0)
std_test_losses = np.std(all_test_losses, axis=0)

# Plot the mean and stddev as shaded areas
num_epochs = all_train_losses.shape[1]
epochs = np.arange(num_epochs)
eval_epochs = np.arange(0, num_epochs)

plt.figure()

# Plot training loss
plt.semilogy(epochs, mean_train_losses, label='Mean Train Loss', color='blue')
plt.fill_between(epochs, mean_train_losses - std_train_losses, mean_train_losses + std_train_losses, color='blue',
                 alpha=0.2)
plt.semilogy(epochs, mean_test_losses, label='Mean Test Loss', color='red')
plt.fill_between(epochs, mean_test_losses - std_test_losses, mean_test_losses + std_test_losses, color='red',
                 alpha=0.2)



plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train and Test Loss Over Multiple Runs")
plt.legend()
# plt.show()
plt.savefig('loss.png')
