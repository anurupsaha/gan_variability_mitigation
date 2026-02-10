from pytorch_image_generation_metrics import get_inception_score, get_fid
from pytorch_image_generation_metrics.inception import InceptionV3

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import os
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
import numpy as np
import gc
import copy
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

class Generator(nn.Module):
    def __init__(self, z_dim, M):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.z_dim, 1024, M, 1, 0, bias=False),  # 4, 4
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 3, 4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.main(z.view(-1, self.z_dim, 1, 1))

class Generator32(Generator):
    def __init__(self, z_dim):
        super().__init__(z_dim, M=2)
        
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def generate_imgs(net_eval, noise, device):
    net_eval.eval()
    batch_size = 128
    generated_imgs = []
    for i in range(0,noise.shape[0],batch_size):
        cur_batch_size = min(batch_size, noise.shape[0] - i)
        #z = torch.randn(cur_batch_size, z_dim).to(device)
        z = noise[i:i+cur_batch_size].to(device)
        with torch.no_grad():
            fake = net_eval(z).detach()
            fake = (fake + 1.0)/2.0
            generated_imgs.append(fake)
    generated_imgs = torch.cat(generated_imgs, dim=0)
    assert generated_imgs.shape == (10000,3,32,32), "mismatch from generated_imgs module"
    return generated_imgs

def custom_inception_score(imgs, inception, device):
    epsilon = 1e-15
    numofimages = imgs.shape[0]
    output_list = []
    batch_inception = 64
    with torch.no_grad():
        for st in range(0, numofimages, batch_inception):
            end = min(st+batch_inception, numofimages)
            cur_imgs = imgs[st:end]
            #cur_imgs = cur_imgs.half()
            output = inception(cur_imgs)[0]
            output = output.detach().cpu().numpy()
            output_list.append(output)
    output_list = np.concatenate(output_list, axis=0)

    cur_prob = output_list

    e1 = cur_prob * np.log(cur_prob + epsilon)
    e1 = -np.sum(e1, axis = 1)
    e1 = np.mean(e1)

    mean_prob = np.mean(cur_prob, axis = 0)
    e2 = -np.sum ( mean_prob * np.log(mean_prob + epsilon))
    inception_score = np.exp(e2 - e1)

    return cur_prob, inception_score

def process_duts(model,
                 perturb_keys,
                 sigma_tot,
                 prop_sys,
                 prop_rand,
                 numofdut,
                 noise,
                 start_seed,
                 device,
                 inception):
    n_class_inception = 1008
    inception_score_list = np.zeros(numofdut).astype(float)
    inception_probs = np.zeros((numofdut,noise.shape[0],n_class_inception))

    frac_sys = prop_sys / (prop_sys + prop_rand)
    frac_rand = prop_rand / (prop_sys + prop_rand)
    frac_sys = np.sqrt(frac_sys)
    frac_rand = np.sqrt(frac_rand)
    sigma_sys = sigma_tot * frac_sys
    sigma_rand = sigma_tot * frac_rand
    
    # --- FIX: Save the original model state ONCE before the loop ---
    original_state_dict = copy.deepcopy(model.state_dict())
    
    # We will now modify the 'model' object directly
    cur_dut = model 
    cur_dut.eval()


    for ii in range(numofdut):
        print(f"seed = {start_seed + ii}")
        set_seed(start_seed + ii)
        #cur_dut = copy.deepcopy(model).to(device)
        #cur_dut.eval()

        sys_coef = torch.randn(size = [1]).to(device) * sigma_sys

        with torch.no_grad():
            for name, param in cur_dut.named_parameters():
                if name in perturb_keys:
                    #rand_coef = sigma_rand * torch.randn(param.data.shape).to(device)
                    #param.data = param.data * (1 + sys_coef + rand_coef)
                    rand_coef = sigma_rand * torch.randn(param.data.shape).to(device)
                    # Apply perturbation to the original parameter
                    original_param = original_state_dict[name]
                    param.data = original_param * (1 + sys_coef + rand_coef)

        imgs = generate_imgs(cur_dut, noise, device)
        out_probs, cur_inception_score = custom_inception_score(imgs, inception, device)
        inception_score_list[ii] = cur_inception_score
        inception_probs[ii] = out_probs
        
        
        
        del imgs
        #del cur_dut
        torch.cuda.empty_cache()
        gc.collect()
    
    # --- FIX: Restore the model to its original state after the batch is done ---
    model.load_state_dict(original_state_dict)
    
    return inception_probs, inception_score_list

class Calculate_Log_IS(nn.Module):
    
    """
    Implements the custom mathematical operation as a Pytorch nn.Module.

    Args:
        T (int): The size of the time/sequence dimension of the input x.
                 This is also the length of the learnable parameter vector w.
        S (int): The constant integer S from the mathematical operation.
    """
    def __init__(self, T: int, S: int):
        super().__init__()

        if not isinstance(T, int) or T <= 0:
            raise ValueError("T must be a positive integer.")
        if not isinstance(S, int) or S <= 0:
            raise ValueError("S must be a positive integer.")

        self.T = T
        self.S = S

        # (Parameter w) Initialize w as a learnable parameter of size T
        # nn.Parameter ensures that this tensor is registered as a model parameter,
        # so it will be included in optimizer.step() updates.
        self.w = nn.Parameter(torch.randn(T))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the custom operation to a batch of data.

        Args:
            x (torch.Tensor): The input tensor of shape (B, T, C), where B is the
                              batch size, T is the time dimension, and C is the
                              channel dimension.

        Returns:
            torch.Tensor: The output tensor y of shape (B,), where each element
                          is the scalar result for the corresponding item in the batch.
        """
        # Ensure input tensor has the correct T dimension
        assert x.shape[1] == self.T, f"Input tensor dim 1 has size {x.shape[1]}, but expected {self.T}"

        # Add a small epsilon for numerical stability in log operations
        epsilon = 1e-15

        # --- Step (1): Calculate m_t ---
        # m_t = S * exp(w_t) / sum(exp(w_t'))
        # This is equivalent to S * softmax(w).
        # self.w has shape (T,). The result m will also have shape (T,).
        m = self.S * F.softmax(self.w, dim=0)

        # --- Step (2): Calculate x_hat_c ---
        # x_hat_c = (1/S) * sum_{t=1 to T} (m_t * x_{t,c})
        # We can perform this efficiently using Einstein summation (einsum).
        # 'btc,t->bc' means: multiply batch 'b', time 't', channel 'c' of x
        # with 't' of m, and sum over the 't' dimension, resulting in a
        # tensor of shape (B, C).
        x_hat = (1 / self.S) * torch.einsum('btc, t -> bc', x, m)
        # x_hat has shape (B, C)

        # --- Step (3): Calculate h_sample ---
        # h_sample = - sum_{c=1 to C} (x_hat_c * log(x_hat_c))
        # This is the entropy of x_hat. We sum over the C dimension (dim=1).
        h_sample = -torch.sum(x_hat * torch.log(x_hat + epsilon), dim=1)
        # h_sample has shape (B,)

        # --- Step (4): Calculate h_class ---
        # h_class = -(1/S) * sum_{t=1 to T} (m_t * sum_{c=1 to C} (x_{t,c} * log(x_{t,c})))
        # Inner part: sum over C -> entropy for each time step t
        inner_entropy_sum = torch.sum(x * torch.log(x + epsilon), dim=2) # Shape: (B, T)

        # Outer part: weighted sum of these entropies using m_t as weights
        weighted_entropy_sum = torch.sum(m * inner_entropy_sum, dim=1) # Shape: (B,)

        h_class = -(1 / self.S) * weighted_entropy_sum
        # h_class has shape (B,)

        # --- Step (5): Calculate the final output y ---
        # y = h_sample - h_class
        y = h_sample - h_class
        # y has shape (B,)

        return y

def read_data(froot, start_batch, end_batch):
    log_prob_mtx_list = []
    inception_score_list = []
    for ii in range(start_batch, end_batch):
        #print(ii)
        flog_prob = froot + 'log_prob_' + str(ii) + '.npy'
        finception = froot + 'inception_' + str(ii) + '.txt'
        cur_log_prob = np.load(flog_prob).astype(np.float32)
        cur_inception = np.loadtxt(finception)
        log_prob_mtx_list.append(cur_log_prob)
        inception_score_list.append(cur_inception)
    log_prob_mtx = np.concatenate(log_prob_mtx_list, axis = 0).astype(np.float32)
    inception_score = np.concatenate(inception_score_list, axis = 0)
    #print(log_prob_mtx.shape)
    #print(inception_score.shape)
    del log_prob_mtx_list
    del inception_score_list
    prob_mtx = np.exp(log_prob_mtx)
    del log_prob_mtx
    return prob_mtx, inception_score

class Calculate_True_Log_IS(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        epsilon = 1e-15
        x_hat = torch.mean(x, dim = 1)
        h_sample = -torch.sum(x_hat * torch.log(x_hat + epsilon), dim=1)
        inner_entropy_sum = torch.sum(x * torch.log(x + epsilon), dim=2)
        h_class = -torch.mean(inner_entropy_sum, dim = 1)
        y = h_sample - h_class

        return y

def modify_tensor_inplace(data, S, xhigh, xlow):
    """
    Identifies the top S values in a 1D tensor, setting them to xhigh and all
    other values to xlow. The modification is done in-place.

    Args:
        data (torch.Tensor): The input 1D tensor to be modified.
        S (int): The number of highest values to identify and set to xhigh.
        xhigh (float): The value to assign to the top S elements.
        xlow (float): The value to assign to the rest of the T-S elements.

    Returns:
        None: This function modifies the tensor in-place and does not return anything.

    Raises:
        ValueError: If S is greater than the total number of elements in the tensor,
                    or if the input tensor is not one-dimensional.
    """
    # Validate that the input tensor is one-dimensional
    if data.dim() != 1:
        raise ValueError("Input tensor must be one-dimensional.")

    # Get the total number of elements in the tensor
    T = data.numel()

    # Validate that S is not larger than the total number of elements
    if S >= T or S == 0:
        return

    with torch.no_grad():
        _, top_indices = torch.topk(data, S)
        data.fill_(float(xlow))
        data[top_indices] = float(xhigh)


def calculate_classification_metrics(true_is, pred_is, cutoff): 
    # --- Input Validation ---
    if true_is.shape != pred_is.shape:
        raise ValueError("Input arrays 'true_is' and 'pred_is' must have the same shape.")
    
    n_samples = len(true_is)
    true_label = (true_is >= cutoff).astype(int)
    pred_label = (pred_is >= cutoff).astype(int)
    tp_count = np.sum((true_label == 1) & (pred_label == 1))
    tn_count = np.sum((true_label == 0) & (pred_label == 0))
    test_escape_count = np.sum((true_label == 0) & (pred_label == 1))
    yield_loss_count = np.sum((true_label == 1) & (pred_label == 0))

    # --- Step 3: Calculate percentages ---
    tp_percent = (tp_count / n_samples) * 100
    tn_percent = (tn_count / n_samples) * 100
    test_escape_percent = (test_escape_count / n_samples) * 100
    yield_loss_percent = (yield_loss_count / n_samples) * 100

    # --- Step 4: Return results in a dictionary ---
    results = {
        'True Positive': tp_percent,
        'True Negative': tn_percent,
        'Test Escape': test_escape_percent,
        'Yield Loss': yield_loss_percent,
    }
    
    return tp_percent, tn_percent, test_escape_percent, yield_loss_percent

def create_and_save_heatmap(data_matrix, metric_name, variability, S_values, cutoff_arr, output_dir, desired_xticks, dataset, test_framework):
    """
    Generates a heatmap for a given metric and saves it to a file.

    Args:
        data_matrix (np.array): The 2D array of metric values.
        metric_name (str): The name of the metric (e.g., 'te' or 'yl').
        variability (str): The variability setting (e.g., '0_100').
        S_values (list): The list of S values for the y-axis.
        cutoff_arr (np.array): The array of cutoff values for the x-axis.
        output_dir (str): The directory where the image will be saved.
    """
    plt.figure(figsize=(3, 2))

    # Create the heatmap using Seaborn
    sns.heatmap(
        data_matrix,
        yticklabels=S_values,
        annot=False,  # Numbers are not displayed on the heatmap
        cmap='viridis',
        vmin = 0.0,
        vmax = 17.0,
        cbar_kws={'label': f'Value of {metric_name}'}
    )

    # Set custom x-axis ticks
    xtick_indices = [np.where(np.isclose(cutoff_arr, tick))[0][0] for tick in desired_xticks]
    plt.xticks(ticks=xtick_indices, labels=desired_xticks, rotation=0)
    plt.yticks(rotation=0)

    # Add dynamic titles and labels
    #plt.xlabel("Cutoff Value")
    #plt.ylabel("S Value")
    #plt.title(f"Heatmap of '{metric_name}' for Variability: {variability}")
    
    # Construct the filename and save the plot
    filename = f"{output_dir}/heatmap_{dataset}_{test_framework}_{metric_name}_{variability}.png"
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    #print(f"Saved plot to {filename}")
    
    # Close the plot to free up memory
    plt.close()