import torch
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def spectral_information_divergence(x, y, epsilon=1e-10):
    """
    Calculate the Spectral Information Divergence (SID) between two spectral vectors x and y.
    Args:
    - x: Tensor of shape (B, N), where B is the batch size and N is the number of spectral bands.
    - y: Tensor of shape (B, N), where B is the batch size and N is the number of spectral bands.
    - epsilon: Small value to avoid log(0)

    Returns:
    - sid: Tensor of shape (B,), the SID values for each pair of vectors in the batch.
    """

    # Ensure non-negative values
    x = torch.clamp(x, min=epsilon)
    y = torch.clamp(y, min=epsilon)

    # Normalize the vectors to get the probability distributions
    P_x = x / torch.sum(x, dim=1, keepdim=True)
    P_y = y / torch.sum(y, dim=1, keepdim=True)

    # Calculate the Kullback-Leibler divergence
    D_kl_x_y = torch.sum(P_x * torch.log(P_x / P_y), dim=1)
    D_kl_y_x = torch.sum(P_y * torch.log(P_y / P_x), dim=1)

    # Calculate SID
    sid = D_kl_x_y + D_kl_y_x

    # Handle potential nan values
    sid = torch.where(torch.isnan(sid), torch.tensor(0.0, device=sid.device), sid)

    return sid


def compute_distance_matrix(tensor):
    B, H, _, C = tensor.shape
    similarity_matrix = torch.zeros(H, H, device=tensor.device)

    tensor = tensor.squeeze(2)  # Shape: B * H * C

    for i in range(H):
        for j in range(i, H):
            s1 = tensor[:, i, :]
            s2 = tensor[:, j, :]
            ssim_value = spectral_information_divergence(s1, s2)
            similarity_matrix[i, j] = ssim_value
            similarity_matrix[j, i] = ssim_value  # SSIM is symmetric

    return similarity_matrix


