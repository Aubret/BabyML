import torch


def compute_gram_matrix(X):
    """Compute the Gram matrix (dot product) for the vectors in X."""
    return torch.mm(X.t(), X)


def center_kernel(K):
    """Center the kernel matrix using the centering formula."""
    N = K.size(0)
    one_matrix = torch.ones(N, N, device=K.device, dtype=K.dtype) / N
    return K - torch.mm(one_matrix, K) - torch.mm(K, one_matrix) + torch.mm(one_matrix, torch.mm(K, one_matrix))


def normalize_kernel(K):
    """Normalize the kernel matrix using its Frobenius norm."""
    norm = torch.norm(K, 'fro')
    return K / norm


def centered_kernel_alignment(X, Y):
    """
    Compute the Centered Kernel Alignment (CKA) between two sets of vectors X and Y.

    X, Y: torch tensors of shape (N, D), where N is the number of samples and D is the dimensionality.
    """
    # Step 1: Compute Gram matrices (dot products)
    K_X = compute_gram_matrix(X)
    K_Y = compute_gram_matrix(Y)

    # Step 2: Center the kernel matrices
    K_X_centered = center_kernel(K_X)
    K_Y_centered = center_kernel(K_Y)


    # Step 3: Normalize the kernel matrices
    K_X_normalized = normalize_kernel(K_X_centered)
    K_Y_normalized = normalize_kernel(K_Y_centered)

    # Step 4: Compute the CKA metric (trace of the product of the centered kernels)
    cka_score = torch.trace(torch.mm(K_X_normalized, K_Y_normalized))

    return cka_score


def compare_batches(X_batch, Y_batch):
    """
    Compare two batches of representations pair-wise using CKA.

    X_batch, Y_batch: torch tensors of shape (N, D), where N is the batch size and D is the dimensionality.
    Returns a tensor of pair-wise CKA scores.
    """
    N = X_batch.size(0)
    cka_scores = torch.zeros(N, device=X_batch.device)  # Initialize tensor to hold pair-wise CKA scores

    for i in range(N):
        # Select the i-th vector from each batch
        X_i = X_batch[i:i + 1, :]  # Shape (1, D)
        Y_i = Y_batch[i:i + 1, :]  # Shape (1, D)

        # Compute the pair-wise CKA score for the i-th pair of representations
        cka_scores[i] = centered_kernel_alignment(X_i, Y_i)

    return cka_scores

