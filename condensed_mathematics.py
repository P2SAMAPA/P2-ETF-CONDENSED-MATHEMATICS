import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform

def compute_composite_macro_factor(macro_df):
    """Compute composite macro factor from all macro variables."""
    if len(macro_df) < 2:
        return np.ones(len(macro_df)) * 0.5
    scaler = StandardScaler()
    macro_scaled = scaler.fit_transform(macro_df)
    pca = PCA(n_components=1)
    factor = pca.fit_transform(macro_scaled).flatten()
    factor = (factor - factor.min()) / (factor.max() - factor.min() + 1e-8)
    return factor

def discrete_continuous_interface(returns, macro_factor, discrete_levels=5):
    """
    Construct the discrete-continuous interface using condensed sets.
    This mixes tick-level (discrete) and diffusion (continuous) phenomena.
    """
    if len(returns) < 10:
        return np.zeros(discrete_levels)
    # Discrete component: quantise returns into tick levels
    quantiles = np.linspace(0, 100, discrete_levels + 1)[1:-1]
    bins = np.percentile(returns, quantiles)
    discrete = np.digitize(returns, bins)
    # Continuous component: smoothed returns (diffusion)
    from scipy.ndimage import uniform_filter1d
    continuous = uniform_filter1d(returns, size=5, mode='nearest')
    # Condensed set: pair (discrete, continuous) at each point
    condensed = []
    for d, c in zip(discrete, continuous):
        condensed.append([d, c])
    condensed = np.array(condensed)
    # Scale by macro factor
    macro_scaled = macro_factor[:len(condensed)] if len(macro_factor) >= len(condensed) else np.ones(len(condensed)) * 0.5
    condensed = condensed * (1 + macro_scaled.reshape(-1, 1) * 0.5)
    return condensed

def condensed_profunctor(condensed_set, dim=10):
    """
    Compute the condensed profunctor (a sheaf-like structure) from the condensed set.
    This captures the mixing of discrete and continuous phenomena.
    """
    if len(condensed_set) < 5:
        return np.zeros(dim)
    # Compute the Gram matrix (similarity) of the condensed set
    gram = condensed_set @ condensed_set.T
    # Compute the eigenvectors (spectral decomposition)
    try:
        eigvals, eigvecs = np.linalg.eigh(gram)
        # Sort by eigenvalue magnitude
        idx = np.argsort(np.abs(eigvals))[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        # Profunctor = top dim eigenvectors weighted by eigenvalues
        profunctor = np.zeros(dim)
        for i in range(min(dim, len(eigvals))):
            profunctor[i] = np.abs(eigvals[i]) * np.mean(np.abs(eigvecs[:, i]))
        return profunctor
    except:
        return np.zeros(dim)

def condensed_cohomology_rank(returns, macro_factor, discrete_levels=5, dim=10):
    """
    Compute the condensed cohomology rank: a measure of structure at the
    discrete-continuous interface.
    """
    if len(returns) < 10:
        return 0
    # Construct condensed set
    condensed = discrete_continuous_interface(returns, macro_factor, discrete_levels)
    if len(condensed) < 5:
        return 0
    # Compute condensed profunctor
    profunctor = condensed_profunctor(condensed, dim)
    # Cohomology rank = sum of absolute values of profunctor
    rank = np.sum(np.abs(profunctor))
    return rank

def condensed_score(returns, macro_df, discrete_levels=5, dim=10):
    """
    Compute per-ETF condensed cohomology score.
    Higher score = more structure at the discrete-continuous interface.
    """
    if len(returns) < 15 or macro_df is None or len(macro_df) < 15:
        return 0.0
    # Align lengths
    min_len = min(len(returns), len(macro_df))
    returns = returns[:min_len]
    macro_df = macro_df.iloc[:min_len]
    # Remove NaN
    mask = ~(np.isnan(returns) | np.isnan(macro_df).any(axis=1))
    returns = returns[mask]
    macro_df = macro_df[mask]
    if len(returns) < 15:
        return 0.0
    # Compute macro factor
    macro_factor = compute_composite_macro_factor(macro_df)
    # Compute condensed cohomology rank
    rank = condensed_cohomology_rank(returns, macro_factor, discrete_levels, dim)
    return float(rank)
