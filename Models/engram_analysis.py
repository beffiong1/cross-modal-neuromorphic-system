"""
Engram Analysis Module
Comprehensive cross-modal memory formation analysis

Analyzes:
1. Engram quality (silhouette, Davies-Bouldin, Calinski-Harabasz)
2. Cross-modal alignment
3. Zero-shot transferability
4. Dimensionality and sparsity
"""

import numpy as np
import torch
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA


def analyze_cross_modal_engrams(
    model_visual,
    model_audio,
    loader_visual,
    loader_audio,
    model_name,
    device='cuda',
    samples_per_class=100,
    num_classes=10
):
    """
    Comprehensive engram analysis with balanced class sampling
    
    Args:
        model_visual: Trained model for visual modality
        model_audio: Trained model for auditory modality
        loader_visual: DataLoader for visual data (N-MNIST)
        loader_audio: DataLoader for auditory data (SHD)
        model_name: Name of model for reporting
        device: Device to run on
        samples_per_class: Number of samples per class for analysis
        num_classes: Number of classes (default: 10)
        
    Returns:
        results: Dictionary containing all analysis results
    """
    
    print(f"\n{'='*70}")
    print(f"Engram Analysis: {model_name}")
    print(f"{'='*70}")
    
    model_visual.eval()
    model_audio.eval()
    
    # ========================================================================
    # EXTRACT FEATURES WITH BALANCED SAMPLING
    # ========================================================================
    
    features_v_by_class = {c: [] for c in range(num_classes)}
    features_a_by_class = {c: [] for c in range(num_classes)}
    
    print(f"\nExtracting visual engrams ({samples_per_class} per class)...")
    with torch.no_grad():
        for data, target in tqdm(loader_visual):
            data, target = data.to(device), target.to(device)
            _, features = model_visual(data)
            
            # Handle temporal dimension (rate encoding)
            if len(features.shape) == 3:
                # Mean firing rate across time
                features = features.mean(dim=1)
            
            features = features.cpu().numpy()
            labels = target.cpu().numpy()
            
            # Store by class
            for feat, label in zip(features, labels):
                if len(features_v_by_class[label]) < samples_per_class:
                    features_v_by_class[label].append(feat)
            
            # Check if done
            if all(len(v) >= samples_per_class for v in features_v_by_class.values()):
                break
    
    print(f"Extracting auditory engrams ({samples_per_class} per class)...")
    with torch.no_grad():
        for data, target in tqdm(loader_audio):
            data, target = data.to(device), target.to(device)
            _, features = model_audio(data)
            
            # Handle temporal dimension (rate encoding)
            if len(features.shape) == 3:
                features = features.mean(dim=1)
            
            features = features.cpu().numpy()
            labels = (target.cpu().numpy() % num_classes)  # Map to unified space
            
            # Store by class
            for feat, label in zip(features, labels):
                if len(features_a_by_class[label]) < samples_per_class:
                    features_a_by_class[label].append(feat)
            
            # Check if done
            if all(len(v) >= samples_per_class for v in features_a_by_class.values()):
                break
    
    # Combine into arrays
    features_v = []
    labels_v = []
    features_a = []
    labels_a = []
    
    for c in range(num_classes):
        # Visual
        class_feats = np.array(features_v_by_class[c])
        if len(class_feats) > 0:
            features_v.append(class_feats)
            labels_v.extend([c] * len(class_feats))
        
        # Audio
        class_feats = np.array(features_a_by_class[c])
        if len(class_feats) > 0:
            features_a.append(class_feats)
            labels_a.extend([c] * len(class_feats))
    
    features_v = np.vstack(features_v)
    features_a = np.vstack(features_a)
    labels_v = np.array(labels_v)
    labels_a = np.array(labels_a)
    
    print(f"\n📊 Extracted (BALANCED):")
    print(f"   Visual:   {features_v.shape}")
    print(f"   Auditory: {features_a.shape}")
    
    # ========================================================================
    # 1. ENGRAM QUALITY METRICS
    # ========================================================================
    
    print(f"\n{'='*70}")
    print("1. ENGRAM QUALITY ANALYSIS")
    print(f"{'='*70}")
    
    # Per-modality clustering
    sil_v = silhouette_score(features_v, labels_v)
    db_v = davies_bouldin_score(features_v, labels_v)
    ch_v = calinski_harabasz_score(features_v, labels_v)
    
    sil_a = silhouette_score(features_a, labels_a)
    db_a = davies_bouldin_score(features_a, labels_a)
    ch_a = calinski_harabasz_score(features_a, labels_a)
    
    print(f"\nVisual Engrams:")
    print(f"   Silhouette:        {sil_v:.4f} (higher = better separation)")
    print(f"   Davies-Bouldin:    {db_v:.4f} (lower = tighter clusters)")
    print(f"   Calinski-Harabasz: {ch_v:.1f} (higher = better defined)")
    
    print(f"\nAuditory Engrams:")
    print(f"   Silhouette:        {sil_a:.4f}")
    print(f"   Davies-Bouldin:    {db_a:.4f}")
    print(f"   Calinski-Harabasz: {ch_a:.1f}")
    
    # ========================================================================
    # 2. CROSS-MODAL ALIGNMENT
    # ========================================================================
    
    print(f"\n{'='*70}")
    print("2. CROSS-MODAL ALIGNMENT ANALYSIS")
    print(f"{'='*70}")
    
    # Compute class centroids
    centroids_v = np.array([features_v[labels_v == c].mean(axis=0) 
                           for c in range(num_classes)])
    centroids_a = np.array([features_a[labels_a == c].mean(axis=0) 
                           for c in range(num_classes)])
    
    # Centroid alignment
    alignment_matrix = cosine_similarity(centroids_v, centroids_a)
    
    # Diagonal = same-class alignment
    same_class_sim = alignment_matrix.diagonal().mean()
    # Off-diagonal = different-class alignment
    off_diag_mask = ~np.eye(num_classes, dtype=bool)
    diff_class_sim = alignment_matrix[off_diag_mask].mean()
    
    print(f"\nCentroid Alignment ({num_classes} classes):")
    print(f"   Same-class similarity:     {same_class_sim:.4f}")
    print(f"   Different-class similarity: {diff_class_sim:.4f}")
    print(f"   Alignment ratio:           {same_class_sim/diff_class_sim:.2f}x")
    
    if same_class_sim > 0.3:
        print(f"   ✅ STRONG cross-modal alignment detected!")
    elif same_class_sim > 0.15:
        print(f"   ⚠️  MODERATE cross-modal alignment")
    else:
        print(f"   ❌ WEAK cross-modal alignment (validates parallel architecture)")
    
    # ========================================================================
    # 3. TRANSFERABILITY SCORE
    # ========================================================================
    
    print(f"\n{'='*70}")
    print("3. CROSS-MODAL TRANSFERABILITY")
    print(f"{'='*70}")
    
    # Train classifier on visual features, test on audio
    clf_v2a = LogisticRegression(max_iter=1000, random_state=42)
    clf_v2a.fit(features_v, labels_v)
    transfer_v2a = clf_v2a.score(features_a, labels_a)
    
    # Train on audio, test on visual
    clf_a2v = LogisticRegression(max_iter=1000, random_state=42)
    clf_a2v.fit(features_a, labels_a)
    transfer_a2v = clf_a2v.score(features_v, labels_v)
    
    print(f"\nZero-Shot Transfer:")
    print(f"   Visual → Audio:  {transfer_v2a*100:.2f}%")
    print(f"   Audio → Visual:  {transfer_a2v*100:.2f}%")
    print(f"   Average:         {(transfer_v2a + transfer_a2v)*50:.2f}%")
    
    baseline = 1.0 / num_classes
    if transfer_v2a > 0.5 or transfer_a2v > 0.5:
        print(f"   ✅ STRONG transferability")
    elif transfer_v2a > 0.3 or transfer_a2v > 0.3:
        print(f"   ⚠️  MODERATE transferability")
    else:
        print(f"   ❌ WEAK transferability (expected for parallel architectures)")
    
    # ========================================================================
    # 4. DIMENSIONALITY & SPARSITY
    # ========================================================================
    
    print(f"\n{'='*70}")
    print("4. ENGRAM DIMENSIONALITY")
    print(f"{'='*70}")
    
    # PCA to find effective dimensionality
    pca_v = PCA(n_components=0.95)  # 95% variance
    pca_v.fit(features_v)
    dim_v = pca_v.n_components_
    
    pca_a = PCA(n_components=0.95)
    pca_a.fit(features_a)
    dim_a = pca_a.n_components_
    
    feature_dim = features_v.shape[1]
    print(f"\nEffective Dimensions (95% variance):")
    print(f"   Visual:   {dim_v}/{feature_dim} ({dim_v/feature_dim*100:.1f}%)")
    print(f"   Auditory: {dim_a}/{feature_dim} ({dim_a/feature_dim*100:.1f}%)")
    
    # Sparsity
    sparsity_v = (np.abs(features_v) < 0.01).mean()
    sparsity_a = (np.abs(features_a) < 0.01).mean()
    
    print(f"\nSparsity (near-zero activations):")
    print(f"   Visual:   {sparsity_v*100:.1f}%")
    print(f"   Auditory: {sparsity_a*100:.1f}%")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print(f"\n{'='*70}")
    print("📋 ENGRAM ANALYSIS SUMMARY")
    print(f"{'='*70}")
    
    print(f"\n✅ Visual Engrams:    Quality={sil_v:.3f}, Transfer={transfer_a2v*100:.1f}%")
    print(f"✅ Auditory Engrams:  Quality={sil_a:.3f}, Transfer={transfer_v2a*100:.1f}%")
    print(f"✅ Cross-Modal Align: {same_class_sim:.3f} ({same_class_sim/diff_class_sim:.1f}x ratio)")
    print(f"✅ Average Transfer:  {(transfer_v2a + transfer_a2v)*50:.1f}%")
    
    # Return all results
    results = {
        'visual': {
            'silhouette': sil_v,
            'davies_bouldin': db_v,
            'calinski_harabasz': ch_v,
            'effective_dims': dim_v,
            'sparsity': sparsity_v
        },
        'auditory': {
            'silhouette': sil_a,
            'davies_bouldin': db_a,
            'calinski_harabasz': ch_a,
            'effective_dims': dim_a,
            'sparsity': sparsity_a
        },
        'cross_modal': {
            'same_class_similarity': same_class_sim,
            'diff_class_similarity': diff_class_sim,
            'alignment_ratio': same_class_sim / diff_class_sim,
            'transfer_v2a': transfer_v2a,
            'transfer_a2v': transfer_a2v,
            'avg_transfer': (transfer_v2a + transfer_a2v) / 2
        },
        'features': {
            'visual': features_v,
            'auditory': features_a,
            'labels_visual': labels_v,
            'labels_auditory': labels_a,
            'centroids_visual': centroids_v,
            'centroids_auditory': centroids_a,
            'alignment_matrix': alignment_matrix
        }
    }
    
    return results
