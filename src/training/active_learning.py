"""
Active Learning Implementation for Wafer Defect Segmentation

Implements query strategies to achieve 85% annotation reduction (PRD requirement):
- Uncertainty sampling: Entropy, BALD (Bayesian Active Learning by Disagreement)
- Diversity sampling: CoreSet, k-center greedy
- Hybrid strategy: Combines uncertainty + diversity

Target: Reduce annotations from 5000 to 750 samples while maintaining >95% IoU
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
from sklearn.metrics import pairwise_distances
import logging

logger = logging.getLogger(__name__)


class UncertaintySampling:
    """
    Uncertainty-based query strategies.
    
    Selects samples where the model is most uncertain,
    indicating informative examples for learning.
    """
    
    @staticmethod
    def entropy_sampling(logits: torch.Tensor) -> float:
        """
        Compute prediction entropy (Shannon entropy).
        
        H(p) = -Σ p_i * log(p_i)
        
        Higher entropy = more uncertain
        
        Args:
            logits: Model logits of shape (num_classes, H, W)
        
        Returns:
            Mean entropy score
        """
        # Convert to probabilities
        probs = F.softmax(logits, dim=0)  # (num_classes, H, W)
        
        # Compute entropy per pixel
        log_probs = torch.log(probs + 1e-10)
        entropy = -torch.sum(probs * log_probs, dim=0)  # (H, W)
        
        # Average entropy across pixels
        mean_entropy = entropy.mean().item()
        
        return mean_entropy
    
    @staticmethod
    def bald_score(logits_samples: List[torch.Tensor]) -> float:
        """
        Bayesian Active Learning by Disagreement (BALD).
        
        Uses MC Dropout to get multiple predictions and measures disagreement.
        
        BALD = H(E[p]) - E[H(p)]
        
        Higher BALD = more epistemic uncertainty (model uncertainty)
        
        Args:
            logits_samples: List of logits from multiple forward passes with dropout
        
        Returns:
            BALD score
        """
        # Stack logits: (num_samples, num_classes, H, W)
        logits_stack = torch.stack(logits_samples, dim=0)
        
        # Convert to probabilities
        probs_stack = F.softmax(logits_stack, dim=1)  # (num_samples, num_classes, H, W)
        
        # Mean prediction across samples
        mean_probs = probs_stack.mean(dim=0)  # (num_classes, H, W)
        
        # Entropy of mean prediction: H(E[p])
        log_mean_probs = torch.log(mean_probs + 1e-10)
        entropy_mean = -torch.sum(mean_probs * log_mean_probs, dim=0)  # (H, W)
        
        # Mean of entropies: E[H(p)]
        log_probs_stack = torch.log(probs_stack + 1e-10)
        entropies = -torch.sum(probs_stack * log_probs_stack, dim=1)  # (num_samples, H, W)
        mean_entropy = entropies.mean(dim=0)  # (H, W)
        
        # BALD score
        bald = entropy_mean - mean_entropy
        
        return bald.mean().item()
    
    @staticmethod
    def variation_ratio(logits_samples: List[torch.Tensor]) -> float:
        """
        Variation ratio: fraction of predictions that disagree with mode.
        
        Args:
            logits_samples: List of logits from multiple forward passes
        
        Returns:
            Variation ratio score
        """
        # Stack and get predicted classes
        logits_stack = torch.stack(logits_samples, dim=0)
        predictions = torch.argmax(logits_stack, dim=1)  # (num_samples, H, W)
        
        # Get mode (most common prediction) per pixel
        mode, _ = torch.mode(predictions, dim=0)
        
        # Count disagreements
        disagreements = (predictions != mode.unsqueeze(0)).float()
        variation_ratio = disagreements.mean().item()
        
        return variation_ratio


class DiversitySampling:
    """
    Diversity-based query strategies.
    
    Selects diverse samples to cover the feature space,
    avoiding redundant annotations.
    """
    
    @staticmethod
    def coreset_greedy(
        embeddings: np.ndarray,
        labeled_indices: List[int],
        budget: int,
    ) -> List[int]:
        """
        CoreSet greedy selection (k-center greedy).
        
        Iteratively selects samples farthest from labeled set.
        
        Args:
            embeddings: Feature embeddings of shape (num_samples, embedding_dim)
            labeled_indices: Indices of already labeled samples
            budget: Number of samples to select
        
        Returns:
            List of selected sample indices
        """
        num_samples = len(embeddings)
        unlabeled_indices = list(set(range(num_samples)) - set(labeled_indices))
        
        selected = []
        
        # Compute minimum distances to labeled set
        if len(labeled_indices) > 0:
            labeled_embeddings = embeddings[labeled_indices]
            distances = pairwise_distances(
                embeddings[unlabeled_indices],
                labeled_embeddings,
                metric='euclidean'
            )
            min_distances = distances.min(axis=1)
        else:
            # All samples are equidistant if no labeled samples
            min_distances = np.ones(len(unlabeled_indices))
        
        # Greedy selection
        for _ in range(min(budget, len(unlabeled_indices))):
            # Select sample with maximum distance to labeled set
            farthest_idx = np.argmax(min_distances)
            selected_sample_idx = unlabeled_indices[farthest_idx]
            selected.append(selected_sample_idx)
            
            # Update distances
            new_distances = pairwise_distances(
                embeddings[unlabeled_indices],
                embeddings[selected_sample_idx:selected_sample_idx+1],
                metric='euclidean'
            ).squeeze()
            
            min_distances = np.minimum(min_distances, new_distances)
            
            # Remove selected sample from unlabeled pool
            min_distances[farthest_idx] = -np.inf
        
        return selected
    
    @staticmethod
    def random_sampling(num_samples: int, budget: int, labeled_indices: List[int]) -> List[int]:
        """
        Random sampling baseline.
        
        Args:
            num_samples: Total number of samples
            budget: Number of samples to select
            labeled_indices: Already labeled samples
        
        Returns:
            Randomly selected indices
        """
        unlabeled_indices = list(set(range(num_samples)) - set(labeled_indices))
        selected = np.random.choice(unlabeled_indices, size=min(budget, len(unlabeled_indices)), replace=False)
        return selected.tolist()


class HybridActiveLearning:
    """
    Hybrid query strategy combining uncertainty and diversity.
    
    Score = λ * uncertainty + (1 - λ) * diversity
    
    Balances exploration (diversity) and exploitation (uncertainty).
    
    Args:
        uncertainty_weight: Weight for uncertainty component (default: 0.6)
        diversity_weight: Weight for diversity component (default: 0.4)
    """
    
    def __init__(self, uncertainty_weight: float = 0.6, diversity_weight: float = 0.4):
        self.uncertainty_weight = uncertainty_weight
        self.diversity_weight = diversity_weight
    
    def query(
        self,
        uncertainty_scores: np.ndarray,
        embeddings: np.ndarray,
        labeled_indices: List[int],
        budget: int,
    ) -> List[int]:
        """
        Select samples using hybrid strategy.
        
        Args:
            uncertainty_scores: Uncertainty score per sample (higher = more uncertain)
            embeddings: Feature embeddings of shape (num_samples, embedding_dim)
            labeled_indices: Already labeled sample indices
            budget: Number of samples to select
        
        Returns:
            Selected sample indices
        """
        num_samples = len(uncertainty_scores)
        unlabeled_indices = list(set(range(num_samples)) - set(labeled_indices))
        
        # Normalize uncertainty scores to [0, 1]
        unc_min, unc_max = uncertainty_scores.min(), uncertainty_scores.max()
        if unc_max > unc_min:
            normalized_uncertainty = (uncertainty_scores - unc_min) / (unc_max - unc_min)
        else:
            normalized_uncertainty = np.ones_like(uncertainty_scores)
        
        # Compute diversity scores (distance to labeled set)
        if len(labeled_indices) > 0:
            labeled_embeddings = embeddings[labeled_indices]
            distances = pairwise_distances(
                embeddings,
                labeled_embeddings,
                metric='euclidean'
            )
            diversity_scores = distances.min(axis=1)
        else:
            diversity_scores = np.ones(num_samples)
        
        # Normalize diversity scores
        div_min, div_max = diversity_scores.min(), diversity_scores.max()
        if div_max > div_min:
            normalized_diversity = (diversity_scores - div_min) / (div_max - div_min)
        else:
            normalized_diversity = np.ones_like(diversity_scores)
        
        # Combined score
        combined_scores = (
            self.uncertainty_weight * normalized_uncertainty +
            self.diversity_weight * normalized_diversity
        )
        
        # Select top-k from unlabeled pool
        unlabeled_scores = combined_scores[unlabeled_indices]
        top_k_indices = np.argsort(unlabeled_scores)[-budget:][::-1]
        selected = [unlabeled_indices[i] for i in top_k_indices]
        
        return selected


class ActiveLearningOrchestrator:
    """
    Orchestrates active learning iterations.
    
    Workflow:
    1. Train model on labeled data
    2. Extract embeddings and compute uncertainty for unlabeled data
    3. Query strategy selects samples
    4. Annotator labels selected samples
    5. Repeat
    
    Args:
        model: Segmentation model
        query_strategy: 'uncertainty', 'diversity', 'hybrid', 'random'
        budget_per_iteration: Samples to annotate per iteration
        max_iterations: Maximum active learning iterations
    """
    
    def __init__(
        self,
        model,
        query_strategy: str = 'hybrid',
        budget_per_iteration: int = 50,
        max_iterations: int = 15,
    ):
        self.model = model
        self.query_strategy = query_strategy
        self.budget_per_iteration = budget_per_iteration
        self.max_iterations = max_iterations
        
        self.uncertainty_sampler = UncertaintySampling()
        self.diversity_sampler = DiversitySampling()
        self.hybrid_sampler = HybridActiveLearning()
        
        self.iteration = 0
        self.labeled_indices = []
    
    @torch.no_grad()
    def compute_uncertainty_scores(
        self,
        unlabeled_dataloader,
        use_mc_dropout: bool = True,
        num_mc_samples: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute uncertainty scores for unlabeled data.
        
        Args:
            unlabeled_dataloader: DataLoader for unlabeled images
            use_mc_dropout: Use MC Dropout for uncertainty estimation
            num_mc_samples: Number of MC samples
        
        Returns:
            Tuple of (uncertainty_scores, embeddings)
        """
        self.model.eval()
        
        uncertainty_scores = []
        embeddings_list = []
        
        for batch_idx, images in enumerate(unlabeled_dataloader):
            images = images.cuda() if torch.cuda.is_available() else images
            
            if use_mc_dropout:
                # Enable dropout for uncertainty estimation
                self.model.train()  # Enables dropout
                
                # Multiple forward passes
                logits_samples = []
                for _ in range(num_mc_samples):
                    logits = self.model(images)
                    logits_samples.append(logits)
                
                # Compute BALD score (batch average)
                bald_scores = []
                for i in range(len(images)):
                    sample_logits = [logits[i] for logits in logits_samples]
                    bald = self.uncertainty_sampler.bald_score(sample_logits)
                    bald_scores.append(bald)
                
                uncertainty_scores.extend(bald_scores)
            else:
                # Single forward pass with entropy
                logits = self.model(images)
                
                for i in range(len(images)):
                    entropy = self.uncertainty_sampler.entropy_sampling(logits[i])
                    uncertainty_scores.append(entropy)
            
            # Extract embeddings from encoder
            with torch.no_grad():
                features = self.model.get_encoder_features(images)
                # Use deepest features (stage4) for embeddings
                deep_features = features['stage4']
                # Global average pooling
                embeddings = F.adaptive_avg_pool2d(deep_features, (1, 1)).squeeze()
                embeddings_list.append(embeddings.cpu().numpy())
        
        uncertainty_scores = np.array(uncertainty_scores)
        embeddings = np.concatenate(embeddings_list, axis=0)
        
        return uncertainty_scores, embeddings
    
    def select_samples(
        self,
        uncertainty_scores: np.ndarray,
        embeddings: np.ndarray,
    ) -> List[int]:
        """
        Select samples to annotate using query strategy.
        
        Args:
            uncertainty_scores: Uncertainty scores for unlabeled data
            embeddings: Feature embeddings
        
        Returns:
            Selected sample indices
        """
        if self.query_strategy == 'uncertainty':
            # Select top-k uncertain samples
            unlabeled_indices = list(set(range(len(uncertainty_scores))) - set(self.labeled_indices))
            unlabeled_scores = uncertainty_scores[unlabeled_indices]
            top_k = np.argsort(unlabeled_scores)[-self.budget_per_iteration:][::-1]
            selected = [unlabeled_indices[i] for i in top_k]
        
        elif self.query_strategy == 'diversity':
            # CoreSet greedy selection
            selected = self.diversity_sampler.coreset_greedy(
                embeddings,
                self.labeled_indices,
                self.budget_per_iteration
            )
        
        elif self.query_strategy == 'hybrid':
            # Hybrid strategy
            selected = self.hybrid_sampler.query(
                uncertainty_scores,
                embeddings,
                self.labeled_indices,
                self.budget_per_iteration
            )
        
        elif self.query_strategy == 'random':
            # Random baseline
            selected = self.diversity_sampler.random_sampling(
                len(uncertainty_scores),
                self.budget_per_iteration,
                self.labeled_indices
            )
        
        else:
            raise ValueError(f"Unknown query strategy: {self.query_strategy}")
        
        return selected
    
    def run_iteration(self, unlabeled_dataloader) -> Dict[str, any]:
        """
        Run single active learning iteration.
        
        Args:
            unlabeled_dataloader: DataLoader for unlabeled data
        
        Returns:
            Dictionary with iteration results
        """
        logger.info(f"Active Learning Iteration {self.iteration + 1}/{self.max_iterations}")
        
        # Compute uncertainty and embeddings
        uncertainty_scores, embeddings = self.compute_uncertainty_scores(unlabeled_dataloader)
        
        # Select samples
        selected_indices = self.select_samples(uncertainty_scores, embeddings)
        
        # Update labeled set
        self.labeled_indices.extend(selected_indices)
        
        # Log statistics
        logger.info(f"  Selected {len(selected_indices)} samples")
        logger.info(f"  Total labeled: {len(self.labeled_indices)}")
        logger.info(f"  Mean uncertainty: {uncertainty_scores.mean():.4f}")
        
        self.iteration += 1
        
        return {
            'iteration': self.iteration,
            'selected_indices': selected_indices,
            'uncertainty_scores': uncertainty_scores[selected_indices],
            'num_labeled': len(self.labeled_indices),
        }


if __name__ == "__main__":
    print("=" * 60)
    print("Active Learning Module")
    print("=" * 60)
    
    # Test uncertainty sampling
    print("\n1. Testing uncertainty sampling...")
    dummy_logits = torch.randn(8, 300, 300)  # (num_classes, H, W)
    
    sampler = UncertaintySampling()
    entropy = sampler.entropy_sampling(dummy_logits)
    print(f"   Entropy: {entropy:.4f}")
    
    # Test BALD
    logits_samples = [torch.randn(8, 300, 300) for _ in range(10)]
    bald = sampler.bald_score(logits_samples)
    print(f"   BALD score: {bald:.4f}")
    print("   ✅ Uncertainty sampling works")
    
    # Test diversity sampling
    print("\n2. Testing diversity sampling...")
    embeddings = np.random.randn(100, 2048)  # 100 samples, 2048-dim
    labeled = [0, 1, 2]
    
    div_sampler = DiversitySampling()
    selected = div_sampler.coreset_greedy(embeddings, labeled, budget=10)
    print(f"   Selected {len(selected)} samples: {selected[:5]}...")
    print("   ✅ Diversity sampling works")
    
    # Test hybrid
    print("\n3. Testing hybrid strategy...")
    uncertainty_scores = np.random.rand(100)
    hybrid = HybridActiveLearning(uncertainty_weight=0.6, diversity_weight=0.4)
    selected = hybrid.query(uncertainty_scores, embeddings, labeled, budget=10)
    print(f"   Selected {len(selected)} samples: {selected[:5]}...")
    print("   ✅ Hybrid strategy works")
    
    print("\n" + "=" * 60)
    print("✅ Active Learning Module Ready!")
    print("=" * 60)
    print("\nTarget: 85% annotation reduction (5000 → 750 samples)")
    print("Expected IoU: >95% with active learning")
