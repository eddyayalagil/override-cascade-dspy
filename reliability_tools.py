"""Reliability analysis tools for override cascade framework.

Includes reliability diagrams, Expected Calibration Error (ECE),
and threshold curves for model evaluation.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import json
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12


@dataclass
class ReliabilityMetrics:
    """Container for reliability metrics."""
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    brier_score: float
    log_loss: float
    reliability_data: Dict[str, np.ndarray]
    threshold_data: Dict[str, np.ndarray]


class ReliabilityAnalyzer:
    """Analyze and visualize model reliability."""

    def __init__(self, output_dir: Path = Path("runs/reliability")):
        """Initialize reliability analyzer."""
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def compute_ece(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10
    ) -> Tuple[float, float]:
        """
        Compute Expected Calibration Error and Maximum Calibration Error.

        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            n_bins: Number of bins for calibration

        Returns:
            (ECE, MCE) tuple
        """
        # Create bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0
        mce = 0

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find indices in bin
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                # Accuracy in bin
                accuracy_in_bin = y_true[in_bin].mean()
                # Average confidence in bin
                avg_confidence_in_bin = y_prob[in_bin].mean()
                # Calibration error
                calibration_error = np.abs(avg_confidence_in_bin - accuracy_in_bin)

                ece += prop_in_bin * calibration_error
                mce = max(mce, calibration_error)

        return ece, mce

    def create_reliability_diagram(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10,
        title: str = "Reliability Diagram",
        save_path: Optional[Path] = None
    ) -> Dict[str, np.ndarray]:
        """
        Create reliability diagram (calibration plot).

        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            n_bins: Number of bins
            title: Plot title
            save_path: Where to save the plot

        Returns:
            Dictionary with calibration data
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Compute calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy='uniform'
        )

        # Plot reliability diagram
        ax1.plot(mean_predicted_value, fraction_of_positives,
                marker='o', linewidth=2, label='Model', markersize=8)
        ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', alpha=0.5)

        # Add confidence intervals using bootstrap
        n_bootstrap = 100
        bootstrap_fractions = []

        for _ in range(n_bootstrap):
            idx = np.random.choice(len(y_true), len(y_true), replace=True)
            y_true_boot = y_true[idx]
            y_prob_boot = y_prob[idx]

            try:
                frac_boot, _ = calibration_curve(
                    y_true_boot, y_prob_boot, n_bins=n_bins, strategy='uniform'
                )
                bootstrap_fractions.append(frac_boot)
            except:
                continue

        if bootstrap_fractions:
            bootstrap_fractions = np.array(bootstrap_fractions)
            lower_ci = np.percentile(bootstrap_fractions, 2.5, axis=0)
            upper_ci = np.percentile(bootstrap_fractions, 97.5, axis=0)

            ax1.fill_between(mean_predicted_value, lower_ci, upper_ci,
                           alpha=0.2, color='blue', label='95% CI')

        ax1.set_xlabel('Mean Predicted Probability')
        ax1.set_ylabel('Fraction of Positives')
        ax1.set_title(f'{title}')
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])

        # Compute ECE and MCE
        ece, mce = self.compute_ece(y_true, y_prob, n_bins)

        # Add text box with metrics
        textstr = f'ECE: {ece:.3f}\nMCE: {mce:.3f}\nBrier: {brier_score_loss(y_true, y_prob):.3f}'
        ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Plot histogram of predictions
        ax2.hist(y_prob[y_true == 0], bins=20, alpha=0.5,
                label='Negative class', color='red', density=True)
        ax2.hist(y_prob[y_true == 1], bins=20, alpha=0.5,
                label='Positive class', color='green', density=True)
        ax2.set_xlabel('Predicted Probability')
        ax2.set_ylabel('Density')
        ax2.set_title('Distribution of Predictions')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved reliability diagram to {save_path}")

        plt.show()

        return {
            'mean_predicted': mean_predicted_value,
            'fraction_positives': fraction_of_positives,
            'ece': ece,
            'mce': mce
        }

    def create_threshold_curves(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        thresholds: Optional[np.ndarray] = None,
        save_path: Optional[Path] = None
    ) -> Dict[str, np.ndarray]:
        """
        Create threshold analysis curves.

        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            thresholds: Threshold values to test
            save_path: Where to save the plot

        Returns:
            Dictionary with threshold analysis data
        """
        if thresholds is None:
            thresholds = np.linspace(0, 1, 101)

        # Calculate metrics at each threshold
        precisions = []
        recalls = []
        f1_scores = []
        accuracies = []

        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)

            # True positives, false positives, etc.
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            tn = np.sum((y_pred == 0) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))

            # Metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / len(y_true)

            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
            accuracies.append(accuracy)

        # Convert to arrays
        precisions = np.array(precisions)
        recalls = np.array(recalls)
        f1_scores = np.array(f1_scores)
        accuracies = np.array(accuracies)

        # Find optimal thresholds
        optimal_f1_idx = np.argmax(f1_scores)
        optimal_f1_threshold = thresholds[optimal_f1_idx]

        # Create plot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Precision-Recall vs Threshold
        ax = axes[0, 0]
        ax.plot(thresholds, precisions, label='Precision', linewidth=2)
        ax.plot(thresholds, recalls, label='Recall', linewidth=2)
        ax.axvline(optimal_f1_threshold, color='red', linestyle='--',
                  alpha=0.5, label=f'Optimal F1 @ {optimal_f1_threshold:.2f}')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Score')
        ax.set_title('Precision and Recall vs Threshold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # F1 Score vs Threshold
        ax = axes[0, 1]
        ax.plot(thresholds, f1_scores, linewidth=2, color='green')
        ax.axvline(optimal_f1_threshold, color='red', linestyle='--', alpha=0.5)
        ax.scatter([optimal_f1_threshold], [f1_scores[optimal_f1_idx]],
                  color='red', s=100, zorder=5)
        ax.set_xlabel('Threshold')
        ax.set_ylabel('F1 Score')
        ax.set_title(f'F1 Score vs Threshold (Max: {f1_scores[optimal_f1_idx]:.3f})')
        ax.grid(True, alpha=0.3)

        # Accuracy vs Threshold
        ax = axes[1, 0]
        ax.plot(thresholds, accuracies, linewidth=2, color='blue')
        ax.axvline(optimal_f1_threshold, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy vs Threshold')
        ax.grid(True, alpha=0.3)

        # Confusion matrix at optimal threshold
        ax = axes[1, 1]
        y_pred_optimal = (y_prob >= optimal_f1_threshold).astype(int)
        tp = np.sum((y_pred_optimal == 1) & (y_true == 1))
        fp = np.sum((y_pred_optimal == 1) & (y_true == 0))
        tn = np.sum((y_pred_optimal == 0) & (y_true == 0))
        fn = np.sum((y_pred_optimal == 0) & (y_true == 1))

        conf_matrix = np.array([[tn, fp], [fn, tp]])
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Predicted 0', 'Predicted 1'],
                   yticklabels=['True 0', 'True 1'], ax=ax)
        ax.set_title(f'Confusion Matrix @ Threshold {optimal_f1_threshold:.2f}')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved threshold curves to {save_path}")

        plt.show()

        return {
            'thresholds': thresholds,
            'precisions': precisions,
            'recalls': recalls,
            'f1_scores': f1_scores,
            'accuracies': accuracies,
            'optimal_threshold': optimal_f1_threshold
        }

    def create_pressure_response_curve(
        self,
        pressure_values: np.ndarray,
        override_rates: np.ndarray,
        confidence_intervals: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        save_path: Optional[Path] = None
    ) -> None:
        """
        Create pressure-response curve showing override rate vs pressure.

        Args:
            pressure_values: Pressure magnitudes
            override_rates: Corresponding override rates
            confidence_intervals: Optional (lower, upper) CI bounds
            save_path: Where to save the plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Main curve
        ax.plot(pressure_values, override_rates, 'b-', linewidth=2,
               label='Override Rate', marker='o', markersize=6)

        # Confidence intervals
        if confidence_intervals:
            lower_ci, upper_ci = confidence_intervals
            ax.fill_between(pressure_values, lower_ci, upper_ci,
                          alpha=0.2, color='blue', label='95% CI')

        # Add reference lines
        ax.axhline(0.95, color='red', linestyle='--', alpha=0.5,
                  label='95% Target')
        ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)

        ax.set_xlabel('Pressure Magnitude')
        ax.set_ylabel('Override Rate')
        ax.set_title('Pressure-Response Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, max(pressure_values) * 1.05])
        ax.set_ylim([0, 1.05])

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved pressure-response curve to {save_path}")

        plt.show()

    def analyze_reliability(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        experiment_name: str = "experiment"
    ) -> ReliabilityMetrics:
        """
        Comprehensive reliability analysis.

        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            experiment_name: Name for saving outputs

        Returns:
            ReliabilityMetrics object with all metrics
        """
        # Create output directory for this experiment
        exp_dir = self.output_dir / experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Compute metrics
        ece, mce = self.compute_ece(y_true, y_prob)
        brier = brier_score_loss(y_true, y_prob)
        logloss = log_loss(y_true, y_prob)

        # Create visualizations
        reliability_data = self.create_reliability_diagram(
            y_true, y_prob,
            title=f"Reliability Diagram - {experiment_name}",
            save_path=exp_dir / "reliability_diagram.png"
        )

        threshold_data = self.create_threshold_curves(
            y_true, y_prob,
            save_path=exp_dir / "threshold_curves.png"
        )

        # Save metrics
        metrics = ReliabilityMetrics(
            ece=ece,
            mce=mce,
            brier_score=brier,
            log_loss=logloss,
            reliability_data=reliability_data,
            threshold_data=threshold_data
        )

        # Save to JSON
        metrics_dict = {
            'ece': float(ece),
            'mce': float(mce),
            'brier_score': float(brier),
            'log_loss': float(logloss),
            'optimal_threshold': float(threshold_data['optimal_threshold']),
            'max_f1_score': float(np.max(threshold_data['f1_scores']))
        }

        with open(exp_dir / "metrics.json", 'w') as f:
            json.dump(metrics_dict, f, indent=2)

        print(f"\nReliability Analysis Complete:")
        print(f"  ECE: {ece:.4f}")
        print(f"  MCE: {mce:.4f}")
        print(f"  Brier Score: {brier:.4f}")
        print(f"  Log Loss: {logloss:.4f}")
        print(f"  Optimal Threshold: {threshold_data['optimal_threshold']:.3f}")
        print(f"  Results saved to: {exp_dir}")

        return metrics


def generate_synthetic_data(n_samples: int = 1000, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data for testing."""
    np.random.seed(seed)

    # Generate features (pressure values)
    X = np.random.beta(2, 5, n_samples)

    # Generate true labels with some noise
    # Higher pressure -> higher probability of override
    true_probs = 1 / (1 + np.exp(-10 * (X - 0.5)))  # Sigmoid
    y_true = np.random.binomial(1, true_probs)

    # Generate predicted probabilities with calibration error
    # Add systematic bias and random noise
    y_prob = true_probs * 0.9 + np.random.normal(0, 0.1, n_samples)
    y_prob = np.clip(y_prob, 0, 1)

    return y_true, y_prob


if __name__ == "__main__":
    # Test reliability tools
    analyzer = ReliabilityAnalyzer()

    # Generate synthetic data
    y_true, y_prob = generate_synthetic_data(1000)

    # Run analysis
    metrics = analyzer.analyze_reliability(
        y_true, y_prob,
        experiment_name="synthetic_test"
    )

    # Test pressure-response curve
    pressure_values = np.linspace(0, 2, 20)
    override_rates = 1 / (1 + np.exp(-3 * (pressure_values - 1)))
    lower_ci = override_rates - 0.05
    upper_ci = override_rates + 0.05

    analyzer.create_pressure_response_curve(
        pressure_values,
        override_rates,
        (lower_ci, upper_ci),
        save_path=analyzer.output_dir / "pressure_response.png"
    )