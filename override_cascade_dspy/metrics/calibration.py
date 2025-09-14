"""
Calibration metrics for override probability predictions
Includes Brier score, reliability diagrams, and confidence intervals
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import scipy.stats as stats
import warnings


@dataclass
class CalibrationMetrics:
    """Calibration metrics for probabilistic predictions"""

    brier_score: float
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    reliability_data: Dict[str, List[float]]
    confidence_interval: Tuple[float, float]
    n_samples: int


def calculate_brier_score(
    probabilities: np.ndarray,
    outcomes: np.ndarray
) -> float:
    """
    Calculate Brier score for probabilistic predictions

    Args:
        probabilities: Predicted probabilities [0, 1]
        outcomes: Binary outcomes (0 or 1)

    Returns:
        Brier score (lower is better)
    """
    return np.mean((probabilities - outcomes) ** 2)


def calculate_confidence_interval(
    values: np.ndarray,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate confidence interval for mean

    Args:
        values: Sample values
        confidence: Confidence level (default 0.95)

    Returns:
        (lower_bound, upper_bound)
    """
    if len(values) < 2:
        return (values[0], values[0]) if len(values) == 1 else (0.0, 0.0)

    mean = np.mean(values)
    sem = stats.sem(values)

    if sem == 0:
        return (mean, mean)

    interval = stats.t.interval(
        confidence,
        len(values) - 1,
        loc=mean,
        scale=sem
    )

    return interval


def calculate_calibration_error(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10
) -> Tuple[float, float, Dict[str, List[float]]]:
    """
    Calculate Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)

    Args:
        probabilities: Predicted probabilities
        outcomes: Binary outcomes
        n_bins: Number of bins for calibration

    Returns:
        (ECE, MCE, reliability_data)
    """
    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0
    mce = 0
    reliability_data = {
        'bin_centers': [],
        'bin_accuracy': [],
        'bin_confidence': [],
        'bin_counts': []
    }

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find predictions in this bin
        in_bin = (probabilities > bin_lower) & (probabilities <= bin_upper)

        if np.sum(in_bin) > 0:
            # Calculate accuracy and confidence in this bin
            bin_accuracy = np.mean(outcomes[in_bin])
            bin_confidence = np.mean(probabilities[in_bin])
            bin_count = np.sum(in_bin)

            # Store for reliability diagram
            reliability_data['bin_centers'].append((bin_lower + bin_upper) / 2)
            reliability_data['bin_accuracy'].append(bin_accuracy)
            reliability_data['bin_confidence'].append(bin_confidence)
            reliability_data['bin_counts'].append(bin_count)

            # Calculate calibration error
            calibration_error = np.abs(bin_accuracy - bin_confidence)
            ece += (bin_count / len(probabilities)) * calibration_error
            mce = max(mce, calibration_error)

    return ece, mce, reliability_data


def calculate_metrics_with_bootstrap(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95
) -> CalibrationMetrics:
    """
    Calculate calibration metrics with bootstrap confidence intervals

    Args:
        probabilities: Predicted probabilities
        outcomes: Binary outcomes
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level

    Returns:
        CalibrationMetrics with confidence intervals
    """
    n_samples = len(probabilities)

    # Calculate point estimates
    brier = calculate_brier_score(probabilities, outcomes)
    ece, mce, reliability_data = calculate_calibration_error(probabilities, outcomes)

    # Bootstrap for confidence intervals
    brier_scores = []
    ece_scores = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        boot_probs = probabilities[indices]
        boot_outcomes = outcomes[indices]

        # Calculate metrics for bootstrap sample
        boot_brier = calculate_brier_score(boot_probs, boot_outcomes)
        boot_ece, _, _ = calculate_calibration_error(boot_probs, boot_outcomes)

        brier_scores.append(boot_brier)
        ece_scores.append(boot_ece)

    # Calculate confidence interval for Brier score
    brier_ci = np.percentile(
        brier_scores,
        [(1 - confidence) / 2 * 100, (1 + confidence) / 2 * 100]
    )

    return CalibrationMetrics(
        brier_score=brier,
        ece=ece,
        mce=mce,
        reliability_data=reliability_data,
        confidence_interval=tuple(brier_ci),
        n_samples=n_samples
    )


class ExperimentMetrics:
    """Calculate and report experiment metrics with proper statistics"""

    def __init__(self):
        self.results = []

    def add_result(
        self,
        condition_id: str,
        predicted_prob: float,
        actual_outcome: bool,
        latency_ms: float,
        tokens_in: int,
        tokens_out: int,
        provider: str,
        model: str,
        seed: int
    ):
        """Add a single experimental result"""
        self.results.append({
            'condition_id': condition_id,
            'predicted_prob': predicted_prob,
            'actual_outcome': int(actual_outcome),
            'latency_ms': latency_ms,
            'tokens_in': tokens_in,
            'tokens_out': tokens_out,
            'provider': provider,
            'model': model,
            'seed': seed
        })

    def calculate_summary_statistics(
        self,
        group_by: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate summary statistics with confidence intervals

        Args:
            group_by: Optional grouping variable (e.g., 'provider', 'model')

        Returns:
            Summary statistics dictionary
        """
        if not self.results:
            return {}

        results_array = np.array(self.results)
        probs = np.array([r['predicted_prob'] for r in self.results])
        outcomes = np.array([r['actual_outcome'] for r in self.results])

        if group_by:
            # Group results
            groups = {}
            for result in self.results:
                key = result[group_by]
                if key not in groups:
                    groups[key] = {'probs': [], 'outcomes': []}
                groups[key]['probs'].append(result['predicted_prob'])
                groups[key]['outcomes'].append(result['actual_outcome'])

            # Calculate per-group statistics
            group_stats = {}
            for key, data in groups.items():
                group_probs = np.array(data['probs'])
                group_outcomes = np.array(data['outcomes'])

                metrics = calculate_metrics_with_bootstrap(
                    group_probs,
                    group_outcomes
                )

                group_stats[key] = {
                    'n': len(group_probs),
                    'mean_prob': np.mean(group_probs),
                    'std_prob': np.std(group_probs),
                    'mean_outcome': np.mean(group_outcomes),
                    'brier_score': metrics.brier_score,
                    'brier_ci': metrics.confidence_interval,
                    'ece': metrics.ece,
                    'mce': metrics.mce,
                    'ci_95': calculate_confidence_interval(group_probs)
                }

            return group_stats

        else:
            # Overall statistics
            metrics = calculate_metrics_with_bootstrap(probs, outcomes)

            return {
                'n': len(probs),
                'mean_prob': np.mean(probs),
                'std_prob': np.std(probs),
                'mean_outcome': np.mean(outcomes),
                'brier_score': metrics.brier_score,
                'brier_ci': metrics.confidence_interval,
                'ece': metrics.ece,
                'mce': metrics.mce,
                'ci_95': calculate_confidence_interval(probs),
                'reliability_data': metrics.reliability_data
            }

    def calculate_effect_sizes(
        self,
        treatment_condition: str,
        control_condition: str
    ) -> Dict[str, float]:
        """
        Calculate Cohen's d effect size between conditions

        Args:
            treatment_condition: Treatment condition ID
            control_condition: Control condition ID

        Returns:
            Effect size metrics
        """
        treatment = [r['predicted_prob'] for r in self.results
                    if r['condition_id'] == treatment_condition]
        control = [r['predicted_prob'] for r in self.results
                  if r['condition_id'] == control_condition]

        if not treatment or not control:
            return {}

        treatment = np.array(treatment)
        control = np.array(control)

        # Cohen's d
        pooled_std = np.sqrt(
            ((len(treatment) - 1) * np.var(treatment, ddof=1) +
             (len(control) - 1) * np.var(control, ddof=1)) /
            (len(treatment) + len(control) - 2)
        )

        if pooled_std == 0:
            cohens_d = 0.0
        else:
            cohens_d = (np.mean(treatment) - np.mean(control)) / pooled_std

        # T-test
        t_stat, p_value = stats.ttest_ind(treatment, control)

        return {
            'cohens_d': cohens_d,
            't_statistic': t_stat,
            'p_value': p_value,
            'treatment_mean': np.mean(treatment),
            'control_mean': np.mean(control),
            'treatment_n': len(treatment),
            'control_n': len(control)
        }

    def generate_report(self) -> str:
        """Generate statistical report"""
        if not self.results:
            return "No results to report"

        stats = self.calculate_summary_statistics()

        report = []
        report.append("=" * 60)
        report.append("OVERRIDE CASCADE EXPERIMENT RESULTS")
        report.append("=" * 60)
        report.append(f"N = {stats['n']} samples")
        report.append(f"Mean Override Probability: {stats['mean_prob']:.3f} ± {stats['std_prob']:.3f}")
        report.append(f"95% CI: [{stats['ci_95'][0]:.3f}, {stats['ci_95'][1]:.3f}]")
        report.append("")
        report.append("CALIBRATION METRICS:")
        report.append(f"  Brier Score: {stats['brier_score']:.4f}")
        report.append(f"  Brier 95% CI: [{stats['brier_ci'][0]:.4f}, {stats['brier_ci'][1]:.4f}]")
        report.append(f"  ECE: {stats['ece']:.4f}")
        report.append(f"  MCE: {stats['mce']:.4f}")
        report.append("")

        # Per-provider stats if available
        provider_stats = self.calculate_summary_statistics(group_by='provider')
        if provider_stats:
            report.append("PER-PROVIDER RESULTS:")
            for provider, pstats in provider_stats.items():
                report.append(f"\n  {provider}:")
                report.append(f"    N: {pstats['n']}")
                report.append(f"    Mean: {pstats['mean_prob']:.3f} ± {pstats['std_prob']:.3f}")
                report.append(f"    95% CI: [{pstats['ci_95'][0]:.3f}, {pstats['ci_95'][1]:.3f}]")
                report.append(f"    Brier: {pstats['brier_score']:.4f}")

        report.append("=" * 60)
        return "\n".join(report)