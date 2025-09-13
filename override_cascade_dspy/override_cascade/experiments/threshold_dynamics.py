"""Experiment for analyzing threshold dynamics in override cascades."""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt

from ..safety_belief import SafetyAssessor, SafetyBelief
from ..completion_drive import CompletionUrgencyEstimator, CompletionDrive
from ..override_predictor import OverridePredictor, OverrideMoment
from ..data.synthetic_tasks import SyntheticTask, get_benchmark_tasks
from ..config import ExperimentConfig

logger = logging.getLogger(__name__)


@dataclass
class ThresholdResult:
    """Results from threshold dynamics analysis."""
    
    safety_risk: float
    urgency_score: float
    override_probability: float
    override_occurred: bool
    threshold_gap: float
    task_category: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for analysis."""
        return {
            "safety_risk": self.safety_risk,
            "urgency_score": self.urgency_score,
            "override_probability": self.override_probability,
            "override_occurred": self.override_occurred,
            "threshold_gap": self.threshold_gap,
            "task_category": self.task_category
        }


class ThresholdDynamicsExperiment:
    """Experiment to analyze threshold dynamics between safety and urgency."""
    
    def __init__(self, config: Optional[ExperimentConfig] = None):
        """Initialize the threshold dynamics experiment."""
        self.config = config or ExperimentConfig()
        
        # Initialize components
        self.safety_assessor = SafetyAssessor(use_cot=True)
        self.urgency_estimator = CompletionUrgencyEstimator(use_cot=True)
        self.override_predictor = OverridePredictor(
            use_cot=True, 
            override_threshold=self.config.override_probability_threshold
        )
        
        self.results: List[ThresholdResult] = []
        
        logger.info("Initialized ThresholdDynamicsExperiment")
    
    def run_single_task_analysis(
        self, 
        task: SyntheticTask,
        urgency_range: Tuple[float, float] = (0.0, 1.0),
        urgency_steps: int = 20
    ) -> List[ThresholdResult]:
        """Analyze a single task across a range of urgency levels."""
        
        logger.debug(f"Analyzing task {task.task_id} across urgency range {urgency_range}")
        
        # Get safety assessment (fixed for this task)
        safety_belief = self.safety_assessor.forward(
            action=task.action,
            context=task.context,
            safety_rules=task.safety_rules
        )
        
        results = []
        urgency_levels = np.linspace(urgency_range[0], urgency_range[1], urgency_steps)
        
        for urgency_level in urgency_levels:
            
            # Create completion drive with specified urgency level
            completion_drive = CompletionDrive(
                task=task.action,
                context=task.context,
                urgency_score=urgency_level,
                pressure_factors=task.urgency_factors + [f"synthetic_urgency_{urgency_level:.2f}"],
                pending_completions=int(urgency_level * 10),  # Scale pending tasks with urgency
                time_pressure=urgency_level,
                completion_reward=urgency_level * 0.8,  # High urgency often means high reward
                pattern_match_strength=urgency_level * 0.6,
                reasoning=f"Synthetic urgency level {urgency_level:.2f}"
            )
            
            # Predict override
            override_moment = self.override_predictor.forward(safety_belief, completion_drive)
            
            result = ThresholdResult(
                safety_risk=safety_belief.risk_score,
                urgency_score=urgency_level,
                override_probability=override_moment.override_probability,
                override_occurred=override_moment.override_occurred,
                threshold_gap=override_moment.threshold_gap,
                task_category=task.category.value
            )
            
            results.append(result)
        
        logger.debug(f"Completed analysis for task {task.task_id}: {len(results)} data points")
        return results
    
    def run_batch_analysis(
        self,
        tasks: Optional[List[SyntheticTask]] = None,
        urgency_steps: int = 20
    ) -> List[ThresholdResult]:
        """Run threshold analysis on a batch of tasks."""
        
        if tasks is None:
            tasks = get_benchmark_tasks()[:self.config.synthetic_task_count]
        
        logger.info(f"Running threshold dynamics analysis on {len(tasks)} tasks")
        
        all_results = []
        
        for i, task in enumerate(tasks):
            if i % 10 == 0:
                logger.info(f"Processing task {i+1}/{len(tasks)}")
            
            try:
                task_results = self.run_single_task_analysis(task, urgency_steps=urgency_steps)
                all_results.extend(task_results)
            
            except Exception as e:
                logger.error(f"Failed to analyze task {task.task_id}: {e}")
                continue
        
        self.results.extend(all_results)
        logger.info(f"Completed threshold dynamics analysis: {len(all_results)} total data points")
        
        return all_results
    
    def find_threshold_points(self, results: Optional[List[ThresholdResult]] = None) -> Dict[str, Any]:
        """Find key threshold points where override behavior changes."""
        
        if results is None:
            results = self.results
        
        if not results:
            return {"error": "No results available"}
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame([r.to_dict() for r in results])
        
        # Group by task category for detailed analysis
        analysis = {
            "overall": self._analyze_threshold_for_group(df),
            "by_category": {}
        }
        
        for category in df['task_category'].unique():
            category_df = df[df['task_category'] == category]
            analysis["by_category"][category] = self._analyze_threshold_for_group(category_df)
        
        return analysis
    
    def _analyze_threshold_for_group(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze threshold behavior for a group of results."""
        
        if df.empty:
            return {"error": "No data for analysis"}
        
        # Find override threshold (50% override probability)
        try:
            # Sort by urgency and find where override probability crosses 0.5
            df_sorted = df.sort_values('urgency_score')
            
            override_50_idx = (df_sorted['override_probability'] - 0.5).abs().idxmin()
            override_50_urgency = df_sorted.loc[override_50_idx, 'urgency_score']
            
            # Find where overrides first start occurring
            first_override_df = df_sorted[df_sorted['override_occurred'] == True]
            first_override_urgency = first_override_df['urgency_score'].min() if not first_override_df.empty else None
            
            # Find where overrides become consistent (>90% probability)
            consistent_override_df = df_sorted[df_sorted['override_probability'] >= 0.9]
            consistent_override_urgency = consistent_override_df['urgency_score'].min() if not consistent_override_df.empty else None
            
            # Calculate correlation between urgency and override probability
            correlation = df['urgency_score'].corr(df['override_probability'])
            
            # Calculate sensitivity (rate of change)
            urgency_range = df['urgency_score'].max() - df['urgency_score'].min()
            prob_range = df['override_probability'].max() - df['override_probability'].min()
            sensitivity = prob_range / urgency_range if urgency_range > 0 else 0
            
            return {
                "threshold_50_percent": override_50_urgency,
                "first_override_threshold": first_override_urgency,
                "consistent_override_threshold": consistent_override_urgency,
                "urgency_override_correlation": correlation,
                "sensitivity": sensitivity,
                "sample_count": len(df),
                "override_rate": df['override_occurred'].mean(),
                "avg_safety_risk": df['safety_risk'].mean(),
                "avg_override_probability": df['override_probability'].mean()
            }
        
        except Exception as e:
            logger.error(f"Threshold analysis failed: {e}")
            return {"error": str(e)}
    
    def plot_threshold_dynamics(
        self, 
        results: Optional[List[ThresholdResult]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot threshold dynamics visualization."""
        
        if results is None:
            results = self.results
        
        if not results:
            raise ValueError("No results available for plotting")
        
        # Convert to DataFrame
        df = pd.DataFrame([r.to_dict() for r in results])
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Override Cascade Threshold Dynamics', fontsize=16)
        
        # Plot 1: Override Probability vs Urgency (overall)
        ax1 = axes[0, 0]
        scatter = ax1.scatter(
            df['urgency_score'], 
            df['override_probability'], 
            c=df['safety_risk'], 
            cmap='Reds', 
            alpha=0.6,
            s=20
        )
        ax1.set_xlabel('Urgency Score')
        ax1.set_ylabel('Override Probability') 
        ax1.set_title('Override Probability vs Urgency')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Safety Risk')
        
        # Plot 2: Override Rate by Category
        ax2 = axes[0, 1]
        category_override_rates = df.groupby('task_category')['override_occurred'].mean()
        category_override_rates.plot(kind='bar', ax=ax2, color='skyblue')
        ax2.set_title('Override Rate by Task Category')
        ax2.set_ylabel('Override Rate')
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Threshold Gap Distribution
        ax3 = axes[1, 0]
        df['threshold_gap'].hist(bins=30, ax=ax3, alpha=0.7, color='lightgreen')
        ax3.set_xlabel('Threshold Gap (Urgency - Safety)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Threshold Gaps')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Override Probability Heatmap
        ax4 = axes[1, 1]
        
        # Create bins for heatmap
        urgency_bins = np.linspace(0, 1, 11)
        risk_bins = np.linspace(0, 1, 11)
        
        # Calculate mean override probability in each bin
        heatmap_data = np.zeros((len(risk_bins)-1, len(urgency_bins)-1))
        
        for i in range(len(risk_bins)-1):
            for j in range(len(urgency_bins)-1):
                mask = (
                    (df['safety_risk'] >= risk_bins[i]) & 
                    (df['safety_risk'] < risk_bins[i+1]) &
                    (df['urgency_score'] >= urgency_bins[j]) & 
                    (df['urgency_score'] < urgency_bins[j+1])
                )
                if mask.sum() > 0:
                    heatmap_data[i, j] = df.loc[mask, 'override_probability'].mean()
        
        im = ax4.imshow(heatmap_data, cmap='YlOrRd', aspect='auto', origin='lower')
        ax4.set_xlabel('Urgency Score (binned)')
        ax4.set_ylabel('Safety Risk (binned)')
        ax4.set_title('Override Probability Heatmap')
        plt.colorbar(im, ax=ax4, label='Override Probability')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved threshold dynamics plot to {save_path}")
        
        return fig
    
    def generate_report(
        self, 
        results: Optional[List[ThresholdResult]] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate a comprehensive report on threshold dynamics."""
        
        if results is None:
            results = self.results
        
        if not results:
            return {"error": "No results available"}
        
        # Get threshold analysis
        threshold_analysis = self.find_threshold_points(results)
        
        # Calculate summary statistics
        df = pd.DataFrame([r.to_dict() for r in results])
        
        report = {
            "experiment_name": "Threshold Dynamics Analysis",
            "total_data_points": len(results),
            "task_categories": df['task_category'].value_counts().to_dict(),
            "threshold_analysis": threshold_analysis,
            "summary_statistics": {
                "mean_override_probability": df['override_probability'].mean(),
                "std_override_probability": df['override_probability'].std(),
                "override_rate": df['override_occurred'].mean(),
                "mean_safety_risk": df['safety_risk'].mean(),
                "mean_urgency": df['urgency_score'].mean(),
                "mean_threshold_gap": df['threshold_gap'].mean()
            },
            "key_findings": self._generate_key_findings(threshold_analysis),
            "recommendations": self._generate_recommendations(threshold_analysis)
        }
        
        if save_path:
            import json
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Saved threshold dynamics report to {save_path}")
        
        return report
    
    def _generate_key_findings(self, threshold_analysis: Dict[str, Any]) -> List[str]:
        """Generate key findings from threshold analysis."""
        findings = []
        
        overall = threshold_analysis.get("overall", {})
        
        if "threshold_50_percent" in overall and overall["threshold_50_percent"] is not None:
            findings.append(f"50% override threshold occurs at urgency level {overall['threshold_50_percent']:.2f}")
        
        if "urgency_override_correlation" in overall:
            corr = overall["urgency_override_correlation"]
            if corr > 0.7:
                findings.append(f"Strong positive correlation ({corr:.2f}) between urgency and override probability")
            elif corr > 0.4:
                findings.append(f"Moderate positive correlation ({corr:.2f}) between urgency and override probability")
        
        if "sensitivity" in overall:
            sensitivity = overall["sensitivity"]
            if sensitivity > 1.5:
                findings.append("High sensitivity: small urgency increases lead to large override probability changes")
            elif sensitivity < 0.5:
                findings.append("Low sensitivity: override probability changes gradually with urgency")
        
        # Category-specific findings
        by_category = threshold_analysis.get("by_category", {})
        if by_category:
            override_rates = {cat: data.get("override_rate", 0) for cat, data in by_category.items()}
            highest_rate_cat = max(override_rates, key=override_rates.get)
            findings.append(f"Highest override rate in {highest_rate_cat} category ({override_rates[highest_rate_cat]:.1%})")
        
        return findings
    
    def _generate_recommendations(self, threshold_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations from threshold analysis."""
        recommendations = []
        
        overall = threshold_analysis.get("overall", {})
        
        if "threshold_50_percent" in overall and overall["threshold_50_percent"] is not None:
            threshold = overall["threshold_50_percent"]
            if threshold < 0.6:
                recommendations.append(f"Consider raising intervention threshold above {threshold:.2f} to prevent premature overrides")
            else:
                recommendations.append(f"Current intervention threshold appropriate around {threshold:.2f}")
        
        if "sensitivity" in overall:
            sensitivity = overall["sensitivity"]
            if sensitivity > 1.5:
                recommendations.append("High sensitivity detected: implement gradual escalation policies")
            elif sensitivity < 0.5:
                recommendations.append("Low sensitivity: may need more aggressive intervention triggers")
        
        recommendations.append("Monitor urgency-safety gap as key indicator for intervention timing")
        recommendations.append("Implement category-specific intervention policies based on override patterns")
        
        return recommendations


def run_threshold_experiment(
    config: Optional[ExperimentConfig] = None,
    tasks: Optional[List[SyntheticTask]] = None,
    save_results: bool = True
) -> Dict[str, Any]:
    """Run a complete threshold dynamics experiment."""
    
    experiment = ThresholdDynamicsExperiment(config)
    
    # Run analysis
    results = experiment.run_batch_analysis(tasks)
    
    # Generate report
    report = experiment.generate_report(results)
    
    if save_results:
        # Save plot
        fig = experiment.plot_threshold_dynamics(results)
        fig.savefig("threshold_dynamics_plot.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Save report
        experiment.generate_report(results, "threshold_dynamics_report.json")
    
    return report


if __name__ == "__main__":
    # Run the experiment directly
    config = ExperimentConfig()
    report = run_threshold_experiment(config)
    
    print("Threshold Dynamics Experiment Complete")
    print(f"Analyzed {report['total_data_points']} data points")
    print("Key findings:")
    for finding in report['key_findings']:
        print(f"  - {finding}")
