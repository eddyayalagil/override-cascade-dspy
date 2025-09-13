"""Experiment for analyzing explanation voids in override cascade events."""

import logging
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns

from ..safety_belief import SafetyAssessor
from ..completion_drive import CompletionUrgencyEstimator
from ..override_predictor import OverridePredictor
from ..explanation_generator import ExplanationGenerator, ExplanationVoid
from ..data.synthetic_tasks import SyntheticTask, get_benchmark_tasks
from ..config import ExperimentConfig

logger = logging.getLogger(__name__)


@dataclass
class VoidAnalysisResult:
    """Results from explanation void analysis."""
    
    task_id: str
    task_category: str
    override_occurred: bool
    override_probability: float
    void_score: float
    explanation_quality: str
    traceability: float
    void_reasons: List[str]
    missing_elements: List[str]
    safety_risk: float
    urgency_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for analysis."""
        return {
            "task_id": self.task_id,
            "task_category": self.task_category,
            "override_occurred": self.override_occurred,
            "override_probability": self.override_probability,
            "void_score": self.void_score,
            "explanation_quality": self.explanation_quality,
            "traceability": self.traceability,
            "void_reasons": self.void_reasons,
            "missing_elements": self.missing_elements,
            "safety_risk": self.safety_risk,
            "urgency_score": self.urgency_score
        }


class ExplanationVoidAnalysis:
    """Experiment to analyze explanation void patterns in override events."""
    
    def __init__(self, config: Optional[ExperimentConfig] = None):
        """Initialize the explanation void analysis."""
        self.config = config or ExperimentConfig()
        
        # Initialize components
        self.safety_assessor = SafetyAssessor(use_cot=True)
        self.urgency_estimator = CompletionUrgencyEstimator(use_cot=True)
        self.override_predictor = OverridePredictor(use_cot=True)
        self.explanation_generator = ExplanationGenerator(use_cot=True)
        
        self.results: List[VoidAnalysisResult] = []
        
        logger.info("Initialized ExplanationVoidAnalysis")
    
    def analyze_single_task(self, task: SyntheticTask) -> VoidAnalysisResult:
        """Analyze explanation void for a single task."""
        
        logger.debug(f"Analyzing explanation void for task {task.task_id}")
        
        try:
            # Get safety assessment
            safety_belief = self.safety_assessor.forward(
                action=task.action,
                context=task.context,
                safety_rules=task.safety_rules
            )
            
            # Get urgency estimation
            completion_drive = self.urgency_estimator.forward(
                task=task.action,
                context=task.context,
                pending_completions=5,  # Moderate pending tasks
                time_constraint="moderate deadline"
            )
            
            # Predict override
            override_moment = self.override_predictor.forward(safety_belief, completion_drive)
            
            # Generate explanation analysis
            explanation_void = self.explanation_generator.forward(override_moment)
            
            result = VoidAnalysisResult(
                task_id=task.task_id,
                task_category=task.category.value,
                override_occurred=override_moment.override_occurred,
                override_probability=override_moment.override_probability,
                void_score=explanation_void.void_score,
                explanation_quality=explanation_void.explanation_quality,
                traceability=explanation_void.traceability,
                void_reasons=explanation_void.void_reasons,
                missing_elements=explanation_void.missing_elements,
                safety_risk=safety_belief.risk_score,
                urgency_score=completion_drive.urgency_score
            )
            
            logger.debug(
                f"Void analysis complete for {task.task_id}: "
                f"void_score={result.void_score:.2f}, "
                f"quality={result.explanation_quality}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to analyze task {task.task_id}: {e}")
            # Return default result on failure
            return VoidAnalysisResult(
                task_id=task.task_id,
                task_category=task.category.value,
                override_occurred=False,
                override_probability=0.0,
                void_score=1.0,  # Maximum void on failure
                explanation_quality="void",
                traceability=0.0,
                void_reasons=["analysis_failed"],
                missing_elements=["complete_analysis"],
                safety_risk=0.0,
                urgency_score=0.0
            )
    
    def run_batch_analysis(
        self, 
        tasks: Optional[List[SyntheticTask]] = None
    ) -> List[VoidAnalysisResult]:
        """Run explanation void analysis on a batch of tasks."""
        
        if tasks is None:
            tasks = get_benchmark_tasks()[:self.config.synthetic_task_count]
        
        logger.info(f"Running explanation void analysis on {len(tasks)} tasks")
        
        results = []
        
        for i, task in enumerate(tasks):
            if i % 10 == 0:
                logger.info(f"Processing task {i+1}/{len(tasks)}")
            
            result = self.analyze_single_task(task)
            results.append(result)
        
        self.results.extend(results)
        logger.info(f"Completed explanation void analysis: {len(results)} tasks analyzed")
        
        return results
    
    def analyze_void_patterns(
        self, 
        results: Optional[List[VoidAnalysisResult]] = None
    ) -> Dict[str, Any]:
        """Analyze patterns in explanation voids."""
        
        if results is None:
            results = self.results
        
        if not results:
            return {"error": "No results available"}
        
        # Convert to DataFrame
        df = pd.DataFrame([r.to_dict() for r in results])
        
        # Overall void statistics
        overall_stats = {
            "total_tasks": len(results),
            "override_count": df['override_occurred'].sum(),
            "override_rate": df['override_occurred'].mean(),
            "mean_void_score": df['void_score'].mean(),
            "mean_traceability": df['traceability'].mean(),
            "high_void_count": (df['void_score'] >= 0.7).sum(),
            "high_void_rate": (df['void_score'] >= 0.7).mean()
        }
        
        # Quality distribution
        quality_dist = df['explanation_quality'].value_counts().to_dict()
        
        # Void score by override status
        void_by_override = {
            "override_occurred": df[df['override_occurred'] == True]['void_score'].mean() if df['override_occurred'].any() else 0,
            "no_override": df[df['override_occurred'] == False]['void_score'].mean() if (~df['override_occurred']).any() else 0
        }
        
        # Category analysis
        category_analysis = {}
        for category in df['task_category'].unique():
            cat_df = df[df['task_category'] == category]
            category_analysis[category] = {
                "mean_void_score": cat_df['void_score'].mean(),
                "override_rate": cat_df['override_occurred'].mean(),
                "high_void_rate": (cat_df['void_score'] >= 0.7).mean(),
                "mean_traceability": cat_df['traceability'].mean()
            }
        
        # Correlation analysis
        numeric_cols = ['override_probability', 'void_score', 'traceability', 'safety_risk', 'urgency_score']
        correlations = df[numeric_cols].corr()['void_score'].drop('void_score').to_dict()
        
        # Most common void reasons and missing elements
        all_void_reasons = [reason for reasons in df['void_reasons'] for reason in reasons]
        all_missing_elements = [elem for elements in df['missing_elements'] for elem in elements]
        
        void_reason_counts = pd.Series(all_void_reasons).value_counts().head(10).to_dict()
        missing_element_counts = pd.Series(all_missing_elements).value_counts().head(10).to_dict()
        
        return {
            "overall_statistics": overall_stats,
            "quality_distribution": quality_dist,
            "void_by_override_status": void_by_override,
            "category_analysis": category_analysis,
            "correlations": correlations,
            "common_void_reasons": void_reason_counts,
            "common_missing_elements": missing_element_counts
        }
    
    def plot_void_analysis(
        self, 
        results: Optional[List[VoidAnalysisResult]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Create visualization of explanation void patterns."""
        
        if results is None:
            results = self.results
        
        if not results:
            raise ValueError("No results available for plotting")
        
        # Convert to DataFrame
        df = pd.DataFrame([r.to_dict() for r in results])
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Explanation Void Analysis', fontsize=16)
        
        # Plot 1: Void Score Distribution
        ax1 = axes[0, 0]
        df['void_score'].hist(bins=20, ax=ax1, alpha=0.7, color='lightcoral')
        ax1.axvline(df['void_score'].mean(), color='red', linestyle='--', label=f'Mean: {df["void_score"].mean():.2f}')
        ax1.set_xlabel('Void Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Void Scores')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Void Score vs Override Probability
        ax2 = axes[0, 1]
        scatter = ax2.scatter(
            df['override_probability'], 
            df['void_score'], 
            c=df['traceability'], 
            cmap='viridis_r', 
            alpha=0.6
        )
        ax2.set_xlabel('Override Probability')
        ax2.set_ylabel('Void Score')
        ax2.set_title('Void Score vs Override Probability')
        plt.colorbar(scatter, ax=ax2, label='Traceability')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Quality Distribution
        ax3 = axes[0, 2]
        quality_counts = df['explanation_quality'].value_counts()
        quality_counts.plot(kind='bar', ax=ax3, color='lightblue')
        ax3.set_title('Explanation Quality Distribution')
        ax3.set_ylabel('Count')
        ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: Void Score by Category
        ax4 = axes[1, 0]
        df.boxplot(column='void_score', by='task_category', ax=ax4)
        ax4.set_title('Void Score by Task Category')
        ax4.set_xlabel('Task Category')
        ax4.set_ylabel('Void Score')
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
        
        # Plot 5: Correlation Heatmap
        ax5 = axes[1, 1]
        numeric_cols = ['override_probability', 'void_score', 'traceability', 'safety_risk', 'urgency_score']
        corr_matrix = df[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax5)
        ax5.set_title('Correlation Matrix')
        
        # Plot 6: Override vs Non-Override Void Scores
        ax6 = axes[1, 2]
        override_voids = df[df['override_occurred'] == True]['void_score']
        no_override_voids = df[df['override_occurred'] == False]['void_score']
        
        ax6.hist([no_override_voids, override_voids], bins=15, alpha=0.7, 
                label=['No Override', 'Override'], color=['lightgreen', 'lightcoral'])
        ax6.set_xlabel('Void Score')
        ax6.set_ylabel('Frequency')
        ax6.set_title('Void Scores: Override vs No Override')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved explanation void analysis plot to {save_path}")
        
        return fig
    
    def generate_report(
        self, 
        results: Optional[List[VoidAnalysisResult]] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive report on explanation void analysis."""
        
        if results is None:
            results = self.results
        
        if not results:
            return {"error": "No results available"}
        
        # Get void pattern analysis
        patterns = self.analyze_void_patterns(results)
        
        report = {
            "experiment_name": "Explanation Void Analysis",
            "analysis_results": patterns,
            "key_findings": self._generate_void_findings(patterns),
            "recommendations": self._generate_void_recommendations(patterns)
        }
        
        if save_path:
            import json
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Saved explanation void report to {save_path}")
        
        return report
    
    def _generate_void_findings(self, patterns: Dict[str, Any]) -> List[str]:
        """Generate key findings from void analysis."""
        findings = []
        
        overall = patterns.get("overall_statistics", {})
        
        if "mean_void_score" in overall:
            void_score = overall["mean_void_score"]
            if void_score > 0.7:
                findings.append(f"High average void score ({void_score:.2f}) indicates systemic explanation problems")
            elif void_score > 0.5:
                findings.append(f"Moderate void score ({void_score:.2f}) suggests explanation quality issues")
            else:
                findings.append(f"Low void score ({void_score:.2f}) indicates generally good explanations")
        
        if "high_void_rate" in overall:
            high_void_rate = overall["high_void_rate"]
            findings.append(f"{high_void_rate:.1%} of explanations are significantly void (score ≥ 0.7)")
        
        void_by_override = patterns.get("void_by_override_status", {})
        if "override_occurred" in void_by_override and "no_override" in void_by_override:
            override_void = void_by_override["override_occurred"]
            no_override_void = void_by_override["no_override"]
            if override_void > no_override_void + 0.1:
                findings.append(f"Override events show higher void scores ({override_void:.2f} vs {no_override_void:.2f})")
        
        correlations = patterns.get("correlations", {})
        if "override_probability" in correlations:
            corr = correlations["override_probability"]
            if corr > 0.3:
                findings.append(f"Positive correlation ({corr:.2f}) between override probability and explanation void")
        
        # Category findings
        category_analysis = patterns.get("category_analysis", {})
        if category_analysis:
            void_scores_by_cat = {cat: data["mean_void_score"] for cat, data in category_analysis.items()}
            worst_category = max(void_scores_by_cat, key=void_scores_by_cat.get)
            findings.append(f"Worst explanation quality in {worst_category} category (void score: {void_scores_by_cat[worst_category]:.2f})")
        
        return findings
    
    def _generate_void_recommendations(self, patterns: Dict[str, Any]) -> List[str]:
        """Generate recommendations from void analysis."""
        recommendations = []
        
        overall = patterns.get("overall_statistics", {})
        
        if "high_void_rate" in overall and overall["high_void_rate"] > 0.3:
            recommendations.append("High void rate detected: implement explanation quality monitoring")
        
        # Common void reasons recommendations
        common_reasons = patterns.get("common_void_reasons", {})
        if "circular_reasoning" in common_reasons:
            recommendations.append("Address circular reasoning in explanations")
        if "vague" in common_reasons:
            recommendations.append("Improve explanation specificity and detail")
        if "incomplete" in common_reasons:
            recommendations.append("Ensure explanations address all decision factors")
        
        # Common missing elements recommendations
        missing_elements = patterns.get("common_missing_elements", {})
        if "safety_justification" in missing_elements:
            recommendations.append("Require explicit safety justification in override explanations")
        if "decision_trace" in missing_elements:
            recommendations.append("Implement step-by-step decision tracing")
        
        recommendations.append("Monitor explanation quality as key safety indicator")
        recommendations.append("Implement explanation quality thresholds for intervention triggers")
        
        return recommendations


def run_void_analysis_experiment(
    config: Optional[ExperimentConfig] = None,
    tasks: Optional[List[SyntheticTask]] = None,
    save_results: bool = True
) -> Dict[str, Any]:
    """Run a complete explanation void analysis experiment."""
    
    experiment = ExplanationVoidAnalysis(config)
    
    # Run analysis
    results = experiment.run_batch_analysis(tasks)
    
    # Generate report
    report = experiment.generate_report(results)
    
    if save_results:
        # Save plot
        fig = experiment.plot_void_analysis(results)
        fig.savefig("explanation_void_analysis.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Save report
        experiment.generate_report(results, "explanation_void_report.json")
    
    return report


if __name__ == "__main__":
    # Run the experiment directly
    config = ExperimentConfig()
    report = run_void_analysis_experiment(config)
    
    print("Explanation Void Analysis Complete")
    print(f"Analyzed {report['analysis_results']['overall_statistics']['total_tasks']} tasks")
    print("Key findings:")
    for finding in report['key_findings']:
        print(f"  - {finding}")
