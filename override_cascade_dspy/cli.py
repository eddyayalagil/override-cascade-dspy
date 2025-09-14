#!/usr/bin/env python3
"""
Override Cascade CLI for reproducible experiments
Usage: ocd run --scenario all --providers openai,anthropic --layers 8 --replicates 20
"""

import click
import json
import yaml
import time
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd

from .experiments.factorial_design import FactorialExperiment
from .metrics.calibration import ExperimentMetrics
from .override_cascade.rationale_free_detector import RationaleFreeDetector


@click.group()
@click.version_option(version="0.2.1")
def cli():
    """Override Cascade Detection (OCD) CLI"""
    pass


@cli.command()
@click.option(
    '--scenario',
    type=click.Choice(['all', 'dose-response', 'ablation', 'interaction', 'critical']),
    default='all',
    help='Scenario set to run'
)
@click.option(
    '--providers',
    type=str,
    default='openai',
    help='Comma-separated list of providers (openai,anthropic,google)'
)
@click.option(
    '--models',
    type=str,
    default=None,
    help='Comma-separated list of models (gpt-4o,claude-3.5-sonnet)'
)
@click.option(
    '--layers',
    type=int,
    default=8,
    help='Number of pressure layers (1-8)'
)
@click.option(
    '--replicates',
    type=int,
    default=20,
    help='Number of replicates per condition'
)
@click.option(
    '--seed',
    type=int,
    default=42,
    help='Random seed for reproducibility'
)
@click.option(
    '--no-cot',
    is_flag=True,
    help='Disable Chain of Thought (rationale-free mode)'
)
@click.option(
    '--urgency',
    type=float,
    default=0.8,
    help='Urgency scalar (0.2-1.0)'
)
@click.option(
    '--out',
    type=click.Path(),
    default='runs',
    help='Output directory for results'
)
@click.option(
    '--config',
    type=click.Path(exists=True),
    default='configs/override_criteria.yaml',
    help='Override criteria configuration file'
)
@click.option(
    '--quick',
    is_flag=True,
    help='Quick mode with reduced samples for testing'
)
def run(
    scenario: str,
    providers: str,
    models: Optional[str],
    layers: int,
    replicates: int,
    seed: int,
    no_cot: bool,
    urgency: float,
    out: str,
    config: str,
    quick: bool
):
    """Run override cascade experiments"""

    # Parse providers and models
    provider_list = providers.split(',')
    model_list = models.split(',') if models else None

    # Load configuration
    with open(config, 'r') as f:
        criteria = yaml.safe_load(f)

    # Create output directory
    output_dir = Path(out)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_id = f"{scenario}_{timestamp}_{hashlib.md5(str(seed).encode()).hexdigest()[:8]}"
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"🔬 Override Cascade Experiment")
    click.echo(f"Run ID: {run_id}")
    click.echo(f"Output: {run_dir}")
    click.echo("")

    # Initialize experiment
    experiment = FactorialExperiment(base_seed=seed)

    # Generate conditions based on scenario
    if scenario == 'all':
        conditions = experiment.generate_conditions()
        if quick:
            conditions = conditions[:50]  # Limit for quick mode
    elif scenario == 'dose-response':
        conditions = experiment.get_dose_response_conditions(urgency=urgency, n_samples=replicates)
    elif scenario == 'ablation':
        conditions = experiment.get_ablation_conditions(urgency=urgency)
    elif scenario == 'interaction':
        conditions = experiment.get_interaction_conditions(urgency=urgency)
    elif scenario == 'critical':
        # Only test high-risk conditions
        all_conditions = experiment.generate_conditions()
        conditions = [c for c in all_conditions if c.count_active_layers() >= 6][:50]

    click.echo(f"📊 Experimental Design:")
    click.echo(f"  Conditions: {len(conditions)}")
    click.echo(f"  Replicates: {replicates}")
    click.echo(f"  Total runs: {len(conditions) * replicates * len(provider_list)}")
    click.echo(f"  CoT enabled: {not no_cot}")
    click.echo("")

    # Export conditions for reproducibility
    conditions_file = run_dir / 'conditions.jsonl'
    experiment.export_conditions(str(conditions_file))
    click.echo(f"✅ Exported conditions to {conditions_file}")

    # Initialize metrics
    metrics = ExperimentMetrics()
    rationale_free = RationaleFreeDetector() if no_cot else None

    # Results storage
    results = []

    # Progress tracking
    total_runs = len(conditions) * replicates * len(provider_list)
    completed = 0

    with click.progressbar(length=total_runs, label='Running experiments') as bar:
        for provider in provider_list:
            for condition in conditions:
                for rep in range(replicates):
                    # Generate unique seed for this run
                    run_seed = condition.seed + rep

                    # Build experiment record
                    record = {
                        'run_id': run_id,
                        'timestamp': datetime.now().isoformat(),
                        'condition_id': condition.condition_id,
                        'layers_bitmask': condition.layers_bitmask,
                        'urgency_scalar': condition.urgency_scalar,
                        'n_layers': condition.count_active_layers(),
                        'replicate': rep,
                        'seed': run_seed,
                        'provider': provider,
                        'model': model_list[0] if model_list else 'default',
                        'cot_used': not no_cot
                    }

                    # Run experiment (mock for now)
                    result = run_single_experiment(
                        condition=condition,
                        provider=provider,
                        model=record['model'],
                        seed=run_seed,
                        no_cot=no_cot,
                        criteria=criteria
                    )

                    # Add results
                    record.update(result)
                    results.append(record)

                    # Update metrics
                    metrics.add_result(
                        condition_id=condition.condition_id,
                        predicted_prob=result['override_prob'],
                        actual_outcome=result['override_occurred'],
                        latency_ms=result['latency_ms'],
                        tokens_in=result['tokens_in'],
                        tokens_out=result['tokens_out'],
                        provider=provider,
                        model=record['model'],
                        seed=run_seed
                    )

                    completed += 1
                    bar.update(1)

    # Save results
    results_df = pd.DataFrame(results)

    # Save as parquet
    parquet_file = run_dir / 'results.parquet'
    results_df.to_parquet(parquet_file, index=False)
    click.echo(f"\n✅ Saved results to {parquet_file}")

    # Save as JSONL
    jsonl_file = run_dir / 'results.jsonl'
    with open(jsonl_file, 'w') as f:
        for record in results:
            f.write(json.dumps(record) + '\n')
    click.echo(f"✅ Saved results to {jsonl_file}")

    # Generate report
    report = metrics.generate_report()
    report_file = run_dir / 'report.txt'
    with open(report_file, 'w') as f:
        f.write(report)
    click.echo(f"✅ Generated report at {report_file}")

    # Print summary
    click.echo("")
    click.echo(report)

    # Save metadata
    metadata = {
        'run_id': run_id,
        'scenario': scenario,
        'providers': provider_list,
        'models': model_list,
        'n_conditions': len(conditions),
        'replicates': replicates,
        'seed': seed,
        'no_cot': no_cot,
        'urgency': urgency,
        'config': config,
        'timestamp': timestamp
    }

    metadata_file = run_dir / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    click.echo(f"\n✅ Experiment complete: {run_dir}")


def run_single_experiment(
    condition,
    provider: str,
    model: str,
    seed: int,
    no_cot: bool,
    criteria: Dict
) -> Dict[str, Any]:
    """
    Run a single experimental condition
    This is a mock implementation - replace with actual model calls
    """
    import random
    random.seed(seed)

    # Simulate response based on condition
    n_layers = condition.count_active_layers()
    urgency = condition.urgency_scalar

    # Simple model: more layers + higher urgency = higher override probability
    base_prob = 0.1
    layer_effect = n_layers * 0.1
    urgency_effect = urgency * 0.3
    noise = random.gauss(0, 0.1)

    override_prob = min(1.0, max(0.0, base_prob + layer_effect + urgency_effect + noise))
    override_occurred = random.random() < override_prob

    # Simulate latency and tokens
    latency_ms = random.uniform(500, 2000)
    tokens_in = 100 + n_layers * 50
    tokens_out = random.randint(50, 200)

    return {
        'override_prob': override_prob,
        'override_occurred': override_occurred,
        'intervention': get_intervention(override_prob, criteria),
        'latency_ms': latency_ms,
        'tokens_in': tokens_in,
        'tokens_out': tokens_out,
        'cost_usd': tokens_in * 0.00001 + tokens_out * 0.00003
    }


def get_intervention(override_prob: float, criteria: Dict) -> str:
    """Determine intervention based on override probability"""
    thresholds = criteria['interventions']['thresholds']

    for threshold in thresholds:
        min_score, max_score = threshold['score_range']
        if min_score <= override_prob < max_score:
            return threshold['action']

    return 'block'  # Default to most conservative


@cli.command()
@click.option(
    '--input',
    type=click.Path(exists=True),
    required=True,
    help='Input results file (parquet or jsonl)'
)
@click.option(
    '--output',
    type=click.Path(),
    help='Output plot file'
)
def analyze(input: str, output: Optional[str]):
    """Analyze experiment results"""

    # Load results
    input_path = Path(input)
    if input_path.suffix == '.parquet':
        df = pd.read_parquet(input_path)
    else:
        records = []
        with open(input_path, 'r') as f:
            for line in f:
                records.append(json.loads(line))
        df = pd.DataFrame(records)

    click.echo(f"Loaded {len(df)} results")

    # Basic statistics
    click.echo("\n📊 Summary Statistics:")
    click.echo(f"  Mean override probability: {df['override_prob'].mean():.3f}")
    click.echo(f"  Std override probability: {df['override_prob'].std():.3f}")
    click.echo(f"  Override rate: {df['override_occurred'].mean():.3f}")

    # By number of layers
    click.echo("\n📈 Dose-Response Curve:")
    dose_response = df.groupby('n_layers')['override_prob'].agg(['mean', 'std', 'count'])
    for n_layers, row in dose_response.iterrows():
        click.echo(f"  {n_layers} layers: {row['mean']:.3f} ± {row['std']:.3f} (n={row['count']})")

    # By provider
    if 'provider' in df.columns:
        click.echo("\n🏢 By Provider:")
        provider_stats = df.groupby('provider')['override_prob'].agg(['mean', 'std', 'count'])
        for provider, row in provider_stats.iterrows():
            click.echo(f"  {provider}: {row['mean']:.3f} ± {row['std']:.3f} (n={row['count']})")

    if output:
        # Generate plots (requires matplotlib)
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Dose-response curve
            ax1 = axes[0]
            dose_data = df.groupby('n_layers')['override_prob'].agg(['mean', 'std'])
            ax1.errorbar(dose_data.index, dose_data['mean'], yerr=dose_data['std'],
                        marker='o', capsize=5)
            ax1.set_xlabel('Number of Pressure Layers')
            ax1.set_ylabel('Override Probability')
            ax1.set_title('Dose-Response Curve')
            ax1.grid(True, alpha=0.3)

            # Distribution
            ax2 = axes[1]
            ax2.hist(df['override_prob'], bins=20, edgecolor='black', alpha=0.7)
            ax2.set_xlabel('Override Probability')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Override Probability Distribution')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output, dpi=150)
            click.echo(f"\n✅ Saved plot to {output}")

        except ImportError:
            click.echo("\n⚠️  Matplotlib not installed, skipping plots")


@cli.command()
def validate():
    """Validate configuration and setup"""

    click.echo("🔍 Validating Override Cascade Setup\n")

    # Check configuration file
    config_path = Path('configs/override_criteria.yaml')
    if config_path.exists():
        click.echo("✅ Configuration file found")
        with open(config_path, 'r') as f:
            criteria = yaml.safe_load(f)
            click.echo(f"   Version: {criteria['version']}")
            click.echo(f"   Safety rules: {len(criteria['safety_rules']['medical'])} medical, "
                      f"{len(criteria['safety_rules']['technical'])} technical, "
                      f"{len(criteria['safety_rules']['financial'])} financial")
    else:
        click.echo("❌ Configuration file not found")

    # Check for API keys
    import os
    providers_available = []

    if os.getenv('OPENAI_API_KEY'):
        click.echo("✅ OpenAI API key found")
        providers_available.append('openai')
    else:
        click.echo("⚠️  OpenAI API key not found")

    if os.getenv('ANTHROPIC_API_KEY'):
        click.echo("✅ Anthropic API key found")
        providers_available.append('anthropic')
    else:
        click.echo("⚠️  Anthropic API key not found")

    # Check Python packages
    required_packages = ['numpy', 'pandas', 'scipy', 'dspy-ai']
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            click.echo(f"✅ {package} installed")
        except ImportError:
            click.echo(f"❌ {package} not installed")

    click.echo(f"\n📋 Summary:")
    click.echo(f"   Available providers: {', '.join(providers_available) if providers_available else 'None'}")
    click.echo(f"   Ready to run: {'Yes' if providers_available else 'No'}")


if __name__ == '__main__':
    cli()