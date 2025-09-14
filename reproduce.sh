#!/bin/bash

# Override Cascade Reproducibility Script
# This script reproduces all experiments with fixed seeds and saves artifacts

set -e  # Exit on error

echo "================================================"
echo "Override Cascade Framework - Reproducibility Run"
echo "================================================"

# Configuration
SEED=42
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUNS_DIR="runs/reproduce_${TIMESTAMP}"
CACHE_DIR="cache"
PYTHON=${PYTHON:-python3}

# Create directories
mkdir -p "${RUNS_DIR}"
mkdir -p "${CACHE_DIR}"
mkdir -p "${RUNS_DIR}/artifacts"
mkdir -p "${RUNS_DIR}/logs"

# Check Python version
echo "Python version:"
${PYTHON} --version

# Install dependencies if needed
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    ${PYTHON} -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

echo "Installing dependencies..."
pip install -q -r requirements.txt

# Export environment variables
export PYTHONHASHSEED=${SEED}
export REPRODUCIBLE_RUN=1
export RUNS_DIR=${RUNS_DIR}

# Log configuration
cat > "${RUNS_DIR}/config.json" << EOF
{
    "seed": ${SEED},
    "timestamp": "${TIMESTAMP}",
    "python_version": "$(${PYTHON} --version 2>&1)",
    "platform": "$(uname -a)",
    "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo 'not-in-git')",
    "runs_directory": "${RUNS_DIR}"
}
EOF

echo ""
echo "Starting reproducibility suite..."
echo "================================"

# 1. Calibration Suite
echo ""
echo "[1/6] Running calibration suite..."
${PYTHON} calibration_suite.py \
    --seed ${SEED} \
    --output "${RUNS_DIR}/calibration" \
    2>&1 | tee "${RUNS_DIR}/logs/calibration.log"

# 2. Reliability Analysis
echo ""
echo "[2/6] Running reliability analysis..."
${PYTHON} reliability_tools.py \
    --seed ${SEED} \
    --output "${RUNS_DIR}/reliability" \
    2>&1 | tee "${RUNS_DIR}/logs/reliability.log"

# 3. Math Model Fitting
echo ""
echo "[3/6] Running math model fitting..."
${PYTHON} math_model.py \
    --seed ${SEED} \
    --output "${RUNS_DIR}/model" \
    2>&1 | tee "${RUNS_DIR}/logs/math_model.log"

# 4. Provider Comparison
echo ""
echo "[4/6] Running provider comparison..."
${PYTHON} provider_comparison.py \
    --seed ${SEED} \
    --output "${RUNS_DIR}/providers" \
    2>&1 | tee "${RUNS_DIR}/logs/providers.log"

# 5. Core Experiments (if test data available)
if [ -f "test_data/scenarios.json" ]; then
    echo ""
    echo "[5/6] Running core experiments..."
    ${PYTHON} -m override_cascade_dspy.override_cascade.main \
        --seed ${SEED} \
        --scenarios test_data/scenarios.json \
        --output "${RUNS_DIR}/experiments" \
        2>&1 | tee "${RUNS_DIR}/logs/experiments.log"
else
    echo "[5/6] Skipping core experiments (no test data found)"
fi

# 6. Generate final report
echo ""
echo "[6/6] Generating final report..."
${PYTHON} - << 'PYTHON_SCRIPT' > "${RUNS_DIR}/final_report.md"
import json
import sys
from pathlib import Path

runs_dir = Path("${RUNS_DIR}")

print("# Override Cascade Reproducibility Report")
print(f"\nGenerated: {runs_dir.name}")
print("\n## Configuration")

with open(runs_dir / "config.json") as f:
    config = json.load(f)
    for key, value in config.items():
        print(f"- **{key}**: {value}")

print("\n## Results Summary")

# Check for calibration results
cal_report = runs_dir / "calibration" / "calibration_report.json"
if cal_report.exists():
    with open(cal_report) as f:
        cal = json.load(f)
        if "validation" in cal:
            print("\n### Calibration")
            print(f"- 95% Claim Validated: {cal['validation']['claim_validated']}")
            print(f"- Achieved Rate: {cal['validation']['achieved_rate']:.3f}")

# Check for model diagnostics
model_metrics = runs_dir / "model" / "metrics.json"
if model_metrics.exists():
    with open(model_metrics) as f:
        metrics = json.load(f)
        print("\n### Math Model")
        print(f"- R-squared: {metrics.get('r_squared', 'N/A')}")
        print(f"- RMSE: {metrics.get('rmse', 'N/A')}")

print("\n## Artifacts")
print("\nThe following artifacts were generated:")

for artifact in runs_dir.glob("**/*.json"):
    print(f"- {artifact.relative_to(runs_dir)}")

for artifact in runs_dir.glob("**/*.png"):
    print(f"- {artifact.relative_to(runs_dir)}")

for artifact in runs_dir.glob("**/*.csv"):
    print(f"- {artifact.relative_to(runs_dir)}")

print("\n## Reproduction Instructions")
print("\nTo reproduce these exact results:")
print("```bash")
print(f"SEED={config['seed']} ./reproduce.sh")
print("```")
PYTHON_SCRIPT

# Create artifact manifest
echo ""
echo "Creating artifact manifest..."
find "${RUNS_DIR}" -type f \( -name "*.json" -o -name "*.png" -o -name "*.csv" -o -name "*.txt" \) | \
    while read -r file; do
        sha256sum "$file"
    done > "${RUNS_DIR}/artifacts/manifest.txt"

# Create compressed archive
echo ""
echo "Creating archive..."
tar -czf "${RUNS_DIR}.tar.gz" "${RUNS_DIR}"

echo ""
echo "================================================"
echo "Reproducibility run complete!"
echo "Results saved to: ${RUNS_DIR}"
echo "Archive created: ${RUNS_DIR}.tar.gz"
echo "================================================"

# Deactivate virtual environment
deactivate

# Display summary
echo ""
echo "Summary:"
echo "--------"
ls -lh "${RUNS_DIR}"/*.md 2>/dev/null || true
echo ""
tail -20 "${RUNS_DIR}/final_report.md" 2>/dev/null || true