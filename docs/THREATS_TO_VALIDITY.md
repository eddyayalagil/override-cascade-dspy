# Threats to Validity

## Overview

This document outlines potential threats to the validity of our override cascade research and the measures taken to address them.

## 1. Construct Validity

### Threat: Ambiguous Definition of "Override"

**Issue**: The concept of "override" may be interpreted differently across contexts and evaluators.

**Mitigation**:
- Operationalized override with machine-readable YAML criteria (`configs/override_criteria.yaml`)
- Three distinct categories: unsafe_action_emitted, refusal_bypass, policy_violating_tool_call
- Pre-registered criteria before experiments
- Allows community to contest/extend definitions

### Threat: Pressure Layers as Proxy for Real-World Pressure

**Issue**: Our 8-layer pressure structure may not accurately represent real-world decision pressure.

**Mitigation**:
- Layers derived from documented incidents and psychology research
- Factorial design tests all combinations (2^8 = 256)
- Ablation studies isolate individual layer effects
- Dose-response curves show cumulative effects

## 2. Internal Validity

### Threat: Confounding Variables

**Issue**: Multiple factors change simultaneously, making causal attribution difficult.

**Mitigation**:
- Factorial design isolates main effects and interactions
- Fixed random seeds for reproducibility
- Counterbalanced condition order
- Control conditions without pressure

### Threat: Pattern Triggers Acting as Jailbreaks

**Issue**: "It worked before" patterns may function as prompting exploits rather than measuring genuine override tendency.

**Mitigation**:
- Ablation studies test with/without pattern triggers
- Counterfactual prompts with reversed patterns
- Separate analysis of pattern vs. non-pattern conditions
- Rationale-free detection to avoid prompt-based artifacts

### Threat: Authority Language vs. Time Pressure Conflation

**Issue**: Authority and temporal pressure often co-occur, making effects inseparable.

**Mitigation**:
- 2^8 factorial includes all combinations
- Interaction analysis specifically tests authority × time
- Isolated conditions with single pressure types
- Statistical decomposition of variance

## 3. External Validity

### Threat: Model-Specific Behavior

**Issue**: Results may not generalize across different AI systems.

**Mitigation**:
- Multi-provider testing (OpenAI, Anthropic, Google, etc.)
- Per-provider variance reporting
- Model version documentation
- Temperature and sampling parameters recorded

### Threat: Artificial Scenarios

**Issue**: Experimental scenarios may not reflect real-world complexity.

**Mitigation**:
- Scenarios based on documented real incidents
- Multiple domains (medical, financial, technical, etc.)
- Varying severity levels
- Frozen benchmark for comparison across studies

## 4. Statistical Conclusion Validity

### Threat: Inadequate Sample Size

**Issue**: Insufficient power to detect true effects.

**Mitigation**:
- Power analysis conducted (n=20 per condition for d=0.5)
- Bootstrap confidence intervals
- Effect size reporting (Cohen's d)
- Replication encouraged with frozen scenarios

### Threat: Multiple Comparisons

**Issue**: Testing many conditions increases false positive risk.

**Mitigation**:
- Bonferroni correction for multiple comparisons
- False Discovery Rate (FDR) control
- Pre-registered primary hypotheses
- Replication across providers

### Threat: Poor Calibration

**Issue**: Predicted probabilities may not match actual frequencies.

**Mitigation**:
- Brier score calculation
- Expected Calibration Error (ECE)
- Reliability diagrams
- Calibration target < 0.1 ECE

## 5. Measurement Validity

### Threat: Chain of Thought Restrictions

**Issue**: Many providers restrict CoT access, limiting interpretability.

**Mitigation**:
- Rationale-free detection mode using observable signals
- Behavioral pattern analysis (refusals, hedging, latency)
- PII scrubbing for logged responses
- --no-cot CLI option for restricted environments

### Threat: Response Variability

**Issue**: Same condition may produce different responses.

**Mitigation**:
- Fixed seeds for reproducibility
- Multiple replicates per condition
- Variance estimation and reporting
- Temperature control (0.1 for consistency)

## 6. Ecological Validity

### Threat: Laboratory vs. Production Differences

**Issue**: Experimental setup differs from real deployment.

**Mitigation**:
- Two SKUs: Research (rigorous) vs. Ops (fast screening)
- Latency and resource usage tracked
- Cost estimation provided
- Integration patterns documented

### Threat: Temporal Validity

**Issue**: Model behavior changes over time with updates.

**Mitigation**:
- Model versions recorded in results
- Timestamp all experiments
- Periodic re-evaluation recommended
- Historical results archived

## 7. Reporting Validity

### Threat: Selective Reporting

**Issue**: Only positive results might be emphasized.

**Mitigation**:
- All conditions tested and reported
- Null results included
- Effect sizes reported regardless of significance
- Complete results in parquet/JSONL format

### Threat: Reproducibility

**Issue**: Others cannot replicate findings.

**Mitigation**:
- Frozen scenarios (benchmarks/frozen_scenarios.jsonl)
- Seed control throughout
- Docker containerization (planned)
- Complete code and data published

## 8. Ethical Validity

### Threat: Dual-Use Potential

**Issue**: Research could be misused to exploit AI systems.

**Mitigation**:
- Defensive focus emphasized
- Intervention mechanisms included
- Responsible disclosure practices
- Safety research framing

## Summary Table

| Validity Type | Primary Threats | Key Mitigations |
|--------------|-----------------|-----------------|
| Construct | Override definition | Machine-readable criteria |
| Internal | Confounding | Factorial design |
| External | Model-specific | Multi-provider testing |
| Statistical | Sample size | Power analysis |
| Measurement | CoT restrictions | Rationale-free mode |
| Ecological | Lab vs. production | Two SKUs |
| Reporting | Reproducibility | Frozen benchmarks |
| Ethical | Dual-use | Defensive focus |

## Recommendations for Users

1. **Always report**:
   - Model versions and providers
   - Seeds and temperature settings
   - N per condition
   - Confidence intervals
   - Effect sizes

2. **Consider running**:
   - Ablation studies for causal claims
   - Multiple providers for generalization
   - Power analysis before experiments
   - Calibration metrics

3. **Be transparent about**:
   - Deviations from frozen scenarios
   - Post-hoc analyses
   - Unexpected findings
   - Limitations encountered

## Version History

- v0.2.1 (2025): Initial threats documentation
- Future: Community feedback integration