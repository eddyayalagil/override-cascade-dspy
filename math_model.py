"""Identifiable mathematical model for override cascade dynamics.

Implements S(t) = S0 * exp(-λ P(t)) * (1 - σ I(t)) + ε R(t)
with proper constraints, fitting procedures, and diagnostics.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ModelParameters:
    """Parameters for the override cascade model."""
    S0: float = 1.0  # Initial safety level
    lambda_p: float = 0.5  # Pressure decay rate (λ > 0)
    sigma_i: float = 0.3  # Intervention effectiveness (0 ≤ σ ≤ 1)
    epsilon_r: float = 0.1  # Recovery rate (ε > 0)

    # Bounds for optimization
    bounds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'S0': (0.5, 1.0),
        'lambda_p': (0.01, 5.0),
        'sigma_i': (0.0, 1.0),
        'epsilon_r': (0.0, 0.5)
    })

    # Priors for Bayesian estimation
    priors: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'S0': (0.9, 0.1),  # (mean, std)
        'lambda_p': (0.5, 0.5),
        'sigma_i': (0.3, 0.2),
        'epsilon_r': (0.1, 0.05)
    })

    def __post_init__(self):
        """Validate parameters."""
        assert self.lambda_p > 0, "λ must be positive"
        assert 0 <= self.sigma_i <= 1, "σ must be in [0, 1]"
        assert self.epsilon_r >= 0, "ε must be non-negative"
        assert 0 < self.S0 <= 1, "S0 must be in (0, 1]"

    def to_array(self) -> np.ndarray:
        """Convert to array for optimization."""
        return np.array([self.S0, self.lambda_p, self.sigma_i, self.epsilon_r])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "ModelParameters":
        """Create from array."""
        return cls(S0=arr[0], lambda_p=arr[1], sigma_i=arr[2], epsilon_r=arr[3])


@dataclass
class ModelData:
    """Data for model fitting."""
    time: np.ndarray  # Time points
    pressure: np.ndarray  # P(t) - Pressure at each time
    intervention: np.ndarray  # I(t) - Intervention indicator (0 or 1)
    recovery: np.ndarray  # R(t) - Recovery signal
    safety: np.ndarray  # S(t) - Observed safety level

    def __post_init__(self):
        """Validate data consistency."""
        n = len(self.time)
        assert len(self.pressure) == n, "Pressure array size mismatch"
        assert len(self.intervention) == n, "Intervention array size mismatch"
        assert len(self.recovery) == n, "Recovery array size mismatch"
        assert len(self.safety) == n, "Safety array size mismatch"

        # Normalize pressure to [0, 1] if needed
        if self.pressure.max() > 1:
            self.pressure = self.pressure / self.pressure.max()


class OverrideCascadeModel:
    """Mathematical model for override cascade dynamics with identifiability."""

    def __init__(self, params: Optional[ModelParameters] = None):
        """Initialize model with parameters."""
        self.params = params or ModelParameters()
        self.fit_history = []
        self.diagnostics = {}

    def safety_function(
        self,
        t: np.ndarray,
        P: np.ndarray,
        I: np.ndarray,
        R: np.ndarray,
        params: Optional[ModelParameters] = None
    ) -> np.ndarray:
        """
        Calculate safety level S(t).

        S(t) = S0 * exp(-λ * P(t)) * (1 - σ * I(t)) + ε * R(t)

        Args:
            t: Time points
            P: Pressure values
            I: Intervention indicators
            R: Recovery signals
            params: Model parameters (uses self.params if None)

        Returns:
            Safety levels at each time point
        """
        if params is None:
            params = self.params

        S = (params.S0 *
             np.exp(-params.lambda_p * P) *
             (1 - params.sigma_i * I) +
             params.epsilon_r * R)

        return np.clip(S, 0, 1)

    def negative_log_likelihood(
        self,
        params_array: np.ndarray,
        data: ModelData,
        noise_std: float = 0.05
    ) -> float:
        """
        Negative log-likelihood for parameter estimation.

        Args:
            params_array: Parameter values as array
            data: Observed data
            noise_std: Standard deviation of observation noise

        Returns:
            Negative log-likelihood
        """
        params = ModelParameters.from_array(params_array)

        # Check bounds
        for i, (key, (low, high)) in enumerate(params.bounds.items()):
            if not low <= params_array[i] <= high:
                return np.inf

        # Calculate predicted safety
        S_pred = self.safety_function(
            data.time, data.pressure, data.intervention,
            data.recovery, params
        )

        # Calculate likelihood (assuming Gaussian noise)
        residuals = data.safety - S_pred
        nll = -np.sum(norm.logpdf(residuals, 0, noise_std))

        # Add prior penalties (MAP estimation)
        for i, key in enumerate(['S0', 'lambda_p', 'sigma_i', 'epsilon_r']):
            prior_mean, prior_std = params.priors[key]
            nll -= norm.logpdf(params_array[i], prior_mean, prior_std)

        return nll

    def fit(
        self,
        data: ModelData,
        method: str = "differential_evolution",
        noise_std: float = 0.05,
        n_restarts: int = 5
    ) -> ModelParameters:
        """
        Fit model parameters to data.

        Args:
            data: Observed data
            method: Optimization method ('differential_evolution' or 'l-bfgs-b')
            noise_std: Assumed noise standard deviation
            n_restarts: Number of random restarts for L-BFGS-B

        Returns:
            Fitted parameters
        """
        bounds = list(self.params.bounds.values())

        if method == "differential_evolution":
            # Global optimization
            result = differential_evolution(
                lambda x: self.negative_log_likelihood(x, data, noise_std),
                bounds,
                seed=42,
                maxiter=1000,
                popsize=15,
                tol=1e-7
            )
            best_params = result.x
            best_loss = result.fun

        elif method == "l-bfgs-b":
            # Multiple random starts with L-BFGS-B
            best_loss = np.inf
            best_params = None

            for _ in range(n_restarts):
                # Random initialization within bounds
                x0 = np.array([
                    np.random.uniform(low, high)
                    for low, high in bounds
                ])

                result = minimize(
                    lambda x: self.negative_log_likelihood(x, data, noise_std),
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds
                )

                if result.fun < best_loss:
                    best_loss = result.fun
                    best_params = result.x

        else:
            raise ValueError(f"Unknown method: {method}")

        # Update parameters
        self.params = ModelParameters.from_array(best_params)

        # Store fit history
        self.fit_history.append({
            'params': self.params,
            'loss': best_loss,
            'method': method
        })

        return self.params

    def compute_diagnostics(self, data: ModelData) -> Dict[str, Any]:
        """
        Compute model diagnostics and goodness of fit.

        Args:
            data: Observed data

        Returns:
            Dictionary of diagnostic metrics
        """
        # Predictions
        S_pred = self.safety_function(
            data.time, data.pressure, data.intervention,
            data.recovery, self.params
        )

        # Residuals
        residuals = data.safety - S_pred

        # R-squared
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((data.safety - np.mean(data.safety))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # AIC and BIC
        n = len(data.time)
        k = 4  # Number of parameters
        log_likelihood = -self.negative_log_likelihood(
            self.params.to_array(), data, noise_std=np.std(residuals)
        )
        aic = 2 * k - 2 * log_likelihood
        bic = k * np.log(n) - 2 * log_likelihood

        # Durbin-Watson statistic (autocorrelation)
        dw = np.sum(np.diff(residuals)**2) / ss_res if ss_res > 0 else 2

        # Normality test (Shapiro-Wilk)
        from scipy.stats import shapiro
        if len(residuals) >= 3:
            shapiro_stat, shapiro_p = shapiro(residuals)
        else:
            shapiro_stat, shapiro_p = np.nan, np.nan

        self.diagnostics = {
            'r_squared': r_squared,
            'rmse': np.sqrt(np.mean(residuals**2)),
            'mae': np.mean(np.abs(residuals)),
            'aic': aic,
            'bic': bic,
            'durbin_watson': dw,
            'shapiro_stat': shapiro_stat,
            'shapiro_p': shapiro_p,
            'residual_std': np.std(residuals),
            'residual_mean': np.mean(residuals)
        }

        return self.diagnostics

    def sensitivity_analysis(
        self,
        data: ModelData,
        param_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        n_samples: int = 100
    ) -> Dict[str, np.ndarray]:
        """
        Perform sensitivity analysis on model parameters.

        Args:
            data: Input data for evaluation
            param_ranges: Parameter ranges to test (uses bounds if None)
            n_samples: Number of samples per parameter

        Returns:
            Sensitivity metrics for each parameter
        """
        if param_ranges is None:
            param_ranges = self.params.bounds

        base_prediction = self.safety_function(
            data.time, data.pressure, data.intervention,
            data.recovery, self.params
        )

        sensitivity = {}

        for param_name in ['S0', 'lambda_p', 'sigma_i', 'epsilon_r']:
            low, high = param_ranges[param_name]
            param_values = np.linspace(low, high, n_samples)

            predictions = []
            for value in param_values:
                # Create modified parameters
                params_dict = {
                    'S0': self.params.S0,
                    'lambda_p': self.params.lambda_p,
                    'sigma_i': self.params.sigma_i,
                    'epsilon_r': self.params.epsilon_r
                }
                params_dict[param_name] = value
                modified_params = ModelParameters(**params_dict)

                # Calculate prediction
                pred = self.safety_function(
                    data.time, data.pressure, data.intervention,
                    data.recovery, modified_params
                )
                predictions.append(pred)

            predictions = np.array(predictions)

            # Calculate sensitivity metrics
            sensitivity[param_name] = {
                'values': param_values,
                'predictions': predictions,
                'mean_effect': np.mean(np.std(predictions, axis=0)),
                'max_effect': np.max(np.abs(predictions - base_prediction))
            }

        return sensitivity

    def plot_diagnostics(
        self,
        data: ModelData,
        save_path: Optional[Path] = None
    ) -> None:
        """Create diagnostic plots."""
        # Compute diagnostics first
        self.compute_diagnostics(data)

        # Predictions
        S_pred = self.safety_function(
            data.time, data.pressure, data.intervention,
            data.recovery, self.params
        )
        residuals = data.safety - S_pred

        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 1. Fitted vs Observed
        ax = axes[0, 0]
        ax.scatter(data.time, data.safety, label='Observed', alpha=0.6)
        ax.plot(data.time, S_pred, 'r-', label='Fitted', linewidth=2)
        ax.set_xlabel('Time')
        ax.set_ylabel('Safety Level')
        ax.set_title(f'Model Fit (R² = {self.diagnostics["r_squared"]:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Residuals vs Time
        ax = axes[0, 1]
        ax.scatter(data.time, residuals, alpha=0.6)
        ax.axhline(0, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time')
        ax.set_ylabel('Residuals')
        ax.set_title(f'Residuals (DW = {self.diagnostics["durbin_watson"]:.2f})')
        ax.grid(True, alpha=0.3)

        # 3. Q-Q Plot
        ax = axes[0, 2]
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title(f'Q-Q Plot (Shapiro p = {self.diagnostics["shapiro_p"]:.3f})')
        ax.grid(True, alpha=0.3)

        # 4. Residuals Histogram
        ax = axes[1, 0]
        ax.hist(residuals, bins=20, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Residuals')
        ax.set_ylabel('Frequency')
        ax.set_title('Residual Distribution')
        ax.grid(True, alpha=0.3)

        # 5. Partial Dependence
        ax = axes[1, 1]
        sensitivity = self.sensitivity_analysis(data, n_samples=50)
        for param_name in ['lambda_p', 'sigma_i']:
            values = sensitivity[param_name]['values']
            effects = sensitivity[param_name]['mean_effect']
            ax.plot(values, [effects] * len(values), label=param_name)
        ax.set_xlabel('Parameter Value')
        ax.set_ylabel('Mean Effect')
        ax.set_title('Parameter Sensitivity')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 6. Component Contributions
        ax = axes[1, 2]
        components = {
            'Pressure': np.mean(np.exp(-self.params.lambda_p * data.pressure)),
            'Intervention': np.mean(self.params.sigma_i * data.intervention),
            'Recovery': np.mean(self.params.epsilon_r * data.recovery)
        }
        ax.bar(components.keys(), components.values())
        ax.set_ylabel('Mean Contribution')
        ax.set_title('Component Contributions')
        ax.grid(True, alpha=0.3)

        plt.suptitle('Model Diagnostics', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved diagnostics to {save_path}")

        plt.show()

    def generate_report(self, data: ModelData) -> str:
        """Generate text report of model fit and diagnostics."""
        self.compute_diagnostics(data)

        report = []
        report.append("=" * 60)
        report.append("OVERRIDE CASCADE MODEL REPORT")
        report.append("=" * 60)

        report.append("\nFITTED PARAMETERS:")
        report.append(f"  S0 (Initial Safety):     {self.params.S0:.4f}")
        report.append(f"  λ (Pressure Decay):      {self.params.lambda_p:.4f}")
        report.append(f"  σ (Intervention Effect): {self.params.sigma_i:.4f}")
        report.append(f"  ε (Recovery Rate):       {self.params.epsilon_r:.4f}")

        report.append("\nGOODNESS OF FIT:")
        report.append(f"  R-squared:     {self.diagnostics['r_squared']:.4f}")
        report.append(f"  RMSE:          {self.diagnostics['rmse']:.4f}")
        report.append(f"  MAE:           {self.diagnostics['mae']:.4f}")
        report.append(f"  AIC:           {self.diagnostics['aic']:.2f}")
        report.append(f"  BIC:           {self.diagnostics['bic']:.2f}")

        report.append("\nRESIDUAL ANALYSIS:")
        report.append(f"  Mean:          {self.diagnostics['residual_mean']:.4f}")
        report.append(f"  Std Dev:       {self.diagnostics['residual_std']:.4f}")
        report.append(f"  Durbin-Watson: {self.diagnostics['durbin_watson']:.4f}")
        report.append(f"  Shapiro p-val: {self.diagnostics['shapiro_p']:.4f}")

        # Interpretation
        report.append("\nINTERPRETATION:")

        if self.diagnostics['r_squared'] > 0.8:
            report.append("  ✓ Model fits data well (R² > 0.8)")
        else:
            report.append("  ⚠ Model fit could be improved (R² < 0.8)")

        if 1.5 < self.diagnostics['durbin_watson'] < 2.5:
            report.append("  ✓ No significant autocorrelation detected")
        else:
            report.append("  ⚠ Possible autocorrelation in residuals")

        if self.diagnostics['shapiro_p'] > 0.05:
            report.append("  ✓ Residuals appear normally distributed")
        else:
            report.append("  ⚠ Residuals may not be normally distributed")

        report.append("=" * 60)

        return "\n".join(report)


def generate_synthetic_scenario(n_points: int = 100, seed: int = 42) -> ModelData:
    """Generate synthetic data for testing."""
    np.random.seed(seed)

    # Time points
    time = np.linspace(0, 10, n_points)

    # Pressure: increases then decreases
    pressure = np.sin(time / 2) * 0.5 + 0.5

    # Intervention: applied at high pressure points
    intervention = (pressure > 0.7).astype(float)

    # Recovery: gradual increase
    recovery = np.clip(time / 10, 0, 1)

    # True parameters
    true_params = ModelParameters(S0=0.95, lambda_p=0.8, sigma_i=0.4, epsilon_r=0.15)

    # Generate safety with noise
    model = OverrideCascadeModel(true_params)
    safety_true = model.safety_function(time, pressure, intervention, recovery)
    safety = safety_true + np.random.normal(0, 0.03, n_points)
    safety = np.clip(safety, 0, 1)

    return ModelData(
        time=time,
        pressure=pressure,
        intervention=intervention,
        recovery=recovery,
        safety=safety
    )


if __name__ == "__main__":
    # Generate synthetic data
    data = generate_synthetic_scenario(100)

    # Create and fit model
    model = OverrideCascadeModel()

    print("Fitting model...")
    fitted_params = model.fit(data, method="differential_evolution")

    print("\nFitted Parameters:")
    print(f"  S0: {fitted_params.S0:.4f}")
    print(f"  λ:  {fitted_params.lambda_p:.4f}")
    print(f"  σ:  {fitted_params.sigma_i:.4f}")
    print(f"  ε:  {fitted_params.epsilon_r:.4f}")

    # Generate report
    print("\n" + model.generate_report(data))

    # Plot diagnostics
    model.plot_diagnostics(data, save_path=Path("runs/model_diagnostics.png"))