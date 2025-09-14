"""Artifact manifest system for tracking experiment outputs.

Provides checksums, metadata, and versioning for reproducibility.
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import platform
import sys
import git


@dataclass
class Artifact:
    """Represents a single artifact."""
    path: str
    checksum: str  # SHA-256
    size_bytes: int
    created_at: float
    artifact_type: str  # "data", "model", "plot", "report"
    metadata: Dict[str, Any]


@dataclass
class Manifest:
    """Complete manifest for an experiment run."""
    run_id: str
    timestamp: float
    git_commit: Optional[str]
    python_version: str
    platform_info: str
    seed: int
    artifacts: List[Artifact]
    config: Dict[str, Any]
    metrics: Dict[str, Any]


class ArtifactManager:
    """Manage artifacts and manifests for experiments."""

    def __init__(self, base_dir: Path = Path("runs")):
        """Initialize artifact manager."""
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def compute_checksum(self, file_path: Path) -> str:
        """Compute SHA-256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def create_artifact(self, file_path: Path, artifact_type: str, metadata: Dict[str, Any] = None) -> Artifact:
        """Create an artifact entry for a file."""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        return Artifact(
            path=str(file_path),
            checksum=self.compute_checksum(file_path),
            size_bytes=file_path.stat().st_size,
            created_at=file_path.stat().st_mtime,
            artifact_type=artifact_type,
            metadata=metadata or {}
        )

    def get_git_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            repo = git.Repo(search_parent_directories=True)
            return repo.head.object.hexsha
        except:
            return None

    def create_manifest(
        self,
        run_id: str,
        seed: int,
        config: Dict[str, Any],
        metrics: Dict[str, Any],
        artifact_paths: List[Tuple[Path, str, Dict[str, Any]]]
    ) -> Manifest:
        """Create a complete manifest for an experiment run.

        Args:
            run_id: Unique identifier for the run
            seed: Random seed used
            config: Configuration dictionary
            metrics: Metrics dictionary
            artifact_paths: List of (path, type, metadata) tuples
        """
        artifacts = []
        for path, artifact_type, metadata in artifact_paths:
            if path.exists():
                artifact = self.create_artifact(path, artifact_type, metadata)
                artifacts.append(artifact)

        manifest = Manifest(
            run_id=run_id,
            timestamp=time.time(),
            git_commit=self.get_git_commit(),
            python_version=sys.version,
            platform_info=platform.platform(),
            seed=seed,
            artifacts=artifacts,
            config=config,
            metrics=metrics
        )

        return manifest

    def save_manifest(self, manifest: Manifest, output_path: Path) -> None:
        """Save manifest to JSON file."""
        manifest_dict = asdict(manifest)

        # Make paths relative to manifest location
        manifest_dir = output_path.parent
        for artifact in manifest_dict['artifacts']:
            try:
                artifact['relative_path'] = str(
                    Path(artifact['path']).relative_to(manifest_dir)
                )
            except ValueError:
                # Path is not relative to manifest dir
                artifact['relative_path'] = artifact['path']

        with open(output_path, 'w') as f:
            json.dump(manifest_dict, f, indent=2)

    def load_manifest(self, manifest_path: Path) -> Manifest:
        """Load manifest from JSON file."""
        with open(manifest_path) as f:
            data = json.load(f)

        # Convert artifacts back to Artifact objects
        data['artifacts'] = [Artifact(**a) for a in data['artifacts']]

        return Manifest(**data)

    def verify_artifacts(self, manifest: Manifest) -> Dict[str, bool]:
        """Verify all artifacts in manifest still exist with correct checksums."""
        verification = {}

        for artifact in manifest.artifacts:
            path = Path(artifact.path)

            if not path.exists():
                verification[artifact.path] = False
                continue

            # Verify checksum
            current_checksum = self.compute_checksum(path)
            verification[artifact.path] = current_checksum == artifact.checksum

        return verification

    def create_run_directory(self, experiment_name: str) -> Path:
        """Create a new run directory with timestamp."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = self.base_dir / f"{experiment_name}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (run_dir / "data").mkdir(exist_ok=True)
        (run_dir / "models").mkdir(exist_ok=True)
        (run_dir / "plots").mkdir(exist_ok=True)
        (run_dir / "reports").mkdir(exist_ok=True)

        return run_dir

    def collect_run_artifacts(self, run_dir: Path) -> List[Tuple[Path, str, Dict[str, Any]]]:
        """Collect all artifacts from a run directory."""
        artifacts = []

        # Data files
        for data_file in (run_dir / "data").glob("*.json"):
            artifacts.append((data_file, "data", {"format": "json"}))

        for data_file in (run_dir / "data").glob("*.csv"):
            artifacts.append((data_file, "data", {"format": "csv"}))

        # Model files
        for model_file in (run_dir / "models").glob("*.pkl"):
            artifacts.append((model_file, "model", {"format": "pickle"}))

        for model_file in (run_dir / "models").glob("*.pt"):
            artifacts.append((model_file, "model", {"format": "pytorch"}))

        # Plots
        for plot_file in (run_dir / "plots").glob("*.png"):
            artifacts.append((plot_file, "plot", {"format": "png"}))

        for plot_file in (run_dir / "plots").glob("*.pdf"):
            artifacts.append((plot_file, "plot", {"format": "pdf"}))

        # Reports
        for report_file in (run_dir / "reports").glob("*.txt"):
            artifacts.append((report_file, "report", {"format": "text"}))

        for report_file in (run_dir / "reports").glob("*.md"):
            artifacts.append((report_file, "report", {"format": "markdown"}))

        return artifacts

    def generate_index(self, manifests_dir: Path) -> None:
        """Generate an index of all manifests."""
        index = {
            "manifests": [],
            "generated_at": time.time(),
            "total_runs": 0
        }

        for manifest_file in manifests_dir.glob("*/manifest.json"):
            try:
                manifest = self.load_manifest(manifest_file)
                index["manifests"].append({
                    "run_id": manifest.run_id,
                    "timestamp": manifest.timestamp,
                    "path": str(manifest_file),
                    "n_artifacts": len(manifest.artifacts),
                    "git_commit": manifest.git_commit
                })
            except Exception as e:
                print(f"Error loading {manifest_file}: {e}")

        index["total_runs"] = len(index["manifests"])

        # Sort by timestamp
        index["manifests"].sort(key=lambda x: x["timestamp"], reverse=True)

        # Save index
        with open(manifests_dir / "index.json", 'w') as f:
            json.dump(index, f, indent=2)


def create_experiment_manifest(
    experiment_name: str,
    seed: int,
    config: Dict[str, Any],
    results: Dict[str, Any]
) -> Path:
    """Convenience function to create manifest for an experiment."""
    manager = ArtifactManager()

    # Create run directory
    run_dir = manager.create_run_directory(experiment_name)

    # Save config
    config_path = run_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Save results
    results_path = run_dir / "data" / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Collect artifacts
    artifacts = manager.collect_run_artifacts(run_dir)
    artifacts.append((config_path, "config", {}))

    # Create manifest
    manifest = manager.create_manifest(
        run_id=f"{experiment_name}_{run_dir.name}",
        seed=seed,
        config=config,
        metrics=results.get("metrics", {}),
        artifact_paths=artifacts
    )

    # Save manifest
    manifest_path = run_dir / "manifest.json"
    manager.save_manifest(manifest, manifest_path)

    print(f"Manifest created: {manifest_path}")
    print(f"Run directory: {run_dir}")
    print(f"Artifacts tracked: {len(manifest.artifacts)}")

    return manifest_path


if __name__ == "__main__":
    # Example usage
    manager = ArtifactManager()

    # Create example manifest
    config = {
        "model": "test_model",
        "learning_rate": 0.001,
        "batch_size": 32
    }

    results = {
        "metrics": {
            "accuracy": 0.95,
            "loss": 0.05
        }
    }

    manifest_path = create_experiment_manifest(
        experiment_name="test_experiment",
        seed=42,
        config=config,
        results=results
    )

    # Verify artifacts
    manifest = manager.load_manifest(manifest_path)
    verification = manager.verify_artifacts(manifest)

    print("\nArtifact verification:")
    for path, valid in verification.items():
        status = "✓" if valid else "✗"
        print(f"  {status} {Path(path).name}")

    # Generate index
    manager.generate_index(manager.base_dir)
    print(f"\nIndex generated: {manager.base_dir / 'index.json'}")