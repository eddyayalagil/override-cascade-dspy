from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip() for line in fh if line.strip() and not line.startswith("#")
    ]

setup(
    name="override-cascade-dspy",
    version="0.3.0",
    author="EvalOps Research Team",
    author_email="info@evalops.dev",
    description="A DSPy-based framework for detecting, measuring, and preventing safety override cascades in LLM systems.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/evalops/override-cascade-dspy",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.11.0",
            "ruff>=0.1.0",
            "black>=22.0.0",
            "isort>=5.0.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
            "myst-parser>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "override-cascade=override_cascade_dspy.cli:main",
            "ocd=override_cascade_dspy.cli:main",
            "oc-demo=override_cascade_dspy.override_cascade.main:demo",
        ],
    },
)
