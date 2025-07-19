# Adversarial Phishing Defender Framework

A unified repository for implementing and evaluating an adversarial training loop for robust phishing email detection. This project implements a game-theoretic approach where attackers and defenders continuously adapt, producing detectors resilient to evasive phishing strategies.

---

## Overview

This document provides a comprehensive overview of the adversarial phishing email detection system. It uses reinforcement learning (RL)–based attackers to generate adversarial phishing examples, transformer-based defenders for classification, and a game-theoretic controller to orchestrate multi-round adversarial training.

### Key Components

* **Data Pipeline**: Preprocessing and tokenization of real and synthetic email datasets.
* **Attack Generation**: Template-based and perturbation-driven adversarial email creation.
* **Attacker Models**: RLAttacker (policy network + Q-learning) and AdaptiveAttacker (rule-based adaptation).
* **Defender Models**: TransformerDefender (DistilBERT classifier with temperature scaling).
* **Game-Theoretic Planner**: Chooses mixed strategies via Nash equilibria over attack strategies.
* **Adversarial Controller**: Manages iterative training rounds, convergence detection, and payoff updates.

For an interactive walkthrough, see the demonstration notebook:
[DeepWiki Demo](https://deepwiki.com/aayushakumar/adversarial-phishing-defender)

---

## System Architecture

The system adopts a modular, multi-layered architecture:

```text
+-----------------+      +----------------+      +----------------------+    +----------------+
|   Data Layer    | ---> | Attack System  | ---> | Game-Theoretic       | -> |  Defender      |
| (EmailDataset,  |      | (TemplateGen,  |      |  Planner &           |    | (Transformer,  |
|  preprocess)    |      |  Perturbation) |      |  AdversarialCtrl)    |    |  Calibration)   |
+-----------------+      +----------------+      +----------------------+    +----------------+
```

* **Data Layer**: Loads, cleans, and tokenizes email data.
* **Attack System**: Generates adversarial emails via templates and perturbations.
* **Game-Theoretic Planner**: Computes strategy mixtures to challenge the defender.
* **Adversarial Controller**: Orchestrates training rounds, updates both attacker and defender.
* **Defender**: Classifies and calibrates outputs to produce reliable probabilities.

---

## Core Component Mapping

| System Function              | Code Class/Method                                  |
| ---------------------------- | -------------------------------------------------- |
| Template Generation          | `TemplateGenerator.generate()`                     |
| Text Perturbation            | `PerturbationEngine.perturb()`                     |
| RL Attack Generation         | `RLAttacker.generate_attack()`                     |
| Rule-Based Attack Adaptation | `AdaptiveAttacker.generate_attack()`               |
| Email Classification         | `TransformerDefender.predict()`                    |
| Calibration                  | `TransformerDefender.calibrate_with_temperature()` |
| Strategy Selection           | `GameTheoreticPlanner.choose_strategy()`           |
| Training Loop Control        | `AdversarialController.run_round()`                |
| Full Experiment Execution    | `run_full_experiment()`                            |

---

## Core Components

### Data Pipeline

Processes raw email data into model-ready inputs.

| Component                      | Purpose                    | Key Methods                          |
| ------------------------------ | -------------------------- | ------------------------------------ |
| `download_preprocessed_data()` | Generate balanced datasets | Downloads and splits real/synthetic  |
| `EmailDataset`                 | PyTorch dataset wrapper    | Tokenizes text for transformer input |
| `preprocess_emails()`          | Clean and normalize text   | Strips headers, normalizes URLs      |

### Attack Generation System

Creates adversarial phishing emails.

| Component            | Strategy                  | Techniques                                                                                        |
| -------------------- | ------------------------- | ------------------------------------------------------------------------------------------------- |
| `TemplateGenerator`  | Template-based generation | Banking, HR, Government templates                                                                 |
| `PerturbationEngine` | Text perturbations        | `_swap_chars()`, `_add_char()`, `_synonym_replacement()`, `_change_case()`, `_hide_url_in_text()` |

### Attacker Models

| Model              | Approach                        | Key Methods                                               |
| ------------------ | ------------------------------- | --------------------------------------------------------- |
| `RLAttacker`       | Reinforcement Learning (Deep Q) | `generate_attack()`, `update_policy()`, `select_action()` |
| `AdaptiveAttacker` | Rule-Based Adaptive             | `generate_attack()`, `update_model()`                     |

### Defender Architecture

| Model                 | Description                            | Key Methods                                            |
| --------------------- | -------------------------------------- | ------------------------------------------------------ |
| `TransformerDefender` | DistilBERT classifier with calibration | `train()`, `predict()`, `calibrate_with_temperature()` |

### Game-Theoretic Planning

Selects attack strategies using Nash equilibria.

| Component              | Role                       | Key Methods                                                       |
| ---------------------- | -------------------------- | ----------------------------------------------------------------- |
| `GameTheoreticPlanner` | Mixed strategy computation | `choose_strategy()`, `update_payoffs()`, `solve_mixed_strategy()` |

### Adversarial Training Process

Managed by `AdversarialController`:

1. **choose\_strategy()** → select attack mix
2. **generate\_attack(n\_samples, strategy)** → craft emails
3. **predict(adversarial\_emails)** → defender evaluates
4. **\_calculate\_metrics()** → compute robust accuracy
5. **update\_payoffs(results)** → adjust game-theoretic payoffs
6. **update\_model(results, retrain=True)** → retrain attacker/defender
7. **Convergence Check** → stop when accuracy stabilizes

### Full Experiment Runner

`run_full_experiment()` handles multi-seed trials, comprehensive evaluation, and plotting of results.

---

## Key Algorithms and Techniques

### Temperature Scaling Calibration

Performs grid search over temperature values to minimize Expected Calibration Error (ECE):

```python
# In TransformerDefender.calibrate_with_temperature
temperatures = np.linspace(0.5, 3.0, 26)
best_temp = min(
    temperatures,
    key=lambda T: compute_ece(scale_logits(logits, T), labels)
)
```

### Reinforcement Learning Attack Generation

* **State**: Email embeddings
* **Actions**: Perturbation functions
* **Reward**: Evasion success − similarity penalty
* **Updates**: Deep Q-learning with experience replay buffer

### Multi-Trial Experimental Framework

Executes `run_full_experiment()` with:

* Multiple random seeds
* Diverse attack perturbations
* Robustness and calibration metrics
* Reliability diagrams and ECE computation

---

## System Integration and Extensibility

The modular design allows:

* **New Attacker Strategies**: Plug into `GameTheoreticPlanner` and `AdversarialController`.
* **Alternate Defenders**: Replace `TransformerDefender` with custom architectures.
* **Custom Metrics**: Extend `evaluate.py` for additional robustness measures.

---

## Usage Guide

### Relevant Source Files

This section provides comprehensive instructions for using the adversarial phishing defender system. It covers basic setup, component configuration, running experiments, and interpreting results. The guide is designed for researchers and developers who want to reproduce the experiments or extend the system.

For technical details about individual components, see **Core Components**. For architectural insights, see **System Architecture**. For API documentation, see **API Reference**.

### Quick Start

The fastest way to get started is via the pre-configured demonstration script in `ProjectDemo.ipynb`:

```python
# Basic usage with synthetic data
train_df, val_df, test_df = download_preprocessed_data(n_samples=1000)
results = run_full_experiment(
    n_rounds=5,
    samples_per_round=100,
    random_seeds=1
)
```

This will:

* Generate 1,000 synthetic emails (500 legitimate, 500 phishing).
* Run a complete adversarial training experiment.
* Save results and visualizations to the `results/` directory.

### System Requirements

* Python 3.8+
* CUDA-compatible GPU (recommended)
* ≥ 8 GB RAM
* Dependencies: PyTorch, Transformers, scikit-learn, NLTK

### Basic Experiment Flow

Execute the full adversarial training protocol:

```python
# In ProjectDemo.ipynb, lines 2766–2961
run_full_experiment(
    n_rounds=5,
    samples_per_round=100,
    random_seeds=5
)
```

### Data Pipeline Usage

#### Synthetic Data Generation

For quick experiments or unit testing:

```python
# Generate a balanced synthetic dataset
train_df, val_df, test_df = download_preprocessed_data(n_samples=1000)
```

Creates realistic phishing and legitimate email templates with controllable parameters.

#### Real Data Processing

To preprocess real-world datasets (Enron + PhishTank):

```python
train_df, val_df, test_df = load_real_data()
```

Automatically handles:

* Enron corpus download and parsing.
* PhishTank URL integration.
* Email body extraction and normalization.
* Duplicate removal and dataset balancing.

##### Data Processing Pipeline

1. `download_datasets()`
2. `preprocess_emails()`
3. `extract_email_body()`
4. `normalize_urls()`
5. `remove_duplicates()`
6. Train/val/test splits
7. `process_phishtank_to_emails()`

### Component Configuration

#### Defender Configuration

```python
# Basic defender setup
defender = TransformerDefender(model_name='distilbert-base-uncased')
# Advanced configuration
defender = TransformerDefender(
    model_name='bert-base-uncased',
    batch_size=16,
    max_length=512,
    learning_rate=2e-5
)
# Train
def training_metrics = defender.train(train_df, val_df, epochs=3)
# Calibrate
calibration_metrics = defender.calibrate_with_temperature(val_df)
```

#### Attacker Configuration

```python
# Reinforcement Learning Attacker
pert_engine = PerturbationEngine()
rl_attacker = RLAttacker(pert_engine)
# Adaptive Attacker
template_gen = TemplateGenerator()
adaptive_attacker = AdaptiveAttacker(template_gen, pert_engine)
```

#### Attack Strategy Configuration

```python
planner = GameTheoreticPlanner(
    attack_strategies=['template', 'perturbation', 'adaptive', 'mixed']
)
```

Component Initialization Workflow:

```text
TemplateGenerator → PerturbationEngine → RLAttacker → AdaptiveAttacker → TransformerDefender
→ defender.train() → defender.calibrate_with_temperature() → GameTheoreticPlanner → AdversarialController
```


### Running Experiments

#### Basic Adversarial Training

```python
controller = AdversarialController(attacker, defender, planner)
round_history = controller.run_adversarial_loop(
    n_rounds=5,
    samples_per_round=100,
    adapt_attacker=True,
    adapt_defender=True
)
```

#### Single Round Execution

```python
round_results, converged = controller.run_round(n_samples=50)
if not converged:
    update_metrics = controller.adapt_and_update(
        round_results,
        adapt_attacker=True,
        adapt_defender=True
    )
```

#### Multi-Trial Experiments

```python
all_results = run_full_experiment(
    n_rounds=5,
    samples_per_round=100,
    random_seeds=5
)
```

### Advanced Usage Scenarios

* **Custom Attack Generation**: Extend `TemplateGenerator` with custom templates.
* **Custom Perturbations**: Subclass `PerturbationEngine` to implement new methods.
* **Evaluation & Analysis**: Use `plot_metrics_over_rounds()`, `visualize_calibration()`, `generate_report()`.
* **Advanced Workflow**: Integrate with CI pipelines and cloud environments.

### Configuration Parameters

| Parameter                                                 | Default                   | Description                           |
| --------------------------------------------------------- | ------------------------- | ------------------------------------- |
| `n_rounds`                                                | 5                         | Number of adversarial training rounds |
| `samples_per_round`                                       | 100                       | Attack samples generated per round    |
| `random_seeds`                                            | 5                         | Number of experimental trials         |
| `epochs`                                                  | 3                         | Training epochs for defender          |
| `batch_size`                                              | 8                         | Batch size for model training         |
| `learning_rate`                                           | 2e-5                      | Learning rate for transformer         |
| **Attacker Params**                                       |                           |                                       |
| `max_steps`                                               | 50                        | Maximum RL attack steps               |
| `exploration_rate`                                        | 0.1 (ε-greedy)            | RL exploration rate                   |
| `intensity`                                               | 0.1–0.5                   | Perturbation intensity range          |
| `gamma`                                                   | 0.99                      | RL discount factor                    |
| `buffer_size`                                             | 1000                      | Experience replay buffer size         |
| **Defender Params**                                       |                           |                                       |
| `model_name`                                              | 'distilbert-base-uncased' | Transformer model                     |
| `max_length`                                              | 512                       | Maximum sequence length               |
| `threshold`                                               | 0.5                       | Classification threshold              |
| `calibration_temp`                                        | 1.0                       | Temperature scaling parameter         |
| `weight_decay`                                            | 0.01                      | Regularization parameter              |

### Output Analysis

* **Round History:** `controller.round_history`
* **Metrics Files:** `results/round_N.json`
* **Visualizations:** PNG files in `results/`
* **Final Report:** `results/final_report.md`

Key Metrics:

* `detection_rate`
* `evasion_rate`
* `accuracy`
* `f1_score`
* `ece`
* `improvement`

#### Interpreting Convergence

The system flags convergence when the detection rate changes <2% across 2 consecutive rounds.

```python
if abs(current_acc - previous_acc) < 0.02:
    convergence_counter += 1
    if convergence_counter >= 2:
        print("Converged!")
```


### Troubleshooting

**Common Issues**:

* **GPU Memory Errors:** Reduce `batch_size` or use smaller model.
* **Slow Training:** Switch to `AdaptiveAttacker` or lower `samples_per_round`, `n_rounds`.
* **Poor Convergence:** Increase `learning_rate`, inspect data balance.
* **Import Errors:** Ensure dependencies installed; run `nltk.download('punkt')`, `nltk.download('wordnet')`.

**Debugging Mode**:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
try:
    results = run_full_experiment(n_rounds=1, samples_per_round=10)
except Exception as e:
    traceback.print_exc()
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
