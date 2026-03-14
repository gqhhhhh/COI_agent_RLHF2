# CoI-Select: Chain-of-Intention Based Synthetic Data Selection for Multi-Turn Goal-Oriented Dialogue Agents

> **Core thesis**: Synthetic data selected via user Chain-of-Intention (CoI) dual-layer filtering (instance + global) produces better training data than random sampling, and trains agents that perform better on multi-turn goal-oriented tasks.

## 1. Dataset Recommendation

**Primary dataset: PersuasionForGood** (Wang et al., 2019)

| Criterion | PersuasionForGood | MultiWOZ | TopDial |
|---|---|---|---|
| CoI suitability | ★★★★★ Clear user intention progression | ★★★ Slot-filling oriented | ★★★★ Topic-oriented |
| Success/fail signals | ★★★★★ Donation amount = clear outcome | ★★★ Task completion | ★★★ Dialogue quality |
| Goal-oriented | ★★★★★ Persuasion = goal-driven | ★★★★ Task-oriented | ★★★ Mixed |
| Data size | ~1,017 dialogues | ~10,000 | ~3,000 |
| Best for this project | ✅ **Main dataset** | Supplementary | Supplementary |

**Why PersuasionForGood?**
- Natural user intention progression: Inquiry → Concern → Positive → Action (or → Reject)
- Clear binary outcome (donated vs. not), perfect for success/fail analysis
- Rich conversational dynamics suitable for CoI modeling
- Manageable size for research validation

## 2. Project Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA FLOW OVERVIEW                           │
│                                                                 │
│  Raw Data ──→ Unified Schema ──→ Intent Labels ──→ CoI Graph   │
│                                                                 │
│  CoI Graph ──→ Route Consistency Scorer                         │
│                                                                 │
│  Simulator + Agent ──→ Synthetic Pool                           │
│                                                                 │
│  Synthetic Pool ──→ Instance Eval ──→ Global Eval               │
│                                                                 │
│  Evaluated Pool ──→ Selection (Random/Instance/CoI)             │
│                                                                 │
│  Selected Data ──→ Agent SFT ──→ Comparison Experiment          │
│                                                                 │
│  (Optional) ──→ Preference Pairs ──→ RM ──→ PPO/DPO            │
└─────────────────────────────────────────────────────────────────┘
```

## 3. Project Structure

```
coi-select/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── configs/
│   ├── project.yaml                   # Master configuration
│   └── taxonomy.yaml                  # Intent taxonomy (extensible)
├── data/
│   ├── raw/                           # Raw dataset files
│   ├── processed/                     # Unified schema JSONL
│   │   └── coi_graph/                 # CoI transition matrices and graph
│   ├── synthetic_pool/                # Generated synthetic dialogues
│   ├── selected/                      # Selected subsets per strategy
│   │   ├── random_k/
│   │   ├── instance_top_k/
│   │   └── coi_selected_k/
│   └── eval/                          # Experiment results
├── src/
│   ├── utils.py                       # Shared utilities
│   ├── data/
│   │   └── preprocess.py              # Phase 1: Data loading & preprocessing
│   ├── intent/
│   │   ├── taxonomy.py                # Dynamic taxonomy loader
│   │   ├── rule_labeler.py            # Phase 2A: Rule-based intent labeler
│   │   └── llm_labeler.py             # Phase 2B: LLM-based intent labeler
│   ├── graph/
│   │   ├── coi_graph.py               # Phase 2.2: CoI graph construction
│   │   └── route_consistency.py       # Phase 3: Route consistency scoring
│   ├── simulator/
│   │   ├── train.py                   # Phase 4: Simulator SFT training
│   │   └── rollout.py                 # Phase 5: Synthetic pool generation
│   ├── eval/
│   │   ├── instance_eval.py           # Phase 6.1: Instance-level metrics
│   │   └── global_eval.py             # Phase 6.2: Global-level metrics
│   ├── select/
│   │   └── selector.py                # Phase 7: Selection strategies
│   ├── agent/
│   │   └── train.py                   # Phase 8.1: Agent SFT training
│   ├── rm/
│   │   └── preference_builder.py      # Phase 8.2: Preference pair builder
│   └── experiments/
│       └── runner.py                  # Phase 9: Experiment evaluation
├── scripts/
│   ├── run_all.py                     # Run full pipeline
│   ├── run_phase1.py                  # Data preprocessing
│   ├── run_phase2.py                  # Intent labeling + CoI graph
│   ├── run_phase3.py                  # Route consistency validation
│   ├── run_phase5_7.py                # Pool generation + eval + selection
│   └── run_experiment.py              # Main experiment comparison
└── tests/
    └── test_pipeline.py               # 32 unit tests
```

## 4. User Intent Taxonomy (7-class CoI)

The **user-side Chain-of-Intention (CoI)** is the core analytical framework:

| Intent | ID | Description | Boundary Notes |
|---|---|---|---|
| **Inquiry** | 0 | Asks for information/clarification | Rhetorical doubt questions → Concern |
| **Positive** | 1 | Expresses interest/agreement/willingness | Polite 'okay' without sentiment → Neutral |
| **Concern** | 2 | Hesitation/doubt/worry (door still open) | Explicit refusal → Reject |
| **Reject** | 3 | Explicit refusal/decline | Mild pushback → Concern |
| **Neutral** | 4 | Small talk/acknowledgement | No clear intent direction |
| **Action** | 5 | Commits to goal action | 'Maybe I will' → Positive |
| **EndSuccess** | 6 | Goal verifiably completed | Positive sentiment alone ≠ EndSuccess |

Defined in `configs/taxonomy.yaml` (not hard-coded). To switch to 9-class, add entries to the YAML.

## 5. Phase-by-Phase Guide

### Phase 1: Data Preprocessing

**Goal**: Convert raw data to unified schema.

```bash
# Generate demo data (for quick testing)
python scripts/run_phase1.py --demo

# Use real PersuasionForGood data
python scripts/run_phase1.py --raw-dir data/raw/persuasionforgood
```

**Output**: `data/processed/{train,dev,test}.jsonl` with format:
```json
{
  "dialogue_id": "xxx",
  "dataset": "persuasionforgood",
  "split": "train",
  "meta": {"outcome": "success"},
  "turns": [
    {"role": "agent", "text": "..."},
    {"role": "user", "text": "..."}
  ]
}
```

### Phase 2: Intent Labeling + CoI Graph

**Goal**: Label user turns, build transition matrix and path lexicon.

```bash
python scripts/run_phase2.py
```

**Output**:
- `data/processed/coi_graph/intent_sequences.jsonl`
- `data/processed/coi_graph/coi_matrix_real.{npy,json}`
- `data/processed/coi_graph/valid_edges.json`
- `data/processed/coi_graph/path_ngrams.json`
- `data/processed/coi_graph/coi_heatmap.png`
- `data/processed/coi_graph/analysis_report.txt`

### Phase 3: Route Consistency Validation

**Goal**: Validate that route scoring differentiates valid vs. invalid paths.

```bash
python scripts/run_phase3.py
```

**4 Route Metrics**:
1. **Edge Validity** – fraction of edges found in real graph
2. **Path-k Validity** – fraction of 3-gram sub-paths in real lexicon
3. **Reachability** – path consistency with observed outcome
4. **Prefix Validity** – average validity over all prefix sub-paths

**Composite**: `route_score = a·edge + b·path3 + c·reach + d·prefix` (weights in config)

### Phases 5-7: Pool Generation → Evaluation → Selection

```bash
python scripts/run_phase5_7.py --num-dialogues 3000 --K 500
```

**Phase 5** generates synthetic dialogues using simulator + baseline agent.
**Phase 6** scores each dialogue on 5 instance metrics + 6 global metrics.
**Phase 7** applies three selection strategies:
- **Random-K**: Random sampling from pool
- **InstanceTopK**: Top K by instance quality score
- **CoI-Selected-K**: Two-stage (instance filter → global distribution optimization)

### Phase 8: Agent Training

```bash
# Trains agents for all three strategies
python -c "
from src.agent.train import train_all_agents
train_all_agents('data/selected')
"
```

### Phase 9: Main Experiment

```bash
python scripts/run_experiment.py
```

### Full Pipeline (All Phases)

```bash
# Quick run with demo data
python scripts/run_all.py --demo --num-dialogues 200 --K 50

# Full run (requires real data + GPU for training)
python scripts/run_all.py --num-dialogues 3000 --K 500
```

## 6. Installation

```bash
# Clone
git clone https://github.com/gqhhhhh/COI_agent_RLHF2.git
cd COI_agent_RLHF2

# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/ -v

# Quick validation with demo data
python scripts/run_all.py --demo --num-dialogues 200 --K 50
```

### Data Download (Optional)

To use real PersuasionForGood data:
1. Download from [PersuasionForGood repository](https://gitlab.com/ucdavisnlp/persuasionforgood)
2. Place CSV files in `data/raw/persuasionforgood/`
3. Run `python scripts/run_phase1.py`

## 7. Evaluation Metrics

### Data-Level Metrics

| Metric | Direction | Description |
|---|---|---|
| KL Divergence | ↓ | Real vs. synthetic CoI transition distribution |
| JS Divergence | ↓ | Symmetric version of KL |
| Intent Coverage | ↑ | Fraction of intents/transitions covered |
| Diversity | ↑ | Shannon entropy of intent distribution |
| Turn-length Match | ↑ | Distribution similarity to real data |
| Outcome Ratio Match | ↑ | Success/fail ratio similarity |

### Agent-Level Metrics (Multi-Turn)

| Metric | Direction | Description |
|---|---|---|
| Success Rate | ↑ | Goal completion rate |
| Avg Turns | — | Average dialogue length |
| Route Validity | ↑ | CoI path validity score |
| Prefix Validity | ↑ | Average prefix path validity |
| Result Consistency | ↑ | Path/outcome agreement |
| Recovery Rate | ↑ | Recovery after Concern/Reject |
| Efficient Success | ↑ | Success without excessive turns |

## 8. Ablation Experiments

The experiment framework supports these ablations:
1. **No route consistency** – Remove route scoring from instance evaluation
2. **No global KL/JS** – Remove distribution alignment from global optimization
3. **No diversity** – Remove diversity component
4. **Instance-only** – Skip global optimization stage

These prove that CoI structural constraints and distribution alignment (not just text quality) drive the improvement.

## 9. Configuration

All parameters are in `configs/project.yaml`:
- Dataset paths and splits
- Intent labeler settings
- CoI graph parameters (smoothing, n-gram sizes)
- Route consistency weights
- Simulator/agent model and training hyperparameters
- Selection parameters (K, M_ratio, MC iterations)

Intent taxonomy is in `configs/taxonomy.yaml` – fully extensible without code changes.

## 10. Priority Roadmap

If resources are limited, priority order:
1. ✅ Real data preprocessing
2. ✅ User CoI labeling
3. ✅ Real CoI graph
4. ✅ Route consistency
5. ✅ Synthetic pool
6. ✅ Random / Instance / CoI three-way selection
7. ✅ SFT comparison experiment
8. ✅ Preference pair builder
9. ⬜ Reward Model training
10. ⬜ PPO / DPO training

## License

This project is for research purposes.