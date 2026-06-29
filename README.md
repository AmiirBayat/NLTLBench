# NLTLBench

This repository contains the code and benchmark used in the paper:

**End-to-End Abstraction-Based Control with LLM-Enhanced NL-to-LTL Translation**

**Authors:** Amir Bayat\*, Necmiye Ozay, Alessandro Abate, Raphaël M. Jungers

\*ICTEAM, UCLouvain, Louvain-la-Neuve, Belgium  
EECS, University of Michigan, Ann Arbor, Michigan, USA  
Department of Computer Science, University of Oxford, Oxford, UK

**Video:** https://youtu.be/X5sc8U-o7RI

---

## Benchmark

The benchmark is available in:

```text
benchmark/DatasetWithNaturalNL_plus_simplified.json
```

This file contains the generated LTL formulas, their verbatim back-translations, LLM-generated paraphrases, and structural metadata used in the experiments.

---

## Repository Structure

The main files are:

- `main.jl`  
  Generates candidate LTL formulas.

- `BackTranslation.jl`  
  Translates LTL formulas into verbatim natural-language descriptions using rule-based back-translation.

- `Paraphrase_LLMName.jl`  
  Paraphrases the verbatim descriptions using the corresponding LLM. These files are used to generate linguistically diverse NL realizations of the same LTL specification.

- `LLM_performnce.jl`  
  Evaluates the performance of LLMs on the benchmark by translating NL descriptions back into LTL formulas and checking semantic equivalence with the ground-truth formulas.

- `ResultplusNesting.jl`  
  Analyzes and plots the experimental results, including performance trends with respect to formula complexity and nesting/temporal-depth measures.

---

## Evaluating LLMs

`LLM_performnce.jl` is used to evaluate LLM translation performance on the benchmark.

To run this file, create a local `.env` file and add your own API key for the corresponding LLM provider. For example:

```text
OPENAI_API_KEY=your_api_key_here
ANTHROPIC_API_KEY=your_api_key_here
MISTRAL_API_KEY=your_api_key_here
DEEPSEEK_API_KEY=your_api_key_here
GEMINI_API_KEY=your_api_key_here
```

Do **not** commit the `.env` file to GitHub.

The evaluation script translates each NL input into an LTL formula and checks whether the predicted formula is semantically equivalent to the ground-truth LTL formula.

---

## Result Analysis

`ResultplusNesting.jl` is used to study and plot the results. In particular, it can be used to analyze how LLM performance changes with respect to:

- formula size,
- temporal depth,
- nesting depth,
- Büchi automaton size,
- NL input length.

---

## Notes

For details on the benchmark generation procedure and experimental setup, please refer to the paper.