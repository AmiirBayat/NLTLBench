# NL2LTLBench

We synthesize pairs of **LTL formulas** and their corresponding **natural language descriptions** to construct a benchmark.

---

## Algorithm

**Input:**  
- `AP`: Set of atomic propositions  
  `AP = {prop_1, prop_2, ..., prop_n}`  
- `TO`: Set of temporal and Boolean operators  
  `TO = {F, G, X, U, |, &, ->, <->, !}`

**Output:**  
- A set of **unique, satisfiable LTL formulas** with correct syntax  
- Their corresponding **natural language descriptions** obtained via back-translation  

---

## Procedure

1. **Sampling of symbols**  
   Randomly select subsets:
   - `AP_1 ⊆ AP`
   - `TO_1 ⊆ TO`

2. **Formula generation (AST-based)**  
   Generate candidate LTL formulas by constructing **Abstract Syntax Trees (ASTs)** using elements from `AP_1` and `TO_1`.  
   - Ensure **syntactic correctness** 
   - Control:
     - maximum formula size (AST size)
     - temporal nesting depth  

3. **Filtering and validation**  
   For each generated formula `f`:

   - **Simplification**  
     Simplify `f` using the Spot toolbox to obtain a canonical form `f_s`  
     *(e.g., `f = prop_1 & G(prop_1) => f_s = G(prop_1)`)*
     
   - **Satisfiability check**  
     Verify that `f_s` is satisfiable  
     *(e.g., `F(prop_1) & G(!prop_1)` is unsatisfiable and thus discarded)*

   - **Redundancy check**  
     Reject `f_s` if it is:
     - syntactically duplicate, or
     - semantically equivalent to an existing formula in the dataset  

4. **Dataset construction**  
   Add all validated formulas to the dataset, along with metadata such as:
   - AST size
   - AST depth
   - temporal depth
   - operator statistics  

5. **Back-translation to natural language**  
   Generate a canonical natural language description for each formula using a **rule-based translation** applied to the AST structure.

---

## Summary

This pipeline ensures that the resulting dataset:
- contains **valid and non-trivial LTL formulas**
- avoids **syntactic and semantic redundancy**
- provides **aligned NL–LTL pairs**
- supports **controlled complexity and analysis**
# NL2LTLBench

This repository contains the code used to generate and evaluate **NLTLBench**, a benchmark of paired **natural-language (NL) descriptions** and **Linear Temporal Logic (LTL) formulas**.

The benchmark is available in:

```text
benchmark/DatasetWithNaturalNL_plus_simplified.json
```

This file contains the generated LTL formulas, their verbatim back-translations, LLM-generated paraphrases, and structural metadata used in the experiments.

---

## Repository Structure

The main files are:

- `main.jl`  
  Generates candidate LTL formulas and applies the filtering pipeline.

- `BackTranslation.jl`  
  Translates LTL formulas into verbatim natural-language descriptions using rule-based back-translation.

- `Paraphrase_LLMName.jl`  
  Paraphrases the verbatim descriptions using the corresponding LLM. These files are used to generate linguistically diverse NL realizations of the same LTL specification.

- `LLM_performnce.jl`  
  Evaluates the performance of LLMs on the benchmark by translating NL descriptions back into LTL formulas and checking semantic equivalence with the ground-truth formulas.

- `ResultplusNesting.jl`  
  Analyzes and plots the experimental results, including performance trends with respect to formula complexity and nesting/temporal-depth measures.

---

## Benchmark Generation Pipeline

The benchmark construction follows the steps below.

### 1. LTL Formula Generation

`main.jl` is used to generate candidate LTL formulas from a set of atomic propositions and temporal/Boolean operators.

The atomic propositions are of the form:

```text
prop_1, prop_2, ..., prop_n
```

The operators include:

```text
F, G, X, U, |, &, ->, <->, !
```

The generation process controls structural properties such as AST size and temporal nesting depth.

### 2. Filtering and Validation

Generated formulas are filtered to keep formulas that are valid for the benchmark. The filtering includes:

- syntactic correctness,
- satisfiability,
- non-triviality,
- syntactic duplicate removal,
- semantic duplicate checking.

Spot is used for LTL manipulation and equivalence checking.

### 3. Back-Translation

`BackTranslation.jl` converts each retained LTL formula into a verbatim NL description. This translation follows the structure of the LTL formula using fixed grammatical templates.

### 4. Paraphrasing

Files of the form:

```text
Paraphrase_LLMName.jl
```

are used to paraphrase the verbatim NL descriptions with different LLMs. This introduces linguistic diversity while preserving the underlying temporal semantics.

### 5. Benchmark Dataset

The final benchmark is stored in:

```text
benchmark/DatasetWithNaturalNL_plus_simplified.json
```

Each record contains the LTL formula, verbatim back-translation, paraphrases, and metadata such as formula size, temporal depth, number of propositions, and automaton statistics.

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

## Summary

This repository supports the full NLTLBench workflow:

1. generate LTL formulas,
2. filter and validate formulas,
3. back-translate formulas into verbatim NL,
4. paraphrase NL descriptions using LLMs,
5. evaluate LLMs on NL-to-LTL translation,
6. analyze and plot the resulting performance trends.