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