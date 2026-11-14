# State of the Art in Ensemble Models for Data Stream Mining

Undergraduate research (**PIBITI**, PUCPR, 2025, **CNPq scholarship**) advised by Prof. Dr. Fabrício Enembreck: a fair, standardized comparison of six ensemble algorithms for data stream classification.

## Problem

Stream mining has no standardized protocol for comparing algorithms, so it is hard to know which ensemble is best for which situation.

## Method

- Algorithms: **ARF, ARE, ARTE, SGBT, SRP, ROSE**.
- **MOA** (Java) with the **prequential** (test-then-train) protocol on real and synthetic streams (e.g. Airlines, CovtypeNorm, Electricity, Gas Sensor, KDD99, Rialto; RBF and SEA generators).
- Four criteria: accuracy, Kappa, processing time and memory. Runs capped at 72 h.

## Results

There is **no universally best algorithm**:

- **ARTE**: best overall on real-world data (rank 1 by accuracy), most balanced.
- **SRP**: second on real data, but weaker on controlled synthetic streams.
- **ARE**: fastest; **SGBT**: lowest memory, but both less accurate (SGBT also hit the time limit on some datasets).
- **ROSE**: strong on synthetic data but did not generalize to real streams and used the most memory.
- **ARF**: reliable, predictable middle ground.

**Main lesson:** performance on ideal synthetic benchmarks does not necessarily translate to real data.

## Reproduce

All experiments were run in the **MOA graphical interface** (Classification tab) with the `EvaluatePrequential` task, one run per algorithm and dataset, loading each stream from an ARFF file (`ArffFileStream`) or from a synthetic generator. Example for one real stream:

```
EvaluatePrequential -l meta.StreamingGradientBoostedTrees -s (ArffFileStream -f <path>/gassensor.arff)
```

The same task was used for every algorithm, with accuracy, Kappa, CPU time and RAM-hours read from the evaluation panel. The exact MOA version and the raw result files were not kept.

## My contribution

I wrote the code and experiment scripts, curated the datasets, and wrote the final report.

## Report

<!-- TODO: add report.pdf (clean version, without advisor comments) only if the advisor and PUCPR allow it to be public. -->
The final report is not published here.
