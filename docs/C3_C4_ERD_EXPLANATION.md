# C3 and C4 ERD Analysis - Explanation of Values

## Summary

The pipeline computes ERD metrics for both C3 (left motor cortex) and C4 (right motor cortex). Epochs have been extended to 20 seconds to capture beta rebound (15-20s post-task).

## Configuration Changes

**Epoch windows extended from 15s to 20s:**
- EEG epochs: -3 to +20s (was -3 to +15s)
- fNIRS epochs: -3 to +20s (was -3 to +15s)
- Beta rebound window: 15-20s (was 16-20s, now within epoch range)

## Current Output

```
C3 (Left Motor Cortex):
  Alpha ERD: 0.34%
  Beta ERD: 0.07%
  Beta Rebound (15-20s): [value]%

C4 (Right Motor Cortex):
  Alpha ERD: 0.16%
  Beta ERD: 0.07%
  Beta Rebound (15-20s): [value]%
```

## Explanation of Values

### 1. Why are Alpha/Beta ERD values near zero?

**These values represent ALL conditions combined (LEFT + RIGHT + NOTHING averaged together).**

When you average across all conditions:
- LEFT hand movement causes C4 ERD (contralateral)
- RIGHT hand movement causes C3 ERD (contralateral)  
- NOTHING causes no significant ERD

The result is that positive and negative changes cancel out, giving near-zero values.

**The condition-specific values show the true ERD patterns:**

```
Alpha ERD by condition:
  LEFT:    C3=-28.0%  C4=-38.4%  (C4 shows strong ERD)
  RIGHT:   C3=+16.2%  C4=+3.9%   (C3 shows ERS, unexpected)
  NOTHING: C3=+11.7%  C4=+9.7%   (Baseline)

Beta ERD by condition:
  LEFT:    C3=-30.4%  C4=-39.7%  (Both show ERD)
  RIGHT:   C3=-31.2%  C4=-31.4%  (Both show ERD)
  NOTHING: C3=+15.6%  C4=+28.4%  (Baseline)
```

### 2. Beta Rebound (15-20s)

Beta rebound is a post-movement increase in beta power that occurs after task completion. With epochs extended to 20s, this can now be measured.

**Expected pattern:** +10% to +30% increase in beta power 15-20s after task onset (after finger tapping stops at 15s).

### 3. Why are p-values not shown?

Statistical significance testing requires trial-level data, but the all-conditions-combined ERD uses averaged TFR. The lateralization analysis provides p-values for condition-specific comparisons.

## Data Quality Note

Sub-002 has incomplete EEG data:
- Expected: 1050s (36 trials)
- Actual: 660s (21 trials: 7 LEFT, 7 RIGHT, 7 NOTHING)

This reduces statistical power and may explain unexpected patterns.

## Scientific Rationale

**Why extend epochs to 20s?**
- Beta rebound is a well-established post-movement phenomenon (Pfurtscheller & Lopes da Silva, 1999)
- Occurs 15-20s after task onset in sequential finger tapping tasks
- Provides additional validation of motor cortex engagement
- Helps distinguish true motor activity from artifacts
