# C3 and C4 ERD Analysis - Sub-002 Data Limitations

## Summary

Sub-002 EEG data has only **4 good channels: C3, C4, F3, F4**. The other 28 channels have poor data quality. The pipeline has been adapted to work with this limited channel set.

## Data Limitations

**Available EEG channels:**
- **C3, C4**: Motor cortex (primary analysis channels)
- **F3, F4**: Frontal cortex (for reference)
- **28 other channels**: Marked as bad (poor data quality)

**Implications:**
- Cannot use motor ROI clusters (neighbors FC1, CP1, T7, FC2, CP2, T8 unavailable)
- Cannot run ICA (requires minimum 15-20 channels)
- Cannot interpolate bad channels (too many bad channels: 28/32)
- Using single electrodes (C3, C4) instead of clusters

## Configuration Changes

**Epoch windows extended to 20s:**
- EEG epochs: -3 to +20s
- fNIRS epochs: -3 to +20s
- Beta rebound window: 15-20s

**EEG preprocessing adapted:**
- Bad channels marked automatically (28 channels)
- ICA disabled (insufficient channels)
- Interpolation skipped (too many bad channels)
- CAR computed using only good channels (C3, C4, F3, F4)

**Motor ROI clusters simplified:**
- Left motor: C3 only (was C3 + FC1 + CP1 + T7)
- Right motor: C4 only (was C4 + FC2 + CP2 + T8)

## Current Output

```
C3 (Left Motor Cortex):
  Alpha ERD: [value]%
  Beta ERD: [value]%
  Beta Rebound (15-20s): [value]%

C4 (Right Motor Cortex):
  Alpha ERD: [value]%
  Beta ERD: [value]%
  Beta Rebound (15-20s): [value]%
```

## Explanation of Values

### 1. Why are Alpha/Beta ERD values near zero?

**These values represent ALL conditions combined (LEFT + RIGHT + NOTHING averaged together).**

The condition-specific values (from lateralization analysis) show the true ERD patterns:

```
Alpha ERD by condition:
  LEFT:    C3=-28.0%  C4=-38.4%  (C4 shows strong ERD)
  RIGHT:   C3=+16.2%  C4=+3.9%   (C3 shows ERS, unexpected)
  NOTHING: C3=+11.7%  C4=+9.7%   (Baseline)
```

### 2. Beta Rebound (15-20s)

Beta rebound is a post-movement increase in beta power that occurs after task completion. Expected pattern: +10% to +30% increase 15-20s after task onset.

### 3. Impact of Limited Channels

**Reduced spatial resolution:**
- Single electrodes instead of clusters → lower SNR
- No spatial averaging → more susceptible to local artifacts
- Cannot validate findings with neighboring channels

**Preprocessing limitations:**
- No ICA → Cannot remove eye blinks, muscle artifacts systematically
- No interpolation → Bad channels remain bad
- Limited CAR → Reference computed from only 4 channels

**Recommendations:**
- Interpret results with caution due to limited channel set
- Focus on C3 vs C4 lateralization (most robust finding)
- Consider replicating with full 32-channel data if available

## Data Quality Note

Sub-002 has multiple data quality issues:
- **EEG**: Only 4/32 channels have good data
- **Recording duration**: 660s of expected 1050s (incomplete)
- **Trials**: 21 of expected 36 (7 per condition instead of 12)

These limitations reduce statistical power and spatial resolution.

## Scientific Rationale

**Why continue analysis with limited channels?**
- C3 and C4 are the **primary motor cortex electrodes**
- Lateralization analysis (C3 vs C4) is still valid
- Provides proof-of-concept for pipeline functionality
- Identifies data quality issues for future recordings
