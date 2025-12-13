# Dataset Issue Found: Class Imbalance

## Problem Identified ✅

The confusion matrix shows the model is **predicting everything as FAKE**. This is because:

### Root Cause
1. **Severe class imbalance**: 
   - Training: 86.4% fake, 13.6% real
   - Validation: 86.4% fake, 13.6% real
   - Test: 86.3% fake, 13.7% real

2. **Model behavior**:
   - Always predicts FAKE (class 1)
   - Gets 86% accuracy (because 86% of data is fake)
   - But AUC = 0.56 (no discrimination)
   - Recall = 100% (catches all fakes) but Precision = 86% (many false positives)

### Confusion Matrix Analysis
```
                Predicted
              Real  Fake
True Real       0   133  ← All real videos misclassified!
True Fake       0   845  ← All fake videos correct (but only because model always says fake)
```

## Solution Applied

Updated `train.py` to use **class-weighted loss**:
- Real class (minority): Higher weight
- Fake class (majority): Lower weight
- Forces model to learn both classes

## Next Steps

1. **Retrain with weighted loss**:
   ```bash
   python train.py --config config.yaml
   ```

2. **Expected improvement**:
   - Model should learn to distinguish real vs fake
   - AUC should improve significantly
   - Confusion matrix should show predictions for both classes

3. **Monitor training**:
   - Check that model predicts both classes
   - AUC should be > 0.7 for good performance
   - Confusion matrix should have non-zero values in all cells

## Alternative Solutions

If weighted loss doesn't work well:

1. **Oversample minority class** (real videos)
2. **Undersample majority class** (fake videos)  
3. **Use Focal Loss** instead of BCE
4. **Use balanced batches** with WeightedRandomSampler

## Current Status

- ✅ Issue identified: Class imbalance causing model to always predict fake
- ✅ Fix applied: Class-weighted loss added to training
- ⏳ Need to retrain model with fix



