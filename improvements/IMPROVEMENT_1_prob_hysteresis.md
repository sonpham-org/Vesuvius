# Improvement 1: Probability-Based Hysteresis

## The Issue
Current code applies hysteresis to class indices (0,1,2) not probabilities.
This accidentally works but loses probability information.

## The Fix
In your forked notebook, make these changes:

### Change 1: Modify `predict_with_tta` to return both class map AND probabilities

```python
def predict_with_tta(inputs, swi):
    logits = []

    # Original
    logits.append(swi(inputs))

    # Flips (spatial only)
    for axis in [1, 2, 3]:
        img_f = np.flip(inputs, axis=axis)
        p = swi(img_f)
        p = np.flip(p, axis=axis)
        logits.append(p)

    # Axial rotations (H, W)
    for k in [1, 2, 3]:
        img_r = np.rot90(inputs, k=k, axes=(2, 3))
        p = swi(img_r)
        p = np.rot90(p, k=-k, axes=(2, 3))
        logits.append(p)

    mean_logits = np.mean(logits, axis=0)
    mean_prob = ops.softmax(mean_logits, axis=-1)

    # Return BOTH class map AND foreground probability
    class_map = mean_prob.argmax(-1).astype(np.uint8).squeeze()
    fg_prob = np.array(mean_prob[0, ..., 1])  # Class 1 probability

    return class_map, fg_prob
```

### Change 2: Modify `topo_postprocess` to use foreground probability

```python
def topo_postprocess(
    fg_prob,  # Now foreground probability (0.0-1.0)
    T_low=0.30,
    T_high=0.80,
    z_radius=3,
    xy_radius=1,
    dust_min_size=150,
):
    # Hysteresis on actual probabilities
    strong = fg_prob >= T_high
    weak = fg_prob >= T_low

    if not strong.any():
        return np.zeros_like(fg_prob, dtype=np.uint8)

    struct_hyst = ndi.generate_binary_structure(3, 3)
    mask = ndi.binary_propagation(strong, mask=weak, structure=struct_hyst)

    if not mask.any():
        return np.zeros_like(fg_prob, dtype=np.uint8)

    if z_radius > 0 or xy_radius > 0:
        struct_close = build_anisotropic_struct(z_radius, xy_radius)
        if struct_close is not None:
            mask = ndi.binary_closing(mask, structure=struct_close)

    if dust_min_size > 0:
        mask = remove_small_objects(mask.astype(bool), min_size=dust_min_size)

    return mask.astype(np.uint8)
```

### Change 3: Update `inference_pipelines`

```python
def inference_pipelines(volume):
    class_map, fg_prob = predict_with_tta(volume, swi)
    final = topo_postprocess(
        fg_prob,
        T_low=0.30,
        T_high=0.80,
        z_radius=3,
        xy_radius=1,
        dust_min_size=150,
    )
    return final
```

## Expected Impact
- Better control over sensitivity/specificity tradeoff
- Probability thresholds (0.30/0.80) are more intuitive than class index thresholds
- More aggressive closing (z=3, xy=1) fills gaps
