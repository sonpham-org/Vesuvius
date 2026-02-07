# Improvement 2: Per-Slice Hole Filling

## The Issue
The current post-processing does morphological closing but doesn't fill interior holes.
Interior holes hurt β₁ (tunnels/holes) in TopoScore which is 33% of the 30% TopoScore weight.

## The Fix
Add per-slice `binary_fill_holes` after closing. This is a simple one-line addition.

### Change: Modify `topo_postprocess` to add hole filling

```python
def topo_postprocess(
    probs,
    T_low=0.50,
    T_high=0.90,
    z_radius=1,
    xy_radius=0,
    dust_min_size=100,
):
    # Step 1: 3D Hysteresis
    strong = probs >= T_high
    weak = probs >= T_low

    if not strong.any():
        return np.zeros_like(probs, dtype=np.uint8)

    struct_hyst = ndi.generate_binary_structure(3, 3)
    mask = ndi.binary_propagation(strong, mask=weak, structure=struct_hyst)

    if not mask.any():
        return np.zeros_like(probs, dtype=np.uint8)

    # Step 2: 3D Anisotropic Closing
    if z_radius > 0 or xy_radius > 0:
        struct_close = build_anisotropic_struct(z_radius, xy_radius)
        if struct_close is not None:
            mask = ndi.binary_closing(mask, structure=struct_close)

    # Step 3: NEW - Per-slice hole filling (targets β₁)
    for z in range(mask.shape[0]):
        mask[z] = ndi.binary_fill_holes(mask[z])

    # Step 4: Dust Removal
    if dust_min_size > 0:
        mask = remove_small_objects(mask.astype(bool), min_size=dust_min_size)

    return mask.astype(np.uint8)
```

## Expected Impact
- Fills interior holes in each 2D slice
- Directly improves β₁ score (holes/tunnels)
- Simple change, low risk
- May slightly increase foreground voxel count

## Variant: Combine with More Aggressive Closing

For maximum topology preservation:

```python
def topo_postprocess(
    probs,
    T_low=0.50,
    T_high=0.90,
    z_radius=3,      # Increase from 1
    xy_radius=1,     # Increase from 0
    dust_min_size=100,
):
    # ... same as above but with larger closing radii
```

This fills both gaps (via closing) and holes (via fill_holes).
