# SUBMISSION 2: Minimal Change - Just Add Hole Filling
#
# Fork ipythonx's notebook, then REPLACE only:
# - Cell 20 (topo_postprocess)
#
# This is the SAFEST change - just adds one feature to the working baseline.

# ============================================
# CELL 20 - REPLACE topo_postprocess
# ============================================
def topo_postprocess(
    probs,
    T_low=0.50,
    T_high=0.90,
    z_radius=1,
    xy_radius=0,
    dust_min_size=100,
):
    # Step 1: 3D Hysteresis (unchanged)
    strong = probs >= T_high
    weak = probs >= T_low

    if not strong.any():
        return np.zeros_like(probs, dtype=np.uint8)

    struct_hyst = ndi.generate_binary_structure(3, 3)
    mask = ndi.binary_propagation(strong, mask=weak, structure=struct_hyst)

    if not mask.any():
        return np.zeros_like(probs, dtype=np.uint8)

    # Step 2: 3D Anisotropic Closing (unchanged)
    if z_radius > 0 or xy_radius > 0:
        struct_close = build_anisotropic_struct(z_radius, xy_radius)
        if struct_close is not None:
            mask = ndi.binary_closing(mask, structure=struct_close)

    # Step 3: NEW - Per-slice hole filling
    for z in range(mask.shape[0]):
        mask[z] = ndi.binary_fill_holes(mask[z])

    # Step 4: Dust Removal (unchanged)
    if dust_min_size > 0:
        mask = remove_small_objects(mask.astype(bool), min_size=dust_min_size)

    return mask.astype(np.uint8)
