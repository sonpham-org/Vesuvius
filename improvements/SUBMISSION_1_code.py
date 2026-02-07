# SUBMISSION 1: Probability Hysteresis + Aggressive Closing
#
# Fork ipythonx's notebook, then REPLACE these cells:
# - Cell 18 (predict_with_tta)
# - Cell 20 (topo_postprocess)
# - Cell 22 (inference_pipelines)

# ============================================
# CELL 18 - REPLACE predict_with_tta
# ============================================
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

    # Return foreground probability (class 1)
    fg_prob = np.array(mean_prob[0, ..., 1])
    return fg_prob


# ============================================
# CELL 20 - REPLACE topo_postprocess
# ============================================
def topo_postprocess(
    fg_prob,
    T_low=0.30,
    T_high=0.80,
    z_radius=3,
    xy_radius=1,
    dust_min_size=150,
):
    # Hysteresis on foreground probability
    strong = fg_prob >= T_high
    weak = fg_prob >= T_low

    if not strong.any():
        return np.zeros_like(fg_prob, dtype=np.uint8)

    struct_hyst = ndi.generate_binary_structure(3, 3)
    mask = ndi.binary_propagation(strong, mask=weak, structure=struct_hyst)

    if not mask.any():
        return np.zeros_like(fg_prob, dtype=np.uint8)

    # Anisotropic Closing
    if z_radius > 0 or xy_radius > 0:
        struct_close = build_anisotropic_struct(z_radius, xy_radius)
        if struct_close is not None:
            mask = ndi.binary_closing(mask, structure=struct_close)

    # Dust Removal
    if dust_min_size > 0:
        mask = remove_small_objects(mask.astype(bool), min_size=dust_min_size)

    return mask.astype(np.uint8)


# ============================================
# CELL 22 - REPLACE inference_pipelines
# ============================================
def inference_pipelines(volume):
    fg_prob = predict_with_tta(volume, swi)
    final = topo_postprocess(fg_prob)
    return final
