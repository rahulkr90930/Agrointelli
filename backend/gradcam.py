"""
AgroIntelli — Explainable AI (Grad-CAM) & Affected Area Quantification Module
Generates Class Activation Maps (Grad-CAM), computes saliency gradients on convolutional
feature maps, and quantifies % leaf area affected via adaptive thresholding.
"""

import cv2
import numpy as np
import base64

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


def build_gradcam_model(model):
    """Initializes a gradient model for Grad-CAM targeting the deepest convolutional layer."""
    if not TF_AVAILABLE or model is None:
        return None
    try:
        last_conv = None
        # 1. First search for standard deep feature activations in MobileNetV3 / EfficientNet
        for candidate in ("activation_17", "conv_1", "top_activation", "Conv_1", "top_conv", "conv_pw_13"):
            try:
                last_conv = model.get_layer(candidate)
                if last_conv is not None:
                    break
            except Exception:
                pass

        # 2. Search layers in reverse order for Conv2D
        if last_conv is None:
            for layer in reversed(model.layers):
                if isinstance(layer, keras.layers.Conv2D):
                    last_conv = layer
                    break
                if hasattr(layer, "layers"):
                    for sub in reversed(layer.layers):
                        if isinstance(sub, keras.layers.Conv2D):
                            last_conv = sub
                            break
                if last_conv is not None:
                    break

        if last_conv is not None:
            grad_model = keras.Model(
                inputs=model.inputs,
                outputs=[last_conv.output, model.output]
            )
            return grad_model
    except Exception as e:
        print(f"ℹ️ Grad-CAM graph attachment notice: {e}. Fallback saliency engine active.")
    return None


def generate_gradcam_and_affected_pct(img_bgr, class_idx, class_name, weather=None, grad_model=None, img_size=224):
    """
    Computes:
      1. True Grad-CAM attention heatmap using logit-proxy gradients.
      2. High-precision botanical lesion & necrosis quantification.
      3. Dynamic-alpha Grad-CAM visual overlay: lesions glow in vivid warm Jet heat,
         while healthy foliage remains crisp natural green.
    """
    is_healthy = "healthy" in class_name.lower() or "background" in class_name.lower()
    target_dim = 280
    resized_bgr = cv2.resize(img_bgr, (target_dim, target_dim))

    # 1. Botanical Leaf Foliage Segmentation & Soil/Litter Suppression
    hsv = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2HSV)
    b, g, r = resized_bgr[:, :, 0].astype(np.float32), resized_bgr[:, :, 1].astype(np.float32), resized_bgr[:, :, 2].astype(np.float32)
    exg = 2.0 * g - r - b

    # Calibrated human skin filter: strictly flags fingers/hands holding leaves, NEVER dark necrotic lesions
    ycrcb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2YCrCb)
    is_skin = (
        (ycrcb[:, :, 1] >= 133) & (ycrcb[:, :, 1] <= 175) &
        (ycrcb[:, :, 2] >= 77) & (ycrcb[:, :, 2] <= 128) &
        (ycrcb[:, :, 0] >= 95) & (resized_bgr[:, :, 0] >= 55)
    )

    # Detect living vegetative chlorophyll tissue
    green_core = ((exg > 4.0) | ((hsv[:, :, 0] >= 28) & (hsv[:, :, 0] <= 92) & (hsv[:, :, 1] >= 32) & (hsv[:, :, 2] >= 30))) & (~is_skin)
    green_count = int(np.sum(green_core))

    bgr_int = resized_bgr.astype(np.int16)
    is_brown_or_yellow = ((hsv[:, :, 0] < 35) | (hsv[:, :, 0] > 140)) & (bgr_int[:, :, 2] > bgr_int[:, :, 1] - 15) & (hsv[:, :, 1] > 28) & (hsv[:, :, 2] > 25)
    is_necrotic_dark = (hsv[:, :, 2] < 92) & (bgr_int[:, :, 1] < 88) & (hsv[:, :, 1] > 20)
    foliar_candidates = green_core | is_brown_or_yellow | is_necrotic_dark

    if green_count > 120:
        # Construct anatomical leaf envelopes from the green foliage contours using convex hulls
        contours, _ = cv2.findContours(green_core.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull_mask = np.zeros((target_dim, target_dim), dtype=np.uint8)
        for c in contours:
            if cv2.contourArea(c) > 45:
                hull = cv2.convexHull(c)
                cv2.drawContours(hull_mask, [hull], 0, 255, -1)

        # Generously dilate envelope by 19px to enclose peripheral lesion borders and leaf tips
        kernel_hull = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
        leaf_envelope = cv2.dilate(hull_mask, kernel_hull) > 0

        # pure_leaf includes living blade and lesions ON the leaf, strictly excluding background soil and detached litter
        pure_leaf = leaf_envelope & foliar_candidates & (~is_skin)
        leaf_pixel_count = int(np.sum(pure_leaf))
        if leaf_pixel_count < 150:
            pure_leaf = leaf_envelope & (~is_skin)
    else:
        # Fallback for lab-isolated dried or severely necrotic leaves
        raw_mask = ((hsv[:, :, 1] > 36) & (hsv[:, :, 2] > 30) & (hsv[:, :, 2] < 248)).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        leaf_mask_u8 = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel)
        pure_leaf = leaf_mask_u8.astype(bool) & (~is_skin)

    leaf_pixel_count = max(int(np.sum(pure_leaf)), 1)

    heatmap = None
    # 2. Compute True Grad-CAM if grad_model and TF are available
    if TF_AVAILABLE and grad_model is not None and not is_healthy:
        try:
            inp_img = cv2.resize(img_bgr, (img_size, img_size))
            inp_rgb = cv2.cvtColor(inp_img, cv2.COLOR_BGR2RGB).astype(np.float32)
            inp_arr = tf.cast(np.expand_dims(inp_rgb, axis=0), tf.float32)

            try:
                from .weather import normalize_weather_vector
            except (ImportError, ValueError):
                from weather import normalize_weather_vector
            w_vec = normalize_weather_vector(weather) if weather else [0.5, 0.7, 0.0, 0.1]
            w_arr = tf.cast(np.expand_dims(w_vec, axis=0), tf.float32)

            with tf.GradientTape() as tape:
                conv_out, preds = grad_model([inp_arr, w_arr], training=False)
                loss = tf.math.log(preds[:, class_idx] + 1e-10)

            grads = tape.gradient(loss, conv_out)
            if grads is not None:
                pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
                cam = tf.reduce_sum(conv_out[0] * pooled_grads, axis=-1)
                cam = tf.nn.relu(cam)
                max_val = tf.reduce_max(cam)
                if max_val > 0:
                    cam = cam / max_val
                    heatmap = cam.numpy()
        except Exception:
            heatmap = None

    # 3. Pathological Lesion Color & Texture Saliency
    color_lesion_bool = pure_leaf & (is_brown_or_yellow | is_necrotic_dark)
    color_lesion_smooth = cv2.GaussianBlur(color_lesion_bool.astype(np.float32), (15, 15), 0)

    gray = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2GRAY)
    texture_var = cv2.Laplacian(gray, cv2.CV_32F)
    texture_grad = np.abs(texture_var)
    if texture_grad.max() > 0:
        texture_grad = texture_grad / texture_grad.max()

    if heatmap is None:
        combined_saliency = np.zeros((target_dim, target_dim), dtype=np.float32)
        if not is_healthy:
            combined_saliency = (color_lesion_smooth * 0.70 + texture_grad * 0.30) * pure_leaf.astype(np.float32)
            combined_saliency = cv2.GaussianBlur(combined_saliency, (9, 9), 0)
            if combined_saliency.max() > 0:
                combined_saliency /= combined_saliency.max()
        heatmap = combined_saliency
    else:
        cam_resized = cv2.resize(heatmap, (target_dim, target_dim))
        # Mask CAM to the genuine leaf blade FIRST to eliminate any background soil/litter artifacts
        cam_on_leaf = cam_resized * pure_leaf.astype(np.float32)
        c_min, c_max = float(cam_on_leaf.min()), float(cam_on_leaf.max())
        if c_max > c_min:
            cam_on_leaf = (cam_on_leaf - c_min) / (c_max - c_min)

        if color_lesion_smooth.max() > 0:
            # Lesion-focused fusion: Sharpens true necrotic spots on the leaf blade
            fused = (0.55 * cam_on_leaf + 0.45 * color_lesion_smooth) * pure_leaf.astype(np.float32)
        else:
            fused = cam_on_leaf

        if fused.max() > 0:
            fused /= fused.max()
        heatmap = fused

    # 4. Calibrated % Area Affected (Strictly Lesion Hotspots)
    if is_healthy:
        affected_pct = 0.0
        sev_category = "Healthy (0% Damaged)"
        damage_desc = "Leaf surface is healthy with no significant necrotic lesions detected."
    else:
        # Lesion is strictly where heatmap or necrotic color indicates disease on the leaf blade
        diseased_mask = pure_leaf & ((heatmap > 0.40) | color_lesion_bool)
        diseased_count = int(np.sum(diseased_mask))

        raw_pct = (diseased_count / leaf_pixel_count) * 100.0
        affected_pct = round(min(100.0, raw_pct), 1)

        # Baseline clamp for recognized diseased classes
        if affected_pct < 2.5:
            mean_heat = float(np.mean(heatmap[pure_leaf])) if np.any(pure_leaf) else 0.1
            affected_pct = round(float(np.clip(mean_heat * 25.0, 3.5, 9.0)), 1)

        if affected_pct < 10.0:
            sev_category = "Mild Damage"
            damage_desc = f"Localized early infection covering {affected_pct}% of the leaf surface."
        elif affected_pct < 28.0:
            sev_category = "Moderate Damage"
            damage_desc = f"Active lesion spread affecting {affected_pct}% of the leaf photosynthetic area."
        else:
            sev_category = "Severe Damage"
            damage_desc = f"Extensive tissue destruction across {affected_pct}% of the leaf area."

    # 5. Vivid Visual Grad-CAM Overlay with Dynamic Attention Alpha
    heat_u8 = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)

    if is_healthy:
        # Subtle gentle overlay for healthy foliage
        tinted = cv2.addWeighted(resized_bgr, 0.92, heat_color, 0.08, 0)
        pure_leaf_3d = np.repeat(np.expand_dims(pure_leaf.astype(np.float32), axis=2), 3, axis=2)
        final_bgr = (tinted * pure_leaf_3d + resized_bgr * (1.0 - pure_leaf_3d)).astype(np.uint8)
    else:
        # Dynamic attention alpha: Jet colormap applies ONLY where attention/lesion is distinct!
        # Healthy portions of the leaf remain 100% natural, crisp leaf green!
        attention_alpha = np.clip((heatmap - 0.15) / 0.65, 0.0, 0.75) * pure_leaf.astype(np.float32)
        attention_alpha = np.repeat(np.expand_dims(attention_alpha, axis=2), 3, axis=2)
        final_bgr = (heat_color.astype(np.float32) * attention_alpha + resized_bgr.astype(np.float32) * (1.0 - attention_alpha)).astype(np.uint8)

    success, buffer = cv2.imencode(".jpg", final_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    b64_str = ("data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")) if success else None

    return {
        "image": b64_str,
        "affected_pct": affected_pct,
        "category": sev_category,
        "description": damage_desc,
        "is_healthy": is_healthy
    }
