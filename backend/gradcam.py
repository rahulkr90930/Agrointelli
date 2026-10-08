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
      3. Dynamic-alpha Grad-CAM visual overlay at 512px resolution: lesions glow in vivid warm Jet heat,
         while underlying leaf veins and natural foliage remain crisp.
    """
    is_healthy = "healthy" in class_name.lower() or "background" in class_name.lower()
    target_dim = 512
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

    # Detect vegetative foliage
    green_core = ((exg > 4.0) | ((hsv[:, :, 0] >= 28) & (hsv[:, :, 0] <= 92) & (hsv[:, :, 1] >= 32) & (hsv[:, :, 2] >= 30))) & (~is_skin)
    green_count = int(np.sum(green_core))

    bgr_int = resized_bgr.astype(np.int16)
    is_brown_or_yellow = ((hsv[:, :, 0] < 35) | (hsv[:, :, 0] > 140)) & (bgr_int[:, :, 2] > bgr_int[:, :, 1] - 15) & (hsv[:, :, 1] > 28) & (hsv[:, :, 2] > 25)
    is_necrotic_dark = (hsv[:, :, 2] < 92) & (bgr_int[:, :, 1] < 88) & (hsv[:, :, 1] > 20)
    # Powdery mildews: white/ash fungal efflorescence (squash, cherry, grape)
    is_powdery_white = (hsv[:, :, 1] < 55) & (hsv[:, :, 2] > 155)
    # Chlorotic virus / mite patterns: yellow mottling, leaf curl chlorosis
    is_chlorotic_yellow = (hsv[:, :, 0] >= 18) & (hsv[:, :, 0] <= 44) & (hsv[:, :, 1] >= 35) & (hsv[:, :, 2] > 95)
    # Rust pustules: orange / cinnamon spots (corn rust, apple rust)
    is_rust_orange = (hsv[:, :, 0] >= 6) & (hsv[:, :, 0] <= 24) & (hsv[:, :, 1] >= 85) & (hsv[:, :, 2] >= 65)

    foliar_candidates = green_core | is_brown_or_yellow | is_necrotic_dark | is_powdery_white | is_chlorotic_yellow | is_rust_orange

    if green_count > 250:
        contours, _ = cv2.findContours(green_core.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull_mask = np.zeros((target_dim, target_dim), dtype=np.uint8)
        for c in contours:
            if cv2.contourArea(c) > 100:
                hull = cv2.convexHull(c)
                cv2.drawContours(hull_mask, [hull], 0, 255, -1)

        kernel_hull = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
        leaf_envelope = cv2.dilate(hull_mask, kernel_hull) > 0
        pure_leaf = leaf_envelope & foliar_candidates & (~is_skin)
        leaf_pixel_count = int(np.sum(pure_leaf))
        if leaf_pixel_count < 300:
            pure_leaf = leaf_envelope & (~is_skin)
    else:
        raw_mask = ((hsv[:, :, 1] > 25) & (hsv[:, :, 2] > 25) & (hsv[:, :, 2] < 252)).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        leaf_mask_u8 = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel)
        pure_leaf = leaf_mask_u8.astype(bool) & (~is_skin)

    leaf_pixel_count = max(int(np.sum(pure_leaf)), 1)

    heatmap = None
    # 2. Compute True Grad-CAM on ALL classes (both diseased and healthy)
    if TF_AVAILABLE and grad_model is not None:
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
                cam_relu = tf.nn.relu(cam)
                max_val = tf.reduce_max(cam_relu)
                if max_val > 0:
                    cam_norm = cam_relu / max_val
                    heatmap = cam_norm.numpy()
                else:
                    cam_abs = tf.abs(cam)
                    max_abs = tf.reduce_max(cam_abs)
                    if max_abs > 0:
                        heatmap = (cam_abs / max_abs).numpy()
        except Exception as e:
            print(f"⚠ Grad-CAM failed for {class_name}: {e}. Using image-saliency fallback.")
            heatmap = None

    # 3. Pathological Lesion Color & Texture Saliency
    color_lesion_bool = (is_brown_or_yellow | is_necrotic_dark | is_powdery_white | is_chlorotic_yellow | is_rust_orange) & pure_leaf & (~is_skin)
    color_lesion_smooth = cv2.GaussianBlur(color_lesion_bool.astype(np.float32), (21, 21), 0)

    gray = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2GRAY)
    texture_var = cv2.Laplacian(gray, cv2.CV_32F)
    texture_grad = np.abs(texture_var)
    if texture_grad.max() > 0:
        texture_grad = texture_grad / texture_grad.max()

    if heatmap is None:
        if is_healthy:
            # Healthy foliage attention: smooth coverage across leaf blade
            leaf_float = pure_leaf.astype(np.float32)
            combined_saliency = cv2.GaussianBlur(leaf_float, (25, 25), 0)
        else:
            combined_saliency = (color_lesion_smooth * 0.70 + texture_grad * 0.30)
            combined_saliency = cv2.GaussianBlur(combined_saliency, (15, 15), 0)

        if combined_saliency.max() > 0:
            combined_saliency /= combined_saliency.max()
        heatmap = combined_saliency
    else:
        cam_resized = cv2.resize(heatmap, (target_dim, target_dim))
        c_min, c_max = float(cam_resized.min()), float(cam_resized.max())
        if c_max > c_min:
            cam_resized = (cam_resized - c_min) / (c_max - c_min)

        if not is_healthy and color_lesion_smooth.max() > 0:
            fused = 0.55 * cam_resized + 0.45 * color_lesion_smooth
        else:
            fused = cam_resized

        if fused.max() > 0:
            fused /= fused.max()
        heatmap = fused

    # Constrain heatmap attention strictly to leaf surface
    heatmap = heatmap * pure_leaf.astype(np.float32)
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    # 4. Calibrated % Area Affected
    if is_healthy:
        affected_pct = 0.0
        sev_category = "Healthy Foliage (0% Damaged)"
        damage_desc = "Photosynthetic blade shows healthy green tissue with no significant fungal or bacterial lesions."
    else:
        diseased_mask = ((heatmap > 0.28) | color_lesion_bool) & pure_leaf
        diseased_count = int(np.sum(diseased_mask))
        raw_pct = (diseased_count / leaf_pixel_count) * 100.0
        affected_pct = round(min(100.0, max(2.0, raw_pct)), 1)

        if affected_pct < 10.0:
            sev_category = "Mild Damage"
            damage_desc = f"Localized early infection covering {affected_pct}% of the leaf blade."
        elif affected_pct < 28.0:
            sev_category = "Moderate Damage"
            damage_desc = f"Active lesion spread affecting {affected_pct}% of the photosynthetic leaf surface."
        else:
            sev_category = "Severe Damage"
            damage_desc = f"Extensive symptomatic tissue necrosis covering {affected_pct}% of the leaf surface."

    # 5. Vivid High-Definition Grad-CAM Visual Overlay
    heat_u8 = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)

    if is_healthy:
        # Gentle translucent emerald/teal attention glow across healthy photosynthetic leaf
        attention_alpha = np.clip(heatmap * 0.32 + 0.04, 0.04, 0.38) * pure_leaf.astype(np.float32)
        attention_alpha = np.repeat(np.expand_dims(attention_alpha, axis=2), 3, axis=2)
        final_bgr = (heat_color.astype(np.float32) * attention_alpha + resized_bgr.astype(np.float32) * (1.0 - attention_alpha)).astype(np.uint8)
    else:
        # Balanced attention alpha: warm glowing hotspots over lesions, natural foliage preserved
        attention_alpha = np.clip(heatmap * 0.55 + 0.14, 0.14, 0.68) * pure_leaf.astype(np.float32)
        attention_alpha = np.repeat(np.expand_dims(attention_alpha, axis=2), 3, axis=2)
        final_bgr = (heat_color.astype(np.float32) * attention_alpha + resized_bgr.astype(np.float32) * (1.0 - attention_alpha)).astype(np.uint8)

    success, buffer = cv2.imencode(".jpg", final_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    b64_str = ("data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")) if success else None

    return {
        "image": b64_str,
        "affected_pct": affected_pct,
        "category": sev_category,
        "description": damage_desc,
        "is_healthy": is_healthy
    }
