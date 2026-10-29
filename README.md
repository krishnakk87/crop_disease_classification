# Production Deployment Manual: Crop Disease Classification System

This manual details the technical specifications, architectural details, and edge deployment strategy for Syngenta's offline crop-disease diagnosis system.

---

## 1. Specifications & Architectural Strategy

To allow remote offline diagnostics by field agronomists, we employ a mobile-first design leveraging a customized **MobileNetV3-Small** architecture, initialized with ImageNet weights and fine-tuned on target pathologies.

- **Base Neural Model:** MobileNetV3-Small (Keras implementation with customized GlobalAveragePooling2D and Dropout regularization)
- **Input Vector Specification:** RGB channels, scale-normalized to standard size `[224, 224, 3]` via MobileNetV3 preprocess standards.
- **Quantization Pipeline:** TensorFlow Lite (TFLite) using Post-Training Dynamic Range Quantization (compressing 32-bit floating-point parameters to 8-bit integers).

---

## 2. Supported Species & Disease Taxonomy

The deployable model covers **15 unique target classes** representing crucial crop varieties and key plant pathologies:

### Peppers (Capsicum annuum)
- **Pepper Bell - Bacterial Spot** (*Xanthomonas campestris pv. vesicatoria*)
- **Pepper Bell - Healthy**

### Potatoes (Solanum tuberosum)
- **Potato - Early Blight** (*Alternaria solani*)
- **Potato - Late Blight** (*Phytophthora infestans*)
- **Potato - Healthy**

### Tomatoes (Solanum lycopersicum)
- **Tomato - Bacterial Spot** (*Xanthomonas campestris pv. vesicatoria*)
- **Tomato - Early Blight** (*Alternaria solani*)
- **Tomato - Late Blight** (*Phytophthora infestans*)
- **Tomato - Leaf Mold** (*Passalora fulva*)
- **Tomato - Septoria Leaf Spot** (*Septoria lycopersici*)
- **Tomato - Spider Mites (Two-Spotted Spider Mite)** (*Tetranychus urticae*)
- **Tomato - Target Spot** (*Corynespora cassiicola*)
- **Tomato - Yellow Leaf Curl Virus** (TYLCV)
- **Tomato - Mosaic Virus** (ToMV)
- **Tomato - Healthy**

---

## 3. Data Augmentation Policy & Biological Rationale

In field-level environments, smartphone capture angles and distances fluctuate significantly. To build resilience to these operational shifts, we utilize a highly strict data augmentation policy:

- **Authorized Augmentations:** Geometric modifications including random horizontal/vertical flips and minor rotations up to 5%.
- **Restricted Augmentations:** No brightness, contrast, hue, saturation, or color jitter transformations are applied. Since crop disease symptoms are highly color-dependent (e.g., distinct brown necrotic rings, chlorotic yellow margins, or powdery white leaf patches), altering original colors would distort critical biological signatures and introduce false diagnostic signals.

---

## 4. Optimization & Edge Performance Benchmarks

| Deployment Parameter | Baseline Keras Model (.h5) | Edge Optimized Model (.tflite) |
| :--- | :--- | :--- |
| **Model Size** | 11.21 MB | **1.09 MB** |
| **Footprint Reduction** | Baseline (1.0x) | **10.3x smaller** |
| **Validation Accuracy** | **91.0%** | **89.0%** (Only a 2.0% delta) |
| **Average Local Latency** | ~4.0 ms | **Sub-millisecond local execution** |
| **Deployment Target** | Server / Cloud APIs | On-device Mobile CPU / GPU Accelerators |
| **Parameter Precision** | 32-bit Floating-Point (Float32) | Post-Training Dynamic Range (Int8 Weights) |

---

## 5. Local Quality Assurance & Continuous Learning Loops

1. **Diagnostic Thresholding:** If the top predictive class returns a confidence score of **less than 75%**, the diagnostic assistant labels the scan as "Low Confidence - Requires Agronomist Audit".
2. **Active Learning Feedback Loop:** Flagged raw leaf images and low-confidence prediction vectors are temporarily stored inside an encrypted local sandbox. Once the device connects to an active network (Wi-Fi), these files are automatically synced to our main database for future active-learning training runs.
