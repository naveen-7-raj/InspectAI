# Explainable AI (XAI) Methodology in InspectAI

## 1. Rationale for Explainability in Predictive Maintenance

Deep learning models applied to raw vibration signals often function as "black boxes". In mission-critical industrial manufacturing, plant maintenance engineers cannot dispatch repair technicians or halt a production line without verifiable evidence.

**InspectAI** provides mathematical and physical explainability to bridge the gap between deep neural network predictions and physical fault dynamics.

---

## 2. Implemented XAI Methods

### 2.1 1D & 2D Grad-CAM (Gradient-Weighted Class Activation Mapping)
- **Concept**: Computes the gradients of the target class score $y^c$ with respect to feature activation maps $A^k$ of the final convolutional layer:
  $$\alpha_k^c = \frac{1}{Z} \sum_{i} \frac{\partial y^c}{\partial A_i^k}$$
  $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
- **Physical Interpretation in InspectAI**:
  - For **1D Waveforms**: Highlights transient periodic impact bursts in the time domain, matching the exact timestamps when a bearing ball strikes an inner or outer race fault.
  - For **2D Spectrograms**: Highlights spectral energy concentrations at specific fault harmonics (e.g., BPFO, BPFI) and their sidebands.

### 2.2 Integrated Gradients (Axiomatic Attribution)
- Evaluates the path integral of gradients along the straight line from a baseline signal $x'$ (e.g., zero baseline or a baseline healthy vibration profile) to input $x$:
  $$IG_i(x) = (x_i - x'_i) \times \int_{0}^{1} \frac{\partial F(x' + \alpha (x - x'))}{\partial x_i} d\alpha$$
- Satisfies **Completeness** and **Implementation Invariance**, ensuring attribution values accurately sum up to the difference between prediction and baseline score.

---

## 3. Physical Validation Loop
Explainability outputs are not merely visualized—they are mapped to physical frequency bands calculated in `docs/data_dictionary.md`. An alert is considered "Confirmed with High Confidence" when high Grad-CAM activations align with theoretical BPFO/BPFI resonance zones.

