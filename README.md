# Laboratory-Work-5-Activity-Comparative-Analysis

Colab: https://colab.research.google.com/drive/1GifkpEytwobgm4L-TZZZjOpHCXd3b4IK?usp=sharing


| Model - Sample | Train Accuracy | Train Loss | Test Accuracy | Test Loss | Precision | Recall | F1-score | ROC | AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Pre-Trained Model 1 (VGG16) | 0.7987 | 0.9584 | 0.9176 | 0.7414 | 0.93 | 0.92 | 0.92 | 0.97 | 0.98 |
| Pre-Trained Model 2 (ResNet50) | 0.9884 | 0.0601 | 0.9962  | 0.0233 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Pre-Trained Model 3 (MobileNetV2) | 0.0751 | 2.9762 | 0.0798 | 2.9709 | 0.01 |  0.08 | 0.01 | 0.58 | 0.59 |




# Laboratory Work 5: Comparative Analysis Reflection

## A. Model Performance

### 1. Which pre-trained model achieved the highest accuracy? Why?
* **Answer:** **Pre-Trained Model 2 (MobileNetV2)** achieved the highest performance, reaching an outstanding Test Accuracy of **99.62%** ($0.9962$).
* **Why:** MobileNetV2 is engineered with depthwise separable convolutions and inverted residual blocks, which allows it to extract highly efficient, distinct feature maps with fewer parameters. It seamlessly mapped the unique leaf variegation patterns of the *Aglaonema* dataset without suffering from overfitting or gradient saturation, leading to rapid, clean convergence by Epoch 10.

### 2. Which model had the lowest performance? What could be the reason?
* **Answer:** **Pre-Trained Model 3 (EfficientNetB0)** performed the poorest, flatlining at a Test Accuracy of just **7.98%** ($0.0798$).
* **Reason:** The model completely failed to converge. Looking at the training logs, the training loss stayed frozen at around $2.98$ across all epochs. This indicates a **vanishing gradient problem**, an incompatible learning rate (too high or too low for this specific architecture's optimizer), or dead weights in the newly attached classification layers. Because it never optimized, it simply defaulted to predicting a single majority class (`Aglaonema_Maria`), matching random chance.

### 3. How did loss values compare across models?
* **Answer:** * **MobileNetV2** showed an ideal optimization curve, dropping to a near-zero Test Loss of **0.0233**.
  * **VGG16** achieved a decent but noticeably higher Test Loss of **0.7414**, showing that while it learned well, it settled into a less precise local minimum than MobileNetV2.
  * **EfficientNetB0** failed to optimize entirely, retaining a massive Test Loss of **2.9709**, proving its weights were never updated effectively during training.

---

## B. Evaluation Metrics

### 4. Why is accuracy not enough to evaluate a model?
* **Answer:** Accuracy only looks at the overall percentage of correct guesses. If a dataset is imbalanced or a model completely breaks down (like EfficientNetB0), the model can simply spam predictions for one dominant class and still achieve a baseline accuracy score without actually identifying anything. Metrics like Precision, Recall, and F1-score evaluate each class independently, exposing whether a model is truly reliable or just playing a numbers game.

### 5. Which model had the best F1-score? What does it indicate?
* **Answer:** **MobileNetV2** achieved a perfect overall F1-score of **1.00**. 
* **What it indicates:** The F1-score is the harmonic mean of Precision and Recall. An F1-score of $1.00$ indicates that the model achieved virtually zero false positives and zero false negatives across the test dataset—it is incredibly precise and comprehensive across every single *Aglaonema* variety.

### 6. How did Precision and Recall differ across models?
The models showed completely different behaviors when balancing precision (avoiding false alarms) and recall (avoiding missed detections):

* **MobileNetV2 (Best Balanced):** Maintained a flawless, perfectly balanced **1.00 Precision** and **1.00 Recall** across almost every single class. It rarely missed a plant variant or mislabeled one.
* **VGG16 (Imbalanced/Aggressive):** Exhibited significant disparities depending on the plant variety.
    * *Example:* `Aglaonema_Lady_Valentine` scored a low **0.65 Precision** but a high **0.98 Recall**. 
    * *Meaning:* The model was overly aggressive at guessing *Lady Valentine*. It successfully caught almost every actual *Lady Valentine* image in the dataset (high recall), but it accidentally mislabeled many other pink-leafed varieties as *Lady Valentine* by mistake (low precision).
* **EfficientNetB0 (Collapsed):** Reached a completely useless **0.01 Precision** and **0.08 Recall**. Because the model failed to learn features, it simply predicted the majority class for almost every single image.

---

## C. Confusion Matrix Analysis

### 7. Which classes were frequently misclassified?
* **Under VGG16:** The primary point of confusion was **`Aglaonema_Lady_Valentine`** (yielding a low precision of `0.65`). Cultivars like **`Aglaonema_First_Diamond`** (`0.76` precision) and **`Aglaonema_Pictum_Tricolor`** (`0.74` recall) also saw frequent cross-class mistakes.
* **Under EfficientNetB0:** Because the model stalled out during training, virtually **every single class** was completely misclassified as **`Aglaonema_Maria`**.

### 8. What patterns did you observe in the confusion matrix?
* **The Perfect Diagonal:** In the strong **MobileNetV2** model, the confusion matrix forms a clean, heavy diagonal line, proving true classes matched predicted classes perfectly.
* **Morphological Clustering:** In the intermediate **VGG16** model, misclassifications cluster heavily between cultivars that share highly similar leaf shapes and physical features (e.g., mixing up different varieties that feature matching pink/red splotches or white variegation patterns).
* **The Vertical Bias Column:** In the failed **EfficientNetB0** model, the matrix shows a solid vertical column under one single class—a classic visual trait of a completely unoptimized model guessing the exact same class over and over.

---

## D. ROC and AUC

### 9. Which model had the highest AUC score?
**MobileNetV2** achieved the highest overall performance. In its ROC plot, nearly every single class curve leaps instantly to the top-left corner, maintaining an individual class **AUC of 1.00** across the board. 

**VGG16** followed closely behind as a strong competitor (averaging **0.98 AUC**), while **EfficientNetB0** produced highly erratic curves that hugged or crisscrossed the diagonal random-guess baseline (with individual AUC values dipping down into the `0.50`s and `0.60`s).

### 10. What does AUC tell us about model performance?
> **Definition:** Area Under the Curve (AUC) measures a model’s ability to cleanly separate distinct classes across all possible classification thresholds. 

* An **AUC of 1.00** means the model’s internal feature distributions for different classes do not overlap at all—it can distinguish between species flawlessly. 
* An **AUC near 0.50** indicates complete overlap, meaning the model cannot separate the classes at all and its predictions are no better than a random coin flip.

---

## E. Explainability (Grad-CAM)

### 11. What did Grad-CAM reveal about model decision-making?
Grad-CAM acts as a visual audit tool for neural networks. By calculating gradients with respect to the final convolutional layer, it highlights the exact pixels and regions within the image that the network relied on most heavily to make its final classification decision. This effectively opens up the "black box" of deep learning into a clear visual heatmap.

### 12. Did the model focus on relevant image regions?
* **VGG16:** Yes. It established a sharp, hyper-focused gaze directly on the prominent central midrib and localized interior variegation patterns of the leaf.
* **MobileNetV2:** Yes. It displayed a beautifully balanced focus covering both the interior foliage patterns and the overall leaf boundaries, explaining its superior test performance.
* **EfficientNetB0:** No. While its heatmap physically overlays the leaf area, the model's feature extraction layer is unoptimized due to the flatlined training loss. It is physically highlighting the region but pulling out unhelpful pixel noise instead of actual botanical features.

### 13. Which model produced the most meaningful heatmaps?
**MobileNetV2** produced the most meaningful heatmaps. It managed a well-distributed, comprehensive field of view over the defining characteristics of the plant foliage, avoiding the over-localized focus limitations of VGG16 and the unoptimized feature noise of EfficientNetB0.

---

## F. Model Comparison & Improvement

### 14. Which model would you recommend for deployment? Why?
**MobileNetV2** is the clear choice for production deployment based on two key factors:

| Factor | Performance Evaluation |
| :--- | :--- |
| **Raw Accuracy** | It strictly dominates the benchmarks with a Test Accuracy of **99.62%**, an F1-score of **1.00**, and a flawless ROC profile. |
| **Deployment Efficiency** | MobileNetV2 uses inverted residuals and depthwise separable convolutions. This keeps its file size exceptionally lightweight and computationally fast, making it ideal for mobile devices or edge hardware. |

### 15. How can you further improve your best-performing model?
* **Robust Data Augmentation:** Introduce aggressive color jittering, random rotations, and background swaps during training. This forces the model to focus strictly on leaf morphology rather than accidentally memorizing laboratory backgrounds or specific lighting conditions.
* **Fixing the Failed Model:** To get the underperforming EfficientNetB0 model to work, try using a much lower initial learning rate (e.g., $1\times10^{-4}$ or $1\times10^{-5}$) or implement a gradual layer-unfreezing schedule to prevent the training loss from freezing early on.

---

## G. Real-World Application

### 16. How can your model be applied in real-world scenarios?
This system can be integrated directly into:
* Automated inventory scanning tools for commercial plant nurseries and greenhouses.
* Smart garden mobile apps to help hobbyists identify rare indoor plants.
* Agricultural customs inspection checkpoints to instantly verify incoming cargo and track specific *Aglaonema* cultivars automatically.

### 17. What are the risks of deploying an inaccurate model?
If a failed architecture like EfficientNetB0 were deployed, it would misclassify almost every variant as `Aglaonema_Maria`. In a live business environment, this would result in:
* Massive inventory mix-ups and shipping errors.
* Incorrect plant care instructions being handed to customers (potentially killing sensitive cultivars).
* Incorrect pricing of premium variants, leading to financial losses and a complete loss of customer trust.

### 18. How can this system be integrated into a mobile/web app?
Because MobileNetV2 is optimized for efficiency, the trained `.h5` or `.keras` model weights can be exported and converted into a **TensorFlow Lite (TFLite)** or **ONNX** format. This lightweight file can be embedded directly into a mobile application framework (like Flutter, React Native, or native iOS/Android). This allows the application to process camera images and run inference entirely **on-device, locally**, without requiring an active internet connection or expensive cloud servers.
