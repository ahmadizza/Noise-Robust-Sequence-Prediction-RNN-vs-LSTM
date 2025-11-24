# **Noise-Robust Sequence Prediction: RNN vs LSTM**

This project investigates the ability of **RNN** and **LSTM** models to perform sequence prediction under the presence of noise. The dataset consists of sequences containing **true input digits (0–9)** followed by **blank tokens (10)** acting as noise. The task is to reconstruct the original input sequence despite the blank tokens.

Two model architectures are implemented to evaluate **short-term** and **long-term** memory capabilities of RNNs and LSTMs.

---

## **Problem Description**

You are given a sequential dataset with:

* **INPUT**: a sequence of digits from 0 to 9
* **BLANK**: noise tokens (digit 10) appended after the input
* **TARGET**: identical to INPUT
* The model receives:
  **SAMPLE_DATA = INPUT + BLANK**

### Example

```
INPUT  = [2, 9, 2, 0, 0]
BLANK  = [10, 10, 10]
SAMPLE = [2, 9, 2, 0, 0, 10, 10, 10]
TARGET = [2, 9, 2, 0, 0]
```

The model must reconstruct the INPUT from SAMPLE, even though the last part of the sequence is noise.

Gradient clipping is used to prevent exploding gradients:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

---

## **Objectives**

1. Train **RNN** and **LSTM** models to predict the clean sequence.
2. Compare:

   * Training loss curves
   * Prediction accuracy
   * Gradient magnitude (optional)
3. Evaluate both models on **5 test samples**.
4. Analyze performance differences on short-term vs long-term memory tasks.

---

## **Model Architectures**

Two types of forward-pass architectures are implemented.

---

### **1. Standard Model (Predict from the Front)**

Uses the **first `INPUT_LEN` outputs**:

```python
output = self.fc(rnn_out[:, :INPUT_LEN, :])
```

* Tests **short-term dependency**
* Model predicts immediately after reading INPUT
* Task is easy because target is close to input

**Analogy:**
See 10 digits → directly write down the same 10 digits.

---

### **2. Memory Model (Predict from the Back)**

Uses the **last `INPUT_LEN` outputs**:

```python
output = self.fc(rnn_out[:, -INPUT_LEN:, :])
```

* Tests **long-term dependency**
* Model must remember INPUT after reading 20 blank tokens
* Much harder than the Standard model

**Analogy:**
See 10 digits → read 20 irrelevant tokens → THEN write down the original digits.

---

## **Components**

1. **Data Generation**
   Synthetic sequences with configurable input length and blank length.

2. **Two Model Types**

   * RNN (vanilla)
   * LSTM

   Each implemented twice:

   * Standard version
   * Memory version

3. **Training Process**

   * Adam optimizer
   * Cross-entropy loss
   * Gradient clipping (norm = 1.0)

4. **Visualization**

   * Loss curves (RNN vs LSTM)
   * Optional: gradient magnitude

5. **Evaluation**

   * 5 sample predictions for each model
   * Count matching digits (Correct / Total)

---

## **Results Summary**

### **Model 1 — Standard (Predict from Front)**

Both RNN and LSTM:

* Achieve **100% accuracy**
* Reach **near-zero loss within a few epochs**
* Have diminishing gradients as the task converges
* Easily memorize short dependencies

**Both models succeed completely**

---

### **Model 2 — Memory (Predict from Back)**

#### **RNN Results**

* Fails to learn
* Predictions collapse to a repeating digit (e.g., `[3,3,3,...]`)
* Accuracy: **0–30%**
* Loss stagnates at ~2.3
* Suffers from **vanishing gradients**

#### **LSTM Results**

* Successfully learns long-term dependencies
* Accuracy: **80–100%**
* Loss decreases consistently
* Gradient magnitudes remain healthy

---

## **Key Findings**

1. **RNN can handle only short-term dependencies**

   * Works perfectly in the Standard model
   * Fails completely in the Memory model

2. **LSTM handles both short-term and long-term dependencies**

   * Solves both tasks
   * Retains information despite long sequences of noise

3. **Vanishing gradient is the main failure mode of RNN**

   * Confirmed by gradient magnitude plots
   * LSTM overcomes this due to gating mechanisms

4. **Gradient clipping prevents exploding gradients**

   * Helps stabilize training
   * Especially important for long sequences

---

## **Example Evaluation Output**

```
Sample 1:
Target:  [1 6 8 3 7 3 8 0 3 4]
RNN:     [3 3 3 3 3 3 3 3 3 3]  (30%)
LSTM:    [1 6 8 3 7 3 8 0 3 4]  (100%)
```

---

## **Project Files**

```
/data_generation.py
/models_rnn_lstm.py
/train.py
/evaluate.py
/plots/
README.md
```

---

## **Conclusion**

This project clearly demonstrates:

* **RNN** is unreliable for long sequences due to vanishing gradients.
* **LSTM** is robust for both short and long dependencies thanks to its memory cell architecture.

Thus, **LSTM is the preferred choice** when the model must ignore noise and recall earlier parts of a sequence.

---

## **Keywords**

`RNN`, `LSTM`, `sequence modeling`, `noise robustness`,
`long-term dependency`, `gradient vanishing`, `gradient clipping`

---
