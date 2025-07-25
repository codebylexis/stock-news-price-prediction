# Stock Price Behavior Prediction from News Headlines

Predicting stock price movements using the top 20 daily news headlines through deep learning and NLP.

---

## Features

- End-to-end deep learning pipeline for sentiment-based stock prediction
- CNN + Dense-layer hybrid neural network architecture
- Natural Language Processing using SpaCy, BeautifulSoup, and WordCloud
- Visual performance metrics including accuracy, loss, confusion matrix, ROC curve, and word clouds

---

## Directory Structure

```
MarketInsight-AI/
├── Dataset/
│   └── Data.csv
├── Graphs and Pictures/
├── Model/
│   └── stock_price_behavior_model.py
├── Notebook/
│   └── stock_price_behavior.ipynb
├── requirements.txt
└── README.md
```

---

## Libraries Used

- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- SpaCy
- BeautifulSoup
- WordCloud

---

## Model Architecture

The model consists of:
- Two 1D convolutional layers with ReLU activations, max-pooling, and dropout
- Three dense layers with ReLU → Sigmoid in the final output layer for binary classification

---

## Target Class Distribution

<img src="https://github.com/user-attachments/assets/9dfc9535-c3dd-4b05-948c-846dca6d85b1">
<br>
<p align="center"> 
<img src="https://github.com/user-attachments/assets/0770b64f-147a-405b-8aea-f2a99aa25407">
</p>    

---

## Model Details

<p align="center">
<img src="https://github.com/user-attachments/assets/382ce46b-4bca-4df3-800e-aacf7afa0da4">
</p>

---

## Model Training

<img src="https://github.com/user-attachments/assets/a8057142-d987-4444-88e6-b80688eeb819" alt="loss_accuracy">

The model was trained for 5 epochs using the Adam optimizer and Mean Squared Error (MSE) as the loss function. It achieved:

- Training Accuracy: 98%
- Training Loss: 0.014
- Test Accuracy: 99%
- Test Loss: 0.026

---

## Confusion Matrix

<img src="https://github.com/user-attachments/assets/5e5b24a7-f6d1-459e-81f5-1085d1c74c36" 
     style="display: block; margin-left: auto; margin-right: auto;">

---

## Classification Report on Development Dataset

<img src="https://github.com/user-attachments/assets/08f34bed-76a0-4271-96d7-01969564a889" 
     style="display: block; margin-left: auto; margin-right: auto;">

---

## Evaluation Summary

- Train Accuracy: 98%  
- Test Accuracy: 99%  
- Train Loss: 0.014  
- Test Loss: 0.026  

---

## Setup & Run

```bash
# Step 1: Create a virtual environment (if not using one already)
python -m venv venv
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate          # Windows

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Launch Jupyter Notebook
jupyter notebook Notebook/stock_price_behavior.ipynb
```

---

## Conclusion

This deep learning model predicts stock price behavior based on the sentiment in top financial news headlines with remarkable accuracy, offering data-driven insights for informed trading decisions.
