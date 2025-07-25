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

### Count of Each Class (Bar Chart)

![Class Distribution - Count](Graphs%20and%20Pictures/Distribution%20of%20Dependent%20Variable.png)

### Percentage of Each Class (Pie Chart)

![Class Distribution - Percentage](Graphs%20and%20Pictures/Distribution%2Off%20Dependent%20Variable%20In%20Percentage.png)

---

## Model Details

![Model Architecture](Graphs%20and%20Pictures/model.png)

---

## Model Training

![Model Training - Loss and Accuracy](Graphs%20and%20Pictures/loss-accuracy.png)

The model was trained for 5 epochs using the Adam optimizer and Mean Squared Error (MSE) as the loss function. It achieved:

- Training Accuracy: 98%
- Training Loss: 0.014
- Test Accuracy: 99%
- Test Loss: 0.026

---

## Confusion Matrix

![Confusion Matrix](Graphs%20and%20Pictures/Confusion%20Matrix.png)


---

## Classification Report on Development Dataset

![Classification Report](Graphs%20and%20Pictures/Classification%20Report.png)


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
