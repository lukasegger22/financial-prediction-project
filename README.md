# Financial Risk Management via NLP (FinBERT)

This repository contains the source code, models, and evaluation artifacts for the research paper: 
**"News-Sentiment Based Directional Prediction for Stocks and Bitcoin: A Transfer Learning Approach"**.

## 📌 Project Overview
This project implements a Transfer Learning pipeline using **FinBERT** to predict the directional movement of the DJIA index based on unstructured daily news headlines. It features custom Data Augmentation (Synthetic Signal Injection), a financial backtesting simulation, and a live inference dashboard.

## 📂 Project Structure
* `data/` - Datasets (DJIA and BTC historical news).
* `models/` - Directory for saved PyTorch model weights (`.pth`).
* `plots/` - Generated evaluation charts and HTML dashboards.
* `scripts/` - Executable Python scripts for training and inference.
* `src/` - Core modules (`model.py`, `data_loader.py`).

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <YOUR_GITHUB_LINK_HERE>
   cd <YOUR_FOLDER_NAME>

   ````
2. **Create a virtual environment (recommended)**
3. ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

   pip install -r requirements.txt
   ````
4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ````
5. **Usage / Execution**
   ```bash
   python scripts/train.py

   python scripts/backtest.py

   python scripts/live_dashboard.py
   ````
   
