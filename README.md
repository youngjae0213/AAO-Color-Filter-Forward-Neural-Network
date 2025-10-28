# AAO-Color-Filter-Forward-Neural-Network
Predicting transmittance spectra of nanoporous AAO-based Fabry–Perot color filters using a PyTorch deep learning model.
This repository contains a PyTorch-based deep learning model for predicting transmittance spectra of nanoporous AAO-based Fabry–Perot color filters.

##  Overview
- **Goal:** Predict transmittance spectra (400–700 nm) of porous Al₂O₃ Fabry–Perot structures using Multi-Layer Perceptron(MLP).
- **Input Parameters:** pore radius, pore period, top Ag thickness, AAO thickness, bottom Ag thickness  
- **Output:** 301-point transmittance spectrum  
- **Model:** Fully connected ANN (MLP)
- **Loss Function:** MSE  
- **Data Normalization:** StandardScaler (z-score)

**Visualizations include:**
- Predicted vs Actual Spectra  
- Training and Validation Loss Curves  
- Scatter plot of predicted vs actual transmittance  

## Project Structure
AAO-Color-Filter-Forward-Neural-Network/
- AAO_FNN.py
- dataset_example.xlsx
- results/
  - RMSE_curve.png
  - predicted_vs_actual.png
  - scatter_all.png
