import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import time

input_param = 5
output_param = 301

# 데이터 불러오기 및 전처리
df = pd.read_csv(r'D:\이영재\FDTD\data\FDTD_AAO_DL\merged_results.csv')

X = df.iloc[:, :input_param].values 
Y = df.iloc[:, input_param:].values  


# 정규화
scaler_X = StandardScaler()
scaler_Y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
Y_scaled = scaler_Y.fit_transform(Y)


# 데이터 준비
num_samples = X.shape[0]
idx = np.random.permutation(num_samples)

n_train = int(0.70 * num_samples)
n_val = int(0.15 * num_samples)
n_test = num_samples - n_train - n_val

idx_train = idx[:n_train]
idx_val = idx[n_train:n_train+n_val]
idx_test = idx[n_train+n_val:]

X_train, Y_train = X_scaled[idx_train], Y_scaled[idx_train]
X_val, Y_val     = X_scaled[idx_val], Y_scaled[idx_val]
X_test, Y_test   = X_scaled[idx_test], Y_scaled[idx_test]


X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
Y_val = torch.tensor(Y_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
Y_test = torch.tensor(Y_test, dtype=torch.float32)

batch_size = 16
train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=batch_size, shuffle=False)

# 모델
model = nn.Sequential(
    nn.Linear(input_param, 64),
    nn.ReLU(),
    nn.Linear(64, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),

    nn.Linear(128, output_param)
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        nn.init.zeros_(m.bias)
model.apply(init_weights)
model = model.to(device)


# 학습 설정
optimizer = optim.Adam(model.parameters(), lr=1e-3)
nb_epochs = 2000

train_losses = []
val_losses = []

def average(lst):
    return sum(lst) / len(lst)

start_time = time.time()

for epoch in range(nb_epochs + 1):
    model.train()
    batch_rmses = []
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = F.mse_loss(pred, yb, reduction = 'mean')
        rmse = torch.sqrt(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_rmses.append(rmse.item())
    train_rmse = average(batch_rmses)
    train_losses.append(train_rmse)

    # 검증
    model.eval()
    val_batch_rmses = []
    with torch.no_grad():
        all_preds = []
        all_targets = []
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            all_preds.append(pred.cpu())
            all_targets.append(yb.cpu())

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        mse = F.mse_loss(all_preds, all_targets, reduction='mean')
        val_rmse = torch.sqrt(mse).item()
        val_losses.append(val_rmse)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Train rmse = {train_rmse:.6f} | Val rmse = {val_rmse:.6f}")

# 테스트 및 역정규화
model.eval()
with torch.no_grad():
    pred_test = model(X_test.to(device)).cpu().numpy()

pred_test = scaler_Y.inverse_transform(pred_test)
Y_test_np = scaler_Y.inverse_transform(Y_test.numpy())

# RMSE 출력
mse = np.mean((pred_test - Y_test_np) ** 2)
rmse = np.sqrt(mse)
print(f"Test RMSE: {rmse:.6f}")


torch.save({
    'model_state': model.state_dict(),
    'scaler_X': scaler_X,
    'scaler_Y': scaler_Y,
}, 'FNN_T.pth')
print("Full checkpoint (model + scaler) saved as forward_fnn_lab.pth")


end_time = time.time()
print(f"총 훈련 시간: {end_time - start_time:.2f}초")

# RMSE 엑셀 출력
epochs = list(range(len(train_losses)))
df_rmse = pd.DataFrame({
    'Epoch': epochs,
    'Train_RMSE': train_losses,
    'Val_RMSE': val_losses
})
df_rmse.to_excel('1_rmse_vs_epoch.xlsx', index=False)


# 예측 VS 실제 스펙트럼 엑셀 출력
cols = df.columns[input_param:]

wavelengths = pd.to_numeric(cols.str.replace('T_', '', regex=False), errors='coerce').to_numpy()
if np.any(np.isnan(wavelengths)) or len(wavelengths) != 301:
    raise ValueError(f"Wavelength header parsing mismatch: got {len(wavelengths)} vs out_dim {out_dim}")

actual_df = pd.DataFrame(np.column_stack([wavelengths, Y_test_np.T]))
pred_df   = pd.DataFrame(np.column_stack([wavelengths, pred_test.T]))
actual_df.columns = ['Wavelength'] + [f'Sample_{i+1}' for i in range(Y_test_np.shape[0])]
pred_df.columns   = ['Wavelength'] + [f'Sample_{i+1}' for i in range(pred_test.shape[0])]
with pd.ExcelWriter('2_spectrum_prediction_results.xlsx') as writer:
    actual_df.to_excel(writer, sheet_name='Actual', index=False)
    pred_df.to_excel(writer, sheet_name='Predicted', index=False)

# 산점도 엑셀 출력
y_true = Y_test_np.flatten()       
y_pred = pred_test.flatten()       
df_scatter = pd.DataFrame({
    'Actual_Transmittance': y_true,
    'Predicted_Transmittance': y_pred
})
df_scatter.to_excel('3_scatter_actual_vs_predicted.xlsx', index=False)



plt.figure(figsize=(8, 5))
plt.plot(train_losses, label='Train Loss', color='blue')
plt.plot(val_losses, label='Validation Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training and Validation Loss')
plt.grid()
plt.legend()
plt.tight_layout()


plt.figure(figsize=(10, 8))
for i in range(3):
    plt.subplot(3, 1, i+1)
    plt.plot(wavelengths, Y_test_np[i], label='Actual', color='blue')
    plt.plot(wavelengths, pred_test[i], '--', label='Predicted', color='red')
    plt.title(f'Sample {i+1}')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Transmittance')
    plt.legend()
    plt.grid(True)
plt.tight_layout()



plt.figure(figsize=(6, 6))
plt.scatter(Y_test_np.flatten(), pred_test.flatten(), alpha=0.4, s=10)
plt.plot([0, 1], [0, 1], 'k--', label='Ideal')  # 기준선 y = x
plt.xlabel('Actual Transmittance')
plt.ylabel('Predicted Transmittance')
plt.title('Predicted vs Actual Transmittance')
plt.grid(True)
plt.legend()
plt.axis('equal')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.tight_layout()
plt.show()


