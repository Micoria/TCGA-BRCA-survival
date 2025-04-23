import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# 1. Load processed data
# ------------------------------
with open('data/tmp/X_rna_scaled.pkl', 'rb') as f:
    X_rna_scaled = pickle.load(f)
with open('data/tmp/X_meth_scaled.pkl', 'rb') as f:
    X_meth_scaled = pickle.load(f)
with open('data/tmp/X_cnv_scaled.pkl', 'rb') as f:
    X_cnv_scaled = pickle.load(f)
with open('data/tmp/survival_info.pkl', 'rb') as f:
    y = pickle.load(f)

X_rna_np = X_rna_scaled.values
X_meth_np = X_meth_scaled.values
X_cnv_np = X_cnv_scaled.values
y_time = y['time'].values
y_event = y['event'].values

# ------------------------------
# 2. Split into train / val / test (70/15/15)
# ------------------------------
X_rna_train, X_rna_temp, X_meth_train, X_meth_temp, X_cnv_train, X_cnv_temp, y_time_train, y_time_temp, y_event_train, y_event_temp = train_test_split(
    X_rna_np, X_meth_np, X_cnv_np, y_time, y_event, test_size=0.3, random_state=42
)

X_rna_val, X_rna_test, X_meth_val, X_meth_test, X_cnv_val, X_cnv_test, y_time_val, y_time_test, y_event_val, y_event_test = train_test_split(
    X_rna_temp, X_meth_temp, X_cnv_temp, y_time_temp, y_event_temp, test_size=0.5, random_state=42
)

# Convert to tensors
def to_tensor(*arrays):
    return [torch.FloatTensor(arr) for arr in arrays]

X_rna_train, X_meth_train, X_cnv_train, y_time_train, y_event_train = to_tensor(
    X_rna_train, X_meth_train, X_cnv_train, y_time_train, y_event_train
)
X_rna_val, X_meth_val, X_cnv_val, y_time_val, y_event_val = to_tensor(
    X_rna_val, X_meth_val, X_cnv_val, y_time_val, y_event_val
)
X_rna_test, X_meth_test, X_cnv_test, y_time_test, y_event_test = to_tensor(
    X_rna_test, X_meth_test, X_cnv_test, y_time_test, y_event_test
)

# ------------------------------
# 3. Define model
# ------------------------------
class SubNet(nn.Module):
    def __init__(self, input_dim):
        super(SubNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        return self.model(x)

class MultiOmicsSurvNet(nn.Module):
    def __init__(self, rna_dim, meth_dim, cnv_dim):
        super(MultiOmicsSurvNet, self).__init__()
        self.rna_net = SubNet(rna_dim)
        self.meth_net = SubNet(meth_dim)
        self.cnv_net = SubNet(cnv_dim)
        self.final = nn.Sequential(
            nn.Linear(64 * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, rna, meth, cnv):
        r = self.rna_net(rna)
        m = self.meth_net(meth)
        c = self.cnv_net(cnv)
        return self.final(torch.cat([r, m, c], dim=1))

def cox_ph_loss(risk, time, event):
    order = torch.argsort(time, descending=True)
    risk = risk[order]
    event = event[order]
    hazard_ratio = torch.exp(risk)
    log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0))
    uncensored_likelihood = risk - log_risk
    return -torch.mean(uncensored_likelihood * event)

# ------------------------------
# 4. Train model with EarlyStopping
# ------------------------------
model = MultiOmicsSurvNet(
    rna_dim=X_rna_train.shape[1],
    meth_dim=X_meth_train.shape[1],
    cnv_dim=X_cnv_train.shape[1]
)

optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
n_epochs = 50

train_losses = []
val_losses = []

best_val_loss = float('inf')
best_model_state = None
patience = 5
no_improve = 0

print("\nStart training...")

for epoch in range(1, n_epochs + 1):
    model.train()
    optimizer.zero_grad()
    pred = model(X_rna_train, X_meth_train, X_cnv_train).squeeze()
    loss = cox_ph_loss(pred, y_time_train, y_event_train)
    loss.backward()
    optimizer.step()
    scheduler.step()
    train_losses.append(loss.item())

    # Validation loss
    model.eval()
    with torch.no_grad():
        val_pred = model(X_rna_val, X_meth_val, X_cnv_val).squeeze()
        val_loss = cox_ph_loss(val_pred, y_time_val, y_event_val).item()
        val_losses.append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict()
        no_improve = 0
        torch.save(model.state_dict(), 'best_model.pt')
    else:
        no_improve += 1

    if epoch % 5 == 0:
        print(f"Epoch {epoch}: Train Loss = {loss.item():.4f}, Val Loss = {val_loss:.4f}")

    if no_improve >= patience:
        print(f"\n⏹️ Early stopping at epoch {epoch}, best val loss: {best_val_loss:.4f}")
        break

model.load_state_dict(best_model_state)

# ------------------------------
# 5. Evaluate model + plot
# ------------------------------
model.eval()
with torch.no_grad():
    test_risk = model(X_rna_test, X_meth_test, X_cnv_test).squeeze().numpy()
    c_index = concordance_index(y_time_test.numpy(), -test_risk, y_event_test.numpy())
    print(f"\n✅ C-Index on Test Set: {c_index:.4f}")

# Plot dual loss curves
plt.figure(figsize=(8, 6))
plt.plot(train_losses, label='Train Loss', marker='o', linewidth=2)
plt.plot(val_losses, label='Val Loss', marker='s', linewidth=2)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Cox Loss', fontsize=14)
plt.title('Loss Curve', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig('loss_curve.png', dpi=300)
plt.close()
