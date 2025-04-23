import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index

# load data
with open('data/tmp/X_rna_scaled.pkl', 'rb') as f:
    X_rna_scaled = pickle.load(f)
with open('data/tmp/X_meth_scaled.pkl', 'rb') as f:
    X_meth_scaled = pickle.load(f)
with open('data/tmp/X_cnv_scaled.pkl', 'rb') as f:
    X_cnv_scaled = pickle.load(f)
with open('data/tmp/survival_info.pkl', 'rb') as f:
    y = pickle.load(f)

# split data for training, testing, validation
X_rna_np = X_rna_scaled.values
X_meth_np = X_meth_scaled.values
X_cnv_np = X_cnv_scaled.values
y_time = y['time'].values
y_event = y['event'].values

X_rna_train, X_rna_test, X_meth_train, X_meth_test, X_cnv_train, X_cnv_test, y_time_train, y_time_test, y_event_train, y_event_test = train_test_split(
    X_rna_np, X_meth_np, X_cnv_np, y_time, y_event, test_size=0.2, random_state=42
)

X_rna_train = torch.FloatTensor(X_rna_train)
X_meth_train = torch.FloatTensor(X_meth_train)
X_cnv_train = torch.FloatTensor(X_cnv_train)
y_time_train = torch.FloatTensor(y_time_train)
y_event_train = torch.FloatTensor(y_event_train)

X_rna_test = torch.FloatTensor(X_rna_test)
X_meth_test = torch.FloatTensor(X_meth_test)
X_cnv_test = torch.FloatTensor(X_cnv_test)
y_time_test = torch.FloatTensor(y_time_test)
y_event_test = torch.FloatTensor(y_event_test)

# define the model

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
            nn.ReLU()
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
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, rna, meth, cnv):
        r = self.rna_net(rna)
        m = self.meth_net(meth)
        c = self.cnv_net(cnv)
        out = torch.cat([r, m, c], dim=1)
        return self.final(out)

def cox_ph_loss(risk, time, event):
    order = torch.argsort(time, descending=True)
    risk = risk[order]
    event = event[order]
    hazard_ratio = torch.exp(risk)
    log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0))
    uncensored_likelihood = risk - log_risk
    loss = -torch.mean(uncensored_likelihood * event)
    return loss


#  model training
model = MultiOmicsSurvNet(
    rna_dim=X_rna_train.shape[1],
    meth_dim=X_meth_train.shape[1],
    cnv_dim=X_cnv_train.shape[1]
)

optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
n_epochs = 100

print("\nStart training...")
for epoch in range(1, n_epochs + 1):
    model.train()
    optimizer.zero_grad()
    pred = model(X_rna_train, X_meth_train, X_cnv_train).squeeze()
    loss = cox_ph_loss(pred, y_time_train, y_event_train)
    loss.backward()
    optimizer.step()
    scheduler.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# model assessment
model.eval()
with torch.no_grad():
    test_risk = model(X_rna_test, X_meth_test, X_cnv_test).squeeze().numpy()
    c_index = concordance_index(y_time_test.numpy(), -test_risk, y_event_test.numpy())
    print(f"\n✅ C-Index on Test Set: {c_index:.4f}")
