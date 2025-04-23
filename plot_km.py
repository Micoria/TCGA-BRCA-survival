import pickle
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt
import torch

# Step 1: 加载测试集数据和训练好的模型
with open('data/tmp/X_rna_scaled.pkl', 'rb') as f:
    X_rna = pickle.load(f)
with open('data/tmp/X_meth_scaled.pkl', 'rb') as f:
    X_meth = pickle.load(f)
with open('data/tmp/X_cnv_scaled.pkl', 'rb') as f:
    X_cnv = pickle.load(f)
with open('data/tmp/survival_info.pkl', 'rb') as f:
    survival = pickle.load(f)

# 只用测试集部分重新划分一次（与训练脚本保持一致）
from sklearn.model_selection import train_test_split
X_rna, X_rna_temp, X_meth, X_meth_temp, X_cnv, X_cnv_temp, y_time, y_time_temp, y_event, y_event_temp = train_test_split(
    X_rna.values, X_meth.values, X_cnv.values,
    survival['time'].values, survival['event'].values,
    test_size=0.3, random_state=42
)

X_rna_val, X_rna_test, X_meth_val, X_meth_test, X_cnv_val, X_cnv_test, y_time_val, y_time_test, y_event_val, y_event_test = train_test_split(
    X_rna_temp, X_meth_temp, X_cnv_temp,
    y_time_temp, y_event_temp,
    test_size=0.5, random_state=42
)

# 转换成 tensor
X_rna_test = torch.FloatTensor(X_rna_test)
X_meth_test = torch.FloatTensor(X_meth_test)
X_cnv_test = torch.FloatTensor(X_cnv_test)

# Step 2: 定义模型结构（与训练时保持一致）
class SubNet(torch.nn.Module):
    def __init__(self, input_dim):
        super(SubNet, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3)
        )

    def forward(self, x):
        return self.model(x)

class MultiOmicsSurvNet(torch.nn.Module):
    def __init__(self, rna_dim, meth_dim, cnv_dim):
        super(MultiOmicsSurvNet, self).__init__()
        self.rna_net = SubNet(rna_dim)
        self.meth_net = SubNet(meth_dim)
        self.cnv_net = SubNet(cnv_dim)
        self.final = torch.nn.Sequential(
            torch.nn.Linear(64 * 3, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )

    def forward(self, rna, meth, cnv):
        r = self.rna_net(rna)
        m = self.meth_net(meth)
        c = self.cnv_net(cnv)
        return self.final(torch.cat([r, m, c], dim=1))

# Step 3: 载入最优模型参数
model = MultiOmicsSurvNet(1000, 1000, 1000)
model.load_state_dict(torch.load('best_model.pt'))  # 如果你没保存，可以不写这行
model.eval()

# Step 4: 计算风险评分
with torch.no_grad():
    risks = model(X_rna_test, X_meth_test, X_cnv_test).squeeze().numpy()

# Step 5: 分组 + 生存曲线
df = pd.DataFrame({
    'risk': risks,
    'time': y_time_test,
    'event': y_event_test
})
median_risk = np.median(df['risk'])
df['group'] = (df['risk'] > median_risk).astype(int)

kmf = KaplanMeierFitter()

plt.figure(figsize=(8, 6))

for group in [0, 1]:
    label = 'High Risk' if group == 1 else 'Low Risk'
    kmf.fit(durations=df[df['group'] == group]['time'],
            event_observed=df[df['group'] == group]['event'],
            label=label)
    kmf.plot_survival_function(ci_show=False)

# Log-rank test
results = logrank_test(
    df[df['group'] == 0]['time'], df[df['group'] == 1]['time'],
    event_observed_A=df[df['group'] == 0]['event'],
    event_observed_B=df[df['group'] == 1]['event']
)

plt.title(f'Kaplan-Meier Survival Curve (p = {results.p_value:.4f})')
plt.xlabel('Time (days)')
plt.ylabel('Survival probability')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('km_curve.png', dpi=300)
plt.close()
