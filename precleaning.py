import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# read RNA expression data
rna = pd.read_csv('data/rna.tsv.gz', sep='\t', index_col=0)
print("RNA shape:", rna.shape)

# read DNA methylation data
meth = pd.read_csv('data/methylation.tsv.gz', sep='\t', index_col=0)
print("Methylation shape:", meth.shape)

# read cnv data
cnv = pd.read_csv('data/cnv.tsv.gz', sep='\t', index_col=0)
print("CNV shape:", cnv.shape)

# read survival data
clinical = pd.read_csv('data/clinical.txt', sep='\t')
clinical = clinical[['sample', 'OS.time', 'OS']]
clinical.columns = ['sample', 'time', 'event']
clinical = clinical.dropna()
clinical['event'] = clinical['event'].astype(int)
print("Clinical shape:", clinical.shape)

print(" rna :", rna.columns[:5])
print(" methylation :", meth.columns[:5])
print(" cnv :", cnv.columns[:5])
print(" clinical :", clinical['sample'].values[:5])

# find out the common samples
common_samples = set(rna.columns) & set(meth.columns) & set(cnv.columns) & set(clinical['sample'])
common_samples = sorted(list(common_samples))

X_rna = rna[common_samples].T
X_meth = meth[common_samples].T
X_cnv = cnv[common_samples].T

y = clinical[clinical['sample'].isin(common_samples)].set_index('sample').loc[common_samples]

def clean_features(df):
    df = df.dropna(axis=1, how='all')
    df = df.fillna(0)
    df = df.loc[:, df.std(axis=0) != 0]
    return df

def select_top_variance_features(df, top_n=1000):
    variances = df.var(axis=0)
    top_features = variances.sort_values(ascending=False).head(top_n).index
    return df[top_features]

X_rna_filtered = select_top_variance_features(clean_features(X_rna), top_n=1000)
X_meth_filtered = select_top_variance_features(clean_features(X_meth), top_n=1000)
X_cnv_filtered = select_top_variance_features(clean_features(X_cnv), top_n=1000)

scaler = StandardScaler()
X_rna_scaled = pd.DataFrame(scaler.fit_transform(X_rna_filtered), index=X_rna_filtered.index, columns=X_rna_filtered.columns)
X_meth_scaled = pd.DataFrame(scaler.fit_transform(X_meth_filtered), index=X_meth_filtered.index, columns=X_meth_filtered.columns)
X_cnv_scaled = pd.DataFrame(scaler.fit_transform(X_cnv_filtered), index=X_cnv_filtered.index, columns=X_cnv_filtered.columns)

# save the cleaned data
with open('data/tmp/X_rna_scaled.pkl', 'wb') as f:
    pickle.dump(X_rna_scaled, f)

with open('data/tmp/X_meth_scaled.pkl', 'wb') as f:
    pickle.dump(X_meth_scaled, f)

with open('data/tmp/X_cnv_scaled.pkl', 'wb') as f:
    pickle.dump(X_cnv_scaled, f)

with open('data/tmp/survival_info.pkl', 'wb') as f:
    pickle.dump(y[['time', 'event']], f)

print("Final RNA input shape:", X_rna_scaled.shape)
print("Final Methylation input shape:", X_meth_scaled.shape)
print("Final CNV input shape:", X_cnv_scaled.shape)
print("Survival info shape:", y.shape)
