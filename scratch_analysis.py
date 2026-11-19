import pandas as pd
import numpy as np

# Image scores
df = pd.read_csv('results/image_scores.csv')
print('=== IMAGE EVAL RESULTS ===')
print(f'Total samples: {len(df)}')
print(f'Real samples: {(df.label==0).sum()}, Fake samples: {(df.label==1).sum()}')

real_df = df[df.label==0]
fake_df = df[df.label==1]

for col in ['vision_score', 'forensic_score', 'combined_score']:
    preds = (df[col] >= 0.5).astype(int)
    acc = (preds == df.label).mean()
    real_fp = (real_df[col] >= 0.5).sum()
    fake_tp = (fake_df[col] >= 0.5).sum()
    
    print(f'\n{col}:')
    print(f'  Accuracy @0.5: {acc:.1%}')
    print(f'  Real correctly classified: {(real_df[col] < 0.5).sum()}/{len(real_df)}')
    print(f'  Fake correctly classified: {fake_tp}/{len(fake_df)}')
    print(f'  Real misclassified as Fake (FP): {real_fp}/{len(real_df)}')
    print(f'  Mean score for Real: {real_df[col].mean():.4f}')
    print(f'  Mean score for Fake: {fake_df[col].mean():.4f}')

print('\n=== FORENSIC SCORE DISTRIBUTION ===')
r_f = real_df['forensic_score']
f_f = fake_df['forensic_score']
print(f'Real forensic mean: {r_f.mean():.4f} +/- {r_f.std():.4f}')
print(f'Fake forensic mean: {f_f.mean():.4f} +/- {f_f.std():.4f}')
print(f'Real forensic range: [{r_f.min():.4f}, {r_f.max():.4f}]')
print(f'Fake forensic range: [{f_f.min():.4f}, {f_f.max():.4f}]')

print('\n=== VISION SCORE DISTRIBUTION ===')
r_v = real_df['vision_score']
f_v = fake_df['vision_score']
print(f'Real vision mean: {r_v.mean():.4f}')
print(f'Fake vision mean: {f_v.mean():.4f}')
print(f'Real with score >= 0.999: {(r_v >= 0.999).sum()}/{len(real_df)}')
print(f'Real with score >= 0.5: {(r_v >= 0.5).sum()}/{len(real_df)}')
print(f'Fake with score <= 0.001: {(f_v <= 0.001).sum()}/{len(fake_df)}')
print(f'Fake with score >= 0.5: {(f_v >= 0.5).sum()}/{len(fake_df)}')

# Check video scores too
try:
    vdf = pd.read_csv('results/video_scores.csv')
    print('\n=== VIDEO EVAL RESULTS ===')
    print(f'Total: {len(vdf)}')
    print(vdf.describe())
except Exception as e:
    print(f'\nNo video scores: {e}')
