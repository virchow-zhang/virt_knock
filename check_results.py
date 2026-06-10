import pandas as pd, os

out_dir = r"D:\OneDrive\bioinformatics\MCAO\GSE331114\expression_matrices\output_oligo_tf_batch_v2"
s = pd.read_csv(os.path.join(out_dir, "batch_summary.tsv"), sep="\t")
print(f"=== OLIGODENDROCYTES TF KNOCKOUT RESULTS ===")
print(f"Total genes after QC: 7,656")
print(f"TF knockout genes: {len(s)}")
print(f"Total significant genes (adj.p < 0.05): {s['n_sig'].sum()}")
print(f"Mean sig per TF: {s['n_sig'].mean():.1f}")
print()

top = s.nlargest(15, "n_sig")
print("TOP 15 TFs by perturbation magnitude:")
for _, r in top.iterrows():
    pct = r["n_sig"] / r["n_total"] * 100
    print(f"  {r['TF']:15s}  {r['n_sig']:4d} / {r['n_total']:4d}  ({pct:.2f}%)")

print()
# Distribution
bins = [(0,1), (1,5), (5,10), (10,20), (20,50), (50,1000)]
for lo, hi in bins:
    n = ((s["n_sig"] > lo) & (s["n_sig"] <= hi)).sum()
    if n > 0:
        print(f"  {lo+1}-{hi} sig genes: {n} TFs")
