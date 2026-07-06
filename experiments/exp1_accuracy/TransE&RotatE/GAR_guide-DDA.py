import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# 1. 读取文件
predictions_df = pd.read_csv('/mnt/f/graduate/GARplus/exp1_accuracy/exp1_accuracy/TransE&RotatE/predictions/DDA_10pct/predictions_with_original_ids.csv')
negative_rules_df = pd.read_csv('/mnt/f/graduate/GARplus/exp1_accuracy/exp1_accuracy/TransE&RotatE/negative_pairs/dda_rule_negative_pairs_0626.csv')

# 2. 构建需要修正的边集合
negative_pairs_to_fix = set(zip(negative_rules_df['chemical_index'], 
                                negative_rules_df['disease_index']))

print(f"师姐提供了 {len(negative_pairs_to_fix)} 条需要修正为负边的样本")

# 3. 复制一份数据框用于修正
corrected_predictions_df = predictions_df.copy()
modification_log = []

# 4. 遍历并修正
for index, row in corrected_predictions_df.iterrows():
    pair = (row['head_old'], row['tail_old'])
    if pair in negative_pairs_to_fix:
        if row['pred_relation'] in [0, 1]:
            modification_log.append({
                'head_old': row['head_old'],
                'tail_old': row['tail_old'],
                'original_pred': row['pred_relation'],
                'new_pred': 2,
                'true_relation': row['true_relation']
            })
            corrected_predictions_df.at[index, 'pred_relation'] = 2

# 5. 保存修改日志
pd.DataFrame(modification_log).to_csv('/mnt/f/graduate/GARplus/exp1_accuracy/exp1_accuracy/TransE&RotatE/modified_prediction/DDA_10pct.csv', index=False)
print(f"共修改了 {len(modification_log)} 条预测结果")

# 6. 计算指标（完全复制 predict_triples.py 的输出格式）
true_labels = predictions_df['true_relation']
orig_preds = predictions_df['pred_relation']
corr_preds = corrected_predictions_df['pred_relation']

# 原始结果
orig_acc = accuracy_score(true_labels, orig_preds)
orig_f1 = f1_score(true_labels, orig_preds, average="macro")
orig_pre = precision_score(true_labels, orig_preds, average="macro", zero_division=0)
orig_rec = recall_score(true_labels, orig_preds, average="macro", zero_division=0)
orig_rec_per_class = recall_score(true_labels, orig_preds, average=None, labels=[0, 1, 2], zero_division=0)
orig_pre_per_class = precision_score(true_labels, orig_preds, average=None, labels=[0, 1, 2], zero_division=0)

# 修正后结果
corr_acc = accuracy_score(true_labels, corr_preds)
corr_f1 = f1_score(true_labels, corr_preds, average="macro")
corr_pre = precision_score(true_labels, corr_preds, average="macro", zero_division=0)
corr_rec = recall_score(true_labels, corr_preds, average="macro", zero_division=0)
corr_rec_per_class = recall_score(true_labels, corr_preds, average=None, labels=[0, 1, 2], zero_division=0)
corr_pre_per_class = precision_score(true_labels, corr_preds, average=None, labels=[0, 1, 2], zero_division=0)

# ===== 原始结果输出（完全复制原格式）=====
print("\n" + "=" * 40)
print("原始结果:")
print(f"Accuracy : {orig_acc:.4f}")
print(f"Macro F1 : {orig_f1:.4f}")
print(f"Precision: {orig_pre:.4f}")
print(f"Recall   : {orig_rec:.4f}")
print("-" * 40)
print(f"Class 0 (No Edge) - Prec: {orig_pre_per_class[0]:.4f}, Rec: {orig_rec_per_class[0]:.4f}")
print(f"Class 1 (Pos Edge) - Prec: {orig_pre_per_class[1]:.4f}, Rec: {orig_rec_per_class[1]:.4f}")
print(f"Class 2 (Neg Edge) - Prec: {orig_pre_per_class[2]:.4f}, Rec: {orig_rec_per_class[2]:.4f}")
print("=" * 40)

# ===== 修正后结果输出（完全复制原格式）=====
print("\n" + "=" * 40)
print("修正后结果:")
print(f"Accuracy : {corr_acc:.4f}")
print(f"Macro F1 : {corr_f1:.4f}")
print(f"Precision: {corr_pre:.4f}")
print(f"Recall   : {corr_rec:.4f}")
print("-" * 40)
print(f"Class 0 (No Edge) - Prec: {corr_pre_per_class[0]:.4f}, Rec: {corr_rec_per_class[0]:.4f}")
print(f"Class 1 (Pos Edge) - Prec: {corr_pre_per_class[1]:.4f}, Rec: {corr_rec_per_class[1]:.4f}")
print(f"Class 2 (Neg Edge) - Prec: {corr_pre_per_class[2]:.4f}, Rec: {corr_rec_per_class[2]:.4f}")
print("=" * 40)

# ===== 提升情况（额外但必要，方便汇报）=====
print("\n提升情况:")
print(f"Accuracy : {corr_acc - orig_acc:+.4f}")
print(f"Macro F1 : {corr_f1 - orig_f1:+.4f}")
print(f"Precision: {corr_pre - orig_pre:+.4f}")
print(f"Recall   : {corr_rec - orig_rec:+.4f}")
print(f"Class 2 (Neg Edge) - Prec: {corr_pre_per_class[2] - orig_pre_per_class[2]:+.4f}, Rec: {corr_rec_per_class[2] - orig_rec_per_class[2]:+.4f}")