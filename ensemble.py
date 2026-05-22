# import pickle
# import numpy as np
# from tqdm import tqdm

# # ==============================
# # PATHS
# # ==============================
# LABEL_PATH = '/home/teaching/Desktop/mml/data/WLASL2000/preprocessed/test_label.pkl'

# JOINT_PATH = '../work_dir/wlasl_joint/eval_results/best_acc.pkl'
# BONE_PATH = '../work_dir/wlasl_bone/eval_results/best_acc.pkl'
# JOINT_MOTION_PATH = '../work_dir/wlasl_joint_motion/eval_results/best_acc.pkl'
# BONE_MOTION_PATH = '../work_dir/wlasl_bone_motion/eval_results/best_acc.pkl'

# # ==============================
# # LOAD DATA
# # ==============================
# with open(LABEL_PATH, 'rb') as f:
#     label = np.array(pickle.load(f))

# with open(JOINT_PATH, 'rb') as f:
#     r1 = list(pickle.load(f).items())

# with open(BONE_PATH, 'rb') as f:
#     r2 = list(pickle.load(f).items())

# with open(JOINT_MOTION_PATH, 'rb') as f:
#     r3 = list(pickle.load(f).items())

# with open(BONE_MOTION_PATH, 'rb') as f:
#     r4 = list(pickle.load(f).items())

# # ==============================
# # 🔥 BEST PRACTICAL WEIGHTS
# # ==============================
# #alpha = [1.0, 1.6, 0.3, 0.5]
# #alpha = [1.6, 1.2, 0.3, 0.3]
# alpha = [1.4, 1.3, 0.4, 0.4]
# # ==============================
# # ENSEMBLE
# # ==============================
# right_num = 0
# right_num_5 = 0
# total_num = 0

# for i in tqdm(range(len(label[0]))):
#     name, true_label = label[:, i]

#     _, s1 = r1[i]
#     _, s2 = r2[i]
#     _, s3 = r3[i]
#     _, s4 = r4[i]

#     # weighted sum
#     score = (
#         s1 * alpha[0] +
#         s2 * alpha[1] +
#         s3 * alpha[2] +
#         s4 * alpha[3]
#     ) / sum(alpha)

#     # top-5
#     rank_5 = score.argsort()[-5:]
#     right_num_5 += int(int(true_label) in rank_5)

#     # top-1
#     pred = np.argmax(score)
#     right_num += int(pred == int(true_label))

#     total_num += 1

# # ==============================
# # RESULTS
# # ==============================
# print("\n FINAL RESULTS ")
# print("Top1 Accuracy: {:.2f}%".format((right_num / total_num) * 100))
# print("Top5 Accuracy: {:.2f}%".format((right_num_5 / total_num) * 100))



import pickle
import numpy as np
from tqdm import tqdm
import json

# ==============================
# PATHS
# ==============================
LABEL_PATH = '/home/teaching/Desktop/mml/data/WLASL2000/preprocessed/test_label.pkl'

JOINT_PATH = '../work_dir/wlasl_joint/eval_results/best_acc.pkl'
BONE_PATH = '../work_dir/wlasl_bone/eval_results/best_acc.pkl'
JOINT_MOTION_PATH = '../work_dir/wlasl_joint_motion/eval_results/best_acc.pkl'
BONE_MOTION_PATH = '../work_dir/wlasl_bone_motion/eval_results/best_acc.pkl'

# ==============================
# LOAD DATA
# ==============================
with open(LABEL_PATH, 'rb') as f:
    label = np.array(pickle.load(f))

with open(JOINT_PATH, 'rb') as f:
    r1 = list(pickle.load(f).items())

with open(BONE_PATH, 'rb') as f:
    r2 = list(pickle.load(f).items())

with open(JOINT_MOTION_PATH, 'rb') as f:
    r3 = list(pickle.load(f).items())

with open(BONE_MOTION_PATH, 'rb') as f:
    r4 = list(pickle.load(f).items())

# ==============================
# WEIGHTS
# ==============================
alpha = [1.4, 1.3, 0.4, 0.4]

# ==============================
# STORE RESULTS
# ==============================
results = []

for i in tqdm(range(len(label[0]))):
    name, true_label = label[:, i]
    true_label = int(true_label)

    _, s1 = r1[i]
    _, s2 = r2[i]
    _, s3 = r3[i]
    _, s4 = r4[i]

    score = (
        s1 * alpha[0] +
        s2 * alpha[1] +
        s3 * alpha[2] +
        s4 * alpha[3]
    ) / sum(alpha)

    pred = int(np.argmax(score))
    pred_conf = float(score[pred])
    true_conf = float(score[true_label])

    top5 = score.argsort()[-5:][::-1]
    top5 = [int(x) for x in top5]

    correct = bool(pred == true_label)
    in_top5 = bool(true_label in top5)

    results.append({
        "name": str(name),
        "true_label": int(true_label),
        "pred_label": int(pred),
        "pred_confidence": float(pred_conf),
        "true_label_confidence": float(true_conf),
        "top5": top5,
        "correct": correct,
        "in_top5": in_top5
    })

# ==============================
# SORT + SELECT TOP 25
# ==============================
results = sorted(results, key=lambda x: x["pred_confidence"], reverse=True)
top25 = results[:25]

# ==============================
# SPLIT GROUPS
# ==============================
top1_correct = []
top5_correct = []
others = []

for r in top25:
    if r["correct"]:
        top1_correct.append(r)
    elif r["in_top5"]:
        top5_correct.append(r)
    else:
        others.append(r)

# ==============================
# FINAL JSON
# ==============================
final_output = {
    "Top1_Correct": top1_correct,
    "Top5_Correct_but_Top1_Wrong": top5_correct,
    "Others": others
}

# ==============================
# SAVE JSON
# ==============================
with open("top25_analysis.json", "w") as f:
    json.dump(final_output, f, indent=4)

print("\n✅ JSON saved successfully: top25_analysis.json")

# ==============================
# SUMMARY
# ==============================
print("\n📊 SUMMARY")
print("Top1 Correct:", len(top1_correct))
print("Top5 Correct:", len(top5_correct))
print("Others:", len(others))