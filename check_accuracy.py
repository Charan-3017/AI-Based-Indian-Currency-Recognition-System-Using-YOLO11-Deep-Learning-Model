import pandas as pd

df = pd.read_csv(r"C:\Users\91984\OneDrive\Desktop\SkillArion\runs\Training\classify\train\results.csv")

last = df.iloc[-1]

print("Top-1 Accuracy:", last["metrics/accuracy_top1"] * 100, "%")
print("Top-5 Accuracy:", last["metrics/accuracy_top5"] * 100, "%")