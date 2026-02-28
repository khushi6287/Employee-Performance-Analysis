
"""
**Import Libraries**
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

"""**Load Dataset**"""

df = pd.read_csv("employee_performance.csv")

df.head()

"""# **Central Tendency & Dispersion**

**Mean, Median, Mode (Salary)**
"""

print("Mean Salary:", df['Salary'].mean())
print("Median Salary:", df['Salary'].median())
print("Mode Salary:", df['Salary'].mode()[0])

"""**Variance & Standard Deviation**"""

print("Variance:", df['Projects_Completed'].var())
print("Standard Deviation:", df['Projects_Completed'].std())

"""# **Probability & Events**

**Probability of Promotion**
"""

promotion_prob = (df['Promotion_Status'] == "Yes").mean()
print("Probability of Promotion:", promotion_prob)

"""**Contingency Table**"""

cont_table = pd.crosstab(df['Department'], df['Promotion_Status'])
print(cont_table)

"""**Conditional Probability**"""

high_perf = df[df['Performance_Score'] > 80]

conditional_prob = (high_perf['Promotion_Status'] == "Yes").mean()
print("P(Promotion | Performance>80):", conditional_prob)

"""# **Distributions & Visualization**

**Histogram with Gaussian Curve**
"""

plt.figure(figsize=(8,5))

sns.histplot(df['Performance_Score'], kde=True)

plt.title("Performance Score Distribution")
plt.show()

"""**Skewness & Kurtosis (Salary)**"""

print("Skewness:", df['Salary'].skew())
print("Kurtosis:", df['Salary'].kurt())

"""**Q-Q Plot (Projects Completed)**"""

stats.probplot(df['Projects_Completed'], dist="norm", plot=plt)
plt.title("Q-Q Plot")
plt.show()

"""# **Linear Algebra Application**

**Create Employee Vectors**
"""

vectors = df[['Projects_Completed','Working_Hours']].head(5).values

v1 = vectors[0]
v2 = vectors[1]

"""**Dot Product**"""

dot_product = np.dot(v1, v2)
print("Dot Product:", dot_product)

"""**Norm 1 & Norm 2**"""

norm1 = np.linalg.norm(v1, 1)
norm2 = np.linalg.norm(v1)

print("Norm 1:", norm1)
print("Norm 2:", norm2)

"""**Angle Between Two Vectors**"""

angle = np.arccos(
    np.dot(v1, v2) /
    (np.linalg.norm(v1) * np.linalg.norm(v2))
)

print("Angle (radians):", angle)
