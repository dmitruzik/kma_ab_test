import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("kma_test_task.csv")
print(df.head())

# Count main metrics:

# Кількість користувачів у кожній групі
print("\nUser count:")
print(df['test_group'].value_counts())

# Конверсія в пробний період
print("\nTrial conversion rate:")
print(df.groupby('test_group')['trial'].mean())

# Конверсія у платну підписку
print("\nPaid conversion rate:")
print(df.groupby('test_group')['paid'].mean())

# ARPU — середній дохід на користувача
print("\nARPU:")
print(df.groupby('test_group')['revenue_1m'].mean())

# ARPPU — середній дохід серед тих, хто оплатив
print("\nARPPU:")
print(df[df['paid'] == 1].groupby('test_group')['revenue_1m'].mean())



# Приклад графіка ARPU
arpu = df.groupby('test_group')['revenue_1m'].mean()
arpu.plot(kind='bar', title='ARPU by Test Group')
plt.ylabel('Revenue ($)')
plt.show()

summary = {
    "Trial Conversion Rate": df.groupby('test_group')['trial'].mean(),
    "Paid Conversion Rate": df.groupby('test_group')['paid'].mean(),
    "ARPU": df.groupby('test_group')['revenue_1m'].mean(),
    "ARPPU": df[df['paid'] == 1].groupby('test_group')['revenue_1m'].mean()
}

summary_df = pd.DataFrame(summary)
summary_df.to_csv("ab_test_summary.csv")
