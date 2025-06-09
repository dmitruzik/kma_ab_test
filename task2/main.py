import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('kma_test_task.csv')

# create dates
df['install_date'] = pd.to_datetime(df['install_date'])

# adding cohort week and week of life
df['cohort_week'] = df['install_date'] - pd.to_timedelta(df['install_date'].dt.weekday, unit='d')
reference_date = df['install_date'].min()
df['lifetime_week'] = ((df['install_date'] - reference_date).dt.days // 7) + 1

# find amount of users in first cohort
cohort_sizes = df.groupby('cohort_week')['user_id'].nunique()

# sum of income by cohort and week of life
cohort_revenue = df.groupby(['cohort_week', 'lifetime_week'])['revenue_1m'].sum().reset_index()

# set cohort size
cohort_revenue['cohort_size'] = cohort_revenue['cohort_week'].map(cohort_sizes)

# income sum for cohort
cohort_revenue = df.groupby(['cohort_week', 'lifetime_week'])['revenue_1m'].sum().reset_index()
cohort_revenue['cohort_size'] = cohort_revenue['cohort_week'].map(cohort_sizes)
cohort_revenue['ltv'] = cohort_revenue['revenue_1m'] / cohort_revenue['cohort_size']
cohort_revenue['cumulative_ltv'] = cohort_revenue.groupby('cohort_week')['ltv'].cumsum()

# LTV on 1 week
ltv_start = cohort_revenue[cohort_revenue['lifetime_week'] == 1][['cohort_week', 'cumulative_ltv']]
ltv_start = ltv_start.rename(columns={'cumulative_ltv': 'ltv_actual'})

# LTV forecast for 52 weeks
weeks = np.arange(1, 53).reshape(-1, 1)
growth_factor = 4  # прогнозоване зростання в 4 рази
predicted_ltv_rows = []
for _, row in ltv_start.iterrows():
    base_ltv = row['ltv_actual']
    cohort = row['cohort_week']
    ltv_predicted = base_ltv * np.log1p(weeks) / np.log1p(52) * growth_factor

    for week, pred in zip(weeks.flatten(), ltv_predicted.flatten()):
        predicted_ltv_rows.append({
            'cohort_week': cohort,
            'week': week,
            'ltv_predicted': pred,
            'ltv_actual': base_ltv if week == 1 else None
        })
  # Create table
ltv_forecast = pd.DataFrame(predicted_ltv_rows)
ltv_forecast.to_csv('ltv_forecast.csv', index=False)


# Visualization
plt.figure(figsize=(12, 6))
for cohort in ltv_forecast['cohort_week'].unique():
    data = ltv_forecast[ltv_forecast['cohort_week'] == cohort]
    plt.plot(data['week'], data['ltv_predicted'], label=f'Прогноз {cohort.date()}')
    plt.scatter(1, data['ltv_actual'].dropna(), marker='o', s=100, label=f'Факт {cohort.date()}')

plt.title('Прогнозований та фактичний LTV на 52 тижні')
plt.xlabel('Тиждень життя користувача')
plt.ylabel('LTV')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()