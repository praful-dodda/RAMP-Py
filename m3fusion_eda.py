from preprocess import *
from evaluate_models import *



# Configuration
model = "M3fusion"
eachYear = 2017 # The year you ran the analysis for
figDir = "./figs/modelEval"

# # yearly ozone analysis for M3fusion
# yearStart = 2000
# yearEnd = 2020
# yearDiff = 5

# # run analysis for M3fusion for the years 2000-2020
# for year in range(yearStart, yearEnd + 1, yearDiff):
#     m3_data = analyze_yearly_ozone_dataset(model, year)

# analyze_yearly_ozone_dataset(model, 2023)

# Overall ozone analysis
analyze_ozone_data(source = model, years = np.arange(1990, 2024))

# Model Performance Analysis
[eval_df, collocated_df] = run_evaluation_analysis([model], [eachYear])

final_df = collocated_df.copy()
metrics_df = calculate_performance_metrics(final_df)

# plot a scatter plot for the modeled vs observed ozone
plt.figure(figsize=(10, 8))
sns.scatterplot(data=final_df, x='observed_ozone', y='model_ozone', 
                hue='year', alpha=0.7, edgecolor=None)
plt.plot([0, 100], [0, 100], 'k--', linewidth=2)  # 1:1 line
plt.xlabel('Observed Ozone (ppb)')
plt.ylabel('Modeled Ozone (ppb)')
plt.title('Modeled vs Observed Ozone MDA8')
plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.xlim(0, 100)
# plt.ylim(0, 100)
# annotate the metrics
for metric, value in metrics_df.items():
    plt.annotate(f"{metric}: {value:.2f}", xy=(0.05, 0.95 - 0.05 * list(metrics_df.keys()).index(metric)),
                 xycoords='axes fraction', fontsize=10, color='black', ha='left', va='top')
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(f"{figDir}/modeled_vs_observed_{model}_{eachYear}.png", dpi=300)
plt.show()

# group the data based on the month and plot similar scatter plot for each month as subplots
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

plt.figure(figsize=(20, 15))
months = final_df['month'].unique()
for i, month in enumerate(sorted(months), 1):
    plt.subplot(3, 4, i)
    month_data = final_df[final_df['month'] == month]
    sns.scatterplot(data=month_data, x='observed_ozone', y='model_ozone', 
                    hue='year', alpha=0.7, edgecolor=None)
    # annotate the metrics for each month
    month_metrics = calculate_performance_metrics(month_data)
    for metric, value in month_metrics.items():
        plt.annotate(f"{metric}: {value:.2f}", xy=(0.05, 0.95 - 0.05 * list(month_metrics.keys()).index(metric)),
                     xycoords='axes fraction', fontsize=10, color='black', ha='left', va='top')
    plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
    # plot a 1:1 line   
    plt.plot([0, 100], [0, 100], 'k--', linewidth=2)  # 1:1 line
    plt.xlabel('Observed Ozone (ppb)')
    plt.ylabel('Modeled Ozone (ppb)')
    plt.title(f'Month: {month_names[month-1]}')
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(f"{figDir}/modeled_vs_observed_monthly_{model}_{eachYear}.png", dpi=300)
plt.show()

# similarly plot for each region
regions = final_df['region'].unique()
plt.figure(figsize=(20, 15))
for i, region in enumerate(sorted(regions), 1):
    plt.subplot(3, 4, i)
    region_data = final_df[final_df['region'] == region]
    sns.scatterplot(data=region_data, x='observed_ozone', y='model_ozone', 
                    hue='year', alpha=0.7, edgecolor=None)
    # annotate the metrics for each region
    region_metrics = calculate_performance_metrics(region_data)
    for metric, value in region_metrics.items():
        plt.annotate(f"{metric}: {value:.2f}", xy=(0.05, 0.95 - 0.05 * list(region_metrics.keys()).index(metric)),
                     xycoords='axes fraction', fontsize=10, color='black', ha='left', va='top')
    plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
    # plot a 1:1 line   
    plt.plot([0, 100], [0, 100], 'k--', linewidth=2)  # 1:1 line
    plt.xlabel('Observed Ozone (ppb)')
    plt.ylabel('Modeled Ozone (ppb)')
    plt.title(f'Region: {region}')
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(f"{figDir}/modeled_vs_observed_region_{model}_{eachYear}.png", dpi=300)
plt.show()