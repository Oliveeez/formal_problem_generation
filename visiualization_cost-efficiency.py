# Visualize all valid statements

with open('cost_valid_complexity.data.pkl', 'rb') as f:
    (data, data_line) = pickle.load(f)

sns.set_style("whitegrid")
plt.figure(figsize=(24, 6))

colors = ["red", "blue", "green", "yellow", "purple", 'cyan', 'orange', 'purple']
# 绘制小提琴图

ax = plt.gca()
ax.set_ylim(0, 100)
ax.set_xlabel(r'Token Cost $(\log_{10})')
ax.set_ylabel('Perentage of Valid Statements (%)')

data_mustard_28316 = data[data['方法'] == 'MUSTARD']
data_rest_5000 = data[data['方法'] != 'MUSTARD']

ax2 = ax.twinx()
sns.violinplot(x="预算值", y="难度", hue="方法", data=data_rest_5000, 
    inner=None,  # 不显示均值线，避免干扰
    native_scale=True,
    density_norm='count',
    common_norm=True,
    # log_scale=True,
    width=10,
    palette=colors,
    alpha=0.5,
    # showextrema=False
    cut=0,
    ax=ax2
    )
sns.violinplot(x="预算值", y="难度", hue="方法", data=data_mustard_28316, 
    inner=None,  # 不显示均值线，避免干扰
    native_scale=True,
    density_norm='count',
    common_norm=True,
    # log_scale=True,
    width=10/28316*5000,
    palette=[colors[-1]],
    alpha=0.5,
    # showextrema=False
    cut=0,
    ax=ax2
    )
ax2.set_ylim(0, 1500)
ax2.set_yticks([i * 300 for i in range(6)])
ax2.set_ylabel('Complexity')
ax2.legend(loc='upper left', title='')

for i, (label, (x_data, y_data)) in enumerate(data_line.items()):
    sns.lineplot(x=x_data, y=y_data, label=label, marker='o', color=colors[i], alpha=1.0, ax=ax)
ax.get_legend().remove()

# plt.title("Distribution")
# plt.show()
plt.tight_layout()
plt.savefig('./cost_valid_complexity.data.pdf')


# Visualize problem-solving questions

with open('cost_valid-solving_difficulty.data.pkl', 'rb') as f:
    (data, data_line) = pickle.load(f)

sns.set_style("whitegrid")
plt.figure(figsize=(24, 6))

colors = ["red", "blue", "green", "yellow", "purple", 'cyan', 'orange']
# 绘制小提琴图

ax = plt.gca()
ax.set_ylim(0, 100)
ax.set_xlabel(r'Token Cost $(\log_{10})$')
ax.set_ylabel('Perentage of Valid Problem-solving Questions (%)')

data_mustard_28316 = data[data['方法'] == 'MUSTARD']
data_rest_5000 = data[data['方法'] != 'MUSTARD']

ax2 = ax.twinx()
sns.violinplot(x="预算值", y="难度", hue="方法", data=data_rest_5000, 
    inner=None,  # 不显示均值线，避免干扰
    native_scale=True,
    density_norm='count',
    common_norm=True,
    # log_scale=True,
    width=10,
    palette=colors,
    alpha=0.5,
    # showextrema=False
    cut=0,
    ax=ax2
    )
sns.violinplot(x="预算值", y="难度", hue="方法", data=data_mustard_28316, 
    inner=None,  # 不显示均值线，避免干扰
    native_scale=True,
    density_norm='count',
    common_norm=True,
    # log_scale=True,
    width=10/28316*5000,
    palette=[colors[-1]],
    alpha=0.5,
    # showextrema=False
    cut=0,
    ax=ax2
    )
ax2.set_ylim(0, 1)
# ax2.set_yticks([i * 300 for i in range(6)])
ax2.set_ylabel('Item Difficulty')
ax2.legend(loc='upper left', title='')

for i, (label, (x_data, y_data)) in enumerate(data_line.items()):
    sns.lineplot(x=x_data, y=y_data, label=label, marker='o', color=colors[i], alpha=1.0, ax=ax)
ax.get_legend().remove()

# plt.title("Distribution")
# plt.show()
plt.tight_layout()
plt.savefig('./cost_valid-solving_difficulty.data.pdf')