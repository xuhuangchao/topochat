import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import mendeleev
import pandas as pd

# 绘制热力图数据
df = pd.read_excel('element-ratio.xlsx')
plot_data = dict(zip(df['Element'], df['SOC_Percentage']))

# df = pd.read_excel('element-count.xlsx')
# plot_data = dict(zip(df['Element'], df['SOC_Element_Count']))

# 元素周期表cell的设置
cell_length = 1
cell_gap = 0.1
cell_edge_width = 0.5

# 获取元素信息
elements = []
for i in range(1, 119):
    try:
        ele = mendeleev.element(i)
        ele_group, ele_period = ele.group_id, ele.period

        # 镧系元素处理
        if 57 <= i <= 71:
            ele_group = i - 57 + 3
            ele_period = 8
        # 锕系元素处理
        if 89 <= i <= 103:
            ele_group = i - 89 + 3
            ele_period = 9

        elements.append([i, ele.symbol, ele_group, ele_period,
                        plot_data.setdefault(ele.symbol, 0)])
    except Exception as e:
        print(f"Error processing element {i}: {e}")

# 添加镧系和锕系标签
elements.extend([
    [None, 'LA', 3, 6, None],
    [None, 'AC', 3, 7, None],
    [None, 'LA', 2, 8, None],
    [None, 'AC', 2, 9, None]
])

# 创建图形和子图
fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
xy_length = (20, 11)

# 配置颜色映射
my_cmap = cm.get_cmap('YlOrRd')
norm = mpl.colors.Normalize(1, 100)
# norm = mpl.colors.Normalize(1, 900)

my_cmap.set_under('None')
cmmapable = cm.ScalarMappable(norm, my_cmap)

# 添加颜色条
cbar = plt.colorbar(cmmapable, ax=ax, drawedges=False)
cbar.set_label('Occurrence Rate(%)', rotation=270, labelpad=15)

# 绘制元素
for e in elements:
    ele_number, ele_symbol, ele_group, ele_period, ele_count = e

    if ele_group is None:
        continue

    x = (cell_length + cell_gap) * (ele_group - 1)
    y = xy_length[1] - ((cell_length + cell_gap) * ele_period)

    if ele_period >= 8:
        y -= cell_length * 0.5

    if ele_number:
        fill_color = my_cmap(norm(ele_count)) if ele_count > 0 else 'white'
        rect = patches.Rectangle(
            xy=(x, y),
            width=cell_length,
            height=cell_length,
            linewidth=cell_edge_width,
            edgecolor='k',
            facecolor=fill_color
        )
        ax.add_patch(rect)

    # 添加文本标签
    if ele_number:
        ax.text(x + 0.04, y + 0.8, ele_number,
                va='center', ha='left',
                fontsize=6, color='black')
    ax.text(x + 0.5, y + 0.5, ele_symbol,
            va='center', ha='center',
            fontsize=9, color='black', weight='bold')
    if ele_count:
        ax.text(x + 0.5, y + 0.12, ele_count,
                va='center', ha='center',
                fontsize=6, color='black')

# 设置图形属性
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('Element Distribution in Topological Materials', pad=20)
plt.tight_layout()
ax.set_ylim(0, xy_length[1])
ax.set_xlim(0, xy_length[0])

# 保存图形
# plt.savefig('./periodic_table_count.png', dpi=300, bbox_inches='tight')
# plt.savefig('./periodic_table_count.eps', format='eps', bbox_inches='tight')  # 保存为eps格式

plt.savefig('./periodic_table_rate.png', dpi=300, bbox_inches='tight')
plt.savefig('./periodic_table_rate.eps', format='eps', bbox_inches='tight')  # 保存为eps格式

# plt.show()