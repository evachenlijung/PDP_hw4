# #!/bin/bash

# # 定義要測試的核心檔案清單
# KERNELS=("naive_kernel.cu" "shared_memory_tiling.cu" "2d_thread_bloacktiling_register_cahcing.cu" "student_kernel.cu")
# HINTS=("Hint1" "Hint2" "Hint3_4" "Hint5_6_7")
# RUNNER="kernels/runner.cu"
# LOG="q1.log"

# # 清空舊的 log
# echo "CUDA SGEMM Optimization Benchmark Log" > $LOG
# echo "======================================" >> $LOG

# # 備份原始 runner.cu
# cp $RUNNER ${RUNNER}.bak

# for i in "${!KERNELS[@]}"; do
#     FILE=${KERNELS[$i]}
#     HINT=${HINTS[$i]}
    
#     echo "Testing $HINT: $FILE..."
    
#     # 使用 sed 修改 runner.cu 中的 #include 行
#     # 註解掉所有可能的核心包含行，然後插入目標核心
#     sed -i 's/^#include ".*_kernel.cu"/ \/\/ #include "old"/' $RUNNER
#     sed -i 's/^#include "shared_memory_tiling.cu"/ \/\/ #include "old"/' $RUNNER
#     sed -i 's/^#include "2d_thread_bloacktiling_register_cahcing.cu"/ \/\/ #include "old"/' $RUNNER
#     # 在 0_cublas.cu 之後插入目標包含行
#     sed -i "/#include \"0_cublas.cu\"/a #include \"$FILE\"" $RUNNER

#     # 編譯與執行 [cite: 39, 42]
#     make clean > /dev/null 2>&1
#     make > /dev/null 2>&1
    
#     if [ $? -eq 0 ]; then
#         echo "[$HINT]" >> $LOG
#         srun -N 1 -n 1 --gpus-per-node 1 -A ACD115083 -t 1 ./main >> $LOG 2>&1
#         echo "" >> $LOG
#     else
#         echo "Compilation failed for $FILE"
#     fi
# done

# # 還原 runner.cu
# mv ${RUNNER}.bak $RUNNER

# echo "Benchmark complete. Parsing results and plotting..."

# --- 2. 使用 Python 繪製雙軸圖表 ---
python3 - <<'EOF'
import re
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

# --- 1. 全域風格設定 ---
plt.rcParams.update({
    'font.size': 20, 
    'axes.titlesize': 32, 
    'axes.labelsize': 26,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'legend.fontsize': 18,
    'axes.labelweight': 'bold'
})

def parse_log(filename):
    data = {}
    mapping = {'Hint1': 'Baseline', 'Hint2': 'Opt 1', 'Hint3_4': 'Opt 2', 'Hint5_6_7': 'Opt 3'}
    try:
        with open(filename, 'r') as f:
            for line in f:
                hint_match = re.search(r'\[(Hint.*?)\]', line)
                if hint_match:
                    current_key = mapping.get(hint_match.group(1))
                    if current_key: data[current_key] = {'size': [], 'time': [], 'gflops': []}
                stat_match = re.search(r'Running size: (\d+).*?avg time: ([\d.]+)s, performance:\s+([\d.]+) GFLOPS', line)
                if stat_match and current_key:
                    data[current_key]['size'].append(int(stat_match.group(1)))
                    data[current_key]['time'].append(float(stat_match.group(2)))
                    data[current_key]['gflops'].append(float(stat_match.group(3)))
    except Exception: pass
    return data

results = parse_log('q1.log')
opt_order = ['Baseline', 'Opt 1', 'Opt 2', 'Opt 3']
sizes = [128, 256, 512, 1024, 2048, 4096]
colors_map = {opt: plt.cm.coolwarm(np.linspace(0.05, 0.95, len(opt_order)))[i] for i, opt in enumerate(opt_order)}

# --- 2. 趨勢圖優化 ---
fig, ax1 = plt.subplots(figsize=(26, 14))
ax2 = ax1.twinx()

ax1.set_ylim(-1500, 19500) 
ax2.set_yscale('log')
ax2.set_ylim(1e-7, 10.0)

for opt in opt_order:
    if opt not in results: continue
    d = results[opt]
    color = colors_map[opt]
    # 加深虛線顏色，並區隔於實線
    ax1.plot(d['size'], d['gflops'], marker='o', markersize=14, lw=6, color=color, label=f'{opt} (GFLOPS)', zorder=5)
    ax2.plot(d['size'], d['time'], marker='x', markersize=14, lw=3, ls='--', color=color, alpha=0.6, label=f'{opt} (Time)', zorder=4)

# --- 核心標籤排序放置邏輯 ---
for s in sizes:
    current_stats = []
    for opt in opt_order:
        if opt in results and s in results[opt]['size']:
            idx = results[opt]['size'].index(s)
            current_stats.append({
                'opt': opt, 
                'gflops': results[opt]['gflops'][idx],
                'time': results[opt]['time'][idx], 
                'color': colors_map[opt]
            })
    if not current_stats: continue

    # A. GFLOPS 標籤：數值大 -> 小 (從上到下)
    stats_sorted_g = sorted(current_stats, key=lambda x: x['gflops'], reverse=True)
    for i, item in enumerate(stats_sorted_g):
        y_g_pos = 18500 - (i * 900) 
        ax1.annotate(f"{item['gflops']:.0f}", xy=(s, item['gflops']), xytext=(s, y_g_pos),
                     ha='center', va='top', fontsize=24, color=item['color'], fontweight='black',
                     path_effects=[pe.withStroke(linewidth=3, foreground="white")],
                     arrowprops=dict(arrowstyle="-", color=item['color'], lw=1.0, alpha=0.3))

    # B. Time 標籤：數值大 -> 小 (從上到下)
    stats_sorted_t = sorted(current_stats, key=lambda x: x['time'], reverse=True)
    for i, item in enumerate(stats_sorted_t):
        # Log 尺度垂直間距計算：數值大的 i 較小，排在上面
        y_t_pos = 10**(np.log10(1e-7) + ((len(stats_sorted_t) - i) * 0.55))
        ax2.annotate(f"{item['time']:.5f}s", xy=(s, item['time']), xytext=(s, y_t_pos),
                     ha='center', va='bottom', fontsize=18, color=item['color'], 
                     fontweight='bold', path_effects=[pe.withStroke(linewidth=3, foreground="white")],
                     arrowprops=dict(arrowstyle="-", color=item['color'], lw=1.0, alpha=0.3))

# 圖例放置於左側中央避開標籤
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc='center left', bbox_to_anchor=(0.02, 0.6), ncol=1, frameon=True, edgecolor='black', framealpha=1.0)

ax1.set_xscale('log', base=2); ax1.set_xticks(sizes); ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax1.set_xlabel("Matrix Size (M=N=K)", fontweight='bold'); ax1.set_ylabel("Performance (GFLOPS)", fontweight='bold', color='navy')
ax2.set_ylabel("Avg Time (s) - Log Scale", fontweight='bold', color='maroon')
plt.title("CUDA SGEMM Optimization Scaling Analysis", pad=40)
plt.tight_layout()
plt.savefig('q1_plot.png', dpi=300)

# --- 3. 表格優化 (非均勻欄寬) ---
fig_tab, ax_tab = plt.subplots(figsize=(24, 6))
ax_tab.axis('off')

opt_labels = {
    'Baseline': 'Baseline: Naïve kernel (Global-memory coalescing)',
    'Opt 1': 'Opt. 1: Shared memory tiling (TILE_SIZE=32)',
    'Opt 2': 'Opt. 2: 2D Tiling + Reg caching + Bank-conflict-free',
    'Opt 3': 'Opt. 3: Vec loads/stores + Warp-level + Double buffering'
}

header = ['Optimization \ Size'] + [str(s) for s in sizes]
table_data = []
for opt in opt_order:
    if opt not in results: continue
    row = [opt_labels[opt]]
    d = results[opt]
    for s in sizes:
        if s in d['size']: row.append(f"{d['gflops'][d['size'].index(s)]:.3f}")
        else: row.append("N/A")
    table_data.append(row)

the_table = ax_tab.table(cellText=table_data, colLabels=header, loc='center')
the_table.auto_set_font_size(False); the_table.set_fontsize(18)

# 設定非均勻欄寬比例
col_widths = [0.42] + [0.1] * len(sizes)
for (row, col), cell in the_table.get_celld().items():
    cell.set_width(col_widths[col])
    if col == 0 or row == 0: cell.get_text().set_horizontalalignment('center')
    else: cell.get_text().set_horizontalalignment('right')
    if row == 0: cell.get_text().set_weight('bold'); cell.set_facecolor('#f2f2f2')

the_table.scale(1.0, 4.0)
plt.savefig('q1_table.png', bbox_inches='tight', pad_inches=0.05, dpi=300)
print("Plot and Table with sorted labels generated.")
EOF