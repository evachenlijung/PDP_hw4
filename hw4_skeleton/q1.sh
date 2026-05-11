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

# --- 1. 全域風格微調 ---
plt.rcParams.update({
    'font.size': 18, 
    'axes.titlesize': 30, 
    'axes.labelsize': 24,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 15,
    'axes.labelweight': 'bold'
})

def parse_log(filename):
    data = {}
    current_hint = None
    try:
        with open(filename, 'r') as f:
            for line in f:
                hint_match = re.search(r'\[(Hint.*?)\]', line)
                if hint_match:
                    current_hint = hint_match.group(1)
                    data[current_hint] = {'size': [], 'time': [], 'gflops': []}
                stat_match = re.search(r'Running size: (\d+).*?avg time: ([\d.]+)s, performance:\s+([\d.]+) GFLOPS', line)
                if stat_match and current_hint:
                    data[current_hint]['size'].append(int(stat_match.group(1)))
                    data[current_hint]['time'].append(float(stat_match.group(2)))
                    data[current_hint]['gflops'].append(float(stat_match.group(3)))
    except Exception: pass
    return data

results = parse_log('q1.log')
sizes = [128, 256, 512, 1024, 2048, 4096]
hint_keys = list(results.keys())
colors_map = {hint: plt.cm.coolwarm(np.linspace(0.05, 0.95, len(hint_keys)))[i] for i, hint in enumerate(hint_keys)}

# 加大畫布以容納圖例
fig, ax1 = plt.subplots(figsize=(26, 14))
ax2 = ax1.twinx()

# 調整 Y 軸範圍，下方留更多空間給 Time 標籤
ax1.set_ylim(-1000, 17500) 
ax2.set_yscale('log')
ax2.set_ylim(1e-7, 5.0) # 下限調低到 1e-7 給標籤空間

for hint in hint_keys:
    d = results[hint]
    if not d['size']: continue
    color = colors_map[hint]
    ax1.plot(d['size'], d['gflops'], marker='o', markersize=14, lw=5, color=color, label=f'{hint} (GFLOPS)')
    ax2.plot(d['size'], d['time'], marker='x', markersize=14, lw=2, ls='--', color=color, alpha=0.4, label=f'{hint} (Time)')

# --- 標籤排序與放置邏輯 ---
for s in sizes:
    current_stats = []
    for hint in hint_keys:
        if s in results[hint]['size']:
            idx = results[hint]['size'].index(s)
            current_stats.append({
                'hint': hint, 'gflops': results[hint]['gflops'][idx],
                'time': results[hint]['time'][idx], 'color': colors_map[hint]
            })
    if not current_stats: continue

    # A. GFLOPS 標籤 (由大到小排序)
    stats_sorted_g = sorted(current_stats, key=lambda x: x['gflops'], reverse=True)
    for i, item in enumerate(stats_sorted_g):
        y_pos = 17000 - (i * 750) 
        ax1.annotate(f"{item['gflops']:.0f}", xy=(s, item['gflops']), xytext=(s, y_pos),
                     ha='center', va='top', fontsize=24, color=item['color'], fontweight='black',
                     path_effects=[pe.withStroke(linewidth=3, foreground="white")],
                     arrowprops=dict(arrowstyle="-", color=item['color'], lw=0.8, alpha=0.2))

    # B. Time 標籤 (由大到小排序，且包含所有尺寸)
    stats_sorted_t = sorted(current_stats, key=lambda x: x['time'], reverse=True)
    for i, item in enumerate(stats_sorted_t):
        # 標籤垂直位移：在 1e-7 到 1e-5 之間排隊
        y_t_pos = 10**(np.log10(8e-6) + ((len(stats_sorted_t)-i) * 0.3))
        ax2.annotate(f"{item['time']:.5f}s", xy=(s, item['time']), xytext=(s, y_t_pos),
                     ha='center', va='bottom', fontsize=18, color=item['color'], 
                     fontweight='bold', path_effects=[pe.withStroke(linewidth=2, foreground="white")],
                     arrowprops=dict(arrowstyle="-", color=item['color'], lw=0.6, alpha=0.2))

# --- 圖例位置優化：移至左側垂直中央 ---
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()

# loc='center left' 設定參考點在圖例框的左邊中點
# bbox_to_anchor=(0.02, 0.5) 將參考點放在圖表橫向 2%、縱向 50% 的位置
ax1.legend(h1 + h2, l1 + l2, 
           loc='center left', 
           bbox_to_anchor=(0.02, 0.6), 
           ncol=1,                 # 建議改為 1 欄，垂直排列在左側空間較美觀
           frameon=True, 
           shadow=False, 
           edgecolor='black', 
           framealpha=1.0,
           fontsize=16)            # 確保字體大小適中

ax1.set_xscale('log', base=2)
ax1.set_xticks(sizes)
ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax1.set_xlabel("Matrix Size (M=N=K)", fontweight='bold')
ax1.set_ylabel("Performance (GFLOPS)", fontweight='bold', color='navy')
ax2.set_ylabel("Avg Time (s) - Log Scale", fontweight='bold', color='maroon')

plt.title("CUDA SGEMM Optimization Performance Analysis (Final Fix)", pad=40)
ax1.grid(True, which='both', ls=':', alpha=0.5)
plt.tight_layout()
plt.savefig('q1_plot.png', dpi=300)
print("Final corrected plot saved as q1_plot.png")



# --- 4. 產生無白邊效能數據表格 ---
fig_tab, ax_tab = plt.subplots(figsize=(18, 3)) 
ax_tab.axis('off')

# 必須先宣告這個字典，否則下方迴圈會報 NameError
hint_labels = {
    'Hint1': '1',
    'Hint2': '2',
    'Hint3_4': '3+4',
    'Hint5_6_7': '5+6+7'
}

header = ['Hint \ Size'] + [str(s) for s in sizes]
table_data = []

# 現在這個迴圈可以正確存取 hint_labels 了
for original_key in hint_keys:
    display_name = hint_labels.get(original_key, original_key)
    row = [display_name]
    d = results[original_key]
    for s in sizes:
        if s in d['size']:
            idx = d['size'].index(s)
            row.append(f"{d['gflops'][idx]:.3f}")
        else:
            row.append("N/A")
    table_data.append(row)

# ... 接下來是 the_table 的定義與存檔邏輯 ...

# 建立表格並設定欄位寬度比例
the_table = ax_tab.table(cellText=table_data,
                         colLabels=header,
                         loc='center')

the_table.auto_set_font_size(False)
the_table.set_fontsize(18)
the_table.scale(1, 2.5) 

# 對齊與樣式設定
for (row, col), cell in the_table.get_celld().items():
    if col == 0 or row == 0:
        cell.get_text().set_horizontalalignment('center')
    else:
        cell.get_text().set_horizontalalignment('right')
    
    if row == 0:
        cell.get_text().set_weight('bold')
        cell.set_facecolor('#f2f2f2')

# 關鍵修正：使用 tight 佈局並將填充 (pad) 設為 0
plt.savefig('q1_table.png', 
            bbox_inches='tight', 
            pad_inches=0.05, # 留極小的 0.05 吋邊距避免框線被切到
            dpi=300)
print("Cropped performance table saved as q1_table.png")

EOF