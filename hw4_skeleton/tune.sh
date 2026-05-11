#!/bin/bash

# --- 1. 配置搜尋空間 ---
BM_LIST=(128)
BN_LIST=(128)
BK_LIST=(8 16 32)
THREAD_TILES=("8,8" "8,4" "4,8" "4,4") 

OUTPUT_FILE="tuning_results.csv"
# 寫入標題列
echo "BM,BN,BK,TM,TN,GFLOPS" > $OUTPUT_FILE

# 定義檔案路徑
ORIGINAL_KERNEL="kernels/student_kernel.cu"
TEMP_KERNEL="kernels/student_kernel_temp.cu"

echo "開始參數掃描（已保護原始檔案）..."

for BM in "${BM_LIST[@]}"; do
    for BN in "${BN_LIST[@]}"; do
        for BK in "${BK_LIST[@]}"; do
            for T_TILE in "${THREAD_TILES[@]}"; do
                TM=$(echo $T_TILE | cut -d',' -f1)
                TN=$(echo $T_TILE | cut -d',' -f2)
                
                # 計算 Block 內的執行緒總數
                BT=$(( (BM / TM) * (BN / TN) ))
                
                # 硬體限制檢查：V100 單個 Block 執行緒上限為 1024 [cite: 22, 24]
                if [ $BT -lt 32 ] || [ $BT -gt 1024 ]; then continue; fi

                echo -n "測試: BM=$BM, BN=$BN, BK=$BK, TM=$TM, TN=$TN ($BT threads)... "

                # 關鍵防護：複製原始碼到臨時檔案，不更動原檔
                cp $ORIGINAL_KERNEL $TEMP_KERNEL
                
                # 修改臨時檔案中的參數
                sed -i "s/#define BM [0-9]*/#define BM $BM/" $TEMP_KERNEL
                sed -i "s/#define BN [0-9]*/#define BN $BN/" $TEMP_KERNEL
                sed -i "s/#define BK [0-9]*/#define BK $BK/" $TEMP_KERNEL
                sed -i "s/#define TM [0-9]*/#define TM $TM/" $TEMP_KERNEL
                sed -i "s/#define TN [0-9]*/#define TN $TN/" $TEMP_KERNEL

                # 暫時將原檔更名，讓副本頂替進行編譯
                mv $ORIGINAL_KERNEL ${ORIGINAL_KERNEL}.safe_bak
                mv $TEMP_KERNEL $ORIGINAL_KERNEL

                # 編譯與執行 [cite: 38, 39, 42]
                make clean > /dev/null 2>&1
                make > /dev/null 2>&1
                COMPILE_STATUS=$?

                if [ $COMPILE_STATUS -eq 0 ]; then
                    # 擷取 4096 尺寸的效能數值 [cite: 51]
                    # 使用 sed 確保精確擷取 performance: 之後的數字
                    RESULT=$(srun -N 1 -n 1 --gpus-per-node 1 -A ACD115083 -t 5 ./main 2>/dev/null | \
                             grep "size: 4096" | \
                             sed -n 's/.*performance: \([0-9.]*\) GFLOPS.*/\1/p')
                    
                    if [ -z "$RESULT" ]; then
                        echo "執行失敗（可能超時或記憶體錯誤）"
                    else
                        echo "$RESULT GFLOPS"
                        echo "$BM,$BN,$BK,$TM,$TN,$RESULT" >> $OUTPUT_FILE
                    fi
                else
                    echo "編譯失敗（可能資源超出限制）"
                fi

                # 核心防護：編譯後立即還原原始檔案
                mv $ORIGINAL_KERNEL $TEMP_KERNEL
                mv ${ORIGINAL_KERNEL}.safe_bak $ORIGINAL_KERNEL
                rm -f $TEMP_KERNEL
            done
        done
    done
done

# --- 2. 彙整結果並排序 ---
echo ""
echo "--------------------------------------------------"
echo "掃描完成！效能前五強參數組合 (GFLOPS):"
echo "BM,BN,BK,TM,TN | GFLOPS"
# 跳過 CSV 第一行標題，按第 6 欄位（GFLOPS）由大到小排序
tail -n +2 $OUTPUT_FILE | sort -t, -k6 -nr | head -n 5
echo "--------------------------------------------------"