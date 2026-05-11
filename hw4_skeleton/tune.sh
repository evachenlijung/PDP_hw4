#!/bin/bash
# tune.sh — sweep BM/BN/BK/TM/TN combinations and report top 5 GFLOPS at size=4096

PROJECT="ACD115083"
SRC="kernels/student_kernel.cu"
RESULTS_CSV="tune_results.csv"
TOP_TXT="tune_top5.txt"

if [ ! -f "$SRC" ]; then
    echo "ERROR: $SRC not found. Run from hw4_skeleton/ root."
    exit 1
fi

if ! grep -q "ifndef BM" "$SRC"; then
    echo "ERROR: $SRC needs #ifndef guards around BM/BN/BK/TM/TN macros."
    echo "Wrap each macro like this:"
    echo "    #ifndef BM"
    echo "    #define BM 128"
    echo "    #endif"
    exit 1
fi

# Candidate combinations: BM BN BK TM TN
CANDIDATES=(
    "128 128  8 8 8"
    "128 128 16 8 8"
    "128 128 16 4 8"
    "64  64  16 8 8"
    "64  128 16 4 8"
    "64  64   8 8 8"
)

echo "BM,BN,BK,TM,TN,threads,shmem_kb,gflops_4096,status" > "$RESULTS_CSV"

echo "================================================================"
echo "  SGEMM autotuner — ${#CANDIDATES[@]} combinations"
echo "================================================================"
printf "%-4s %-4s %-3s %-3s %-3s | %-6s %-7s | %-10s %s\n" \
       "BM" "BN" "BK" "TM" "TN" "thrds" "shmem" "GFLOPS" "status"
echo "----------------------------------------------------------------"

for combo in "${CANDIDATES[@]}"; do
    set -- $combo
    BM=$1; BN=$2; BK=$3; TM=$4; TN=$5

    threads=$(( (BM/TM) * (BN/TN) ))
    shmem=$(( 2*(BM*BK + BK*BN)*4 ))
    shmem_kb=$(( shmem/1024 ))

    # Constraint checks
    skip=""
    [ $threads -gt 1024 ] && skip="thrds>1024"
    [ $((threads % 32)) -ne 0 ] && skip="thrds%32"
    [ $shmem -gt 49152 ] && skip="shmem>48KB"
    [ $((BK % 4)) -ne 0 ] && skip="BK%4"
    [ $((BN % 4)) -ne 0 ] && skip="BN%4"
    [ $(( (BM*BK/4) % threads )) -ne 0 ] && skip="a_unbal"
    [ $(( (BK*BN/4) % threads )) -ne 0 ] && skip="b_unbal"

    if [ -n "$skip" ]; then
        printf "%-4s %-4s %-3s %-3s %-3s | %-6s %-7s | %-10s SKIP:%s\n" \
            "$BM" "$BN" "$BK" "$TM" "$TN" "$threads" "${shmem_kb}KB" "-" "$skip"
        echo "$BM,$BN,$BK,$TM,$TN,$threads,$shmem_kb,0,SKIP_$skip" >> "$RESULTS_CSV"
        continue
    fi

    # Build
    rm -f main
    nvcc -O3 -std=c++14 -gencode=arch=compute_70,code=sm_70 \
         -DBM=$BM -DBN=$BN -DBK=$BK -DTM=$TM -DTN=$TN \
         main.cu -o main -lcublas > /tmp/tune_build.log 2>&1

    if [ ! -f main ]; then
        printf "%-4s %-4s %-3s %-3s %-3s | %-6s %-7s | %-10s BUILD_FAIL\n" \
            "$BM" "$BN" "$BK" "$TM" "$TN" "$threads" "${shmem_kb}KB" "-"
        echo "$BM,$BN,$BK,$TM,$TN,$threads,$shmem_kb,0,BUILD_FAIL" >> "$RESULTS_CSV"
        continue
    fi

    # Run
    out=$(srun -N 1 -n 1 --gpus-per-node 1 -A "$PROJECT" -t 2 ./main 2>&1)

    if echo "$out" | grep -q "verification failed"; then
        printf "%-4s %-4s %-3s %-3s %-3s | %-6s %-7s | %-10s VERIFY_FAIL\n" \
            "$BM" "$BN" "$BK" "$TM" "$TN" "$threads" "${shmem_kb}KB" "-"
        echo "$BM,$BN,$BK,$TM,$TN,$threads,$shmem_kb,0,VERIFY_FAIL" >> "$RESULTS_CSV"
        continue
    fi

    if echo "$out" | grep -qE "CUDA error|illegal memory|exit code 1"; then
        printf "%-4s %-4s %-3s %-3s %-3s | %-6s %-7s | %-10s RUNTIME_ERR\n" \
            "$BM" "$BN" "$BK" "$TM" "$TN" "$threads" "${shmem_kb}KB" "-"
        echo "$BM,$BN,$BK,$TM,$TN,$threads,$shmem_kb,0,RUNTIME_ERR" >> "$RESULTS_CSV"
        continue
    fi

    # Parse GFLOPS at size 4096
    gflops=$(echo "$out" | grep "Running size: 4096" | grep -oE "performance: *[0-9.]+" | grep -oE "[0-9.]+")

    if [ -z "$gflops" ]; then
        printf "%-4s %-4s %-3s %-3s %-3s | %-6s %-7s | %-10s NO_OUTPUT\n" \
            "$BM" "$BN" "$BK" "$TM" "$TN" "$threads" "${shmem_kb}KB" "-"
        echo "$BM,$BN,$BK,$TM,$TN,$threads,$shmem_kb,0,NO_OUTPUT" >> "$RESULTS_CSV"
        continue
    fi

    printf "%-4s %-4s %-3s %-3s %-3s | %-6s %-7s | %-10s OK\n" \
        "$BM" "$BN" "$BK" "$TM" "$TN" "$threads" "${shmem_kb}KB" "$gflops"
    echo "$BM,$BN,$BK,$TM,$TN,$threads,$shmem_kb,$gflops,OK" >> "$RESULTS_CSV"
done

echo "----------------------------------------------------------------"
echo ""
echo "================================================================"
echo "  TOP 5 by GFLOPS at size=4096"
echo "================================================================"
printf "%-4s %-4s %-3s %-3s %-3s | %-6s %-7s | %s\n" \
    "BM" "BN" "BK" "TM" "TN" "thrds" "shmem" "GFLOPS"
echo "----------------------------------------------------------------"
{
tail -n +2 "$RESULTS_CSV" | awk -F',' '$9=="OK"' | sort -t',' -k8 -g -r | head -5 | \
    awk -F',' '{ printf "%-4s %-4s %-3s %-3s %-3s | %-6s %-5sKB | %s\n", $1,$2,$3,$4,$5,$6,$7,$8 }'
} | tee "$TOP_TXT"

echo ""
echo "Full log: $RESULTS_CSV"
echo "Top 5:    $TOP_TXT"