#!/bin/bash

# ============================================================
# 1. 环境与网络配置 (针对 AutoDL 和 5090 优化)
# ============================================================
source /etc/network_turbo

# 针对 RTX 5090 的通信修复 (防止 NCCL 报错)
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TORCH_CUDA_ARCH_LIST="12.0"

# 强制使用稳定的 vLLM V0 引擎
export VLLM_USE_V1=0

# ============================================================
# 2. 实验参数定义
# ============================================================

# --- 关键修改：模型路径指向数据盘 (autodl-tmp) ---
# 请确保这两个路径在你的机器上真实存在 (用 ls 命令核对过)
MODEL_BASE="/root/autodl-tmp/models/Qwen/Qwen2.5-Math-1.5B"
MODEL_GRPO="/root/autodl-tmp/models/RLHFlow/Qwen2.5-Math-1.5B-GRPO-n8-easy"

# 将路径放入数组
models=("$MODEL_BASE" "$MODEL_GRPO")

# 数据集列表
datasets=("math" "amc" "aime")

# 温度列表
# GRPO 模型通常能承受较高的温度，但在数学任务上 0.6 和 1.0 是常用基准
temperatures=("0.6" "1.0" "1.2")

# 确保输出目录存在
mkdir -p outputs

# ============================================================
# 3. 自动化实验循环
# ============================================================
for model in "${models[@]}"; do
    # --- 自动设置模型简称 ---
    if [[ "$model" == *"GRPO"* ]]; then
        model_short="grpo"
        echo ">> 正在准备模型: GRPO 版"
    else
        model_short="base"
        echo ">> 正在准备模型: Base 版"
    fi

    for dataset in "${datasets[@]}"; do
        # 根据手册要求设置采样数: Math500 为 16, 其他为 64
        if [ "$dataset" == "math" ]; then
            n=16
        else
            n=64
        fi

        for temp in "${temperatures[@]}"; do
            # 格式化温度用于文件名 (如 0.6 -> t06)
            temp_str="t${temp//./}"
            
            output_file="outputs/${dataset}_${model_short}_${temp_str}.jsonl"
            eval_file="outputs/${dataset}_${model_short}_${temp_str}_eval.jsonl"

            echo "------------------------------------------------------------"
            echo "运行信息: [${model_short}] | 数据集: [${dataset}] | 温度: [${temp}]"
            echo "------------------------------------------------------------"

            # 运行推理 (Task 1)
            python src/inference.py \
              --model "$model" \
              --dataset "$dataset" \
              --dp-size 2 \
              --batch-size 16 \
              --rollout-n $n \
              --temperature "$temp" \
              --top-p 0.9 \
              --output_file "$output_file"

            # 如果推理成功，立即运行评分 (Task 2 & 3)
            if [ $? -eq 0 ]; then
                echo ">> 推理完成，正在评分..."
                python src/evaluate.py \
                  --input_file "$output_file" \
                  --output_file "$eval_file"
            else
                echo "!! 错误: $output_file 推理中断，跳过该项评分。"
            fi
        done
    done
done

# ============================================================
# 4. 自动汇总结果
# ============================================================
if [ -f "collect_results.py" ]; then
    echo "正在汇总所有实验指标至 results_summary.csv..."
    python collect_results.py
else
    echo "未找到 collect_results.py，跳过汇总步骤。"
fi

echo "======== 所有实验任务圆满完成！ ========"