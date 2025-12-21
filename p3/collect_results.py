import json
import csv
import os
import re
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any, Set

def collect_metrics(output_dir: str = "outputs", summary_file: str = "results_summary.csv") -> None:
    """
    自动扫描 outputs 目录下的所有 _eval.jsonl 文件，
    重新计算汇总指标并保存到 CSV 文件中。
    """
    all_results: List[Dict[str, Any]] = []
    
    # 获取目录下所有评估后的文件
    eval_files = [f for f in os.listdir(output_dir) if f.endswith("_eval.jsonl")]
    
    if not eval_files:
        print(f"在 {output_dir} 中未找到任何评估结果文件 (_eval.jsonl)。")
        return

    print(f"正在处理 {len(eval_files)} 个实验结果...")

    for filename in eval_files:
        filepath = os.path.join(output_dir, filename)
        
        # 从文件名解析数据集、模型类型和温度
        # 命名格式假设为: {dataset}_{model}_{temp}_eval.jsonl
        parts = filename.replace("_eval.jsonl", "").split("_")
        dataset = parts[0]
        model_type = parts[1]
        temp = parts[2] if len(parts) > 2 else "unknown"

        # 重新读取数据计算指标
        problem2scores = defaultdict(list)
        problem2preds = defaultdict(list)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                pid = data.get("id")
                score = data.get("score", 0.0)
                pred = data.get("extracted_pred")
                problem2scores[pid].append(score)
                problem2preds[pid].append((pred, score))

        # 计算 pass@1 (即平均准确率)
        all_scores = [sum(s)/len(s) for s in problem2scores.values() if len(s) > 0]
        pass_at_1 = (sum(all_scores) / len(all_scores)) * 100 if all_scores else 0

        # 计算 maj@1 (多数投票)
        maj_correct = 0
        for pid, preds in problem2preds.items():
            from collections import Counter
            counts = Counter([p[0] for p in preds])
            winner = counts.most_common(1)[0][0]
            # 检查胜出者是否正确
            for p, s in preds:
                if p == winner and s > 0.0:
                    maj_correct += 1
                    break
        maj_at_1 = (maj_correct / len(problem2preds)) * 100 if problem2preds else 0

        # 整理成一行数据
        all_results.append({
            "Dataset": dataset,
            "Model": model_type,
            "Temperature": temp,
            "Pass@1 (%)": f"{pass_at_1:.2f}",
            "Maj@1 (%)": f"{maj_at_1:.2f}",
            "FileName": filename
        })

    # 将所有结果写入 CSV
    keys = ["Dataset", "Model", "Temperature", "Pass@1 (%)", "Maj@1 (%)", "FileName"]
    with open(summary_file, 'w', newline='', encoding='utf-8') as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(all_results)

    print(f"成功！汇总指标已保存至: {summary_file}")

if __name__ == "__main__":
    collect_metrics()