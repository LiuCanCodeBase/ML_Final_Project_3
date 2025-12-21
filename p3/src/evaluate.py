import json
import argparse
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict, Counter
from verifier import compute_score
import math
import random
from typing import List, Dict, Any, Tuple, Optional

# 固定随机种子以确保平票处理时的确定性
random.seed(42)

def evaluate_results(input_file: str, output_file: str) -> None:
    """
    读取模型生成的 JSONL 文件，计算分数，并保存增强后的结果。
    计算指标：
    1. 无偏 pass@k 估计。
    2. 多数投票 (Majority Vote @1) 准确率。
    """
    # 按问题 ID 分组分数，用于计算 pass@k
    # Key: problem_id -> Value: List[float] (0.0 或 1.0)
    problem2scores: Dict[Any, List[float]] = defaultdict(list)

    # 按问题 ID 分组预测结果，用于多数投票
    # Key: problem_id -> Value: List[Tuple[Optional[str], float]]
    problem2maj_data: Dict[Any, List[Tuple[Optional[str], float]]] = defaultdict(list)

    input_path: Path = Path(input_file)
    if not input_path.is_file():
        print(f"错误：未找到输入文件：{input_file}")
        return

    output_path: Path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines_processed: int = 0

    print(f"正在计算 {input_file} 的行数...")
    with open(input_path, "r", encoding="utf-8") as f:
        total_lines: int = sum(1 for _ in f)

    print(f"正在对 {input_file} 的生成结果进行评分...")

    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:

        for line_idx, line in enumerate(tqdm(f_in, total=total_lines, desc="处理行")):
            data: Dict[str, Any] = json.loads(line)
            lines_processed += 1

            model_answer: str = data.get("answer", "")
            gold_answer: str = data.get("gold", "")
            
            # 使用 'id' 对同一问题的多次生成（rollouts）进行分组
            problem_id: Any = data.get("id", line_idx)

            try:
                # 计算正确性分数
                score_dict: Dict[str, Any] = compute_score(model_answer, gold_answer)
                data.update(score_dict)

                if score_dict:
                    score_val: float = float(score_dict.get("score", 0.0))
                    pred_val: Optional[str] = score_dict.get("extracted_pred", None)

                    # 1. 存储用于 Pass@k 的分数
                    problem2scores[problem_id].append(score_val)

                    # 2. 存储用于多数投票的数据
                    problem2maj_data[problem_id].append((pred_val, score_val))

            except Exception as e:
                tqdm.write(f"处理第 {line_idx + 1} 行时出错: {e}")
                data["error"] = str(e)

            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")

    if lines_processed == 0:
        print("未处理任何行。")
        return

    num_problems: int = len(problem2scores)
    print(f"处理完成。已对 {num_problems} 个问题的 {lines_processed} 行进行评分。")

    # ==========================================
    # 1. 无偏 Pass@k 计算
    # ==========================================
    k2problem_pass_vals: Dict[int, List[float]] = defaultdict(list)

    for problem_id, scores in problem2scores.items():
        n_resps: int = len(scores)
        if n_resps == 0:
            continue

        # TODO 1: 统计正确响应的数量 (c_correct)
        # scores 是包含 0.0 或 1.0 的列表
        c_correct: int = int(sum(scores))

        # 生成 k 值：1, 2, 4, ... 直到 n_resps
        ks: List[int] = []
        k_val: int = 1
        while k_val <= n_resps:
            ks.append(k_val)
            k_val *= 2
        
        if n_resps not in ks:
            ks.append(n_resps)
        
        ks = sorted(list(set(ks)))

        for k in ks:
            # TODO 2: 实现无偏 pass@k 公式
            # 公式: pass@k = 1 - C(n-c, k) / C(n, k)
            if k > n_resps:
                pass_at_k = 1.0
            else:
                try:
                    # 使用组合数公式计算不包含任何正确答案的概率
                    # 如果 n-c < k，math.comb 返回 0，pass@k 自动为 1.0
                    num_incorrect_ways = math.comb(n_resps - c_correct, k)
                    total_ways = math.comb(n_resps, k)
                    pass_at_k = 1.0 - (num_incorrect_ways / total_ways)
                except (ValueError, ZeroDivisionError):
                    pass_at_k = 0.0

            k2problem_pass_vals[k].append(pass_at_k)

    # ==========================================
    # 2. 多数投票 (Majority Vote) 计算
    # ==========================================
    maj_correct_count: int = 0
    maj_total: int = 0

    for problem_id, data_list in problem2maj_data.items():
        if not data_list:
            continue
        
        maj_total += 1
        
        # 提取预测值用于计数
        preds: List[Optional[str]] = [item[0] for item in data_list]
        
        # TODO 3: 实现 Majority Vote 逻辑
        # 3a. 统计频率
        counts = Counter(preds)
        max_freq = max(counts.values())
        
        # 获取所有获得最高票数的候选答案（处理平票）
        candidates = [p for p, c in counts.items() if c == max_freq]
        
        # 确定最终胜出者 (Winner)
        if len(candidates) == 1:
            winner = candidates[0]
        else:
            # 按固定方式处理平票：排序后选择第一个以保证确定性 
            candidates.sort(key=lambda x: str(x))
            winner = candidates[0]
        
        # 3b. 检查胜出者是否正确
        is_winner_correct: bool = False
        for pred, score in data_list:
            if pred == winner:
                if score > 0.0:
                    is_winner_correct = True
                break
        
        if is_winner_correct:
            maj_correct_count += 1

    # ==========================================
    # 打印结果
    # ==========================================
    if k2problem_pass_vals:
        print("\nPass@k 指标:")
        for k in sorted(k2problem_pass_vals.keys()):
            vals = k2problem_pass_vals[k]
            avg_pass = (sum(vals) / len(vals)) * 100.0
            print(f"  pass@{k:<4}: {avg_pass:.2f}%")

    if maj_total > 0:
        maj_acc = (maj_correct_count / maj_total) * 100.0
        print("\n多数投票指标 (Majority Vote):")
        print(f"  maj@1    : {maj_acc:.2f}%")

    print(f"\n评分结果已保存至 {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估模型生成结果 (Pass@k 和 Majority Vote)。")
    parser.add_argument("--input_file", type=str, required=True, help="输入 JSONL 文件路径。")
    parser.add_argument("--output_file", type=str, required=True, help="输出评分结果路径。")
    args = parser.parse_args()

    evaluate_results(args.input_file, args.output_file)