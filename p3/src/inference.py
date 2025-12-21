import os
import sys
import time
import json
from time import sleep
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse
import multiprocessing

def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="数据并行推理脚本")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-1.5B", help="模型名称或路径")
    parser.add_argument("--dataset", type=str, default="math", help="数据集别名 (math, amc, aime)")
    parser.add_argument("--dp-size", type=int, default=2, help="数据并行大小")
    parser.add_argument("--temperature", type=float, default=1.0, help="采样温度")
    parser.add_argument("--top-p", type=float, default=0.9, help="采样 Top-p")
    parser.add_argument("--tp-size", type=int, default=1, help="张量并行大小")
    parser.add_argument("--seq-len", type=int, default=2048, help="最大序列长度")
    parser.add_argument("--output_file", type=str, required=True, help="输出文件路径")
    parser.add_argument("--batch-size", type=int, default=16, help="推理批大小")
    parser.add_argument("--rollout-n", type=int, default=1, help="每个问题生成的样本数量 (k)")
    return parser.parse_args()

def main(
    dataset: str,
    model: str,
    dp_size: int,
    local_dp_rank: int,
    global_dp_rank: int,
    dp_master_ip: str,
    dp_master_port: int,
    GPUs_per_dp_rank: int,
    output_file: str,
    batch_size: int,
    seq_len: int,
    temperature: float,
    top_p: float,
    n: int,
) -> None:
    """
    单个推理进程的主函数
    """
    # GPU 映射逻辑
    start_gpu_id = local_dp_rank * GPUs_per_dp_rank
    end_gpu_id = start_gpu_id + GPUs_per_dp_rank
    cuda_visible_devices = ",".join(str(i) for i in range(start_gpu_id, end_gpu_id))
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    # 清除干扰环境变量
    for key in ["MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK"]:
        os.environ.pop(key, None)

    try:
        from vllm import LLM, SamplingParams
        import torch
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError as e:
        print(f"导入库失败: {e}")
        return

    # 数据集映射
    dataset_map: Dict[str, str] = {
        "math": "math-ai/math500",
        "amc": "math-ai/amc23",
        "aime": "math-ai/aime25",
    }
    dataset_path = dataset_map.get(dataset.lower(), dataset)
    ds = load_dataset(dataset_path, split="test")

    # 根据数据集结构提取问题和答案
    name = dataset_path.lower()
    if "math500" in name or "aime25" in name:
        problems, gold_answers = ds["problem"], ds["answer"]
    elif "amc23" in name:
        problems, gold_answers = ds["question"], ds["answer"]
    else:
        problems, gold_answers = ds["problem"], ds["answer"]

    # 加载分词器并应用 Chat Template
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    prompt_strs: List[str] = []
    for p in problems:
        conv = [{"role": "user", "content": p}]
        try:
            # 尝试启用思维链模板
            s = tokenizer.apply_chat_template(conv, add_generation_prompt=True, tokenize=False, enable_thinking=True)
        except:
            s = tokenizer.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        prompt_strs.append(s)

    # 数据并行切分：确保每个进程处理不同的题目子集
    num_samples = len(prompt_strs)
    base, rem = divmod(num_samples, dp_size)
    start = global_dp_rank * base + min(global_dp_rank, rem)
    end = start + base + (1 if global_dp_rank < rem else 0)

    rank_prompts = prompt_strs[start:end]
    rank_problems = problems[start:end]
    rank_gold = gold_answers[start:end]

    # 初始化 vLLM 引擎
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=seq_len, n=n)
    llm = LLM(model=model, dtype="bfloat16", tensor_parallel_size=GPUs_per_dp_rank, trust_remote_code=True, max_model_len=3072)

    # 结果保存
    base_path = Path(output_file)
    partial_path = base_path.with_name(f"{base_path.stem}.rank{global_dp_rank}.jsonl")
    
    with partial_path.open("w", encoding="utf-8") as f:
        for i in range(0, len(rank_prompts), batch_size):
            outputs = llm.generate(rank_prompts[i:i+batch_size], sampling_params)
            for j, out in enumerate(outputs):
                for sample_id, seq in enumerate(out.outputs):
                    record = {
                        "id": start + i + j,
                        "sample_id": sample_id,
                        "problem": rank_problems[i+j],
                        "gold": rank_gold[i+j],
                        "answer": seq.text
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    args = parse_args()
    
    # 简单的参数检查
    import torch
    total_gpus = torch.cuda.device_count()
    required_gpus = args.dp_size * args.tp_size
    
    if total_gpus < required_gpus:
        print(f"警告: 检测到 {total_gpus} 张显卡，但配置需要 {required_gpus} (dp={args.dp_size} * tp={args.tp_size})。")
        print("如果是调试模式，请确保显卡 ID 设置正确。")

    print(f"开始推理: Model={args.model}, DP={args.dp_size}, TP={args.tp_size}, Total GPUs={required_gpus}")

    # 设置多进程启动方式
    multiprocessing.set_start_method('spawn', force=True)

    processes = []
    # 假设单机运行，local_rank 等于 global_rank
    for rank in range(args.dp_size):
        p = multiprocessing.Process(
            target=main,
            args=(
                args.dataset,
                args.model,
                args.dp_size,
                rank,          # local_dp_rank
                rank,          # global_dp_rank (单机情况)
                "127.0.0.1",   # dp_master_ip
                12355,         # dp_master_port
                args.tp_size,  # GPUs_per_dp_rank (对应 tp-size)
                args.output_file,
                args.batch_size,
                args.seq_len,
                args.temperature,
                args.top_p,
                args.rollout_n,
            )
        )
        p.start()
        processes.append(p)

    # 等待所有进程结束
    for p in processes:
        p.join()

    # 合并结果文件 (可选，但建议加上方便查看)
    print("正在合并结果...")
    base_path = Path(args.output_file)
    all_records = []
    for rank in range(args.dp_size):
        partial_file = base_path.with_name(f"{base_path.stem}.rank{rank}.jsonl")
        if partial_file.exists():
            with partial_file.open("r", encoding="utf-8") as f:
                for line in f:
                    all_records.append(json.loads(line))
            # 合并后可以删除分片文件
            # partial_file.unlink() 
        else:
            print(f"警告: 缺失分片文件 {partial_file}")
    
    # 根据 id 排序并写入最终文件
    all_records.sort(key=lambda x: x["id"])
    with base_path.open("w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
    print(f"推理完成！结果已保存至 {args.output_file}")