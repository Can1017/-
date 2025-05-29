#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
import itertools
import os


def extract_jsonl_head(input_path, output_path, n):
    """处理 JSON Lines 格式，直接读前 n 行"""
    with open(input_path, 'r', encoding='utf-8') as fin, \
            open(output_path, 'w', encoding='utf-8') as fout:
        for i, line in enumerate(fin):
            if i >= n:
                break
            fout.write(line)


def extract_json_array_head(input_path, output_path, n, json_depth=10):
    """处理 JSON 数组格式，读入整个数组，切片，写出合法数组"""
    with open(input_path, 'r', encoding='utf-8') as fin:
        data = json.load(fin)
        if not isinstance(data, list):
            raise ValueError("检测到的 JSON 并非数组格式")
    head = data[:n]
    with open(output_path, 'w', encoding='utf-8') as fout:
        json.dump(head, fout, ensure_ascii=False, indent=2)


def is_jsonl(input_path, sample_lines=5):
    """
    简单检测：取文件前几行，若每行都是合法的 JSON 对象（不以 '[' 开头），则判为 JSONL
    """
    with open(input_path, 'r', encoding='utf-8') as fin:
        for _ in range(sample_lines):
            line = fin.readline().strip()
            if not line:
                continue
            # 如果整行以 [ 或 ] 开头，极可能是数组格式
            if line.startswith('[') or line.startswith(']'):
                return False
            try:
                json.loads(line)
            except json.JSONDecodeError:
                return False
    return True


def main():
    parser = argparse.ArgumentParser(description="提取 JSON 文件的前 N 条记录")
    parser.add_argument("input",  help="输入 JSON 文件路径")
    parser.add_argument("output", help="输出 JSON 文件路径")
    parser.add_argument("-n", type=int, default=10,
                        help="要提取的记录数，默认 10")
    args = parser.parse_args()

    inp = args.input
    outp = args.output
    n = args.n

    if not os.path.isfile(inp):
        print(f"错误：找不到输入文件 {inp}")
        return

    try:
        if is_jsonl(inp):
            print("检测到 JSON Lines 格式，按行读取前 %d 条..." % n)
            extract_jsonl_head(inp, outp, n)
        else:
            print("检测到 JSON 数组格式，读取并切片前 %d 项..." % n)
            extract_json_array_head(inp, outp, n)
        print("已写入：", outp)
    except Exception as e:
        print("处理出错：", e)


if __name__ == "__main__":
    main()
