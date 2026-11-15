"""
Utility functions for the pmhcuni package.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Optional

import re

import torch
import csv


def normalize_allele_name(name: str) -> str:
    """
    Normalize allele names to the standard format with '*' and ':'.
    E.g. 'HLA-A0201' -> 'HLA-A*02:01', 'HLA-B0702' -> 'HLA-B*07:02'.
    If the name already contains '*' and ':', return as-is.
    """
    name = name.strip()
    # Already in standard format like HLA-A*02:01
    if "*" in name and ":" in name:
        return name
    # Match patterns like HLA-A0201, HLA-B0702, HLA-C0401, etc.
    m = re.match(r'^(HLA-[A-Za-z]+)(\d{4,})$', name)
    if m:
        prefix = m.group(1)
        digits = m.group(2)
        # Insert '*' after prefix and ':' between 2-digit groups
        parts = [digits[i:i+2] for i in range(0, len(digits), 2)]
        return f"{prefix}*{':'.join(parts)}"
    return name


def load_allele_sequences(path: str) -> OrderedDict[str, str]:
    """
    正确处理 CSV/TSV 文件并跳过表头。
    """
    alleles = OrderedDict()
    
    with open(path, "r", encoding="utf-8") as f:
        # 自动检测是逗号还是制表符 (可选，也可以直接指定 delimiter=',')
        # 这里为了稳健性，直接针对你的 CSV 使用 delimiter=','
        reader = csv.reader(f, delimiter=',')
        
        # 获取第一行（表头）并检查
        header = next(reader, None)
        
        for row in reader:
            if len(row) >= 2:
                allele, sequence = row[0], row[1]
                alleles[allele] = sequence
                
    return alleles


def get_device(device_str: Optional[str] = None) -> torch.device:
    if device_str is not None:
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
