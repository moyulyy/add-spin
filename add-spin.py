# -*- coding: utf-8 -*-
"""
add-spin.py —— 一站式 VASP 磁矩初猜工具（POSCAR -> ISPIN / MAGMOM）
================================================================================
把「按元素拍脑袋给磁矩」升级为「LLM 工具调用 + 晶体学/表面分析」：

  * 解析完整 POSCAR（晶格、元素、坐标、Selective dynamics、Cartesian/Direct）；
  * 判别 bulk / slab / molecule（真空层算法与 mk-KPOINTS 一致）；
  * slab 分层，识别 surface / subsurface / interior 等配位不饱和位点；
  * 识别 fcc/bcc、岩盐、闪锌矿、纤锌矿、钙钛矿、双钙钛矿、尖晶石/反尖晶石、
    LDH、氢氧化物、刚玉、金红石等结构；
  * 内置常见磁性材料知识库，按「元素 + 配位数 + 位点角色」给出磁矩；
  * 自动分配铁磁 / 亚铁磁 / 反铁磁（fcc 磁层投影、bcc 二部图染色）符号；
  * 可选调用任意 OpenAI 兼容 LLM 做 function calling，无 Key 时自动回退启发式。

命令行（--help 查看全部）:
  python add-spin.py POSCAR                     # 调 LLM，写 INCAR
  python add-spin.py POSCAR --print             # 只打印
  python add-spin.py POSCAR --no-llm            # 不联网，内置启发式
  python add-spin.py POSCAR --hint "...":       # 给 LLM 的补充说明
  python add-spin.py --self-test                # 运行内置自检
  python add-spin.py --make-examples DIR        # 导出示例 POSCAR

环境变量: LLM_API_KEY / OPENAI_API_KEY / DEEPSEEK_API_KEY,
          LLM_BASE_URL / OPENAI_BASE_URL, LLM_MODEL

依赖: Python >= 3.9 + numpy（HTTP 用标准库，无需 openai SDK）。
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# ============================================================================
# 0. 兼容旧版 add-spin.py 的模块级 API
#    （fast-vasp CLI 会 from add_spin import parse_poscar, build_magmom, append_magmom）
# ============================================================================
# 磁性设置字典: 元素 -> 默认初猜磁矩 (未列出的元素默认为 0)
DEFAULT_MAGMOM = {
    'H': 0, 'He': 0, 'Li': 0, 'Be': 0, 'B': 0, 'C': 0, 'N': 0, 'O': 0,
    'F': 0, 'Ne': 0, 'Na': 0, 'Mg': 0, 'Al': 0, 'Si': 0, 'P': 0, 'S': 0,
    'Cl': 0, 'Ar': 0, 'K': 0, 'Ca': 0, 'Sc': 3, 'Ti': 3, 'V': 3, 'Cr': 3,
    'Mn': 5, 'Fe': 5, 'Co': 5, 'Ni': 5, 'Cu': 3, 'Zn': 0, 'Ga': 0, 'Ge': 0,
    'As': 0, 'Se': 0, 'Br': 0, 'Kr': 0, 'Rb': 0, 'Sr': 0, 'Y': 3, 'Zr': 3,
    'Nb': 3, 'Mo': 5, 'Tc': 0, 'Ru': 5, 'Rh': 5, 'Pd': 3, 'Ag': 3, 'Cd': 3,
    'In': 3, 'Sn': 3, 'Sb': 3, 'Te': 0, 'I': 0, 'Xe': 0, 'Cs': 0, 'Ba': 0,
    'La': 3, 'Ce': 3, 'Pr': 3, 'Nd': 3, 'Pm': 3, 'Sm': 3, 'Eu': 3, 'Gd': 3,
    'Tb': 3, 'Dy': 3, 'Ho': 3, 'Er': 3, 'Tm': 3, 'Yb': 3, 'Lu': 3, 'Hf': 3,
    'Ta': 3, 'W': 3, 'Re': 3, 'Os': 3, 'Ir': 3, 'Pt': 3, 'Au': 3, 'Hg': 3,
    'Tl': 3, 'Pb': 3, 'Bi': 3, 'Po': 3, 'At': 3, 'Rn': 3, 'Fr': 3, 'Ra': 3,
    'Ac': 3, 'Th': 3, 'Pa': 3, 'U': 3, 'Np': 3, 'Pu': 3, 'Am': 3, 'Cm': 3
}


def parse_poscar(path="POSCAR"):
    """解析 POSCAR 第 6/7 行 (元素符号 / 数量), 返回 [(元素, 数量), ...]。
    解析失败或格式不支持时返回空列表。
    """
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
    if len(lines) < 7:
        return []
    elements = re.findall(r'[A-Z][a-z]*', lines[5])
    counts = list(map(int, re.findall(r'\d+', lines[6])))
    if not elements or len(elements) != len(counts):
        return []
    return list(zip(elements, counts))


def build_magmom(pairs, custom=None):
    """生成 'Mag parameter' 文本块 (ISPIN=2 + MAGMOM)。
    pairs : [(元素, 数量), ...]
    custom: {元素: 磁矩值}, 未给出的元素使用 DEFAULT_MAGMOM (缺省 0)
    """
    custom = custom or {}
    magmom_values = '   '.join(
        f"{count}*{custom.get(element, DEFAULT_MAGMOM.get(element, 0))}"
        for element, count in pairs
    )
    elements_str = "   ".join([f"{el:<6}" for el, _ in pairs])
    counts_str = "   ".join([f"{cnt:<6}" for _, cnt in pairs])
    magmom_str = "   ".join([f"{x:<6}" for x in magmom_values.split()])

    spin_msg = "Mag parameter\n   ISPIN = 2\n"
    spin_msg += f"   #Element: {elements_str}\n   #Numbers: {counts_str}\n"
    spin_msg += "   MAGMOM =  " + magmom_str + "\n"
    return spin_msg


def append_magmom(path="POSCAR", incar="INCAR", custom=None):
    """解析 POSCAR 并向 INCAR 追加自旋参数块, 成功返回 True。"""
    pairs = parse_poscar(path)
    if not pairs:
        return False
    msg = build_magmom(pairs, custom)
    with open(incar, 'a', encoding='utf-8') as f:
        f.write("\n" + msg)
    return True

# ============================================================================
# 附录：内置示例结构（--self-test / --make-examples 使用）
# ============================================================================
EXAMPLE_POSCARS: Dict[str, str] = {
    "Pt_fcc": """Pt fcc primitive
1.0
    0.0000000000      1.9600000000      1.9600000000
    1.9600000000      0.0000000000      1.9600000000
    1.9600000000      1.9600000000      0.0000000000
Pt
1
Direct
    0.0000000000      0.0000000000      0.0000000000
""",
    "Fe_bcc": """Fe bcc primitive
1.0
   -1.4350000000      1.4350000000      1.4350000000
    1.4350000000     -1.4350000000      1.4350000000
    1.4350000000      1.4350000000     -1.4350000000
Fe
1
Direct
    0.0000000000      0.0000000000      0.0000000000
""",
    "NiO": """NiO rocksalt conventional
1.0
    4.1700000000      0.0000000000      0.0000000000
    0.0000000000      4.1700000000      0.0000000000
    0.0000000000      0.0000000000      4.1700000000
Ni  O
4  4
Direct
    0.0000000000      0.0000000000      0.0000000000
    0.0000000000      0.5000000000      0.5000000000
    0.5000000000      0.0000000000      0.5000000000
    0.5000000000      0.5000000000      0.0000000000
    0.5000000000      0.0000000000      0.0000000000
    0.5000000000      0.5000000000      0.5000000000
    1.0000000000      0.0000000000      0.5000000000
    1.0000000000      0.5000000000      0.0000000000
""",
    "NiO_001_slab": """NiO(001) slab 3 layers, vacuum 12 A
1.0
    2.9486352775      0.0000000000      0.0000000000
    1.4743176388      2.5535930569      0.0000000000
    0.0000000000      0.0000000000     30.0188765563
Ni  O  Ni  O  Ni  O
1  1  1  1  1  1
Direct
    0.0000000000      0.0000000000      0.3997484708
    0.6666666667      0.6666666667      0.4398490825
    0.3333333333      0.3333333333      0.4799496942
    0.0000000000      0.0000000000      0.5200503058
    0.6666666667      0.6666666667      0.5601509175
    0.3333333333      0.3333333333      0.6002515292
""",
    "ZnS_zincblende": """ZnS zincblende (a=5.41)
1.0
    5.4100000000      0.0000000000      0.0000000000
    0.0000000000      5.4100000000      0.0000000000
    0.0000000000      0.0000000000      5.4100000000
Zn  S
4  4
Direct
    0.0000000000      0.0000000000      0.0000000000
    0.0000000000      0.5000000000      0.5000000000
    0.5000000000      0.0000000000      0.5000000000
    0.5000000000      0.5000000000      0.0000000000
    0.2500000000      0.2500000000      0.2500000000
    0.2500000000      0.7500000000      0.7500000000
    0.7500000000      0.2500000000      0.7500000000
    0.7500000000      0.7500000000      0.2500000000
""",
    "ZnO_wurtzite": """ZnO wurtzite (a=3.25,c=5.21)
1.0
    3.2500000000      0.0000000000      0.0000000000
   -1.6250000000      2.8145825623      0.0000000000
    0.0000000000      0.0000000000      5.2100000000
Zn  O
2  2
Direct
    0.3333333333      0.6666666667      0.0000000000
    0.6666666667      0.3333333333      0.5000000000
    0.3333333333      0.6666666667      0.3820000000
    0.6666666667      0.3333333333      0.8820000000
""",
    "LaFeO3": """LaFeO3 cubic perovskite (a=3.93)
1.0
    3.9300000000      0.0000000000      0.0000000000
    0.0000000000      3.9300000000      0.0000000000
    0.0000000000      0.0000000000      3.9300000000
La  Fe  O
1  1  3
Direct
    0.0000000000      0.0000000000      0.0000000000
    0.5000000000      0.5000000000      0.5000000000
    0.5000000000      0.5000000000      0.0000000000
    0.5000000000      0.0000000000      0.5000000000
    0.0000000000      0.5000000000      0.5000000000
""",
    "NiAl_LDH": """Ni3Al(OH)8 layered double hydroxide monolayer
1.0
    6.2600000000      0.0000000000      0.0000000000
   -3.1300000000      5.4213190277      0.0000000000
    0.0000000000      0.0000000000     20.0000000000
Ni  Al  O  H
3  1  8  8
Direct
    0.0000000000      0.0000000000      0.0000000000
    0.5000000000      0.0000000000      0.0000000000
    0.0000000000      0.5000000000      0.0000000000
    0.5000000000      0.5000000000      0.0000000000
    0.1666666667      0.3333333333      0.0000000000
    0.3333333333      0.1666666667      0.0000000000
    0.6666666667      0.3333333333      0.0000000000
    0.8333333333      0.1666666667      0.0000000000
    0.1666666667      0.8333333333      0.0000000000
    0.3333333333      0.6666666667      0.0000000000
    0.6666666667      0.8333333333      0.0000000000
    0.8333333333      0.6666666667      0.0000000000
    0.1666666667      0.3333333333      0.0600000000
    0.3333333333      0.1666666667      0.0600000000
    0.6666666667      0.3333333333      0.0600000000
    0.8333333333      0.1666666667      0.0600000000
    0.1666666667      0.8333333333      0.0600000000
    0.3333333333      0.6666666667      0.0600000000
    0.6666666667      0.8333333333      0.0600000000
    0.8333333333      0.6666666667      0.0600000000
""",
    "Fe3O4": """Fe3O4 inverse spinel (a=8.3967)
1.0
    8.3967000000      0.0000000000      0.0000000000
    0.0000000000      8.3967000000      0.0000000000
    0.0000000000      0.0000000000      8.3967000000
Fe  O
24  32
Direct
    0.1250000000      0.1250000000      0.1250000000
    0.6250000000      0.1250000000      0.6250000000
    0.1250000000      0.6250000000      0.6250000000
    0.6250000000      0.6250000000      0.1250000000
    0.8750000000      0.3750000000      0.3750000000
    0.8750000000      0.8750000000      0.8750000000
    0.3750000000      0.3750000000      0.8750000000
    0.3750000000      0.8750000000      0.3750000000
    0.5000000000      0.5000000000      0.5000000000
    0.2500000000      0.7500000000      0.0000000000
    0.7500000000      0.0000000000      0.2500000000
    0.0000000000      0.2500000000      0.7500000000
    0.5000000000      0.0000000000      0.0000000000
    0.2500000000      0.2500000000      0.5000000000
    0.7500000000      0.5000000000      0.7500000000
    0.0000000000      0.7500000000      0.2500000000
    0.0000000000      0.5000000000      0.0000000000
    0.7500000000      0.7500000000      0.5000000000
    0.2500000000      0.0000000000      0.7500000000
    0.5000000000      0.2500000000      0.2500000000
    0.0000000000      0.0000000000      0.5000000000
    0.7500000000      0.2500000000      0.0000000000
    0.2500000000      0.5000000000      0.2500000000
    0.5000000000      0.7500000000      0.7500000000
    0.2549000000      0.2549000000      0.2549000000
    0.4951000000      0.9951000000      0.7549000000
    0.9951000000      0.7549000000      0.4951000000
    0.7549000000      0.4951000000      0.9951000000
    0.0049000000      0.5049000000      0.2451000000
    0.7451000000      0.7451000000      0.7451000000
    0.5049000000      0.2451000000      0.0049000000
    0.2451000000      0.0049000000      0.5049000000
    0.2549000000      0.7549000000      0.7549000000
    0.4951000000      0.4951000000      0.2549000000
    0.9951000000      0.2549000000      0.9951000000
    0.7549000000      0.9951000000      0.4951000000
    0.0049000000      0.0049000000      0.7451000000
    0.7451000000      0.2451000000      0.2451000000
    0.5049000000      0.7451000000      0.5049000000
    0.2451000000      0.5049000000      0.0049000000
    0.7549000000      0.2549000000      0.7549000000
    0.9951000000      0.9951000000      0.2549000000
    0.4951000000      0.7549000000      0.9951000000
    0.2549000000      0.4951000000      0.4951000000
    0.5049000000      0.5049000000      0.7451000000
    0.2451000000      0.7451000000      0.2451000000
    0.0049000000      0.2451000000      0.5049000000
    0.7451000000      0.0049000000      0.0049000000
    0.7549000000      0.7549000000      0.2549000000
    0.9951000000      0.4951000000      0.7549000000
    0.4951000000      0.2549000000      0.4951000000
    0.2549000000      0.9951000000      0.9951000000
    0.5049000000      0.0049000000      0.2451000000
    0.2451000000      0.2451000000      0.7451000000
    0.0049000000      0.7451000000      0.0049000000
    0.7451000000      0.5049000000      0.5049000000
""",
    "MgAl2O4": """MgAl2O4 normal spinel (a=8.08)
1.0
    8.0800000000      0.0000000000      0.0000000000
    0.0000000000      8.0800000000      0.0000000000
    0.0000000000      0.0000000000      8.0800000000
Mg  Al  O
8  16  32
Direct
    0.1250000000      0.1250000000      0.1250000000
    0.6250000000      0.1250000000      0.6250000000
    0.1250000000      0.6250000000      0.6250000000
    0.6250000000      0.6250000000      0.1250000000
    0.8750000000      0.3750000000      0.3750000000
    0.8750000000      0.8750000000      0.8750000000
    0.3750000000      0.3750000000      0.8750000000
    0.3750000000      0.8750000000      0.3750000000
    0.5000000000      0.5000000000      0.5000000000
    0.2500000000      0.7500000000      0.0000000000
    0.7500000000      0.0000000000      0.2500000000
    0.0000000000      0.2500000000      0.7500000000
    0.5000000000      0.0000000000      0.0000000000
    0.2500000000      0.2500000000      0.5000000000
    0.7500000000      0.5000000000      0.7500000000
    0.0000000000      0.7500000000      0.2500000000
    0.0000000000      0.5000000000      0.0000000000
    0.7500000000      0.7500000000      0.5000000000
    0.2500000000      0.0000000000      0.7500000000
    0.5000000000      0.2500000000      0.2500000000
    0.0000000000      0.0000000000      0.5000000000
    0.7500000000      0.2500000000      0.0000000000
    0.2500000000      0.5000000000      0.2500000000
    0.5000000000      0.7500000000      0.7500000000
    0.2549000000      0.2549000000      0.2549000000
    0.4951000000      0.9951000000      0.7549000000
    0.9951000000      0.7549000000      0.4951000000
    0.7549000000      0.4951000000      0.9951000000
    0.0049000000      0.5049000000      0.2451000000
    0.7451000000      0.7451000000      0.7451000000
    0.5049000000      0.2451000000      0.0049000000
    0.2451000000      0.0049000000      0.5049000000
    0.2549000000      0.7549000000      0.7549000000
    0.4951000000      0.4951000000      0.2549000000
    0.9951000000      0.2549000000      0.9951000000
    0.7549000000      0.9951000000      0.4951000000
    0.0049000000      0.0049000000      0.7451000000
    0.7451000000      0.2451000000      0.2451000000
    0.5049000000      0.7451000000      0.5049000000
    0.2451000000      0.5049000000      0.0049000000
    0.7549000000      0.2549000000      0.7549000000
    0.9951000000      0.9951000000      0.2549000000
    0.4951000000      0.7549000000      0.9951000000
    0.2549000000      0.4951000000      0.4951000000
    0.5049000000      0.5049000000      0.7451000000
    0.2451000000      0.7451000000      0.2451000000
    0.0049000000      0.2451000000      0.5049000000
    0.7451000000      0.0049000000      0.0049000000
    0.7549000000      0.7549000000      0.2549000000
    0.9951000000      0.4951000000      0.7549000000
    0.4951000000      0.2549000000      0.4951000000
    0.2549000000      0.9951000000      0.9951000000
    0.5049000000      0.0049000000      0.2451000000
    0.2451000000      0.2451000000      0.7451000000
    0.0049000000      0.7451000000      0.0049000000
    0.7451000000      0.5049000000      0.5049000000
""",
}


# ============================================================================
# 1. 元素基础数据
# ============================================================================
_SYMBOLS = (
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm",
)

_MASSES = (
    0.0, 1.008, 4.0026, 6.94, 9.0122, 10.81, 12.011, 14.007, 15.999, 18.998,
    20.180, 22.990, 24.305, 26.982, 28.085, 30.974, 32.06, 35.45, 39.948,
    39.098, 40.078, 44.956, 47.867, 50.942, 51.996, 54.938, 55.845, 58.933,
    58.693, 63.546, 65.38, 69.723, 72.630, 74.922, 78.971, 79.904, 83.798,
    85.468, 87.62, 88.906, 91.224, 92.906, 95.95, 98.0, 101.07, 102.91,
    106.42, 107.87, 112.41, 114.82, 118.71, 121.76, 127.60, 126.90, 131.29,
    132.91, 137.33, 138.91, 140.12, 140.91, 144.24, 145.0, 150.36, 151.96,
    157.25, 158.93, 162.50, 164.93, 167.26, 168.93, 173.05, 174.97, 178.49,
    180.95, 183.84, 186.21, 190.23, 192.22, 195.08, 196.97, 200.59, 204.38,
    207.2, 208.98, 209.0, 210.0, 222.0, 223.0, 226.0, 227.0, 232.04, 231.04,
    238.03, 237.0, 244.0, 243.0, 247.0,
)

ATOMIC_MASS = {s: _MASSES[i + 1] for i, s in enumerate(_SYMBOLS)}

# Cordero 共价半径 (Å)，用于自动判断成键
RCOV = {
    "H": 0.31, "He": 0.28, "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76,
    "N": 0.71, "O": 0.66, "F": 0.57, "Ne": 0.58, "Na": 1.66, "Mg": 1.41,
    "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02, "Ar": 1.06,
    "K": 2.03, "Ca": 1.76, "Sc": 1.70, "Ti": 1.60, "V": 1.53, "Cr": 1.39,
    "Mn": 1.39, "Fe": 1.32, "Co": 1.26, "Ni": 1.24, "Cu": 1.32, "Zn": 1.22,
    "Ga": 1.22, "Ge": 1.20, "As": 1.19, "Se": 1.20, "Br": 1.20, "Kr": 1.16,
    "Rb": 2.20, "Sr": 1.95, "Y": 1.90, "Zr": 1.75, "Nb": 1.64, "Mo": 1.54,
    "Tc": 1.47, "Ru": 1.46, "Rh": 1.42, "Pd": 1.39, "Ag": 1.45, "Cd": 1.44,
    "In": 1.42, "Sn": 1.39, "Sb": 1.39, "Te": 1.38, "I": 1.39, "Xe": 1.40,
    "Cs": 2.44, "Ba": 2.15, "La": 2.07, "Ce": 2.04, "Pr": 2.03, "Nd": 2.01,
    "Pm": 1.99, "Sm": 1.98, "Eu": 1.98, "Gd": 1.96, "Tb": 1.94, "Dy": 1.92,
    "Ho": 1.92, "Er": 1.89, "Tm": 1.90, "Yb": 1.87, "Lu": 1.87, "Hf": 1.75,
    "Ta": 1.70, "W": 1.62, "Re": 1.51, "Os": 1.44, "Ir": 1.41, "Pt": 1.36,
    "Au": 1.36, "Hg": 1.32, "Tl": 1.45, "Pb": 1.46, "Bi": 1.48, "Po": 1.40,
    "At": 1.50, "Rn": 1.50, "Fr": 2.60, "Ra": 2.21, "Ac": 2.15, "Th": 2.06,
    "Pa": 2.00, "U": 1.96, "Np": 1.90, "Pu": 1.87, "Am": 1.80, "Cm": 1.69,
}

# 常见氧化态（用于给出价态 / d 电子数提示）
COMMON_OXIDATION = {
    "Sc": [3], "Ti": [2, 3, 4], "V": [2, 3, 4, 5], "Cr": [2, 3, 6],
    "Mn": [2, 3, 4, 6, 7], "Fe": [2, 3], "Co": [2, 3], "Ni": [2, 3],
    "Cu": [1, 2], "Zn": [2], "Y": [3], "Zr": [4], "Nb": [3, 5],
    "Mo": [3, 4, 6], "Ru": [3, 4], "Rh": [3], "Pd": [2, 4], "Ag": [1],
    "Ce": [3, 4], "Pr": [3], "Nd": [3], "Sm": [2, 3], "Eu": [2, 3],
    "Gd": [3], "Tb": [3], "Dy": [3], "Ho": [3], "Er": [3], "Tm": [3],
    "Yb": [2, 3], "Lu": [3], "Hf": [4], "Ta": [5], "W": [4, 6],
    "Re": [4, 7], "Os": [4], "Ir": [3, 4], "Pt": [2, 4], "U": [4, 6],
}

# 3d/4d/5d/4f 磁性离子：ox -> (d/f 电子数, 高自旋未成对电子数, 低自旋未成对电子数)
MAGNETIC_IONS = {
    "Ti": {2: (2, 2, 0), 3: (1, 1, 1)},
    "V":  {2: (3, 3, 1), 3: (2, 2, 0), 4: (1, 1, 1)},
    "Cr": {2: (4, 4, 2), 3: (3, 3, 3)},
    "Mn": {2: (5, 5, 1), 3: (4, 4, 0), 4: (3, 3, 1)},
    "Fe": {2: (6, 4, 0), 3: (5, 5, 1)},
    "Co": {2: (7, 3, 1), 3: (6, 4, 0), 4: (5, 5, 1)},
    "Ni": {2: (8, 2, 0), 3: (7, 3, 1)},
    "Cu": {2: (9, 1, 1)},
    "Ru": {3: (5, 3, 1), 4: (4, 2, 0)},
    "Rh": {3: (6, 4, 0)},
    "Mo": {3: (3, 3, 3), 4: (2, 2, 0)},
    "W":  {4: (2, 2, 0), 5: (1, 1, 1)},
    "Re": {4: (3, 3, 1), 6: (1, 1, 1)},
    "Ir": {4: (5, 3, 1)},
    "Ce": {3: (1, 1, 1)},
    "Pr": {3: (2, 2, 2)},
    "Nd": {3: (3, 3, 3)},
    "Sm": {3: (5, 5, 5)},
    "Eu": {2: (7, 7, 7), 3: (6, 6, 6)},
    "Gd": {3: (7, 7, 7)},
    "Tb": {3: (8, 6, 6)},
    "Dy": {3: (9, 5, 5)},
    "Ho": {3: (10, 4, 4)},
    "Er": {3: (11, 3, 3)},
    "Tm": {3: (12, 2, 2)},
    "Yb": {3: (13, 1, 1)},
}

# 纯金属铁磁/反铁磁参考磁矩 (μB/atom)，用于简单金属体系
METAL_MOMENT = {
    "Fe": 2.2, "Co": 1.7, "Ni": 0.6,
    "Mn": 1.0, "Cr": 1.0, "Gd": 7.0, "Tb": 6.0, "Dy": 5.0, "Ho": 4.0,
    "Er": 3.0,
}
# 常见非磁性（d0/d10/闭壳层）元素，避免给它们乱加磁矩
NONMAGNETIC = set(
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca "
    "Ga Ge As Se Br Kr Rb Sr In Sn Sb Te I Xe Cs Ba "
    "Zn Cd Hg Pb Bi Ag Cu Au".split()
)
# 氧化物中常见高自旋磁矩（按元素，未知价态时的稳妥初猜）
OXIDE_HS = {
    "Ti": 1, "V": 2, "Cr": 3, "Mn": 5, "Fe": 5, "Co": 3, "Ni": 2, "Cu": 1,
    "Ru": 3, "Rh": 2, "Mo": 2, "W": 2, "Re": 2, "Ir": 2,
    "Ce": 1, "Pr": 2, "Nd": 3, "Sm": 5, "Eu": 6, "Gd": 7, "Tb": 6,
    "Dy": 5, "Ho": 4, "Er": 3, "Tm": 2, "Yb": 1,
}

# 阴离子/非金属集合
ANIONS = set("O F Cl Br I S Se Te N P".split())


def _formula_sort_key(symbol: str):
    """化学式元素顺序：金属在前，阴离子次之，H 最后。"""
    if symbol == "H":
        return (2, 1)
    return (0 if symbol not in ANIONS else 1,
            _SYMBOLS.index(symbol) if symbol in _SYMBOLS else 999)


LEGACY_DEFAULT_MAGMOM = dict(DEFAULT_MAGMOM)


def cov_radius(symbol: str) -> float:
    return RCOV.get(symbol, 1.5)


def default_element_moment(symbol: str) -> float:
    """单一元素、未知环境的兜底磁矩。"""
    if symbol in NONMAGNETIC:
        return 0.0
    if symbol in OXIDE_HS:
        return float(OXIDE_HS[symbol])
    return float(LEGACY_DEFAULT_MAGMOM.get(symbol, 0.0))


# ============================================================================
# 2. POSCAR 解析
# ============================================================================
@dataclass
class Poscar:
    comment: str
    lattice: np.ndarray          # 3x3, 行向量
    species: List[str]           # VASP 第 6 行的元素表（唯一）
    counts: List[int]            # 每个元素的数量
    symbols: List[str]           # 逐原子元素符号
    frac: np.ndarray             # (N,3) 分数坐标
    coord_mode: str              # Direct / Cartesian
    selective: Optional[List[List[bool]]] = None
    velocities: Optional[np.ndarray] = None
    source: str = "POSCAR"

    # ---- 派生属性 -----------------------------------------------------
    @property
    def n_atoms(self) -> int:
        return len(self.symbols)

    @property
    def volume(self) -> float:
        return float(abs(np.linalg.det(self.lattice)))

    def cartesian(self) -> np.ndarray:
        return self.frac @ self.lattice

    def density(self) -> float:
        mass = sum(ATOMIC_MASS.get(s, 0.0) for s in self.symbols)
        # g/cm^3 : u/Å^3 * 1.66054
        return mass * 1.66053906660 / self.volume if self.volume > 0 else 0.0

    def blocks(self) -> List[Tuple[str, int, int]]:
        """返回 [(元素, 起始序号, 结束序号), ...]（结束为开区间）。"""
        out, start = [], 0
        for el, cnt in zip(self.species, self.counts):
            out.append((el, start, start + cnt))
            start += cnt
        return out

    def composition(self) -> "Counter[str]":
        return Counter(self.symbols)

    def reduced_composition(self) -> Dict[str, int]:
        counts = self.composition()
        g = math.gcd(*counts.values()) or 1
        return {el: c // g for el, c in counts.items()}

    def reduced_formula(self) -> str:
        red = self.reduced_composition()
        if not red:
            return ""
        return "".join(
            f"{el}{red[el] if red[el] > 1 else ''}"
            for el in sorted(red, key=_formula_sort_key)
        )

    def formula(self) -> str:
        counts = self.composition()
        return "".join(
            f"{el}{counts[el] if counts[el] > 1 else ''}"
            for el in sorted(counts, key=_formula_sort_key)
        )


def _is_int(token: str) -> bool:
    try:
        int(token)
        return True
    except ValueError:
        return False


def _read_potcar_species(potcar_path: str) -> Optional[List[str]]:
    """VASP4 的 POSCAR 没有元素行，尝试从 POTCAR 的 VRHFIN 读取。"""
    if not os.path.exists(potcar_path):
        return None
    try:
        text = open(potcar_path, "r", encoding="utf-8", errors="ignore").read()
    except OSError:
        return None
    found = re.findall(r"VRHFIN\s*=\s*([A-Za-z]+)", text)
    return found or None


def parse_poscar_full(path: str = "POSCAR") -> Poscar:
    """完整解析 POSCAR 文件。"""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return parse_poscar_text(text, source=path)


def parse_poscar_text(text: str, source: str = "POSCAR") -> Poscar:
    lines = text.splitlines()
    if len(lines) < 7:
        raise ValueError("POSCAR 行数不足，无法解析")

    comment = lines[0].strip()
    scale_tokens = lines[1].split()

    # ---- 缩放因子 -----------------------------------------------------
    coord_scale: np.ndarray | float
    scale_vec: Optional[np.ndarray] = None
    volume_target: Optional[float] = None
    if len(scale_tokens) == 3:
        scale_vec = np.array([float(x) for x in scale_tokens])
        coord_scale = scale_vec
    else:
        s = float(scale_tokens[0])
        if s < 0:
            volume_target = -s
            coord_scale = 1.0
        else:
            coord_scale = s

    raw_lattice = np.array(
        [[float(x) for x in lines[2 + i].split()[:3]] for i in range(3)]
    )
    if volume_target is not None:
        raw_vol = abs(np.linalg.det(raw_lattice))
        factor = (volume_target / raw_vol) ** (1.0 / 3.0) if raw_vol > 0 else 1.0
        lattice = raw_lattice * factor
        coord_scale = factor
    elif scale_vec is not None:
        lattice = raw_lattice * scale_vec[:, None]
    else:
        lattice = raw_lattice * float(coord_scale)

    # ---- 元素 / 数量 --------------------------------------------------
    idx = 5
    tok5 = lines[idx].split()
    source_species: Optional[List[str]] = None
    if tok5 and all(_is_int(t) for t in tok5):
        # VASP4：没有元素符号行
        counts = [int(t) for t in tok5]
        idx += 1
    else:
        source_species = [re.sub(r"[^A-Za-z]", "", t) for t in tok5]
        idx += 1
        counts = [int(t) for t in lines[idx].split()[: len(source_species)]]
        idx += 1

    # ---- Selective dynamics ------------------------------------------
    selective = None
    if idx < len(lines) and lines[idx].strip()[:1] in ("S", "s"):
        selective = []
        idx += 1
    # ---- 坐标模式 -----------------------------------------------------
    if idx >= len(lines):
        raise ValueError("POSCAR 缺少坐标模式行")
    mode = lines[idx].strip()
    if not mode:
        raise ValueError("POSCAR 坐标模式行为空")
    coord_mode = "Direct" if mode[0] in ("D", "d") else "Cartesian"
    idx += 1

    n_atoms = sum(counts)
    coords: List[List[float]] = []
    for k in range(n_atoms):
        if idx + k >= len(lines):
            raise ValueError(f"POSCAR 坐标行不足，期望 {n_atoms} 个原子")
        parts = lines[idx + k].split()
        coords.append([float(parts[0]), float(parts[1]), float(parts[2])])
        if selective is not None:
            flags = [p[:1] in ("T", "t") for p in parts[3:6]]
            while len(flags) < 3:
                flags.append(True)
            selective.append(flags)
    idx += n_atoms

    # ---- 可选速度（MD/NEB）-------------------------------------------
    velocities = None
    remaining = [ln for ln in lines[idx:] if ln.strip()]
    if remaining:
        first = remaining[0].split()
        if len(first) >= 3 and all(_is_float(x) for x in first[:3]):
            try:
                velocities = np.array(
                    [[float(x) for x in ln.split()[:3]] for ln in remaining[:n_atoms]]
                )
            except ValueError:
                velocities = None

    # ---- 元素符号顺序 -------------------------------------------------
    if source_species is None:
        source_species = _read_potcar_species(
            os.path.join(os.path.dirname(os.path.abspath(source)) or ".", "POTCAR")
        )
        if not source_species or len(source_species) != len(counts):
            source_species = [f"X{i + 1}" for i in range(len(counts))]

    symbols: List[str] = []
    for el, cnt in zip(source_species, counts):
        symbols.extend([el] * cnt)

    arr = np.array(coords, dtype=float)
    if coord_mode == "Cartesian":
        if isinstance(coord_scale, np.ndarray):
            cart = arr * coord_scale
        else:
            cart = arr * float(coord_scale)
        try:
            frac = cart @ np.linalg.inv(lattice)
        except np.linalg.LinAlgError as exc:  # pragma: no cover
            raise ValueError("晶格矩阵不可逆，POSCAR 有问题") from exc
    else:
        frac = arr

    return Poscar(
        comment=comment,
        lattice=lattice,
        species=list(source_species),
        counts=counts,
        symbols=symbols,
        frac=frac,
        coord_mode=coord_mode,
        selective=selective,
        velocities=velocities,
        source=source,
    )


def _is_float(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


# ============================================================================
# 3. 周期性近邻 / 配位分析
# ============================================================================
def _image_offsets(lattice: np.ndarray, cutoff: float) -> np.ndarray:
    """返回覆盖半径 cutoff 所需的所有晶格平移向量 (N,3)。"""
    vol = abs(np.linalg.det(lattice))
    a1, a2, a3 = lattice
    b1 = np.cross(a2, a3) / vol
    b2 = np.cross(a3, a1) / vol
    b3 = np.cross(a1, a2) / vol
    n1 = int(math.ceil(cutoff * np.linalg.norm(b1))) + 1
    n2 = int(math.ceil(cutoff * np.linalg.norm(b2))) + 1
    n3 = int(math.ceil(cutoff * np.linalg.norm(b3))) + 1
    offsets = []
    for i in range(-n1, n1 + 1):
        for j in range(-n2, n2 + 1):
            for k in range(-n3, n3 + 1):
                offsets.append(i * a1 + j * a2 + k * a3)
    return np.array(offsets)


def min_image_distances(
    pos: Poscar, cutoff: float, indices: Optional[Sequence[int]] = None
) -> np.ndarray:
    """计算满足 cutoff 的原子对最小镜像距离，返回 (M,3) 的 [i, j, dist]。"""
    idx = list(range(pos.n_atoms)) if indices is None else list(indices)
    cart = pos.cartesian()
    offsets = _image_offsets(pos.lattice, cutoff)
    pairs = []
    for a in range(len(idx)):
        i = idx[a]
        for b in range(a + 1, len(idx)):
            j = idx[b]
            diff = cart[j] - cart[i]
            d = float(np.min(np.linalg.norm(diff + offsets, axis=1)))
            if d <= cutoff:
                pairs.append((i, j, d))
    return np.array(pairs, dtype=float) if pairs else np.zeros((0, 3))


def _first_shell_count(sorted_dists: List[float], ratio: float = 1.25, abs_gap: float = 0.6) -> int:
    """在一维排序距离中找第一个"大间隙"，返回第一配位壳层原子数。"""
    if len(sorted_dists) <= 1:
        return len(sorted_dists)
    for k in range(len(sorted_dists) - 1):
        d0, d1 = sorted_dists[k], sorted_dists[k + 1]
        if d1 > d0 * ratio or (d1 - d0) > abs_gap:
            return k + 1
    return len(sorted_dists)


def coordination_analysis(
    pos: Poscar, factor: float = 1.35
) -> List[List[Tuple[int, float]]]:
    """自动判定第一配位壳层，返回每个原子的 [(近邻序号, 距离), ...]。

    先用共价半径和上限筛出候选接触（含 i==j 的自镜像，例如 fcc 原胞 CN=12），
    再用"距离间隙"截断到第一配位壳层。这样在离子化合物中不会被
    阳离子-阳离子长接触（如 Mg-Mg 3.5 Å）污染，得到正确的四面体/八面体配位。
    """
    radii = np.array([cov_radius(s) for s in pos.symbols])
    max_cut = factor * float(radii.max()) * 2.0 + 0.5
    cart = pos.cartesian()
    offsets = _image_offsets(pos.lattice, max_cut)
    offset_norms = np.linalg.norm(offsets, axis=1)
    cand: List[List[Tuple[int, float]]] = [[] for _ in pos.symbols]
    n = pos.n_atoms
    for i in range(n):
        for j in range(i, n):
            cut = factor * (radii[i] + radii[j])
            diff = cart[j] - cart[i]
            dists = np.linalg.norm(diff + offsets, axis=1)
            if i == j:
                for d in dists[(offset_norms > 1e-6) & (dists <= cut)]:
                    cand[i].append((i, float(d)))
            else:
                # 小晶胞中同一原子的多个周期镜像都可能是真实近邻，全部计入
                for d in dists[dists <= cut]:
                    cand[i].append((j, float(d)))
                    cand[j].append((i, float(d)))

    neigh: List[List[Tuple[int, float]]] = []
    for i in range(n):
        lst = sorted(cand[i], key=lambda t: t[1])
        cn = _first_shell_count([d for _, d in lst])
        neigh.append(lst[:cn])
    return neigh


def geometry_label(cn: int) -> str:
    return {
        2: "linear (2)",
        3: "trigonal-planar (3)",
        4: "tetrahedral (4)",
        5: "trigonal-bipyramidal (5)",
        6: "octahedral (6)",
        7: "7-coordinate",
        8: "cubic/8-coordinate",
        9: "9-coordinate",
        10: "10-coordinate",
        11: "11-coordinate",
        12: "cuboctahedral/fcc (12)",
    }.get(cn, f"{cn}-coordinate")


# ============================================================================
# 4. 结构基元识别
# ============================================================================
# ============================================================================
# 3.5 体系类型 (bulk / slab / mole) 与表面层的数学分析
#     判别算法与 mk-KPOINTS 项目一致：一维分数坐标的最大周期空隙 × 晶格长度，
#     空隙 > 阈值 (默认 5 Å) 即认为该方向存在真空。
# ============================================================================
DEFAULT_VACUUM_THRESHOLD = 5.0
_ACTIVE_VACUUM_THRESHOLD = DEFAULT_VACUUM_THRESHOLD


def largest_periodic_gap(frac_values: Sequence[float]) -> float:
    """一维分数坐标中最大的周期性空隙（分数，0~1）。"""
    wrapped = sorted(float(f) % 1.0 for f in frac_values)
    if not wrapped:
        return 1.0
    largest = wrapped[0] + 1.0 - wrapped[-1]
    for left, right in zip(wrapped, wrapped[1:]):
        largest = max(largest, right - left)
    return largest


def vacuum_gaps(pos: Poscar) -> Dict[str, float]:
    """返回 a/b/c 三个方向的真空层厚度 (Å)。"""
    lengths = np.linalg.norm(pos.lattice, axis=1)
    return {
        ax: largest_periodic_gap(pos.frac[:, i]) * float(lengths[i])
        for i, ax in enumerate("abc")
    }


def detect_system_type(
    pos: Poscar,
    vacuum_threshold: float = DEFAULT_VACUUM_THRESHOLD,
    gaps: Optional[Dict[str, float]] = None,
) -> Tuple[str, Dict[str, float]]:
    """按真空层判别 bulk / slab / mole / unknown（与 mk-KPOINTS 一致）。"""
    if gaps is None:
        gaps = vacuum_gaps(pos)
    has = {a: gaps[a] > vacuum_threshold + 1e-8 for a in gaps}
    if has["a"] and has["b"] and has["c"]:
        return "mole", gaps
    if (not has["a"]) and (not has["b"]) and has["c"]:
        return "slab", gaps
    if (not has["a"]) and (not has["b"]) and (not has["c"]):
        return "bulk", gaps
    return "unknown", gaps


def _vacuum_axis_index(gaps: Dict[str, float]) -> int:
    return int(np.argmax([gaps["a"], gaps["b"], gaps["c"]]))


def slab_layer_analysis(
    pos: Poscar,
    neigh: Optional[List[List[Tuple[int, float]]]] = None,
    gaps: Optional[Dict[str, float]] = None,
    layer_tol: float = 0.35,
) -> Dict:
    """把 slab 沿真空法向分层，识别上下表面 / 次表面 / 内部，并给出层间距。

    - 真空方向由最大 vacuum gap 决定（与 mk-KPOINTS 判别一致）；
    - 层位置 = 原子笛卡尔坐标在“真空轴法向”上的投影；
    - 把最大空隙平移到边界，于是 t<0 一侧是下表面、t>0 一侧是上表面。
    """
    if gaps is None:
        gaps = vacuum_gaps(pos)
    axis = _vacuum_axis_index(gaps)
    lattice = pos.lattice
    other = [x for x in range(3) if x != axis]
    normal = np.cross(lattice[other[0]], lattice[other[1]])
    nrm = float(np.linalg.norm(normal))
    if nrm < 1e-8:
        normal = lattice[axis] / (np.linalg.norm(lattice[axis]) + 1e-12)
    else:
        normal = normal / nrm
    cart = pos.cartesian()
    t = cart @ normal
    period = abs(float(np.dot(lattice[axis], normal)))
    if period < 1e-8:
        period = float(np.linalg.norm(lattice[axis]))
    t = t % period

    n = len(t)
    if n <= 1:
        t_shift = t.copy()
    else:
        order = np.argsort(t)
        ts = t[order]
        diffs = np.diff(ts)
        wrap = ts[0] + period - ts[-1]
        if wrap > diffs.max():
            gap_center = (ts[-1] + ts[0] + period) / 2.0 % period
        else:
            idx = int(np.argmax(diffs))
            gap_center = (ts[idx] + ts[idx + 1]) / 2.0
        # 把“真空层中心”映射到 ±period/2，使 slab 居中于 0
        t_shift = (t - gap_center + period / 2.0) % period
        t_shift = np.where(t_shift > period / 2.0, t_shift - period, t_shift)

    # 一维聚类成“原子层”
    order2 = np.argsort(t_shift)
    sts = t_shift[order2]
    layers = np.zeros(n, dtype=int)
    lvl = 0
    ref = sts[0]
    for i in order2[1:]:
        if sts[i] - ref > layer_tol:
            lvl += 1
            ref = sts[i]
        layers[i] = lvl
    n_layers = lvl + 1
    layer_positions = [
        float(np.mean(t_shift[layers == L])) for L in range(n_layers)
    ]
    depth = np.minimum(t_shift - t_shift.min(), t_shift.max() - t_shift)
    return {
        "vacuum_axis": "abc"[axis],
        "vacuum_axis_index": axis,
        "normal": [round(float(x), 6) for x in normal],
        "layers": layers.tolist(),
        "n_layers": int(n_layers),
        "layer_positions": [round(p, 4) for p in layer_positions],
        "layer_depth": [round(float(x), 4) for x in depth],
        "slab_thickness": round(float(t_shift.max() - t_shift.min()), 4),
    }


def site_context(
    pos: Poscar,
    neigh: Optional[List[List[Tuple[int, float]]]] = None,
    vacuum_threshold: Optional[float] = None,
) -> Dict:
    if vacuum_threshold is None:
        vacuum_threshold = _ACTIVE_VACUUM_THRESHOLD
    """综合位点上下文：体系类型 + 层信息 + 配位参考 + 位点角色。

    位点角色 (role) 取值：
      bulk        体相内部
      surface_top / surface_bottom / surface  表面积配位不饱和位点
      subsurface  次表层
      interior    slab 内部（已是体相配位）
      cluster    分子/团簇
    """
    if neigh is None:
        neigh = coordination_analysis(pos)
    gaps = vacuum_gaps(pos)
    stype, gaps = detect_system_type(pos, vacuum_threshold, gaps)
    cn_list = [len(x) for x in neigh]

    info: Dict = {
        "system_type": stype,
        "vacuum_gaps": {k: round(float(v), 4) for k, v in gaps.items()},
        "cn_list": cn_list,
    }
    if stype == "slab":
        info.update(slab_layer_analysis(pos, neigh, gaps))

    ref_cn: Dict[str, int] = {}
    for s, cn in zip(pos.symbols, cn_list):
        ref_cn[s] = max(ref_cn.get(s, 0), cn)

    layers = info.get("layers")
    roles: List[str] = []
    for i, s in enumerate(pos.symbols):
        if stype == "slab" and layers is not None:
            L = int(layers[i])
            nL = int(info["n_layers"])
            if nL <= 1:
                role = "surface"
            elif L == 0:
                role = "surface_bottom"
            elif L == nL - 1:
                role = "surface_top"
            elif nL >= 4 and (L == 1 or L == nL - 2):
                role = "subsurface"
            else:
                role = "interior"
        elif stype == "mole":
            role = "cluster"
        else:
            role = "bulk"
        roles.append(role)

    info["roles"] = roles
    info["ref_cn"] = ref_cn
    info["cn_deficit"] = [ref_cn[s] - cn for s, cn in zip(pos.symbols, cn_list)]
    info["undercoordinated"] = [d > 0 for d in info["cn_deficit"]]
    info["is_surface"] = [r.startswith("surface") or r == "cluster" for r in roles]
    # 逐 (元素, 配位数) 的位置分类，便于 LLM 区分 bulk 位与 surface 位
    info["site_groups"] = _site_group_table(pos, cn_list, roles, neigh)
    return info


def _site_group_table(
    pos: Poscar,
    cn_list: Sequence[int],
    roles: Sequence[str],
    neigh: Optional[List[List[Tuple[int, float]]]] = None,
) -> List[Dict]:
    groups: Dict[Tuple[str, int, str], Dict] = {}
    for i, s in enumerate(pos.symbols):
        key = (s, int(cn_list[i]), roles[i])
        g = groups.setdefault(
            key,
            {"element": s, "coordination": int(cn_list[i]),
             "geometry": geometry_label(int(cn_list[i])),
             "role": roles[i], "count": 0, "neighbor_composition": Counter(),
             "example_indices": []},
        )
        g["count"] += 1
        if neigh is not None:
            g["neighbor_composition"].update(pos.symbols[j] for j, _ in neigh[i])
        if len(g["example_indices"]) < 3:
            g["example_indices"].append(i)
    out = list(groups.values())
    for g in out:
        g["neighbor_composition"] = dict(g["neighbor_composition"])
    out.sort(key=lambda g: (g["element"], g["coordination"], g["role"]))
    return out


def _lattice_kind(pos: Poscar) -> str:
    """纯元素/单元素近似下判断 fcc / bcc / hcp / sc / diamond。"""
    a = pos.lattice
    lengths = np.linalg.norm(a, axis=1)
    angles = []
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        c = np.dot(a[j], a[k]) / (lengths[j] * lengths[k])
        angles.append(math.degrees(math.acos(max(-1.0, min(1.0, c)))))
    cubic = all(abs(x - 90) < 1.0 for x in angles) and np.allclose(lengths, lengths[0], atol=0.05)
    if cubic:
        return "cubic"
    # 六方判据：gamma=120, alpha=beta=90
    if abs(angles[2] - 120) < 1.0 and abs(angles[0] - 90) < 1.0 and abs(angles[1] - 90) < 1.0:
        return "hexagonal"
    return "other"


def detect_motif(
    pos: Poscar, neigh: List[List[Tuple[int, float]]], system_type: str = "bulk"
) -> Dict:
    """基于化学计量、最大配位数与晶格形状识别常见结构基元。

    使用"最大配位数"而非瞬时配位数，以兼容 slab 上下表面配位不饱和的情况
    （表面 4 配位原子降到 3、八面体 6 配位降到 5 等，内部原子仍保留 bulk 配位）。
    """
    comp = pos.composition()
    cn_list = [len(x) for x in neigh]
    elements = set(comp)

    cn_by_element: Dict[str, Counter] = {}
    max_cn: Dict[str, int] = {}
    for i, s in enumerate(pos.symbols):
        cn_by_element.setdefault(s, Counter())[cn_list[i]] += 1
        max_cn[s] = max(max_cn.get(s, 0), cn_list[i])

    motif: Dict = {
        "family": "unknown",
        "name": "未识别结构",
        "cations": {},
        "anions": sorted(elements & ANIONS),
        "cn_by_element": {el: dict(c) for el, c in cn_by_element.items()},
        "max_cn": max_cn,
        "system_type": system_type,
        "notes": [],
    }
    if system_type == "slab":
        motif["notes"].append(
            "这是由体相切出的 slab，上下表面存在配位不饱和位点；"
            "MAGMOM 需要区分 bulk 内部位点与 surface/次表面位点。"
        )
    elif system_type == "mole":
        motif["notes"].append("这是分子/团簇体系，所有原子都可视为配位不饱和。")

    # ---- 单元素金属 ---------------------------------------------------
    if len(elements) == 1:
        el = next(iter(elements))
        kind = _lattice_kind(pos)
        cn = max_cn[el]
        sub = "unknown"
        if cn == 12:
            sub = "hcp" if kind == "hexagonal" else "fcc"
        elif cn >= 13:
            sub = "bcc"
        elif cn == 8:
            sub = "bcc"
        elif cn == 6:
            sub = "simple-cubic"
        elif cn == 4:
            sub = "diamond-like"
        motif.update(family="metal", name=f"{el} 金属 ({sub})", lattice=kind, metal_lattice=sub)
        return motif

    # ---- 含阴离子/羟基的化合物 ----------------------------------------
    anions = elements & ANIONS
    if anions:
        anion = sorted(anions)[0]
        norm = pos.reduced_composition()
        metals = [el for el in elements if el not in ANIONS and el != "H"]
        n_cations = sum(c for el, c in norm.items() if el != anion and el != "H")
        cation_elements = [el for el in norm if el != anion and el != "H"]
        site_cn = {el: set(cn_by_element.get(el, {})) for el in cation_elements}
        tet_els = [el for el in cation_elements if 4 in site_cn[el]]
        oct_els = [el for el in cation_elements if 6 in site_cn[el]]

        # ---- 氢氧化物 / 层状双氢氧化物 LDH ----------------------------
        if "H" in elements and "O" in elements and metals:
            oct_metals = [el for el in metals if max_cn.get(el, 0) >= 6]
            if len(metals) >= 2 and oct_metals:
                motif.update(
                    family="ldh",
                    name=f"层状双金属氢氧化物 LDH {pos.formula()}",
                    metals=sorted(metals),
                    notes=[
                        "主体为八面体 M(OH)6 共边形成的类水镁石层，层间为阴离子/水；",
                        "磁矩主要来自层板中八面体配位的过渡金属 M2+/M3+，层间物种一律 0。",
                        "LDH 层板内常为铁磁或自旋玻璃，可用正磁矩或反铁磁交替作初猜。",
                    ],
                )
                return motif
            if len(metals) == 1 and max_cn.get(metals[0], 0) >= 6:
                motif.update(
                    family="hydroxide",
                    name=f"氢氧化物 {pos.formula()} (brucite-like)",
                    metals=sorted(metals),
                    notes=["八面体 M(OH)6 层板；H 置 0。"],
                )
                return motif

        # ---- 双钙钛矿 A2 B B' O6 ------------------------------------
        if norm.get(anion) == 6 and len(cation_elements) >= 3:
            counts = sorted(norm[el] for el in cation_elements)
            if counts == [1, 1, 2]:
                b_sites = [el for el in cation_elements if max_cn.get(el, 0) == 6]
                motif.update(
                    family="double_perovskite",
                    name=f"双钙钛矿型 {pos.formula()}",
                    b_sites=b_sites,
                    notes=[
                        "B/B' 位为八面体过渡金属，通常反平行排列（铁磁/亚铁磁）；",
                        "如 Sr2FeMoO6：Fe3+(+5) 与 Mo5+(-1) 反平行。",
                    ],
                )
                return motif

        # ---- 尖晶石 A B2 O4（含 Fe3O4 / Co3O4 这类二元 3:4）-----------
        if norm.get(anion) == 4 and n_cations == 3:
            if len(cation_elements) == 1:
                el = cation_elements[0]
                if 4 in site_cn[el] and 6 in site_cn[el]:
                    stype = "inverse/mixed（同一元素同时占 8a 与 16d）"
                elif 6 in site_cn[el]:
                    stype = "全部八面体（非典型尖晶石）"
                else:
                    stype = "未定"
                motif.update(
                    family="spinel", name=f"尖晶石型 {pos.formula()}",
                    spinel_type=stype, tetrahedral_site=el, octahedral_site=el,
                )
            else:
                one = [el for el in cation_elements if norm[el] == 1]
                two = [el for el in cation_elements if norm[el] == 2]
                tet = tet_els[0] if tet_els else None
                if one and two and tet == one[0]:
                    stype, name = "normal", f"正尖晶石 {pos.formula()}"
                elif two and tet == two[0]:
                    stype, name = "inverse", f"反尖晶石 {pos.formula()}"
                else:
                    stype, name = "mixed/undetermined", f"尖晶石型 {pos.formula()}"
                motif.update(
                    family="spinel", name=name, spinel_type=stype,
                    tetrahedral_site=tet, octahedral_site=oct_els[0] if oct_els else None,
                )
            motif["notes"].append(
                "MAGMOM 需按 8a(四面体) 与 16d(八面体) 位点分别设置；"
                "反尖晶石/混合型中 16d 位由多种阳离子共存。"
            )
            return motif

        # ---- 钙钛矿 A B O3 ------------------------------------------
        if norm.get(anion) == 3 and n_cations == 2:
            a_site = [el for el in cation_elements if max_cn.get(el, 0) >= 8]
            b_site = [el for el in cation_elements if max_cn.get(el, 0) == 6]
            motif.update(
                family="perovskite",
                name=f"钙钛矿型 {pos.formula()}",
                a_site=a_site or [cation_elements[0]],
                b_site=b_site or [cation_elements[-1]],
                notes=[
                    "B 位过渡金属（CN=6）承担磁矩；A 位（La/Sr/Ba/Ca/稀土，CN≥8）"
                    "通常非磁或为 4f 磁矩。",
                ],
            )
            return motif

        # ---- 岩盐 / 闪锌矿 / 纤锌矿 AO -------------------------------
        if len(norm) == 2 and norm.get(anion) == 1:
            cation = cation_elements[0] if cation_elements else None
            family, name = "rocksalt", f"岩盐型 {pos.formula()}"
            if cation and max_cn.get(cation, 0) == 4:
                if _lattice_kind(pos) == "hexagonal":
                    family, name = "wurtzite", f"纤锌矿型 {pos.formula()}"
                else:
                    family, name = "zincblende", f"闪锌矿型 {pos.formula()}"
            motif.update(
                family=family, name=name,
                notes=["岩盐型氧化物（MnO/FeO/CoO/NiO）多为反铁磁，需要正负号交替。"],
            )
            return motif

        # ---- 金红石 MO2 --------------------------------------------
        if norm.get(anion) == 2 and len(cation_elements) == 1:
            cation = cation_elements[0]
            if max_cn.get(cation, 0) == 6:
                motif.update(
                    family="rutile",
                    name=f"金红石型 {pos.formula()}",
                    notes=["阳离子六配位、阴离子三配位；按阳离子价态给高自旋磁矩。"],
                )
                return motif

        # ---- 刚玉 A2O3 ---------------------------------------------
        if norm.get(anion) == 3 and n_cations == 2:
            motif.update(
                family="corundum",
                name=f"刚玉型 {pos.formula()}",
                notes=["Cr2O3 / Fe2O3 / Al2O3 等，磁性时通常为反铁磁。"],
            )
            return motif

        motif.update(
            family="oxide/compound",
            name=f"含 {anion} 化合物 {pos.formula()}",
            notes=["按元素常见价态与配位环境给出高自旋初猜。"],
        )
        return motif

    # ---- 其他金属间化合物 ---------------------------------------------
    motif.update(
        family="intermetallic",
        name=f"金属间/合金化合物 {pos.formula()}",
        notes=["若无明确磁性信息，对 3d 过渡金属给有限初猜，其余置 0。"],
    )
    return motif


# ============================================================================
# 5. 内置磁性材料知识库
# ============================================================================
def canonical_formula(comp: Dict[str, int]) -> str:
    return "".join(f"{el}{comp[el] if comp[el] > 1 else ''}" for el in sorted(comp))


def _kb_entry(comp, name, order, assignments, afm=(), note=""):
    return dict(
        comp=comp,
        name=name,
        order=order,
        assignments=assignments,
        afm=list(afm),
        note=note,
    )


_KB_LIST = [
    # ---------- 尖晶石 / 反尖晶石 ----------
    _kb_entry(
        {"Fe": 3, "O": 4}, "磁铁矿 Fe3O4（反尖晶石）", "ferrimagnetic",
        [
            {"element": "Fe", "coordination": 4, "moment": 5.0, "site": "8a 四面体 Fe3+"},
            {"element": "Fe", "coordination": 6, "moment": -5.0,
             "site": "16d 八面体 Fe2+/Fe3+"},
            {"element": "O", "moment": 0.0, "site": "32e"},
        ],
        note="亚铁磁：8a Fe3+ 与整个 16d 亚晶格反平行（16d 内部同向）；八面体混合价用 |5| 作初猜。",
    ),
    _kb_entry(
        {"Co": 3, "O": 4}, "Co3O4（正尖晶石）", "antiferromagnetic",
        [
            {"element": "Co", "coordination": 4, "moment": 3.0, "afm_group": "tet",
             "site": "8a 四面体 Co2+"},
            {"element": "Co", "coordination": 6, "moment": 0.0, "site": "16d 八面体 Co3+ 低自旋"},
            {"element": "O", "moment": 0.0},
        ],
        afm=["tet"],
        note="Co3+ (3d6, 八面体) 低自旋 S=0；四面体 Co2+ 亚晶格反铁磁。",
    ),
    _kb_entry(
        {"Ni": 1, "Fe": 2, "O": 4}, "镍铁氧体 NiFe2O4（反尖晶石）", "ferrimagnetic",
        [
            {"element": "Fe", "coordination": 4, "moment": 5.0, "site": "8a Fe3+"},
            {"element": "Fe", "coordination": 6, "moment": -5.0, "site": "16d Fe3+"},
            {"element": "Ni", "coordination": 6, "moment": -2.0, "site": "16d Ni2+"},
            {"element": "O", "moment": 0.0},
        ],
        note="(Fe3+)tet [Ni2+Fe3+]oct O4，16d 亚晶格整体与 8a 反平行。",
    ),
    _kb_entry(
        {"Co": 1, "Fe": 2, "O": 4}, "钴铁氧体 CoFe2O4（反尖晶石）", "ferrimagnetic",
        [
            {"element": "Fe", "coordination": 4, "moment": 5.0, "site": "8a Fe3+"},
            {"element": "Fe", "coordination": 6, "moment": -5.0},
            {"element": "Co", "coordination": 6, "moment": -3.0},
            {"element": "O", "moment": 0.0},
        ],
        note="Co2+ 高自旋 S=3/2；16d 亚晶格整体与 8a 反平行。",
    ),
    _kb_entry(
        {"Mn": 1, "Fe": 2, "O": 4}, "锰铁氧体 MnFe2O4（多为反尖晶石）", "ferrimagnetic",
        [
            {"element": "Fe", "coordination": 4, "moment": 5.0},
            {"element": "Fe", "coordination": 6, "moment": -5.0},
            {"element": "Mn", "coordination": 6, "moment": -5.0},
            {"element": "O", "moment": 0.0},
        ],
        note="Mn2+ 高自旋 S=5/2；实际阳离子分布随合成条件变化。",
    ),
    _kb_entry(
        {"Zn": 1, "Fe": 2, "O": 4}, "锌铁氧体 ZnFe2O4（正尖晶石）", "antiferromagnetic",
        [
            {"element": "Zn", "coordination": 4, "moment": 0.0},
            {"element": "Fe", "coordination": 6, "moment": 5.0, "afm_group": "oct"},
            {"element": "O", "moment": 0.0},
        ],
        afm=["oct"],
        note="Zn2+ 3d10 非磁；八面体 Fe3+ 亚晶格反铁磁，需正负交替。",
    ),
    _kb_entry(
        {"Mg": 1, "Fe": 2, "O": 4}, "镁铁氧体 MgFe2O4（部分反尖晶石）", "ferrimagnetic",
        [
            {"element": "Fe", "coordination": 4, "moment": 5.0},
            {"element": "Fe", "coordination": 6, "moment": -5.0},
            {"element": "Mg", "moment": 0.0},
            {"element": "O", "moment": 0.0},
        ],
        note="阳离子分布介于正/反尖晶石之间。",
    ),
    _kb_entry(
        {"Mg": 1, "Al": 2, "O": 4}, "镁铝尖晶石 MgAl2O4（正尖晶石，非磁）", "nonmagnetic",
        [
            {"element": "Mg", "moment": 0.0},
            {"element": "Al", "moment": 0.0},
            {"element": "O", "moment": 0.0},
        ],
        note="全 d0/d10，ISPIN 其实可设为 1；此处 ISPIN=2 时全部置 0。",
    ),
    # ---------- 岩盐反铁磁体 ----------
    _kb_entry(
        {"Ni": 1, "O": 1}, "NiO（岩盐，II 型反铁磁）", "antiferromagnetic",
        [{"element": "Ni", "moment": 2.0}, {"element": "O", "moment": 0.0}],
        afm=["Ni"],
        note="fcc Ni 亚晶格 II 型反铁磁（(111) 面内铁磁、面间反铁磁），用磁层投影给正负交替。",
    ),
    _kb_entry(
        {"Co": 1, "O": 1}, "CoO（岩盐，反铁磁）", "antiferromagnetic",
        [{"element": "Co", "moment": 3.0}, {"element": "O", "moment": 0.0}],
        afm=["Co"],
    ),
    _kb_entry(
        {"Mn": 1, "O": 1}, "MnO（岩盐，II 型反铁磁）", "antiferromagnetic",
        [{"element": "Mn", "moment": 5.0}, {"element": "O", "moment": 0.0}],
        afm=["Mn"],
    ),
    _kb_entry(
        {"Fe": 1, "O": 1}, "FeO（方铁矿，反铁磁）", "antiferromagnetic",
        [{"element": "Fe", "moment": 4.0}, {"element": "O", "moment": 0.0}],
        afm=["Fe"],
        note="Fe2+ 高自旋 S=2。",
    ),
    # ---------- 倍半氧化物 ----------
    _kb_entry(
        {"Fe": 2, "O": 3}, "赤铁矿 Fe2O3（刚玉型，反铁磁）", "antiferromagnetic",
        [{"element": "Fe", "moment": 5.0}, {"element": "O", "moment": 0.0}],
        afm=["Fe"],
        note="Fe3+ 高自旋 S=5/2。",
    ),
    _kb_entry(
        {"Cr": 2, "O": 3}, "Cr2O3（刚玉型，反铁磁）", "antiferromagnetic",
        [{"element": "Cr", "moment": 3.0}, {"element": "O", "moment": 0.0}],
        afm=["Cr"],
        note="Cr3+ S=3/2，奈尔温度约 308 K。",
    ),
    # ---------- 钙钛矿 ----------
    _kb_entry(
        {"La": 1, "Mn": 1, "O": 3}, "LaMnO3（A 型反铁磁）", "antiferromagnetic",
        [
            {"element": "Mn", "coordination": 6, "moment": 4.0},
            {"element": "La", "moment": 0.0},
            {"element": "O", "moment": 0.0},
        ],
        afm=["Mn"],
        note="Mn3+ 高自旋 S=2；A 型反铁磁为面内铁磁、面间反平行，"
             "简单二部图染色仅给近似初猜。",
    ),
    _kb_entry(
        {"La": 1, "Fe": 1, "O": 3}, "LaFeO3（G 型反铁磁）", "antiferromagnetic",
        [
            {"element": "Fe", "coordination": 6, "moment": 5.0},
            {"element": "La", "moment": 0.0},
            {"element": "O", "moment": 0.0},
        ],
        afm=["Fe"],
        note="Fe3+ S=5/2；G 型反铁磁，最近邻 Fe 自旋相反。",
    ),
    _kb_entry(
        {"Bi": 1, "Fe": 1, "O": 3}, "BiFeO3（G 型反铁磁 + 弱铁电）", "antiferromagnetic",
        [
            {"element": "Fe", "coordination": 6, "moment": 5.0},
            {"element": "Bi", "moment": 0.0},
            {"element": "O", "moment": 0.0},
        ],
        afm=["Fe"],
    ),
    _kb_entry(
        {"La": 1, "Co": 1, "O": 3}, "LaCoO3（温致自旋态转变）", "paramagnetic",
        [
            {"element": "Co", "coordination": 6, "moment": 0.0},
            {"element": "La", "moment": 0.0},
            {"element": "O", "moment": 0.0},
        ],
        note="低温 Co3+ 低自旋 S=0；如研究激发态可给非零初猜（如 2~4）。",
    ),
    _kb_entry(
        {"Sr": 1, "Fe": 1, "O": 3}, "SrFeO3（螺旋反铁磁）", "antiferromagnetic",
        [
            {"element": "Fe", "coordination": 6, "moment": 4.0},
            {"element": "Sr", "moment": 0.0},
            {"element": "O", "moment": 0.0},
        ],
        note="Fe4+ (3d4) 高自旋 S=2；螺旋磁序，初猜给有限正磁矩即可。",
    ),
    # ---------- 简单氧化物 / 分子 ----------
    _kb_entry(
        {"Cr": 1, "O": 2}, "CrO2（铁磁金属）", "ferromagnetic",
        [{"element": "Cr", "coordination": 6, "moment": 2.0},
         {"element": "O", "moment": 0.0}],
        note="Cr4+ (3d2)，实验磁矩约 2 μB，铁磁。",
    ),
    _kb_entry(
        {"Mn": 1, "O": 2}, "MnO2（金红石型，反铁磁）", "antiferromagnetic",
        [{"element": "Mn", "coordination": 6, "moment": 3.0},
         {"element": "O", "moment": 0.0}],
        afm=["Mn"],
        note="Mn4+ (3d3) S=3/2。",
    ),
    _kb_entry(
        {"Mn": 2, "O": 3}, "Mn2O3（反铁磁）", "antiferromagnetic",
        [{"element": "Mn", "coordination": 6, "moment": 4.0},
         {"element": "O", "moment": 0.0}],
        afm=["Mn"],
        note="Mn3+ (3d4) 高自旋 S=2。",
    ),
    _kb_entry(
        {"V": 2, "O": 5}, "V2O5（非磁半导体）", "nonmagnetic",
        [{"element": "V", "moment": 0.0}, {"element": "O", "moment": 0.0}],
        note="V5+ 为 d0，非磁。",
    ),
    _kb_entry(
        {"Ti": 1, "O": 2}, "TiO2（非磁）", "nonmagnetic",
        [{"element": "Ti", "moment": 0.0}, {"element": "O", "moment": 0.0}],
        note="Ti4+ d0。",
    ),
    # ---------- 绝缘磷酸盐 ----------
    _kb_entry(
        {"Li": 1, "Fe": 1, "P": 1, "O": 4}, "LiFePO4（橄榄石，反铁磁）", "antiferromagnetic",
        [{"element": "Fe", "moment": 4.0}, {"element": "Li", "moment": 0.0},
         {"element": "P", "moment": 0.0}, {"element": "O", "moment": 0.0}],
        afm=["Fe"],
        note="Fe2+ 高自旋 S=2。",
    ),
    # ---------- 纯金属 ----------
    _kb_entry(
        {"Fe": 1}, "bcc Fe（铁磁金属）", "ferromagnetic",
        [{"element": "Fe", "moment": 2.2}],
        note="bcc Fe 实验磁矩约 2.22 μB。",
    ),
    _kb_entry(
        {"Co": 1}, "hcp/fcc Co（铁磁金属）", "ferromagnetic",
        [{"element": "Co", "moment": 1.7}],
        note="Co 实验磁矩约 1.72 μB。",
    ),
    _kb_entry(
        {"Ni": 1}, "fcc Ni（铁磁金属）", "ferromagnetic",
        [{"element": "Ni", "moment": 0.6}],
        note="Ni 实验磁矩约 0.6 μB；初猜可给 0.6~1.0。",
    ),
    _kb_entry(
        {"Cr": 1}, "bcc Cr（反铁磁）", "antiferromagnetic",
        [{"element": "Cr", "moment": 1.0}],
        afm=["Cr"],
        note="Cr 为反铁磁，需正负交替。",
    ),
    _kb_entry(
        {"Mn": 1}, "α-Mn（复杂反铁磁）", "antiferromagnetic",
        [{"element": "Mn", "moment": 2.0}],
        afm=["Mn"],
        note="α-Mn 磁结构复杂，初猜给交替小磁矩。",
    ),
    _kb_entry(
        {"Pt": 1}, "fcc Pt（体相非磁）", "nonmagnetic",
        [{"element": "Pt", "moment": 0.0}],
        note="体相 Pt 为顺磁/非磁，MAGMOM 应置 0（不要用旧的 3！）。",
    ),
    _kb_entry(
        {"Pd": 1}, "fcc Pd（体相非磁）", "nonmagnetic",
        [{"element": "Pd", "moment": 0.0}],
    ),
    _kb_entry(
        {"Cu": 1}, "fcc Cu（非磁）", "nonmagnetic",
        [{"element": "Cu", "moment": 0.0}],
    ),
    _kb_entry(
        {"Ag": 1}, "fcc Ag（非磁）", "nonmagnetic",
        [{"element": "Ag", "moment": 0.0}],
    ),
    _kb_entry(
        {"Au": 1}, "fcc Au（非磁）", "nonmagnetic",
        [{"element": "Au", "moment": 0.0}],
    ),
    _kb_entry(
        {"Al": 1}, "fcc Al（非磁）", "nonmagnetic",
        [{"element": "Al", "moment": 0.0}],
    ),
    _kb_entry(
        {"Gd": 1}, "hcp Gd（铁磁，4f7）", "ferromagnetic",
        [{"element": "Gd", "moment": 7.0}],
        note="Gd3+ 4f7，S=7/2。",
    ),
    # ---------- 闪锌矿 / 纤锌矿 ----------
    _kb_entry(
        {"Mn": 1, "S": 1}, "MnS（闪锌矿/纤锌矿，反铁磁）", "antiferromagnetic",
        [{"element": "Mn", "coordination": 4, "moment": 5.0},
         {"element": "S", "moment": 0.0}],
        afm=["Mn"],
        note="Mn2+ 高自旋 S=5/2；四配位亚晶格反铁磁。",
    ),
    _kb_entry(
        {"Mn": 1, "Se": 1}, "MnSe（反铁磁）", "antiferromagnetic",
        [{"element": "Mn", "coordination": 4, "moment": 5.0},
         {"element": "Se", "moment": 0.0}],
        afm=["Mn"],
    ),
    _kb_entry(
        {"Mn": 1, "Te": 1}, "MnTe（反铁磁）", "antiferromagnetic",
        [{"element": "Mn", "coordination": 4, "moment": 5.0},
         {"element": "Te", "moment": 0.0}],
        afm=["Mn"],
    ),
    _kb_entry(
        {"Zn": 1, "S": 1}, "ZnS 闪锌矿/纤锌矿（非磁）", "nonmagnetic",
        [{"element": "Zn", "moment": 0.0}, {"element": "S", "moment": 0.0}],
        note="Zn2+ 3d10，全非磁。",
    ),
    _kb_entry(
        {"Zn": 1, "O": 1}, "ZnO 纤锌矿（非磁）", "nonmagnetic",
        [{"element": "Zn", "moment": 0.0}, {"element": "O", "moment": 0.0}],
        note="Zn2+ 3d10、O2- 闭壳层，非磁（缺陷/掺杂才可能有磁矩）。",
    ),
    _kb_entry(
        {"Ga": 1, "As": 1}, "GaAs 闪锌矿（非磁）", "nonmagnetic",
        [{"element": "Ga", "moment": 0.0}, {"element": "As", "moment": 0.0}],
    ),
    # ---------- 双钙钛矿 ----------
    _kb_entry(
        {"Sr": 2, "Fe": 1, "Mo": 1, "O": 6},
        "Sr2FeMoO6（双钙钛矿，亚铁磁）", "ferrimagnetic",
        [
            {"element": "Fe", "coordination": 6, "moment": 5.0},
            {"element": "Mo", "coordination": 6, "moment": -1.0},
            {"element": "Sr", "moment": 0.0},
            {"element": "O", "moment": 0.0},
        ],
        note="B/B' 有序双钙钛矿：Fe3+(↑,5) 与 Mo5+(↓,1) 反平行，居里温度高。",
    ),
    _kb_entry(
        {"Sr": 2, "Fe": 1, "Re": 1, "O": 6},
        "Sr2FeReO6（双钙钛矿，亚铁磁）", "ferrimagnetic",
        [
            {"element": "Fe", "coordination": 6, "moment": 5.0},
            {"element": "Re", "coordination": 6, "moment": -2.0},
            {"element": "Sr", "moment": 0.0},
            {"element": "O", "moment": 0.0},
        ],
        note="Fe3+(↑) 与 Re5+(4d2, ↓) 反平行。",
    ),
    _kb_entry(
        {"La": 2, "Ni": 1, "Mn": 1, "O": 6},
        "La2NiMnO6（双钙钛矿，铁磁）", "ferromagnetic",
        [
            {"element": "Ni", "coordination": 6, "moment": 2.0},
            {"element": "Mn", "coordination": 6, "moment": 3.0},
            {"element": "La", "moment": 0.0},
            {"element": "O", "moment": 0.0},
        ],
        note="Ni2+(↑,2) 与 Mn4+(↑,3) 铁磁耦合（有序双钙钛矿）。",
    ),
    # ---------- 更多钙钛矿 ----------
    _kb_entry(
        {"La": 1, "Cr": 1, "O": 3}, "LaCrO3（G 型反铁磁）", "antiferromagnetic",
        [{"element": "Cr", "coordination": 6, "moment": 3.0},
         {"element": "La", "moment": 0.0}, {"element": "O", "moment": 0.0}],
        afm=["Cr"],
    ),
    _kb_entry(
        {"La": 1, "Ni": 1, "O": 3}, "LaNiO3（金属，顺磁）", "paramagnetic",
        [{"element": "Ni", "coordination": 6, "moment": 1.0},
         {"element": "La", "moment": 0.0}, {"element": "O", "moment": 0.0}],
        note="Ni3+ (3d7) 金属性，给 1 作初猜即可（不期望长程有序）。",
    ),
    # ---------- 氢氧化物 / 层状双氢氧化物 LDH ----------
    _kb_entry(
        {"Ni": 1, "O": 2, "H": 2}, "Ni(OH)2（水镁石型，反铁磁）", "antiferromagnetic",
        [{"element": "Ni", "coordination": 6, "moment": 2.0},
         {"element": "O", "moment": 0.0}, {"element": "H", "moment": 0.0}],
        afm=["Ni"],
    ),
    _kb_entry(
        {"Co": 1, "O": 2, "H": 2}, "Co(OH)2（水镁石型，反铁磁）", "antiferromagnetic",
        [{"element": "Co", "coordination": 6, "moment": 3.0},
         {"element": "O", "moment": 0.0}, {"element": "H", "moment": 0.0}],
        afm=["Co"],
    ),
    _kb_entry(
        {"Ni": 3, "Fe": 1, "O": 8, "H": 8},
        "NiFe-LDH (Ni:Fe=3:1)", "ferromagnetic",
        [
            {"element": "Ni", "coordination": 6, "moment": 2.0},
            {"element": "Fe", "coordination": 6, "moment": 5.0},
            {"element": "O", "moment": 0.0},
            {"element": "H", "moment": 0.0},
        ],
        note="层板八面体 Ni2+/Fe3+，常见铁磁/亚铁磁耦合；层间物种置 0。",
    ),
    _kb_entry(
        {"Co": 3, "Fe": 1, "O": 8, "H": 8},
        "CoFe-LDH (Co:Fe=3:1)", "ferrimagnetic",
        [
            {"element": "Co", "coordination": 6, "moment": 3.0},
            {"element": "Fe", "coordination": 6, "moment": -5.0},
            {"element": "O", "moment": 0.0},
            {"element": "H", "moment": 0.0},
        ],
        note="层板 Fe3+ 与 Co2+ 常反平行（亚铁磁）。",
    ),
    _kb_entry(
        {"Ni": 3, "Al": 1, "O": 8, "H": 8},
        "NiAl-LDH (Ni:Al=3:1)", "antiferromagnetic",
        [
            {"element": "Ni", "coordination": 6, "moment": 2.0, "afm_group": "ni"},
            {"element": "Al", "moment": 0.0},
            {"element": "O", "moment": 0.0},
            {"element": "H", "moment": 0.0},
        ],
        afm=["ni"],
        note="Al3+ 非磁，磁矩来自层板 Ni2+ 亚晶格（反铁磁/自旋玻璃）。",
    ),
    _kb_entry(
        {"Mg": 3, "Al": 1, "O": 8, "H": 8},
        "MgAl-LDH (Mg:Al=3:1)", "nonmagnetic",
        [{"element": "Mg", "moment": 0.0}, {"element": "Al", "moment": 0.0},
         {"element": "O", "moment": 0.0}, {"element": "H", "moment": 0.0}],
        note="全 d0/d10，非磁。",
    ),
]

KNOWLEDGE_BASE: Dict[str, Dict] = {
    canonical_formula(e["comp"]): e for e in _KB_LIST
}


def lookup_known_material(formula_or_name: str) -> Optional[Dict]:
    """按化学式或名称模糊查询知识库。"""
    if not formula_or_name:
        return None
    token = formula_or_name.strip()
    # 1) 规范化学式直接命中（支持括号、水合物等）
    comp = _parse_formula(token)
    if comp:
        key = canonical_formula(comp)
        if key in KNOWLEDGE_BASE:
            return KNOWLEDGE_BASE[key]
    # 2) 名称子串匹配（如“磁铁矿”“NiFe-LDH”）
    low = token.lower()
    for key, entry in KNOWLEDGE_BASE.items():
        if low == key.lower() or (len(low) >= 3 and low in entry["name"].lower()):
            return entry
    return None


def _parse_formula(text: str) -> Optional[Dict[str, int]]:
    """解析化学式，支持嵌套括号与下标，例如 Ni3Fe(OH)8、Ni(OH)2。

    未知元素/字母（如 'LDH' 里的 L、D）会被忽略，避免误报。
    """
    if not text:
        return None
    stack: List[Dict[str, int]] = [{}]
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c in "([":
            stack.append({})
            i += 1
        elif c in ")]":
            group = stack.pop()
            i += 1
            j = i
            while j < n and text[j].isdigit():
                j += 1
            mult = int(text[i:j]) if j > i else 1
            i = j
            for el, cnt in group.items():
                stack[-1][el] = stack[-1].get(el, 0) + cnt * mult
        elif c.isupper():
            j = i + 1
            while j < n and text[j].islower():
                j += 1
            el = text[i:j]
            i = j
            k = i
            while k < n and text[k].isdigit():
                k += 1
            cnt = int(text[i:k]) if k > i else 1
            i = k
            if el in ATOMIC_MASS:
                stack[-1][el] = stack[-1].get(el, 0) + cnt
        else:
            i += 1
    comp = stack[0]
    return comp or None


# ============================================================================
# 6. 反铁磁符号分配（二部图染色）
# ============================================================================
def _candidate_axes(lattice: np.ndarray) -> List[np.ndarray]:
    """生成候选磁序方向：晶格矢量、两两和/差、体对角线。"""
    a, b, c = lattice
    vecs = [a, b, c,
            a + b, a + c, b + c, a - b, a - c, b - c,
            a + b + c, a + b - c, a - b + c, -a + b + c]
    axes = []
    for v in vecs:
        n = np.linalg.norm(v)
        if n > 1e-8:
            axes.append(v / n)
    return axes


def _min_positive_gap(proj: np.ndarray) -> Optional[float]:
    vals = np.sort(np.unique(np.round(proj, 6)))
    if len(vals) < 2:
        return None
    gaps = np.diff(vals)
    gaps = gaps[gaps > 1e-4]
    return float(np.min(gaps)) if len(gaps) else None


def _quantize_levels(proj: np.ndarray, tol: float) -> np.ndarray:
    order = np.argsort(proj)
    levels = np.zeros(len(proj), dtype=int)
    lvl = 0
    ref = proj[order[0]]
    for pos in order[1:]:
        if proj[pos] - ref > tol:
            lvl += 1
            ref = proj[pos]
        levels[pos] = lvl
    return levels


def layered_signs(
    pos: Poscar,
    indices: Sequence[int],
    axis: Optional[Sequence[float]] = None,
    tol_frac: float = 0.4,
) -> Dict[int, int]:
    """按“磁层”交替给符号：把磁性原子投影到某方向，按层号奇偶给 ±。

    这是 fcc 岩盐（NiO/CoO/MnO 的 II 型反铁磁）、尖晶石八面体亚晶格等
    非二部图体系的标准初猜方式。axis=None 时自动在候选方向中挑最佳者。
    """
    idx = list(indices)
    if len(idx) <= 1:
        return {i: 1 for i in idx}
    cart = pos.cartesian()[idx]
    local = {g: k for k, g in enumerate(idx)}

    pairs = min_image_distances(pos, cutoff=8.0, indices=idx)
    if len(pairs) == 0:
        return {i: 1 for i in idx}
    dmin = float(np.min(pairs[:, 2]))
    nn = pairs[pairs[:, 2] <= dmin * 1.2]
    ii = [local[int(a)] for a, _, _ in nn]
    jj = [local[int(b)] for _, b, _ in nn]

    if axis is not None:
        axes = [np.array(axis, dtype=float) / np.linalg.norm(axis)]
    else:
        axes = _candidate_axes(pos.lattice)

    best = None
    for ax in axes:
        proj = cart @ ax
        sp = _min_positive_gap(proj)
        if not sp:
            continue
        levels = _quantize_levels(proj, tol_frac * sp)
        li = levels[ii]
        lj = levels[jj]
        score = float(np.mean((li % 2) != (lj % 2))) if len(li) else 0.0
        # 偏好层数少、两类原子尽量均衡的方案
        balance = 1.0 - abs(np.mean(levels % 2) - 0.5) * 2
        key = (round(score, 4), round(balance, 4), -len(set(levels.tolist())))
        if best is None or key > best[0]:
            best = (key, levels)

    if best is None:
        return {i: 1 for i in idx}
    levels = best[1]
    return {g: (1 if levels[k] % 2 == 0 else -1) for k, g in enumerate(idx)}


def bipartite_signs(
    pos: Poscar,
    element: Optional[str] = None,
    neigh: Optional[List[List[Tuple[int, float]]]] = None,
    nn_tolerance: float = 1.15,
    indices: Optional[Sequence[int]] = None,
) -> Dict[int, int]:
    """对指定磁性亚晶格做近邻图二部图染色，返回 {atom_index: +1/-1}。

    为正确处理小晶胞中的周期镜像（例如 8 原子 NiO 惯用胞），先把亚晶格扩成
    3x3x3 超胞再对该“展开图”二染色，最后取中心拷贝的符号。
    对 fcc / bcc 等二部图晶格，这正好给出面间反平行的反铁磁初猜。
    """
    if indices is not None:
        idx = list(indices)
    elif element is None:
        idx = list(range(pos.n_atoms))
    else:
        idx = [i for i, s in enumerate(pos.symbols) if s == element]
    if len(idx) <= 1:
        return {i: 1 for i in idx}

    base = pos.frac[idx]
    lattice = pos.lattice
    shifts = list(itertools.product((-1, 0, 1), repeat=3))
    shifts.sort(key=lambda s: (abs(s[0]) + abs(s[1]) + abs(s[2]), s))
    home = shifts.index((0, 0, 0))

    fracs = []
    for sh in shifts:
        for f in base:
            fracs.append(f + np.array(sh, dtype=float))
    fracs = np.array(fracs)
    cart = fracs @ lattice
    nnode = len(cart)
    chunk = max(64, min(512, 8_000_000 // max(1, nnode)))

    # 第一遍：分块求展开图内最近邻间距
    def _chunk_dist(s: int, e: int) -> np.ndarray:
        d = np.linalg.norm(cart[s:e, None, :] - cart[None, :, :], axis=2)
        for a in range(e - s):
            d[a, s + a] = np.inf
        return d

    dmin = np.inf
    for s in range(0, nnode, chunk):
        e = min(nnode, s + chunk)
        dmin = min(dmin, float(_chunk_dist(s, e).min()))
    if not np.isfinite(dmin):
        return {i: 1 for i in idx}
    cut = dmin * nn_tolerance

    # 第二遍：建近邻图
    adj: List[List[int]] = [[] for _ in range(nnode)]
    for s in range(0, nnode, chunk):
        e = min(nnode, s + chunk)
        d = _chunk_dist(s, e)
        for a, b in np.argwhere(d <= cut):
            u = s + int(a)
            v = int(b)
            if u != v:
                adj[u].append(v)

    # BFS 二染色
    color = np.full(nnode, 0, dtype=int)
    ok = True
    for start in range(nnode):
        if color[start] != 0:
            continue
        color[start] = 1
        dq = deque([start])
        while dq:
            u = dq.popleft()
            for v in adj[u]:
                if color[v] == 0:
                    color[v] = -color[u]
                    dq.append(v)
                elif color[v] == color[u]:
                    ok = False

    if ok:
        return {idx[k]: int(color[home * len(idx) + k]) for k in range(len(idx))}

    # fcc 等非二部图：退化为按磁层投影交替
    return layered_signs(pos, idx)


# ============================================================================
# 7. 综合分析（给 LLM 的一站式结构报告）
# ============================================================================
def analyze_structure(pos: Poscar, max_sites: int = 400) -> Dict:
    neigh = coordination_analysis(pos)
    ctx = site_context(pos, neigh)
    motif = detect_motif(pos, neigh, ctx["system_type"])
    comp = pos.composition()
    kb = KNOWLEDGE_BASE.get(canonical_formula(pos.reduced_composition()))

    layers = ctx.get("layers")
    depths = ctx.get("layer_depth")
    sites = []
    for i, el in enumerate(pos.symbols):
        nlist = neigh[i]
        n_el = Counter(pos.symbols[j] for j, _ in nlist)
        dists = [d for _, d in nlist]
        cn = len(nlist)
        sites.append(
            {
                "index": i,
                "element": el,
                "frac": [round(float(x), 5) for x in pos.frac[i]],
                "cn": cn,
                "geometry": geometry_label(cn),
                "neighbors": dict(n_el),
                "mean_bond": round(float(np.mean(dists)), 4) if dists else None,
                "min_bond": round(float(min(dists)), 4) if dists else None,
                "role": ctx["roles"][i],
                "layer": (int(layers[i]) if layers is not None else None),
                "depth": (round(float(depths[i]), 4) if depths is not None else None),
                "cn_deficit": ctx["cn_deficit"][i],
                "undercoordinated": ctx["undercoordinated"][i],
                "magnetic_candidate": el not in NONMAGNETIC,
                "default_element_moment": default_element_moment(el),
            }
        )

    species_blocks = [
        {"element": el, "count": b - a, "index_range": [a, b]}
        for el, a, b in pos.blocks()
    ]

    report = {
        "source": pos.source,
        "comment": pos.comment,
        "formula": pos.formula(),
        "reduced_formula": pos.reduced_formula(),
        "n_sites": pos.n_atoms,
        "lattice": [[round(float(x), 5) for x in row] for row in pos.lattice],
        "volume": round(pos.volume, 4),
        "density": round(pos.density(), 4),
        "coord_mode": pos.coord_mode,
        "system_type": ctx["system_type"],
        "vacuum_gaps": ctx["vacuum_gaps"],
        "slab": (
            {
                "vacuum_axis": ctx.get("vacuum_axis"),
                "n_layers": ctx.get("n_layers"),
                "layer_positions": ctx.get("layer_positions"),
                "slab_thickness": ctx.get("slab_thickness"),
            }
            if ctx["system_type"] == "slab"
            else None
        ),
        "species_blocks": species_blocks,
        "motif": motif,
        "knowledge_base_match": (
            {
                "name": kb["name"],
                "magnetic_order": kb["order"],
                "suggested_assignments": kb["assignments"],
                "afm_groups": kb.get("afm", []),
                "note": kb.get("note", ""),
            }
            if kb
            else None
        ),
        "site_groups": ctx["site_groups"],
        "sites": sites[:max_sites],
        "sites_truncated": pos.n_atoms > max_sites,
    }
    return report


# ============================================================================
# 8. 磁矩方案应用 / 校验 / 输出
# ============================================================================
def report_to_llm_view(report: Dict, max_example_sites: int = 12) -> Dict:
    """把 analyze_structure 的完整报告压缩成适合喂给 LLM 的紧凑视图。

    按 (元素, 配位数, 位点角色) 分组，避免把几百个原子逐个塞进上下文；
    同时保留体系类型 (bulk/slab/mole)、真空层、层数等关键信息。
    """
    view = {
        "source": report["source"],
        "formula": report["formula"],
        "reduced_formula": report["reduced_formula"],
        "n_sites": report["n_sites"],
        "volume": report["volume"],
        "density": report["density"],
        "coord_mode": report["coord_mode"],
        "system_type": report.get("system_type", "bulk"),
        "vacuum_gaps": report.get("vacuum_gaps"),
        "slab": report.get("slab"),
        "species_blocks": report["species_blocks"],
        "motif": report["motif"],
        "knowledge_base_match": report["knowledge_base_match"],
        "site_groups": report.get("site_groups", []),
        "example_sites": report["sites"][:max_example_sites],
    }
    return view


def _role_matches(site_role: Optional[str], actual: str) -> bool:
    if not site_role or str(site_role).lower() in ("any", "all", "*"):
        return True
    role = str(site_role).lower()
    if role == "surface":
        return actual.startswith("surface")
    if role in ("top_surface", "surface_top"):
        return actual in ("surface_top", "surface")
    if role in ("bottom_surface", "surface_bottom"):
        return actual in ("surface_bottom", "surface")
    if role in ("interior", "bulk"):
        return actual in ("interior", "bulk")
    if role == "subsurface":
        return actual == "subsurface"
    if role == "cluster":
        return actual == "cluster"
    return actual == role


def build_moments_from_assignments(
    pos: Poscar,
    report: Dict,
    assignments: Sequence[Dict],
    overrides: Optional[Sequence[Dict]] = None,
    afm: Optional[Sequence] = None,
    afm_elementwise: bool = False,
) -> List[float]:
    """根据 LLM 给出的规则计算逐原子磁矩。

    每条 assignment 支持：
      element       元素符号（必需）
      coordination  配位数（可选，用于区分四面体/八面体/表面不饱和位点）
      site_role     位点角色（可选：any/surface/top_surface/bottom_surface/
                    subsurface/interior/bulk/cluster）
      moment        磁矩（可带正负号）
      afm_group     反铁磁组名（可选，供 afm 参数引用）
    匹配按列表顺序，先到先得。
    """
    neigh = coordination_analysis(pos)
    cn_list = [len(x) for x in neigh]
    ctx = site_context(pos, neigh)
    roles = ctx["roles"]

    base = [0.0] * pos.n_atoms
    site_group: List[Optional[str]] = [None] * pos.n_atoms
    for i, el in enumerate(pos.symbols):
        chosen = None
        fallback = None  # (排序键, assignment)
        for oi, a in enumerate(assignments):
            ael = a.get("element")
            if ael and ael != el:
                continue
            if "site_role" in a and not _role_matches(a.get("site_role"), roles[i]):
                continue
            coord = a.get("coordination")
            if coord is None or int(coord) == cn_list[i]:
                chosen = a
                break
            # 配位数不完全匹配时（常见于 slab 表面不饱和位），
            # 退回该元素最接近的指定配位数（优先高配位，即体相“母位点”）
            cand = int(coord)
            above = cand > cn_list[i]
            dist = (cand - cn_list[i]) if above else (10_000 + (cn_list[i] - cand))
            key = (dist, oi)
            if fallback is None or key < fallback[0]:
                fallback = (key, a)
        if chosen is None and fallback is not None:
            chosen = fallback[1]
        if chosen is None:
            continue
        base[i] = float(chosen.get("moment", 0.0))
        g = chosen.get("afm_group")
        site_group[i] = str(g) if g else None

    # ---- 反铁磁/亚铁磁符号：每个位点只应用一次（组名优先于元素名）----
    afm_set = {str(t) for t in (afm or [])}
    for a in assignments:
        if a.get("afm_group"):
            afm_set.add(str(a["afm_group"]))

    token_of: List[Optional[str]] = []
    if afm_elementwise:
        # magnetic_order="afm"：直接对元素做全局正负交替，忽略分组
        for el in pos.symbols:
            token_of.append(el if el in afm_set else None)
    else:
        for i, el in enumerate(pos.symbols):
            if site_group[i]:
                token_of.append(site_group[i])
            elif el in afm_set:
                token_of.append(el)
            else:
                token_of.append(None)

    token_members: Dict[str, List[int]] = {}
    for i, tok in enumerate(token_of):
        if tok:
            token_members.setdefault(tok, []).append(i)

    signs = [1] * pos.n_atoms
    for members in token_members.values():
        s = bipartite_signs(pos, indices=members)
        for atom_i, sign in s.items():
            signs[atom_i] = 1 if sign > 0 else -1

    moments = [
        (abs(base[i]) * signs[i]) if token_of[i] is not None else base[i]
        for i in range(pos.n_atoms)
    ]

    if overrides:
        for ov in overrides:
            i = int(ov["index"])
            if 0 <= i < pos.n_atoms:
                moments[i] = float(ov["moment"])

    return moments


def _sub_poscar(pos: Poscar, indices: Sequence[int]) -> Poscar:
    """抽取部分原子构造子 POSCAR，用于独立做反铁磁染色。"""
    idx = list(indices)
    species: List[str] = []
    counts: List[int] = []
    for i in idx:
        el = pos.symbols[i]
        if species and species[-1] == el:
            counts[-1] += 1
        else:
            species.append(el)
            counts.append(1)
    sub = Poscar(
        comment="subset",
        lattice=pos.lattice.copy(),
        species=species,
        counts=counts,
        symbols=[pos.symbols[i] for i in idx],
        frac=pos.frac[idx].copy(),
        coord_mode=pos.coord_mode,
    )
    return sub


def validate_moments(moments: Sequence[float], n_atoms: int) -> List[str]:
    warnings = []
    if len(moments) != n_atoms:
        warnings.append(f"MAGMOM 长度 {len(moments)} != 原子数 {n_atoms}")
    for i, m in enumerate(moments):
        if not math.isfinite(float(m)):
            warnings.append(f"第 {i} 个磁矩非有限值: {m}")
        elif abs(float(m)) > 20:
            warnings.append(f"第 {i} 个磁矩偏大 ({m})，请确认")
    return warnings


def _fmt(x: float) -> str:
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.3f}".rstrip("0").rstrip(".")


def format_magmom(
    pos: Poscar, moments: Sequence[float], ispin: int = 2, compact: bool = True
) -> str:
    """生成与 add-spin.py 风格一致的 ISPIN / MAGMOM 文本块。"""
    if len(moments) != pos.n_atoms:
        raise ValueError("磁矩数量与原子数不一致")

    if ispin == 1:
        return (
            "Mag parameter\n"
            "   ISPIN = 1\n"
            "   # 非自旋极化：体系判定为非磁，无需 MAGMOM\n"
        )

    # 含负值时（反铁磁/亚铁磁）VASP 的 n*value 缩写不易读，直接逐原子展开
    use_compact = compact and all(float(m) >= 0 for m in moments)
    if use_compact:
        parts = []
        run_val = None
        run_len = 0
        for m in moments:
            if run_val is not None and abs(float(m) - run_val) < 1e-9:
                run_len += 1
            else:
                if run_val is not None:
                    parts.append(f"{run_len}*{_fmt(run_val)}" if run_len > 1 else _fmt(run_val))
                run_val = float(m)
                run_len = 1
        if run_val is not None:
            parts.append(f"{run_len}*{_fmt(run_val)}" if run_len > 1 else _fmt(run_val))
        magmom_line = "   ".join(parts)
    else:
        magmom_line = "   ".join(_fmt(m) for m in moments)

    # 元素注释：优先按元素块压缩
    elem_parts, num_parts = [], []
    uniform = True
    for el, a, b in pos.blocks():
        vals = moments[a:b]
        if vals and all(abs(float(v) - float(vals[0])) < 1e-9 for v in vals):
            elem_parts.append(f"{el}({len(vals)})={_fmt(float(vals[0]))}")
        else:
            uniform = False
            elem_parts.append(f"{el}({len(vals)})")

    header = "# " + "  ".join(elem_parts)
    if not uniform:
        header += "   （同一元素内部存在不同磁矩，MAGMOM 按 POSCAR 原子顺序逐一对应）"

    lines = [
        "Mag parameter",
        f"   ISPIN = {ispin}",
        f"   {header}",
        f"   MAGMOM =  {magmom_line}",
    ]
    return "\n".join(lines) + "\n"


def append_to_incar(block: str, incar: str = "INCAR") -> None:
    with open(incar, "a", encoding="utf-8") as f:
        f.write("\n" + block)


# ============================================================================
# 9. 无 LLM 时的启发式回退方案
# ============================================================================
def heuristic_plan(pos: Poscar, report: Optional[Dict] = None) -> Tuple[List[float], str]:
    """在无法调用 LLM 时，用知识库 + 元素常识给出可用初猜。"""
    if report is None:
        report = analyze_structure(pos)
    n = pos.n_atoms
    comp = dict(pos.composition())
    kb = KNOWLEDGE_BASE.get(canonical_formula(pos.reduced_composition()))

    if kb:
        moments = build_moments_from_assignments(
            pos, report, kb["assignments"], afm=kb.get("afm", [])
        )
        return moments, f"[回退] 命中知识库：{kb['name']}（{kb['order']}）。{kb.get('note','')}"

    assignments = []
    afm = []
    # 单元素金属
    if len(comp) == 1:
        el = next(iter(comp))
        m = METAL_MOMENT.get(el, 0.0)
        assignments.append({"element": el, "moment": m})
        if el in ("Cr", "Mn"):
            afm.append(el)
        reason = f"[回退] 纯 {el} 金属，取 {m} μB。"

    # 含阴离子的化合物
    elif set(comp) & ANIONS:
        for el in comp:
            if el in ANIONS or el in NONMAGNETIC:
                assignments.append({"element": el, "moment": 0.0})
            else:
                m = float(OXIDE_HS.get(el, default_element_moment(el)))
                assignments.append({"element": el, "moment": m})
                # 未知氧化物默认按反铁磁给交替符号（更接近多数氧化物基态）
                if el in ("Cr", "Mn", "Fe", "Co", "Ni", "V"):
                    afm.append(el)
        reason = "[回退] 含阴离子化合物，按常见高自旋磁矩 + 反铁磁交替给初猜。"

    # 合金/金属间化合物
    else:
        for el in comp:
            assignments.append({"element": el, "moment": default_element_moment(el)})
        reason = "[回退] 金属间化合物，按元素常识给初猜。"

    moments = build_moments_from_assignments(pos, report, assignments, afm=afm)
    return moments, reason


def plan_from_llm_args(
    pos: Poscar, report: Dict, args: Dict
) -> Tuple[List[float], List[str]]:
    """把 LLM 调用 submit_magmom 的参数转成逐原子磁矩。

    支持 magnetic_order：
      auto / ferrimagnetic / ferri  按 assignments 的正负号（默认，适合亚铁磁）
      fm / ferromagnetic            全部取正（铁磁）
      afm / antiferromagnetic       对每个磁性元素做正负交替（反铁磁）
      nonmagnetic / nm              全部 0，ISPIN=1
    """
    assignments = args.get("assignments") or []
    overrides = args.get("site_overrides") or []
    afm = list(args.get("afm_elements") or [])
    order = str(args.get("magnetic_order") or "auto").lower()

    if order in ("nonmagnetic", "non-magnetic", "nm", "none"):
        return [0.0] * pos.n_atoms, []

    if order in ("afm", "antiferromagnetic", "antiferro"):
        # 强制对含非零初猜的元素做正负交替
        magnetic_els = sorted({
            a.get("element") for a in assignments
            if a.get("element") and abs(float(a.get("moment", 0.0))) > 1e-9
        })
        for el in magnetic_els:
            if el not in afm:
                afm.append(el)

    moments = build_moments_from_assignments(
        pos, report, assignments, overrides, afm,
        afm_elementwise=(order in ("afm", "antiferromagnetic", "antiferro")),
    )

    if order in ("fm", "ferromagnetic", "ferro"):
        moments = [abs(float(m)) for m in moments]

    warnings = validate_moments(moments, pos.n_atoms)
    return moments, warnings


# ============================================================================
# 10. LLM 工具定义（OpenAI-compatible function calling schema）
# ============================================================================
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "analyze_structure",
            "description": (
                "解析并分析当前 POSCAR：化学式、晶格、体积、体系类型（bulk/slab/mole）、"
                "真空层厚度与 slab 分层、每个位点的配位数/配位几何/近邻元素/位点角色"
                "（bulk/interior/surface/subsurface）、结构基元（fcc/bcc/岩盐/闪锌矿/纤锌矿/"
                "钙钛矿/双钙钛矿/尖晶石/反尖晶石/LDH/刚玉/金红石…）以及内置知识库匹配结果。"
                "这是理解结构的第一步必须调用的工具。"
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_element_info",
            "description": (
                "查询某个元素的磁性参考信息：常见氧化态、不同价态/自旋态下的未成对电子数、"
                "共价半径、以及默认初猜磁矩。用于判断某个过渡金属该给多少磁矩。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "元素符号，如 Fe"}
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_known_material",
            "description": (
                "在内置磁性材料知识库中按化学式或中文/英文名称查询。"
                "例如 'Fe3O4'、'磁铁矿'、'NiO'、'CoFe2O4'。"
                "返回推荐的分位点磁矩、磁序类型与说明。强烈建议对常见磁性材料调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "formula_or_name": {"type": "string", "description": "化学式或名称"}
                },
                "required": ["formula_or_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "preview_magmom",
            "description": (
                "试算磁矩方案但不提交：给定按 (元素, 配位数) 匹配的赋值规则，"
                "返回每种 (元素, 配位数) 得到的具体磁矩以及未覆盖位点/告警。"
                "可在正式提交前用它检查尖晶石 8a/16d、岩盐反铁磁等是否正确。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "assignments": {
                        "type": "array",
                        "description": (
                            "赋值规则列表，按顺序匹配；每条规则可含 element、coordination(可选，"
                            "用于区分同一元素的不同晶体学位点)、site_role(可选：any/surface/"
                            "top_surface/bottom_surface/subsurface/interior/bulk/cluster)、"
                            "moment(可带正负号)、afm_group(可选)。"
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "element": {"type": "string"},
                                "coordination": {"type": "integer"},
                                "site_role": {"type": "string"},
                                "moment": {"type": "number"},
                                "afm_group": {"type": "string"},
                            },
                            "required": ["element", "moment"],
                        },
                    },
                    "afm_elements": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "需要正负号交替的组名或元素符号。组名对应 assignments 中的 afm_group；"
                            "也可直接写元素符号。自动使用磁层投影/二部图染色。"
                        ),
                    },
                    "magnetic_order": {
                        "type": "string",
                        "enum": ["auto", "fm", "afm", "ferrimagnetic", "nonmagnetic"],
                        "description": (
                            "整体磁序：fm=铁磁(全部取正)，afm=反铁磁(对磁性元素正负交替)，"
                            "ferrimagnetic=亚铁磁(按 assignments 正负号)，nonmagnetic=ISPIN=1。"
                            "默认 auto 等同于 ferrimagnetic。"
                        ),
                    },
                    "site_overrides": {
                        "type": "array",
                        "description": "对个别位点强制赋值：[{\"index\": i, \"moment\": m}]",
                        "items": {
                            "type": "object",
                            "properties": {
                                "index": {"type": "integer"},
                                "moment": {"type": "number"},
                            },
                            "required": ["index", "moment"],
                        },
                    },
                },
                "required": ["assignments"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_magmom",
            "description": (
                "提交最终磁矩方案。参数与 preview_magmom 相同，另加 rationale。"
                "调用此工具即表示任务完成，脚本会据此写出 INCAR。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "assignments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "element": {"type": "string"},
                                "coordination": {"type": "integer"},
                                "site_role": {"type": "string"},
                                "moment": {"type": "number"},
                                "afm_group": {"type": "string"},
                            },
                            "required": ["element", "moment"],
                        },
                    },
                    "afm_elements": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "magnetic_order": {
                        "type": "string",
                        "enum": ["auto", "fm", "afm", "ferrimagnetic", "nonmagnetic"],
                    },
                    "site_overrides": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "index": {"type": "integer"},
                                "moment": {"type": "number"},
                            },
                        },
                    },
                    "ispin": {"type": "integer", "description": "默认 2"},
                    "rationale": {
                        "type": "string",
                        "description": "简要说明为什么这样设置（中文即可）",
                    },
                },
                "required": ["assignments"],
            },
        },
    },
]


# ============================================================================
# 2. 工具运行时
# ============================================================================
class ToolRuntime:
    def __init__(self, pos: Poscar):
        self.pos = pos
        self.report = analyze_structure(pos)
        self.view = report_to_llm_view(self.report)
        self.submitted: Optional[Dict] = None

    def call(self, name: str, args: Optional[Dict] = None) -> Dict:
        args = args or {}
        if name == "analyze_structure":
            return {"ok": True, "structure": self.view}
        if name == "get_element_info":
            return {"ok": True, **element_info(args.get("symbol", ""))}
        if name == "lookup_known_material":
            entry = lookup_known_material(args.get("formula_or_name", ""))
            if entry is None:
                return {"ok": True, "found": False,
                        "hint": "知识库无此材料，请依据配位环境与元素价态自行推断。"}
            return {"ok": True, "found": True, **entry}
        if name in ("preview_magmom", "submit_magmom"):
            result = self._apply(args)
            if name == "submit_magmom":
                self.submitted = args
            return result
        return {"ok": False, "error": f"未知工具: {name}"}

    def _apply(self, args: Dict) -> Dict:
        assignments = args.get("assignments") or []
        overrides = args.get("site_overrides") or []
        afm = args.get("afm_elements") or []
        moments, warnings = plan_from_llm_args(self.pos, self.report, args)

        # 按 (元素, 配位数, 位点角色) 汇总，便于 LLM/用户核对
        neigh = coordination_analysis(self.pos)
        ctx = site_context(self.pos, neigh)
        summary: Dict[str, Dict] = {}
        for i, (sym, m) in enumerate(zip(self.pos.symbols, moments)):
            key = f"{sym}(CN={len(neigh[i])},{ctx['roles'][i]})"
            s = summary.setdefault(key, {"count": 0, "moments": set()})
            s["count"] += 1
            s["moments"].add(round(float(m), 4))
        pretty = {
            k: {"count": v["count"], "moments": sorted(v["moments"])}
            for k, v in summary.items()
        }

        covered = {a.get("element") for a in assignments if a.get("element")}
        uncovered = [
            i for i, s in enumerate(self.pos.symbols)
            if s not in covered and s not in NONMAGNETIC
        ]
        return {
            "ok": True,
            "system_type": self.report.get("system_type", "bulk"),
            "per_site_group": pretty,
            "n_atoms": self.pos.n_atoms,
            "warnings": warnings,
            "uncovered_magnetic_sites": uncovered[:20],
            "note": "warnings 非空时请修正方案后重新调用。",
        }


def element_info(symbol: str) -> Dict:
    symbol = (symbol or "").strip().capitalize()
    ions = MAGNETIC_IONS.get(symbol, {})
    ion_table = {
        f"{symbol}{ox}+": {
            "unpaired_electrons_high_spin": hs,
            "unpaired_electrons_low_spin": ls,
            "d_electrons": d,
        }
        for ox, (d, hs, ls) in ions.items()
    }
    return {
        "symbol": symbol,
        "atomic_mass": ATOMIC_MASS.get(symbol),
        "covalent_radius": RCOV.get(symbol),
        "common_oxidation_states": COMMON_OXIDATION.get(symbol, []),
        "magnetic_ions": ion_table,
        "default_guess": default_element_moment(symbol),
        "nonmagnetic": symbol in NONMAGNETIC,
        "metal_moment": METAL_MOMENT.get(symbol),
    }


# ============================================================================
# 3. 系统提示词
# ============================================================================
SYSTEM_PROMPT = """\
你是一位资深的第一性原理计算（VASP）专家，专门为 INCAR 设置自旋极化计算（ISPIN=2）的
MAGMOM 初猜。你的目标不是给出最终收敛值，而是给出**物理上合理、能引导 VASP 收敛到正确
磁基态**的逐原子初猜磁矩。

## 关键背景
- MAGMOM 的顺序必须与 POSCAR 中原子出现的顺序**逐原子一一对应**（先按元素分块，
  每个元素块内部再按坐标行顺序）。
- 你不需要自己数原子：最终你会把“按 (元素, 配位数, 位点角色) 的赋值规则”交给工具，
  由脚本展开到每个原子。
- add-spin.py 里那种“一个元素一个固定值”的粗糙做法在复杂体系会出错。例如：
  * 体相 fcc Pt 是非磁的，MAGMOM 应该是 0（旧表给的 3 是错的）；
  * 尖晶石/反尖晶石中同一元素可能占四面体(8a)和八面体(16d)两种位点，磁矩不同；
  * 岩盐 MnO/FeO/CoO/NiO、刚玉 Cr2O3/Fe2O3 等是反铁磁，需要相邻磁矩正负交替；
  * LDH/氢氧化物只有层板八面体金属有磁矩，层间阴离子与水一律 0。

## 磁矩大小经验
- 3d 过渡金属在氧化物/卤化物中通常高自旋，初猜取“未成对电子数”：
  Ti3+ 1, V3+ 2, V4+ 1, Cr3+ 3, Mn2+ 5, Mn3+ 4, Mn4+ 3, Fe2+ 4, Fe3+ 5,
  Co2+ 3, Ni2+ 2, Cu2+ 1。
- 4d/5d 及强场（八面体 + 强配体）常低自旋：Co3+ 八面体低自旋 0，Ru3+ 1~3，Rh3+ 0；
  但 Mo5+(4d1)=1、Re5+(4d2)=2 等需按 d 电子数给。
- 4f 稀土：Gd3+ 7, Eu2+ 7, Eu3+ 6, Tb3+ 6, Dy3+ 5, Ho3+ 4, Er3+ 3, Tm3+ 2, Yb3+ 1；
  La3+/Y3+/Lu3+ 通常非磁。
- 主族闭壳层（O2-, F-, Al3+, Mg2+, Zn2+, Li+, Ti4+, V5+ 等 d0/d10）一律 0。
- 纯金属：Fe 2.2, Co 1.7, Ni 0.6；Cr、Mn 是反铁磁（正负交替）；Pt/Pd/Cu/Ag/Au/Al 体相非磁=0。
- 初猜值可以比实验值略大一些以利于收敛，但不要离谱（一般 |m| ≤ 7）。

## 常见结构类型的磁性处理
- **尖晶石 AB2O4**：8a 四面体位与 16d 八面体位分别赋值。
  * 正尖晶石：A 在 8a，B 在 16d（MgAl2O4、ZnFe2O4）。
  * 反尖晶石：B 在 8a，A、B 共同占 16d（Fe3O4、NiFe2O4、CoFe2O4）。
  * 亚铁磁（Fe3O4、NiFe2O4）：8a 与整个 16d 亚晶格整体反平行 → 16d 给**负号**，
    16d 内部同向，**不要**在 16d 内部再正负交替。
  * Co3O4：8a Co2+ 反铁磁（正负交替），16d Co3+ 低自旋=0。
- **岩盐 MO**（NiO/CoO/MnO/FeO）：II 型反铁磁，磁性亚晶格 (111) 面内铁磁、面间反铁磁，
  用 magnetic_order="afm" 或 afm_elements 指定。
- **闪锌矿 / 纤锌矿 AB**：阳离子四配位。ZnS/ZnO/GaAs/CdTe 等 d10 体系非磁；
  磁性代表 MnS/MnSe/MnTe（Mn2+ 5，反铁磁）。
- **钙钛矿 ABO3**：B 位过渡金属(CN=6)承担磁矩；A 位(La/Sr/Ba/Ca/稀土, CN≥8)通常 0。
  LaMnO3 (A 型反铁磁)、LaFeO3/BiFeO3/LaCrO3 (G 型反铁磁)。
- **双钙钛矿 A2BB'O6**：B/B' 有序时通常反平行（亚铁磁），如 Sr2FeMoO6：Fe +5、Mo -1；
  La2NiMnO6：Ni2+ +2 与 Mn4+ +3 铁磁（同号）。
- **LDH / 氢氧化物 M(OH)2**：层板为共边八面体 M(OH)6，磁矩只来自层板金属 M2+/M3+，
  层间阴离子（CO3^2-/NO3^-/Cl^-）与水一律 0；LDH 层板内常为铁磁或自旋玻璃，
  可用正磁矩，也可用 magnetic_order 指定反铁磁。
- **刚玉 A2O3**：Cr2O3/Fe2O3 反铁磁。**金红石 MO2**：CrO2 铁磁，MnO2 反铁磁。
- 若知识库 lookup_known_material 命中，优先采用其建议；如与结构信息冲突，以结构信息为准。

## slab / 表面（重要）
- analyze_structure 会给出 system_type（bulk/slab/mole）、真空层方向、分层数与每个位点的
  role：bulk / interior / surface_top / surface_bottom / subsurface / cluster。
- slab 的上下表面必然配位不饱和（如八面体 6 配位降到 5、四面体 4 降到 3）。
  工具在匹配 coordination 时会自动把表面低配位位点回退到体相“母位点”的规则，因此你
  可以只按体相配位数（如 4/6）写规则，表面自动继承。
- 若想让表面与内部不同，可用 site_role 单独写规则，例如：
  {"element":"Fe","site_role":"surface","moment":5.4} 放在体相规则之前。
- 金属 slab 表面磁矩常因配位降低而增强（约 +10%~30%），可在表面规则里适当放大；
  氧化物/离子晶体通常保持价态不变，表面与内部给相同磁矩即可。
- LDH 单层、二维材料都按 slab 处理；分子/团簇按 cluster 处理。

## 磁序设置
- 调用 submit_magmom 时可传 magnetic_order：
  * "ferromagnetic"：所有非零磁矩取正；
  * "antiferromagnetic"：对磁性元素自动正负交替（fcc 用磁层投影，bcc 用二部图）；
  * "ferrimagnetic"：按你写的 assignments 正负号（尖晶石/双钙钛矿常用）；
  * "nonmagnetic"：全部 0，ISPIN=1。
- 也可用 afm_elements 只对特定亚晶格交替，或在 assignments 里直接写带负号的 moment。

## 工作流程（必须遵守）
1. 先调用 analyze_structure 查看结构（体系类型、位点 role、配位数、结构基元、知识库匹配）。
2. 对磁性过渡金属/稀土可调用 get_element_info；对常见材料调用 lookup_known_material。
3. 组装 assignments（元素 + 可选 coordination + 可选 site_role），必要时设定 magnetic_order。
4. 可先调用 preview_magmom 检查（尤其确认尖晶石 8a/16d、slab 表面位点是否给对）。
5. 确认无误后调用 submit_magmom 提交，并写一句中文 rationale。

只输出工具调用，不要输出额外长篇解释。所有数值用 μB。"""


def build_user_prompt(pos: Poscar, poscar_text: str, hint: str = "") -> str:
    text = poscar_text
    if len(text) > 8000:
        text = text[:8000] + "\n... (POSCAR 已截断，完整信息以 analyze_structure 工具结果为准)"
    prompt = (
        "请为下面这个 POSCAR 设置 ISPIN=2 的 MAGMOM 初猜，"
        "重点处理过渡金属的价态与晶体学位点、以及可能的反铁磁/亚铁磁序。\n\n"
        f"POSCAR 内容：\n```\n{text}\n```\n"
    )
    if hint:
        prompt += f"\n用户补充说明：{hint}\n"
    prompt += "\n请先调用 analyze_structure。"
    return prompt


# ============================================================================
# 4. OpenAI 兼容客户端（仅标准库 urllib）
# ============================================================================
class LLMClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com/v1",
        model: str = "deepseek-chat",
        timeout: int = 180,
        temperature: float = 0.2,
        max_retries: int = 3,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.max_retries = max_retries

    def chat(self, messages: List[Dict], tools: Optional[List[Dict]] = None) -> Dict:
        url = f"{self.base_url}/chat/completions"
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                return body["choices"][0]["message"]
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="ignore")
                last_err = RuntimeError(f"HTTP {exc.code}: {detail[:500]}")
                if exc.code in (429, 500, 502, 503, 504):
                    time.sleep(2 ** attempt)
                    continue
                raise last_err from exc
            except (urllib.error.URLError, TimeoutError, KeyError) as exc:
                last_err = exc
                time.sleep(2 ** attempt)
        raise RuntimeError(f"调用 LLM 失败: {last_err}")


def build_client_from_args(args) -> LLMClient:
    api_key = (
        args.api_key
        or os.environ.get("LLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("DEEPSEEK_API_KEY")
        or ""
    )
    base_url = (
        args.base_url
        or os.environ.get("LLM_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or "https://api.deepseek.com/v1"
    )
    model = (
        args.model
        or os.environ.get("LLM_MODEL")
        or "deepseek-chat"
    )
    if not api_key:
        raise RuntimeError(
            "未找到 API Key。请设置 LLM_API_KEY / OPENAI_API_KEY / DEEPSEEK_API_KEY，"
            "或用 --api-key 指定；也可以用 --no-llm 走内置启发式。"
        )
    return LLMClient(api_key, base_url=base_url, model=model, temperature=args.temperature)


# ============================================================================
# 5. Agent 主循环
# ============================================================================
def run_agent(
    client: LLMClient,
    runtime: ToolRuntime,
    user_prompt: str,
    max_steps: int = 8,
    verbose: bool = True,
) -> Optional[Dict]:
    messages: List[Dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    for step in range(max_steps):
        if verbose:
            print(f"[LLM] 第 {step + 1} 轮请求 …", file=sys.stderr)
        msg = client.chat(messages, tools=TOOL_SCHEMAS)
        tool_calls = msg.get("tool_calls") or []
        assistant_msg: Dict[str, Any] = {"role": "assistant", "content": msg.get("content") or ""}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        if not tool_calls:
            content = (msg.get("content") or "").strip()
            if verbose and content:
                print(f"[LLM] 未再调用工具，返回文本：{content[:300]}", file=sys.stderr)
            parsed = _try_parse_content_plan(content)
            if parsed:
                runtime.submitted = parsed
                return parsed
            break

        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            if verbose:
                print(f"[tool] {name}({_short_args(args)})", file=sys.stderr)
            result = runtime.call(name, args)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.get("id", name),
                    "content": json.dumps(result, ensure_ascii=False, default=str)[:12000],
                }
            )
            if name == "submit_magmom" and runtime.submitted is not None:
                return runtime.submitted
    return runtime.submitted


def _short_args(args: Dict) -> str:
    text = json.dumps(args, ensure_ascii=False, default=str)
    return text if len(text) <= 200 else text[:200] + "…"


def _try_parse_content_plan(content: str) -> Optional[Dict]:
    """有些模型会把 submit 参数直接写在正文 JSON 里，这里兜底解析。"""
    if not content:
        return None
    start = content.find("{")
    end = content.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(content[start:end + 1])
    except json.JSONDecodeError:
        return None
    if isinstance(obj, dict) and "assignments" in obj:
        return obj
    return None


# ============================================================================
# 6. 输出 / CLI
# ============================================================================
def finalize(
    pos: Poscar,
    runtime: ToolRuntime,
    plan: Optional[Dict],
    incar: str,
    do_print: bool,
    do_write: bool,
    verbose: bool = True,
) -> str:
    if plan:
        moments, warnings = plan_from_llm_args(pos, runtime.report, plan)
        ispin = int(plan.get("ispin", 2) or 2)
        if str(plan.get("magnetic_order", "")).lower() in (
            "nonmagnetic", "non-magnetic", "nm"
        ):
            ispin = 1
        rationale = plan.get("rationale", "")
        source = "LLM 工具调用"
    else:
        moments, rationale = heuristic_plan(pos, runtime.report)
        warnings = validate_moments(moments, pos.n_atoms)
        ispin = 2
        source = "内置启发式回退"

    block = format_magmom(pos, moments, ispin=ispin)

    print("=" * 72)
    print(f"结构: {runtime.view['formula']}  ({pos.n_atoms} atoms)  体系: {runtime.view['system_type']}")
    print(f"基元: {runtime.view['motif']['name']}")
    slab = runtime.view.get("slab")
    if slab:
        print(
            f"slab: 真空沿 {slab['vacuum_axis']} 轴, {slab['n_layers']} 个原子层, "
            f"厚度 {slab['slab_thickness']} Å；真空层 {runtime.view['vacuum_gaps']}"
        )
    kb = runtime.view.get("knowledge_base_match")
    if kb:
        print(f"知识库: {kb['name']}  [{kb['magnetic_order']}]")
    print(f"来源: {source}")
    if rationale:
        print(f"依据: {rationale}")
    if warnings:
        print("告警: " + "; ".join(warnings))
    print("-" * 72)
    print(block, end="")
    print("=" * 72)

    if do_write:
        if os.path.exists(incar):
            try:
                old = open(incar, "r", encoding="utf-8", errors="ignore").read()
            except OSError:
                old = ""
            if "MAGMOM" in old.upper() or "ISPIN" in old.upper():
                print(
                    f"[提示] {incar} 中已存在 ISPIN/MAGMOM，本次为追加写入；"
                    "如需替换请先手动清理旧参数。"
                )
        append_to_incar(block, incar)
        print(f"已追加写入: {incar}")

    return block


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="用 LLM 工具调用为 POSCAR 生成 VASP MAGMOM 初猜"
    )
    ap.add_argument("poscar", nargs="?", default="POSCAR", help="POSCAR 文件路径")
    ap.add_argument("--incar", default="INCAR", help="要追加写入的 INCAR（默认 INCAR）")
    ap.add_argument("--print", dest="do_print", action="store_true", help="只打印不写文件")
    ap.add_argument("--no-llm", action="store_true", help="不调用 LLM，直接用内置启发式")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--hint", default="", help="给 LLM 的补充说明（如已知价态/磁性）")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--vacuum-threshold", type=float, default=5.0,
                    help="bulk/slab 判别的真空层阈值 (Å)，默认 5.0")
    ap.add_argument("--self-test", action="store_true",
                    help="运行内置物理自检后退出")
    ap.add_argument("--make-examples", metavar="DIR", default=None,
                    help="把内置示例 POSCAR 导出到目录后退出")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    global _ACTIVE_VACUUM_THRESHOLD
    _ACTIVE_VACUUM_THRESHOLD = args.vacuum_threshold
    if args.self_test:
        return run_self_test()
    if args.make_examples:
        return export_examples(args.make_examples)
    if not os.path.exists(args.poscar):
        print(f"找不到 POSCAR: {args.poscar}", file=sys.stderr)
        return 2

    pos = parse_poscar_full(args.poscar)
    runtime = ToolRuntime(pos)
    poscar_text = open(args.poscar, "r", encoding="utf-8", errors="ignore").read()
    plan: Optional[Dict] = None

    if not args.no_llm:
        try:
            client = build_client_from_args(args)
            user_prompt = build_user_prompt(pos, poscar_text, args.hint)
            plan = run_agent(client, runtime, user_prompt, args.max_steps, verbose=not args.quiet)
            if plan is None:
                print("[警告] LLM 未提交方案，改用内置启发式回退。", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            print(f"[警告] LLM 调用失败（{exc}），改用内置启发式回退。", file=sys.stderr)
            plan = None

    finalize(
        pos, runtime, plan, args.incar,
        do_print=args.do_print,
        do_write=not args.do_print,
        verbose=not args.quiet,
    )
    return 0



# ============================================================================
# 12. 内置自检 与 示例导出
# ============================================================================
def run_self_test(verbose: bool = True) -> int:
    """不依赖网络/ase 的内置自检。"""
    failures: List[str] = []

    def check(name, cond, detail):
        status = "OK" if cond else "FAIL"
        if not cond:
            failures.append(f"{name}: {detail}")
        if verbose:
            print(f"[{status}] {name}: {detail}")

    def plan(name):
        text = EXAMPLE_POSCARS[name]
        pos = parse_poscar_text(text)
        rep = analyze_structure(pos)
        mom, _ = heuristic_plan(pos, rep)
        return pos, rep, mom

    # 纯金属
    _, rep, mom = plan("Pt_fcc")
    check("Pt_fcc", all(abs(x) < 1e-9 for x in mom), f"M={mom}")
    _, rep, mom = plan("Fe_bcc")
    check("Fe_bcc", all(x > 0 for x in mom), f"M={mom}")

    # 岩盐反铁磁
    pos, rep, mom = plan("NiO")
    ni = [mom[i] for i, s in enumerate(pos.symbols) if s == "Ni"]
    check("NiO", any(x > 0 for x in ni) and any(x < 0 for x in ni), f"Ni={ni}")

    # slab
    pos, rep, mom = plan("NiO_001_slab")
    ni = [mom[i] for i, s in enumerate(pos.symbols) if s == "Ni"]
    check("NiO(001) slab",
          rep["system_type"] == "slab" and any(x > 0 for x in ni) and any(x < 0 for x in ni),
          f"type={rep['system_type']} Ni={ni}")

    # 闪锌矿 / 纤锌矿
    _, rep, mom = plan("ZnS_zincblende")
    check("ZnS zincblende",
          rep["motif"]["family"] == "zincblende" and all(abs(x) < 1e-9 for x in mom), f"M={sorted(set(mom))}")
    _, rep, mom = plan("ZnO_wurtzite")
    check("ZnO wurtzite",
          rep["motif"]["family"] == "wurtzite" and all(abs(x) < 1e-9 for x in mom), f"M={sorted(set(mom))}")

    # 钙钛矿
    pos, rep, mom = plan("LaFeO3")
    fe = [mom[i] for i, s in enumerate(pos.symbols) if s == "Fe"]
    check("LaFeO3 perovskite",
          rep["motif"]["family"] == "perovskite" and fe and all(abs(x) > 1e-9 for x in fe), f"Fe={fe}")

    # LDH
    pos, rep, mom = plan("NiAl_LDH")
    ni = [mom[i] for i, s in enumerate(pos.symbols) if s == "Ni"]
    others = [mom[i] for i, s in enumerate(pos.symbols) if s in ("O", "H", "Al")]
    check("NiAl-LDH",
          rep["motif"]["family"] == "ldh" and any(abs(x) > 1e-9 for x in ni)
          and all(abs(x) < 1e-9 for x in others),
          f"Ni={ni} others={sorted(set(others))}")

    # 尖晶石 / 反尖晶石
    pos, rep, mom = plan("Fe3O4")
    neigh = coordination_analysis(pos)
    tet = {mom[i] for i, s in enumerate(pos.symbols) if s == "Fe" and len(neigh[i]) == 4}
    octa = {mom[i] for i, s in enumerate(pos.symbols) if s == "Fe" and len(neigh[i]) == 6}
    check("Fe3O4 inverse spinel", tet == {5.0} and octa == {-5.0}, f"tet={tet} oct={octa}")

    pos, rep, mom = plan("MgAl2O4")
    check("MgAl2O4 normal spinel",
          rep["motif"]["family"] == "spinel" and all(abs(x) < 1e-9 for x in mom),
          f"M={sorted(set(mom))}")

    print("-" * 60)
    if failures:
        print("自检失败:")
        for f in failures:
            print("  -", f)
        return 1
    print("全部自检通过 ✅")
    return 0


def export_examples(outdir: str) -> int:
    os.makedirs(outdir, exist_ok=True)
    for name, text in EXAMPLE_POSCARS.items():
        path = os.path.join(outdir, f"POSCAR_{name}")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
