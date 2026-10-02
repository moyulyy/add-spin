# -*- coding: utf-8 -*-
"""
add-spin.py —— 一站式 VASP 磁矩初猜工具（POSCAR -> ISPIN / MAGMOM）
================================================================================
把「按元素拍脑袋给磁矩」升级为「LLM 工具调用 + 晶体学/表面分析」：

  * 解析完整 POSCAR（晶格、元素、坐标、Selective dynamics、Cartesian/Direct）；
  * 由晶格面法向上的周期空隙估计 bulk / slab / molecule；
  * slab 分层，识别 surface / subsurface / interior 等配位不饱和位点；
  * 识别 fcc/bcc、岩盐、闪锌矿、纤锌矿、钙钛矿、双钙钛矿、尖晶石/反尖晶石、
    LDH、氢氧化物、刚玉、金红石等结构；
  * 内置常见磁性材料知识库，按「元素 + 配位数 + 位点角色」给出磁矩；
  * 在校验磁性晶胞相容性后，生成共线磁序候选的正负符号；
  * 可选调用任意 OpenAI 兼容 LLM 做 function calling，无 Key 时自动回退启发式。

LLM 连接及运行配置集中在下方 LLM_* 常量；命令行参数可覆盖。

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
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ============================================================================
# 用户配置：LLM（命令行参数 > 环境变量 > 下列默认值）
# API Key 推荐通过环境变量提供，勿将真实密钥提交到版本库。
# ============================================================================
LLM_API_KEY = ""
LLM_BASE_URL = "https://api.deepseek.com/v1"
LLM_MODEL = "deepseek-chat"
LLM_TEMPERATURE = 0.2
LLM_TIMEOUT = 180
LLM_MAX_RETRIES = 3  # 单轮请求的最大总尝试次数（含首次请求）
LLM_MAX_STEPS = 8


SYSTEM_PROMPT = """\
你为 VASP 生成共线自旋初猜 ISPIN/MAGMOM。只输出工具调用，用中文说明依据。
MAGMOM 是自洽计算初猜，不是磁基态结论；必须比较不同磁序/自旋态的收敛能量。

工作流程：先 analyze_structure，必要时查元素/材料，preview_magmom 校验后再 submit_magmom。
每条 assignment 必须含 element 与有限 moment。所有原子都必须被规则或 site_overrides 覆盖。
规则按列表顺序匹配；coordination 严格匹配，不自动猜测低配位表面的体相母位点。
CN 不能唯一确定几何、氧化态或自旋态；结构名称及按化学式检索的知识库均为候选。
表面规则应显式给出，并放在元素通用规则之前；不要仅凭低配位就增强磁矩或套用体相结论。

3d 离子弱场自旋种子可取未成对电子数 2S（μB），不是顺磁有效磁矩 sqrt(n(n+2))。
八面体 d2、d3、d8 不因“低自旋”而变成0；Ni2+ 平方平面与八面体需分别判断。
按电中性与化学环境推断价态，不能仅按元素判零：Cu2+ 常取1，Cu+ d10通常0；金属Cu通常0。
Co3+ 八面体可能低自旋0或其他自旋态；Fe2+/Fe3+ 常用4/5。4f应标注自旋与轨道/SOC的区别。
Fe3O4 的四面体 Fe3+ 与八面体混合价亚晶格反平行；无法区分Fe2+/Fe3+时4.5仅为平均种子。
稀土A位不一律为0。分子需要电荷/多重度；O2三重态有非零总自旋，不可套用阴离子O2-规则。

magnetic_order:
- fm：将所有非零磁矩取正。
- afm：默认在所有非零磁性位点上建立共同周期近邻图；afm_elements 可选择元素/明确组名。
  只在周期图能二染色时自动赋号，岩盐MnO/FeO/CoO/NiO另检查(111) AFM-II相容性。
  一个磁性位点的原胞、奇周期或受挫图可能不相容；校验失败不能改称成功AFM。
  8原子立方NiO惯用胞也不能容纳AFM-II。A/C/G型不能混用；复杂磁序需显式位点方案。
- ferrimagnetic/auto：保留赋值中的正负号。auto配合site_overrides可指定已知磁序。
- nonmagnetic：所有位点必须明确为0，ISPIN=1。ISPIN=1不得与非零磁矩并存。
afm_group只是分组标签，需在afm_elements中选中才触发自动赋号。
自动AFM不能与site_overrides混用；显式磁序改用auto并完整给出正负磁矩。
若当前晶胞无法容纳已知目标磁序，必须指出需要磁性超胞，不能通过改成FM掩盖问题。
仅支持共线磁性，不处理SOC、非共线向量、自旋螺旋、无序局域矩或自动生成超胞。
所有磁矩单位μB，位点index从0开始，顺序与POSCAR一致。
"""


# ============================================================================
# 0. 兼容旧版 add-spin.py 的模块级 API
#    （fast-vasp CLI 会 from add_spin import parse_poscar, build_magmom, append_magmom）
# ============================================================================
# 未知化学环境下的开壳层自旋初猜（μB）；不是元素固有磁矩。
# 未列出的元素默认为 0；具体价态、晶体场和金属环境应由完整分析确定。
DEFAULT_MAGMOM = {
    'Ti': 1, 'V': 2, 'Cr': 3, 'Mn': 5, 'Fe': 5, 'Co': 3, 'Ni': 2, 'Cu': 1,
    'Nb': 1, 'Mo': 2, 'Tc': 3, 'Ru': 2, 'Rh': 1,
    'W': 2, 'Re': 2, 'Os': 2, 'Ir': 1,
    'Ce': 1, 'Pr': 2, 'Nd': 3, 'Pm': 4, 'Sm': 5, 'Eu': 6, 'Gd': 7,
    'Tb': 6, 'Dy': 5, 'Ho': 4, 'Er': 3, 'Tm': 2, 'Yb': 1,
    'Pa': 1, 'U': 2, 'Np': 3, 'Pu': 4, 'Am': 5, 'Cm': 6,
}


def parse_poscar(path="POSCAR"):
    """兼容入口，复用完整解析器避免元素/数量顺序不一致。"""
    try:
        pos = parse_poscar_full(path)
    except ValueError:
        return []
    return list(zip(pos.species, pos.counts))


def build_magmom(pairs, custom=None):
    """兼容无坐标入口，仅生成元素种子；无法识别位点或 AFM。"""
    custom = custom or {}
    pairs = list(pairs)
    if not pairs or any(el not in ATOMIC_MASS or type(n) is not int or n < 0 for el, n in pairs):
        raise ValueError("无效的元素/原子数量")
    if not sum(n for el, n in pairs):
        raise ValueError("至少需要一个原子")
    pure = len({el for el, n in pairs if n}) == 1
    moments = []
    for el, count in pairs:
        default = METAL_MOMENT.get(el, 0.0) if pure else default_element_moment(el)
        moments.extend([_finite_moment(custom.get(el, default))] * count)
    pos = Poscar("legacy", np.eye(3), [el for el, n in pairs], [n for el, n in pairs],
                 [el for el, n in pairs for _ in range(n)], np.zeros((len(moments), 3)), "Direct")
    return "# 仅按元素的种子：未校验晶体位点、价态或磁序\n" + format_magmom(pos, moments)


def append_magmom(path="POSCAR", incar="INCAR", custom=None):
    """兼容入口，使用完整结构分析并安全更新 INCAR。"""
    pos = parse_poscar_full(path)
    report = analyze_structure(pos)
    if custom is None:
        moments, _ = heuristic_plan(pos, report)
    else:
        assignments = [{"element": el, "moment": custom.get(el, default_element_moment(el))}
                       for el in dict.fromkeys(pos.symbols)]
        moments = build_moments_from_assignments(pos, report, assignments)
    append_to_incar(format_magmom(pos, moments), incar)
    return True

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
    "Sc": [3], "Ti": [2, 3, 4], "V": [2, 3, 4, 5], "Cr": [2, 3, 4, 6],
    "Mn": [2, 3, 4, 6, 7], "Fe": [2, 3, 4], "Co": [2, 3, 4], "Ni": [2, 3],
    "Cu": [1, 2], "Zn": [2], "Y": [3], "Zr": [4], "Nb": [3, 5],
    "Mo": [3, 4, 5, 6], "Ru": [3, 4], "Rh": [3], "Pd": [2, 4], "Ag": [1],
    "Ce": [3, 4], "Pr": [3], "Nd": [3], "Pm": [3], "Sm": [2, 3], "Eu": [2, 3],
    "Gd": [3], "Tb": [3], "Dy": [3], "Ho": [3], "Er": [3], "Tm": [3],
    "Yb": [2, 3], "Lu": [3], "Hf": [4], "Ta": [5], "W": [4, 5, 6],
    "Re": [4, 5, 6, 7], "Os": [4], "Ir": [3, 4], "Pt": [2, 4], "U": [4, 6],
}

# 离子：ox -> (d/f 电子数, 高自旋未成对电子数, 低自旋未成对电子数)。
# d 壳层 HS/LS 均指理想八面体晶体场；d1-d3、d8-d10 的两列相同。
# 四面体、平方平面等环境须另行判断（例如平方平面 d8 可为 S=0）。
# f 壳层两列给 Hund 自旋计数，不代表含 SOC 的总磁矩或实验有效磁矩。
MAGNETIC_IONS = {
    "Sc": {3: (0, 0, 0)},
    "Ti": {2: (2, 2, 2), 3: (1, 1, 1), 4: (0, 0, 0)},
    "V":  {2: (3, 3, 3), 3: (2, 2, 2), 4: (1, 1, 1), 5: (0, 0, 0)},
    "Cr": {2: (4, 4, 2), 3: (3, 3, 3), 4: (2, 2, 2), 6: (0, 0, 0)},
    "Mn": {2: (5, 5, 1), 3: (4, 4, 2), 4: (3, 3, 3), 6: (1, 1, 1), 7: (0, 0, 0)},
    "Fe": {2: (6, 4, 0), 3: (5, 5, 1), 4: (4, 4, 2)},
    "Co": {2: (7, 3, 1), 3: (6, 4, 0), 4: (5, 5, 1)},
    "Ni": {2: (8, 2, 2), 3: (7, 3, 1)},
    "Cu": {1: (10, 0, 0), 2: (9, 1, 1)},
    "Zn": {2: (10, 0, 0)},
    "Y": {3: (0, 0, 0)},
    "Zr": {4: (0, 0, 0)},
    "Nb": {3: (2, 2, 2), 5: (0, 0, 0)},
    "Ru": {3: (5, 5, 1), 4: (4, 4, 2)},
    "Rh": {3: (6, 4, 0)},
    "Pd": {2: (8, 2, 2), 4: (6, 4, 0)},
    "Ag": {1: (10, 0, 0)},
    "Mo": {3: (3, 3, 3), 4: (2, 2, 2), 5: (1, 1, 1), 6: (0, 0, 0)},
    "Hf": {4: (0, 0, 0)},
    "Ta": {5: (0, 0, 0)},
    "W":  {4: (2, 2, 2), 5: (1, 1, 1), 6: (0, 0, 0)},
    "Re": {4: (3, 3, 3), 5: (2, 2, 2), 6: (1, 1, 1), 7: (0, 0, 0)},
    "Os": {4: (4, 4, 2)},
    "Ir": {3: (6, 4, 0), 4: (5, 5, 1)},
    "Pt": {2: (8, 2, 2), 4: (6, 4, 0)},
    "La": {3: (0, 0, 0)},
    "Ce": {3: (1, 1, 1), 4: (0, 0, 0)},
    "Pr": {3: (2, 2, 2)},
    "Nd": {3: (3, 3, 3)},
    "Pm": {3: (4, 4, 4)},
    "Sm": {2: (6, 6, 6), 3: (5, 5, 5)},
    "Eu": {2: (7, 7, 7), 3: (6, 6, 6)},
    "Gd": {3: (7, 7, 7)},
    "Tb": {3: (8, 6, 6)},
    "Dy": {3: (9, 5, 5)},
    "Ho": {3: (10, 4, 4)},
    "Er": {3: (11, 3, 3)},
    "Tm": {3: (12, 2, 2)},
    "Yb": {2: (14, 0, 0), 3: (13, 1, 1)},
    "Lu": {3: (14, 0, 0)},
}

# 纯金属铁磁/反铁磁参考磁矩 (μB/atom)，用于简单金属体系
METAL_MOMENT = {
    "Fe": 2.2, "Co": 1.7, "Ni": 0.6,
    "Mn": 1.0, "Cr": 1.0, "Gd": 7.0, "Tb": 6.0, "Dy": 5.0, "Ho": 4.0,
    "Er": 3.0,
}
# 常见闭壳层化合物的零磁矩兜底；不适用于孤立原子、自由基、缺陷态。
# Cu2+ 为 d9，不能因金属 Cu 非磁就将所有含 Cu 化合物判为非磁。
NONMAGNETIC = set(
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca "
    "Ga Ge As Se Br Kr Rb Sr In Sn Sb Te I Xe Cs Ba "
    "Sc Y La Lu Zn Cd Hg Pb Bi Ag Au".split()
)
# 氧化物中未知价态时的经验自旋初猜；4d/5d 常取较小值，不代表确定自旋态。
OXIDE_HS = {
    "Ti": 1, "V": 2, "Cr": 3, "Mn": 5, "Fe": 5, "Co": 3, "Ni": 2, "Cu": 1,
    "Ru": 2, "Rh": 1, "Mo": 2, "W": 2, "Re": 2, "Ir": 1,
    "Ce": 1, "Pr": 2, "Nd": 3, "Pm": 4, "Sm": 5, "Eu": 6, "Gd": 7, "Tb": 6,
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
    """未知化学环境的自旋初猜；不能用于推断价态或磁基态。"""
    if symbol in NONMAGNETIC:
        return 0.0
    if symbol in OXIDE_HS:
        return float(OXIDE_HS[symbol])
    return float(DEFAULT_MAGMOM.get(symbol, 0.0))


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
    scale_tokens = re.split(r"[!#]", lines[1], maxsplit=1)[0].split()
    if len(scale_tokens) not in (1, 3):
        raise ValueError("POSCAR 缩放行必须包含 1 个或 3 个数值")
    scales = np.array([float(x) for x in scale_tokens], dtype=float)
    if not np.all(np.isfinite(scales)) or np.any(scales == 0):
        raise ValueError("POSCAR 缩放因子必须有限且非零")

    # ---- 缩放因子 -----------------------------------------------------
    coord_scale: np.ndarray | float
    scale_vec: Optional[np.ndarray] = None
    volume_target: Optional[float] = None
    if len(scale_tokens) == 3:
        scale_vec = scales
        if np.any(scale_vec <= 0):
            raise ValueError("POSCAR 的三个缩放因子必须全部为正")
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
    if raw_lattice.shape != (3, 3) or not np.all(np.isfinite(raw_lattice)):
        raise ValueError("POSCAR 晶格必须为有限的 3×3 矩阵")
    raw_vol = abs(float(np.linalg.det(raw_lattice)))
    if raw_vol <= 1e-12:
        raise ValueError("POSCAR 晶格矩阵不可逆或体积过小")
    if volume_target is not None:
        factor = (volume_target / raw_vol) ** (1.0 / 3.0)
        lattice = raw_lattice * factor
        coord_scale = factor
    elif scale_vec is not None:
        # VASP 三因子分别缩放 Cartesian x/y/z 分量，即晶格矩阵的列。
        lattice = raw_lattice * scale_vec[None, :]
    else:
        lattice = raw_lattice * float(coord_scale)

    # ---- 元素 / 数量 --------------------------------------------------
    idx = 5
    tok5 = re.split(r"[!#]", lines[idx], maxsplit=1)[0].split()
    if not tok5:
        raise ValueError("POSCAR 元素/数量行为空")
    source_species: Optional[List[str]] = None
    if tok5 and all(_is_int(t) for t in tok5):
        # VASP4：没有元素符号行
        counts = [int(t) for t in tok5]
        idx += 1
    else:
        source_species = []
        for token in tok5:
            # 保留标准符号；允许 Fe_pv 等常见 POTCAR 标签。
            symbol = token.split("_", 1)[0]
            if symbol not in _SYMBOLS:
                raise ValueError(f"POSCAR 无法识别元素符号: {token}")
            source_species.append(symbol)
        idx += 1
        count_tokens = re.split(r"[!#]", lines[idx], maxsplit=1)[0].split()
        if len(count_tokens) != len(source_species):
            raise ValueError("POSCAR 元素种类数与数量条目数不一致")
        counts = [int(t) for t in count_tokens]
        idx += 1
    if not counts or any(c < 0 for c in counts) or sum(counts) <= 0:
        raise ValueError("POSCAR 原子数量必须为非负整数，且总原子数必须大于零")

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
    if mode[0] not in "DdCcKk":
        raise ValueError(f"POSCAR 未知坐标模式: {mode}")
    coord_mode = "Cartesian" if mode[0] in "CcKk" else "Direct"
    idx += 1

    n_atoms = sum(counts)
    coords: List[List[float]] = []
    for k in range(n_atoms):
        if idx + k >= len(lines):
            raise ValueError(f"POSCAR 坐标行不足，期望 {n_atoms} 个原子")
        parts = lines[idx + k].split()
        if len(parts) < 3:
            raise ValueError(f"POSCAR 第 {k + 1} 个原子坐标不足三个分量")
        coords.append([float(parts[0]), float(parts[1]), float(parts[2])])
        if selective is not None:
            if len(parts) < 6 or any(p[:1] not in "TtFf" for p in parts[3:6]):
                raise ValueError(f"POSCAR 第 {k + 1} 个原子缺少有效的三个 T/F 约束标记")
            flags = [p[:1] in ("T", "t") for p in parts[3:6]]
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
            raise ValueError("VASP4 POSCAR 缺少元素符号，需在同目录提供匹配的 POTCAR")
        if any(s not in _SYMBOLS for s in source_species):
            raise ValueError("POTCAR 包含无法识别的元素符号")

    symbols: List[str] = []
    for el, cnt in zip(source_species, counts):
        symbols.extend([el] * cnt)

    arr = np.array(coords, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError("POSCAR 原子坐标必须为有限数值")
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
    # 平移任一原子整数个晶格矢量不能改变近邻；先折回主晶胞，
    # 才能使用围绕原点构造的有限周期镜像集合。
    cart = (pos.frac % 1.0) @ pos.lattice
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
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("配位截断因子必须为正的有限数值")
    if pos.n_atoms == 0:
        return []
    radii = np.array([cov_radius(s) for s in pos.symbols])
    max_cut = factor * float(radii.max()) * 2.0 + 0.5
    cart = (pos.frac % 1.0) @ pos.lattice
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
                if np.any(dists < 1e-6):
                    raise ValueError(f"原子 {i} 与 {j} 在周期边界条件下重合")
                for d in dists[dists <= cut]:
                    cand[i].append((j, float(d)))
                    cand[j].append((i, float(d)))

    neigh: List[List[Tuple[int, float]]] = []
    nonmetals = set("H He B C N O F Ne Si P S Cl Ar Ge As Se Br Kr Sb Te I Xe At Rn".split())
    pure_metal = len(set(pos.symbols)) == 1 and pos.symbols[0] not in nonmetals
    for i in range(n):
        lst = sorted(cand[i], key=lambda t: t[1])
        # bcc 的第二壳层仅比第一壳层远 15.5%，不能沿用离子配位容差。
        cn = _first_shell_count([d for _, d in lst], ratio=1.10 if pure_metal else 1.25)
        neigh.append(lst[:cn])
    return neigh


def geometry_label(cn: int) -> str:
    return {
        2: "2-coordinate (geometry unverified)",
        3: "3-coordinate (geometry unverified)",
        4: "4-coordinate (tetrahedral/square-planar candidates)",
        5: "5-coordinate (geometry unverified)",
        6: "6-coordinate (octahedral candidate)",
        7: "7-coordinate",
        8: "8-coordinate (geometry unverified)",
        9: "9-coordinate",
        10: "10-coordinate",
        11: "11-coordinate",
        12: "12-coordinate (fcc/hcp candidates)",
    }.get(cn, f"{cn}-coordinate")


# ============================================================================
# 4. 结构基元识别
# ============================================================================
# ============================================================================
# 3.5 体系类型 (bulk / slab / mole) 与表面层的数学分析
#     一维分数坐标的最大周期空隙 × 对应晶格面的法向间距，
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
    """返回各晶格面对的最大空隙 (Å)，作为真空厚度的几何估计。

    分数坐标 f_i 的法向为逆晶格矩阵第 i 列；斜晶胞不能用晶格矢量
    长度替代面间距。该判据不包含原子半径，不能单独证明体系维度。
    """
    heights = 1.0 / np.linalg.norm(np.linalg.inv(pos.lattice), axis=0)
    return {
        ax: largest_periodic_gap(pos.frac[:, i]) * float(heights[i])
        for i, ax in enumerate("abc")
    }


def detect_system_type(
    pos: Poscar,
    vacuum_threshold: float = DEFAULT_VACUUM_THRESHOLD,
    gaps: Optional[Dict[str, float]] = None,
) -> Tuple[str, Dict[str, float]]:
    """按真空空隙估计 bulk / slab / mole / unknown。

    任一单方向有真空均可为 slab；两个方向有真空可能是线状体系，
    暂以 unknown 标记。
    """
    if not np.isfinite(vacuum_threshold) or vacuum_threshold <= 0:
        raise ValueError("真空阈值必须为正的有限数值")
    if gaps is None:
        gaps = vacuum_gaps(pos)
    has = {a: gaps[a] > vacuum_threshold + 1e-8 for a in gaps}
    if has["a"] and has["b"] and has["c"]:
        return "mole", gaps
    if sum(has.values()) == 1:
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

    - 真空方向由最大 vacuum gap 决定；
    - 层位置 = 原子笛卡尔坐标在“真空轴法向”上的投影；
    - 把最大空隙平移到边界，于是 t<0 一侧是下表面、t>0 一侧是上表面。
    """
    if gaps is None:
        gaps = vacuum_gaps(pos)
    if not np.isfinite(layer_tol) or layer_tol <= 0:
        raise ValueError("分层容差必须为正的有限数值")
    if pos.n_atoms == 0:
        raise ValueError("空结构无法分层")
    axis = _vacuum_axis_index(gaps)
    lattice = pos.lattice
    other = [x for x in range(3) if x != axis]
    normal = np.cross(lattice[other[0]], lattice[other[1]])
    nrm = float(np.linalg.norm(normal))
    if nrm < 1e-8:
        normal = lattice[axis] / (np.linalg.norm(lattice[axis]) + 1e-12)
    else:
        normal = normal / nrm
    if np.dot(lattice[axis], normal) < 0:
        normal = -normal
    cart = pos.cartesian()
    t = cart @ normal
    period = abs(float(np.dot(lattice[axis], normal)))
    if period < 1e-8:
        period = float(np.linalg.norm(lattice[axis]))
    t = t % period

    n = len(t)
    if n <= 1:
        t_shift = np.zeros_like(t)
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
    layers = np.zeros(n, dtype=int)
    lvl = 0
    ref = t_shift[order2[0]]
    for i in order2[1:]:
        if t_shift[i] - ref > layer_tol:
            lvl += 1
            ref = t_shift[i]
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
    """基于化学计量、配位数与局部键角提出结构基元候选。

    使用"最大配位数"而非瞬时配位数，以兼容 slab 上下表面配位不饱和的情况
    （表面 4 配位原子降到 3、八面体 6 配位降到 5 等，内部原子仍保留 bulk 配位）。
    """
    comp = pos.composition()
    cn_list = [len(x) for x in neigh]
    elements = set(comp)
    nonmetals = set("H He B C N O F Ne Si P S Cl Ar Ge As Se Br Kr Sb Te I Xe At Rn".split())

    def has_site_geometry(i: int, geometry: str) -> bool:
        """按周期近邻键角区分四面体/八面体，避免仅以 CN 推断。"""
        expected = 4 if geometry == "tetrahedral" else 6
        if len(neigh[i]) != expected:
            return False
        cutoff = max(d for _, d in neigh[i]) + 1e-5
        offsets = _image_offsets(pos.lattice, cutoff)
        cart = (pos.frac % 1.0) @ pos.lattice
        vectors = []
        for j in {j for j, _ in neigh[i]}:
            diff = cart[j] - cart[i] + offsets
            lengths = np.linalg.norm(diff, axis=1)
            for vector, length in zip(diff, lengths):
                if 1e-6 < length <= cutoff:
                    vectors.append(vector / length)
        if len(vectors) != expected:
            return False
        dots = np.array([np.dot(vectors[a], vectors[b])
                         for a in range(expected) for b in range(a + 1, expected)])
        target = np.full(6, -1.0 / 3.0) if expected == 4 else np.array([-1.0] * 3 + [0.0] * 12)
        return bool(np.allclose(np.sort(dots), target, atol=0.20, rtol=0.0))

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
            "真空空隙判据提示 slab；表面可能配位不饱和，"
            "不能仅按表面 CN 推断对应体相晶格或阳离子价态。"
        )
    elif system_type == "mole":
        motif.update(family="molecule", name=f"分子/团簇候选 {pos.formula()}")
        motif["notes"].append("存在三个方向的大空隙；不套用体相金属或晶体结构类型。")
        return motif

    # ---- 单元素金属 ---------------------------------------------------
    if len(elements) == 1:
        el = next(iter(elements))
        kind = _lattice_kind(pos)
        cn = max_cn[el]
        sub = "unknown"
        if cn == 12:
            sub = "hcp" if kind == "hexagonal" else "fcc"
        elif cn == 8:
            sub = "bcc"
        elif cn == 6:
            sub = "simple-cubic"
        elif cn == 4:
            sub = "diamond-like"
        if el in nonmetals:
            motif.update(family="elemental", name=f"{el} 单质（局部 {sub} 候选）", lattice=kind)
        else:
            motif.update(family="metal", name=f"{el} 金属 ({sub} 候选)", lattice=kind, metal_lattice=sub)
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
                    name=f"层状双金属氢氧化物 LDH 候选 {pos.formula()}",
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
                    name=f"氢氧化物候选 {pos.formula()} (brucite-like)",
                    metals=sorted(metals),
                    notes=["八面体 M(OH)6 层板；H 置 0。"],
                )
                return motif

        # ---- 双钙钛矿 A2 B B' O6 ------------------------------------
        if norm.get(anion) == 6 and len(cation_elements) >= 3:
            counts = sorted(norm[el] for el in cation_elements)
            if counts == [1, 1, 2] and len(oct_els) >= 2:
                b_sites = [el for el in cation_elements if max_cn.get(el, 0) == 6]
                motif.update(
                    family="double_perovskite",
                    name=f"双钙钛矿候选 {pos.formula()}",
                    b_sites=b_sites,
                    notes=[
                        "B/B' 位为八面体过渡金属，通常反平行排列（铁磁/亚铁磁）；",
                        "如 Sr2FeMoO6：Fe3+(+5) 与 Mo5+(-1) 反平行。",
                    ],
                )
                return motif

        # ---- 尖晶石 A B2 O4（含 Fe3O4 / Co3O4 这类二元 3:4）-----------
        if (norm.get(anion) == 4 and n_cations == 3 and system_type == "bulk"
                and tet_els and oct_els
                and any(has_site_geometry(i, "tetrahedral") for i, el in enumerate(pos.symbols) if el in tet_els)
                and any(has_site_geometry(i, "octahedral") for i, el in enumerate(pos.symbols) if el in oct_els)):
            if len(cation_elements) == 1:
                el = cation_elements[0]
                if 4 in site_cn[el] and 6 in site_cn[el]:
                    stype = "未定（同一元素同时占四/六配位，需价态区分正/反尖晶石）"
                elif 6 in site_cn[el]:
                    stype = "全部八面体（非典型尖晶石）"
                else:
                    stype = "未定"
                motif.update(
                    family="spinel", name=f"尖晶石候选 {pos.formula()}",
                    spinel_type=stype, tetrahedral_site=el, octahedral_site=el,
                )
            else:
                one = [el for el in cation_elements if norm[el] == 1]
                two = [el for el in cation_elements if norm[el] == 2]
                tet = tet_els[0] if tet_els else None
                if one and two and tet == one[0]:
                    stype, name = "normal", f"正尖晶石候选 {pos.formula()}"
                elif two and tet == two[0]:
                    stype, name = "inverse", f"反尖晶石候选 {pos.formula()}"
                else:
                    stype, name = "mixed/undetermined", f"尖晶石候选 {pos.formula()}"
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
        if (norm.get(anion) == 3 and n_cations == 2 and len(cation_elements) == 2
                and all(norm[el] == 1 for el in cation_elements)
                and any(max_cn.get(el, 0) >= 8 for el in cation_elements)
                and any(max_cn.get(el, 0) == 6 for el in cation_elements)):
            a_site = [el for el in cation_elements if max_cn.get(el, 0) >= 8]
            b_site = [el for el in cation_elements if max_cn.get(el, 0) == 6]
            motif.update(
                family="perovskite",
                name=f"钙钛矿候选 {pos.formula()}",
                a_site=a_site,
                b_site=b_site,
                notes=[
                    "B 位过渡金属（CN=6）承担磁矩；A 位（La/Sr/Ba/Ca/稀土，CN≥8）"
                    "通常非磁或为 4f 磁矩。",
                ],
            )
            return motif

        # ---- 岩盐 / 闪锌矿 / 纤锌矿 AO -------------------------------
        if len(norm) == 2 and norm.get(anion) == 1 and n_cations == 1:
            cation = cation_elements[0] if cation_elements else None
            family, name = None, None
            cation_indices = [i for i, el in enumerate(pos.symbols) if el == cation]
            if (cation and max_cn.get(cation, 0) == 6 and max_cn.get(anion, 0) == 6
                    and any(has_site_geometry(i, "octahedral") for i in cation_indices)):
                family, name = "rocksalt", f"岩盐候选 {pos.formula()}"
            elif (cation and max_cn.get(cation, 0) == 4 and max_cn.get(anion, 0) == 4
                    and any(has_site_geometry(i, "tetrahedral") for i in cation_indices)):
                if _lattice_kind(pos) == "hexagonal":
                    family, name = "wurtzite", f"纤锌矿候选 {pos.formula()}"
                else:
                    family, name = "zincblende", f"闪锌矿候选 {pos.formula()}"
            if family is not None:
                motif.update(family=family, name=name)
                motif["notes"].append("配位和局部键角仅支持结构候选；磁序还需核验周期与材料信息。")
                return motif

        # ---- 金红石 MO2 --------------------------------------------
        if norm.get(anion) == 2 and len(cation_elements) == 1 and n_cations == 1:
            cation = cation_elements[0]
            if max_cn.get(cation, 0) == 6:
                motif.update(
                    family="rutile",
                    name=f"金红石等六配位 MO2 候选 {pos.formula()}",
                    notes=["阳离子六配位、阴离子三配位；按阳离子价态给高自旋磁矩。"],
                )
                return motif

        # ---- 刚玉 A2O3 ---------------------------------------------
        if (norm.get(anion) == 3 and n_cations == 2 and len(cation_elements) == 1
                and max_cn.get(cation_elements[0], 0) == 6):
            motif.update(
                family="corundum",
                name=f"刚玉等六配位 M2O3 候选 {pos.formula()}",
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
            {"element": "Fe", "coordination": 6, "moment": -4.5,
             "site": "16d 八面体 Fe2+/Fe3+"},
            {"element": "O", "moment": 0.0, "site": "32e"},
        ],
        note="亚铁磁：8a Fe3+ 与整个 16d 亚晶格反平行（16d 内部同向）；"
             "八面体等量 Fe2+/Fe3+ 用平均 |4.5| 作初猜，不代表已解析低温电荷有序。",
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
        {"Mn": 1, "Fe": 2, "O": 4}, "锰铁氧体 MnFe2O4（阳离子占位可变）", "ferrimagnetic",
        [
            {"element": "Fe", "coordination": 4, "moment": 5.0},
            {"element": "Fe", "coordination": 6, "moment": -5.0},
            {"element": "Mn", "coordination": 4, "moment": 5.0},
            {"element": "Mn", "coordination": 6, "moment": -5.0},
            {"element": "O", "moment": 0.0},
        ],
        note="Mn2+ 高自旋 S=5/2；正/部分反尖晶石的实际占位随条件变化，"
             "Mn、Fe 均按四面体与八面体亚晶格反平行赋号。",
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
        {"Bi": 1, "Fe": 1, "O": 3}, "BiFeO3（G 型反铁磁，多铁性）", "antiferromagnetic",
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
        {"Mn": 1, "S": 1}, "MnS（岩盐/闪锌矿/纤锌矿多晶型）", "antiferromagnetic",
        [{"element": "Mn", "moment": 5.0},
         {"element": "S", "moment": 0.0}],
        afm=["Mn"],
        note="Mn2+ 高自旋 S=5/2；α-MnS 为岩盐六配位，其他多晶型可四配位，磁序须结合结构。",
    ),
    _kb_entry(
        {"Mn": 1, "Se": 1}, "MnSe（反铁磁）", "antiferromagnetic",
        [{"element": "Mn", "moment": 5.0},
         {"element": "Se", "moment": 0.0}],
        afm=["Mn"],
        note="Mn2+ 高自旋；存在不同多晶型，不能仅由化学式假定为四配位。",
    ),
    _kb_entry(
        {"Mn": 1, "Te": 1}, "MnTe（反铁磁）", "antiferromagnetic",
        [{"element": "Mn", "moment": 5.0},
         {"element": "Te", "moment": 0.0}],
        afm=["Mn"],
        note="常见 α-MnTe 为 NiAs 型六配位，层内铁磁、层间反铁磁；其他多晶型须另行判断。",
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
        note="Fe3+(↑) 与 Re5+(5d2, ↓) 反平行。",
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
            if len(stack) == 1:
                # 多余的右括号：忽略，避免弹空栈后崩溃。
                i += 1
                continue
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
def _magnetic_contacts(pos: Poscar, indices: Sequence[int], cutoff: float = 8.0):
    """磁性位点所有周期接触，保留自镜像及跨边界键。"""
    idx = list(indices)
    cart = (pos.frac % 1.0) @ pos.lattice
    offsets = _image_offsets(pos.lattice, cutoff)
    contacts = []
    for i in idx:
        for j in idx:
            vectors = cart[j] - cart[i] + offsets
            lengths = np.linalg.norm(vectors, axis=1)
            for k in np.flatnonzero((lengths > 1e-7) & (lengths <= cutoff)):
                contacts.append((i, j, float(lengths[k]), vectors[k]))
    return contacts


def _phase_signs(pos: Poscar, indices: Sequence[int], q: np.ndarray) -> Dict[int, int]:
    """q 为 Cartesian 倒空间传播矢量（cycles/Å），验证晶胞平移保持自旋。"""
    translations = pos.lattice @ q
    if not np.allclose(translations, np.rint(translations), atol=1e-5, rtol=0):
        raise ValueError("当前晶胞与目标 AFM 传播矢量不相容；请构造磁性超胞后重试")
    idx = list(indices)
    phase = (pos.frac[idx] % 1.0) @ pos.lattice @ q
    twice = 2 * (phase - phase[0])
    if not np.allclose(twice, np.rint(twice), atol=0.08, rtol=0):
        raise ValueError("磁性位点不在所指定的两类共线磁层上；请提供明确的位点磁矩")
    signs = np.where(np.rint(twice).astype(int) % 2 == 0, 1, -1)
    if len(set(signs)) < 2:
        raise ValueError("当前晶胞无法产生两类相反自旋；请构造磁性超胞")
    return dict(zip(idx, map(int, signs)))


def layered_signs(pos: Poscar, indices: Sequence[int],
                  axis: Optional[Sequence[float]] = None, tol_frac: float = 0.4) -> Dict[int, int]:
    """显式 Cartesian 层法向的交替磁序；不从近邻反号比例猜测磁序类型。"""
    idx = list(indices)
    if len(idx) < 2 or axis is None:
        raise ValueError("层状 AFM 需要至少两个磁性位点和明确的层法向；请指定磁序或磁性超胞")
    normal = np.asarray(axis, dtype=float)
    if normal.shape != (3,) or not np.all(np.isfinite(normal)) or np.linalg.norm(normal) < 1e-9:
        raise ValueError("AFM 层法向必须是有限非零三维向量")
    normal /= np.linalg.norm(normal)
    cart = (pos.frac[idx] % 1.0) @ pos.lattice
    # 包含晶格平移后的层间距，避免只看到有限晶胞中缺失的层。
    proj = np.concatenate([(cart + offset) @ normal for offset in
                           _image_offsets(pos.lattice, 0.0)])
    gaps = np.diff(np.unique(np.round(proj, 5)))
    gaps = gaps[gaps > 1e-4]
    if not len(gaps):
        raise ValueError("未找到不同磁层；请指定磁性超胞")
    return _phase_signs(pos, idx, normal / (2 * float(gaps.min())))


def rocksalt_type2_signs(pos: Poscar, indices: Sequence[int]) -> Dict[int, int]:
    """由 fcc 阳离子次近邻立方轴构造 (111) AFM-II，并验证磁胞相容性。"""
    idx = list(indices)
    if len(idx) < 2:
        raise ValueError("岩盐 AFM-II 至少需要两个磁性位点；请构造磁性超胞")
    contacts = _magnetic_contacts(pos, idx)
    if not contacts:
        raise ValueError("未识别到岩盐磁性亚晶格")
    origin = idx[0]
    local = [(d, v) for i, j, d, v in contacts if i == origin]
    dmin = min(d for d, v in local)
    # fcc 次近邻距离为最近邻的 sqrt(2) 倍。
    vectors = [v for d, v in local if abs(d / dmin - math.sqrt(2)) < 0.04]
    for trio in itertools.combinations(vectors, 3):
        axes = np.array(trio)
        gram = axes @ axes.T
        if not np.allclose(gram, np.eye(3) * gram[0, 0], rtol=0.04, atol=0.04):
            continue
        for signs in itertools.product((-1, 1), repeat=3):
            q = 0.5 * np.sum(axes * np.array(signs)[:, None], axis=0) / gram[0, 0]
            try:
                return _phase_signs(pos, idx, q)
            except ValueError:
                continue
    raise ValueError("未找到与当前晶胞相容的岩盐 (111) AFM-II；请构造磁性超胞或明确给出位点磁矩")


def bipartite_signs(pos: Poscar, element: Optional[str] = None,
                    neigh: Optional[List[List[Tuple[int, float]]]] = None,
                    nn_tolerance: float = 1.15,
                    indices: Optional[Sequence[int]] = None) -> Dict[int, int]:
    """直接染色周期商图；每条周期最近邻键都必须连接相反自旋。"""
    idx = list(indices) if indices is not None else [
        i for i, el in enumerate(pos.symbols) if element is None or el == element]
    if len(idx) < 2:
        raise ValueError("AFM 至少需要两个非零磁性位点；请构造磁性超胞")
    contacts = _magnetic_contacts(pos, idx)
    if not contacts:
        raise ValueError("未找到磁性近邻；不能自动确定 AFM 磁序")
    nearest = {i: min(d for a, b, d, v in contacts if a == i) for i in idx}
    adj = {i: set() for i in idx}
    for i, j, d, vector in contacts:
        if d <= min(nearest[i], nearest[j]) * nn_tolerance:
            if i == j:
                raise ValueError("AFM 最近邻为自身周期镜像，当前晶胞过小；请构造磁性超胞")
            adj[i].add(j)
            adj[j].add(i)
    color = {}
    # 几何排序使原子重排只影响整体自旋翻转，而不改变相对磁序。
    ordered = sorted(idx, key=lambda i: tuple(np.round(pos.frac[i] % 1.0, 8)))
    for start in ordered:
        if start in color:
            continue
        if not adj[start]:
            raise ValueError("磁性近邻图存在孤立位点；请明确给出位点磁矩")
        color[start] = 1
        queue = deque([start])
        while queue:
            i = queue.popleft()
            for j in adj[i]:
                if j not in color:
                    color[j] = -color[i]
                    queue.append(j)
                elif color[j] == color[i]:
                    raise ValueError("周期磁性近邻图不是二部图，无法确定此 AFM 磁序；请提供明确位点磁矩或合适磁性超胞")
    return color


# ============================================================================
# 7. 综合分析（给 LLM 的一站式结构报告）
# ============================================================================
def analyze_structure(pos: Poscar, max_sites: int = 400) -> Dict:
    neigh = coordination_analysis(pos)
    ctx = site_context(pos, neigh)
    motif = detect_motif(pos, neigh, ctx["system_type"])
    kb = _compatible_kb(pos, motif, ctx["system_type"])

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


def _finite_moment(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("moment 必须是有限数值")
    return float(value)


def build_moments_from_assignments(
    pos: Poscar, report: Dict, assignments: Sequence[Dict],
    overrides: Optional[Sequence[Dict]] = None, afm: Optional[Sequence] = None,
    afm_elementwise: bool = False,
) -> List[float]:
    """严格按顺序匹配显式规则；不由低配位数猜测母位点/氧化态。"""
    if not isinstance(assignments, (list, tuple)):
        raise ValueError("assignments 必须是规则列表")
    valid_roles = {"any", "all", "*", "surface", "surface_top", "surface_bottom",
                   "top_surface", "bottom_surface", "subsurface", "interior", "bulk", "cluster"}
    for rule in assignments:
        if not isinstance(rule, dict) or "element" not in rule or "moment" not in rule:
            raise ValueError("每条 assignment 必须包含 element 和 moment")
        if rule["element"] not in pos.symbols:
            raise ValueError(f"规则元素不存在于 POSCAR: {rule['element']}")
        _finite_moment(rule["moment"])
        cn = rule.get("coordination")
        if cn is not None and (type(cn) is not int or cn < 0):
            raise ValueError("coordination 必须是非负整数")
        if rule.get("site_role", "any") not in valid_roles:
            raise ValueError("未知 site_role")
        if "afm_group" in rule and (not isinstance(rule["afm_group"], str) or not rule["afm_group"]):
            raise ValueError("afm_group 必须是非空字符串")
    if not isinstance(overrides or [], (list, tuple)):
        raise ValueError("site_overrides 必须是列表")
    override_map = {}
    for ov in overrides or []:
        if not isinstance(ov, dict) or type(ov.get("index")) is not int or "moment" not in ov:
            raise ValueError("site_overrides 必须包含整数 index 和 moment")
        i = ov["index"]
        if not 0 <= i < pos.n_atoms or i in override_map:
            raise ValueError("site_overrides 索引越界或重复")
        override_map[i] = _finite_moment(ov["moment"])
    neigh = coordination_analysis(pos)
    ctx = site_context(pos, neigh)
    base = [0.0] * pos.n_atoms
    groups = [None] * pos.n_atoms
    missing = []
    for i, el in enumerate(pos.symbols):
        chosen = next((rule for rule in assignments
                       if rule["element"] == el
                       and _role_matches(rule.get("site_role"), ctx["roles"][i])
                       and (rule.get("coordination") is None or rule["coordination"] == len(neigh[i]))), None)
        if chosen is None:
            if i not in override_map:
                missing.append(i)
        else:
            base[i] = float(chosen["moment"])
            groups[i] = chosen.get("afm_group")
    if missing:
        raise ValueError(f"规则未覆盖位点 {missing[:20]}；请提供明确配位/表面规则或逐位点覆盖")
    if not isinstance(afm or [], (list, tuple)) or any(not isinstance(x, str) for x in afm or []):
        raise ValueError("afm_elements 必须是元素或组名列表")
    tokens = set(afm or [])
    known = set(pos.symbols) | {g for g in groups if g}
    if tokens - known:
        raise ValueError(f"未知 AFM 元素/组名: {sorted(tokens - known)}")
    members_by_group = {}
    for i, m in enumerate(base):
        if abs(m) < 1e-9:
            continue
        if afm_elementwise and not tokens:
            token = "__all_magnetic_sites__"
        elif groups[i] in tokens:
            token = groups[i]
        elif pos.symbols[i] in tokens:
            token = "__selected_elements__"
        else:
            continue
        members_by_group.setdefault(token, []).append(i)
    if tokens and not members_by_group:
        raise ValueError("AFM 选择没有非零磁性位点")
    if members_by_group and override_map:
        raise ValueError("自动 AFM 不接受 site_overrides；显式磁序请用 auto 并完整指定正负磁矩")
    for members in members_by_group.values():
        # 岩盐 3d 一氧化物使用指定的 II 型，而非按最大反号键比例猜磁序。
        if (report.get("motif", {}).get("family") == "rocksalt"
                and set(pos.symbols) in ({"Ni", "O"}, {"Mn", "O"}, {"Fe", "O"}, {"Co", "O"})):
            signs = rocksalt_type2_signs(pos, members)
        else:
            signs = bipartite_signs(pos, indices=members)
        for i, sign in signs.items():
            base[i] = abs(base[i]) * sign
    for i, m in override_map.items():
        base[i] = m
    return base


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
    return f"{x:.8g}"


def format_magmom(
    pos: Poscar, moments: Sequence[float], ispin: Optional[int] = None, compact: bool = True
) -> str:
    """生成与 add-spin.py 风格一致的 ISPIN / MAGMOM 文本块。"""
    if len(moments) != pos.n_atoms:
        raise ValueError("磁矩数量与原子数不一致")
    moments = [_finite_moment(m) for m in moments]
    ispin = resolve_ispin(moments, ispin)

    if ispin == 1:
        return (
            "# Mag parameter\n"
            "   ISPIN = 1\n"
            "   # 非自旋极化候选：本方案所有初始磁矩为零，无需 MAGMOM\n"
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
    elem_parts = []
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
        "# Mag parameter",
        f"   ISPIN = {ispin}",
        f"   {header}",
        f"   MAGMOM =  {magmom_line}",
    ]
    return "\n".join(lines) + "\n"


def append_to_incar(block: str, incar: str = "INCAR") -> None:
    """保留其余参数，替换已有共线自旋参数；拒绝覆盖不兼容的计算模式。"""
    old = ""
    if os.path.exists(incar):
        with open(incar, "r", encoding="utf-8") as f:
            old = f.read()
    kept = []
    logical_lines = []
    pending = ""
    for raw in old.splitlines():
        pieces = re.split(r"([#!].*)", raw, maxsplit=1)
        active = pieces[0].rstrip()
        if active.endswith("\\"):
            pending += active[:-1] + " "
            if len(pieces) > 1:
                logical_lines.append(pieces[1])
        else:
            logical_lines.append(pending + raw)
            pending = ""
    if pending:
        raise ValueError("INCAR 存在未结束的续行，拒绝修改")
    for line in logical_lines:
        pieces = re.split(r"([#!].*)", line, maxsplit=1)
        active, comment = pieces[0], "".join(pieces[1:])
        remaining = []
        for segment in active.split(";"):
            match = re.match(r"\s*([A-Za-z_]+)\s*=\s*(.*?)\s*$", segment)
            if not match:
                if segment.strip() and segment.strip() != "Mag parameter":
                    remaining.append(segment.strip())
                continue
            tag, value = match.group(1).upper(), match.group(2)
            if tag in ("LSORBIT", "LNONCOLLINEAR") and value.strip(". ").upper() in ("TRUE", "T"):
                raise ValueError(f"INCAR 启用了 {tag}，本工具仅支持共线磁性")
            if tag == "NUPDOWN":
                try:
                    constrained = float(value) >= 0
                except ValueError:
                    constrained = True
                if constrained:
                    raise ValueError("INCAR 已有固定总自旋 NUPDOWN；请先明确其与新磁矩方案的关系")
            if tag not in ("ISPIN", "MAGMOM"):
                remaining.append(segment.strip())
        # 清理本工具上次的标题，保留用户的其他注释。
        if comment.strip() == "# Mag parameter":
            comment = ""
        content = "; ".join(remaining)
        if comment:
            content += ("  " if content else "") + comment
        if content or not line.strip():
            kept.append(content)
    new = "\n".join(kept).rstrip() + "\n\n" + block
    # 所有校验先完成，再写同目录临时文件并原子替换，避免部分写入。
    import tempfile
    target = os.path.abspath(incar)
    fd, temporary = tempfile.mkstemp(prefix=".add-spin-", dir=os.path.dirname(target), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(new.lstrip("\n"))
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


# ============================================================================
# 9. 无 LLM 时的启发式回退方案
# ============================================================================
def _compatible_kb(pos: Poscar, motif: Dict, system_type: str) -> Optional[Dict]:
    """化学式只是候选检索键，结构类型不符时不自动采用该体相条目。"""
    if system_type in ("mole", "unknown"):
        return None
    kb = KNOWLEDGE_BASE.get(canonical_formula(pos.reduced_composition()))
    if not kb:
        return None
    family = motif.get("family")
    name = kb["name"]
    required = None
    if "尖晶石" in name:
        required = {"spinel"}
    elif "双钙钛矿" in name:
        required = {"double_perovskite"}
    elif any(el in kb["comp"] for el in ("La", "Sr", "Bi")) and kb["comp"].get("O") == 3:
        required = {"perovskite"}
    elif "刚玉" in name or (len(kb["comp"]) == 2 and kb["comp"].get("O") == 3):
        required = {"corundum"}
    elif "LDH" in name:
        required = {"ldh"}
    elif "水镁石" in name:
        required = {"hydroxide"}
    elif "金红石" in name:
        required = {"rutile"}
    elif len(kb["comp"]) == 1:
        required = {"metal"}
        if system_type != "bulk":
            return None  # 表面/团簇不能直接照搬体相零磁矩结论。
        if "bcc" in name and motif.get("metal_lattice") != "bcc":
            return None
        if "fcc" in name and "hcp" not in name and motif.get("metal_lattice") != "fcc":
            return None
    elif (kb["comp"].get("O") == 1
          and set(kb["comp"]) in ({"Ni", "O"}, {"Co", "O"}, {"Mn", "O"}, {"Fe", "O"})):
        required = {"rocksalt"}
    if required is not None and family not in required:
        return None
    return kb


def _ionic_seed_moments(comp: Dict[str, int]) -> Tuple[Dict[str, float], bool]:
    """只有一个未知阳离子时，以常见固定价和电中性求整数价态。"""
    fixed = {"O": -2, "F": -1, "Cl": -1, "Br": -1, "I": -1,
             "H": 1, "Li": 1, "Na": 1, "K": 1, "Rb": 1, "Cs": 1,
             "Be": 2, "Mg": 2, "Ca": 2, "Sr": 2, "Ba": 2, "Zn": 2,
             "Cd": 2, "Al": 3, "Ga": 3, "Sc": 3, "Y": 3, "La": 3, "Lu": 3}
    # S/Se/Te/N/P 可能组成多原子阴离子，不能统一当作单原子负价。
    unknown = [el for el in comp if el not in fixed]
    moments = {el: float(OXIDE_HS.get(el, default_element_moment(el))) for el in comp}
    for el in fixed:
        if el in moments:
            moments[el] = 0.0
    if len(unknown) == 1 and set(comp) & {"O", "F", "Cl", "Br", "I"}:
        el = unknown[0]
        ox = -sum(fixed[e] * n for e, n in comp.items() if e in fixed) / comp[el]
        if abs(ox - round(ox)) < 1e-8:
            ion = MAGNETIC_IONS.get(el, {}).get(int(round(ox)))
            if ion is not None:
                # 仅有组成不能确认晶体场；采用非零高自旋种子，低自旋需另行比较。
                moments[el] = float(ion[1])
                return moments, True
    if not unknown and sum(fixed[e] * n for e, n in comp.items()) != 0:
        raise ValueError("常见固定价模型不满足电中性，可能含过氧/超氧键或带电物种；请明确自旋与电荷")
    return moments, not unknown


def heuristic_plan(pos: Poscar, report: Optional[Dict] = None) -> Tuple[List[float], str]:
    """给出受限、可解释的共线种子，不能从几何唯一判定磁基态。"""
    if report is None:
        report = analyze_structure(pos)
    comp = dict(pos.composition())
    if report["system_type"] == "mole":
        if comp == {"O": 2}:
            distance = min_image_distances(pos, 2.0)
            if len(distance):
                return [1.0, 1.0], "[回退] O2 三重态自旋种子，总自旋磁矩 2 μB；未约束 NUPDOWN。"
        if comp == {"H": 2} and len(min_image_distances(pos, 1.2)):
            return [0.0, 0.0], "[回退] H2 成键单重态种子。"
        if pos.n_atoms == 1 and pos.symbols[0] in {"H", "He", "C", "N", "O", "F", "Ne", "Ar"}:
            spin = {"H": 1., "He": 0., "C": 2., "N": 3., "O": 2., "F": 1., "Ne": 0., "Ar": 0.}
            return [spin[pos.symbols[0]]], "[回退] 孤立原子自旋种子（不含轨道磁矩）。"
        raise ValueError("分子/团簇的电荷和自旋多重度未知，无法套用体相磁性；请提供明确磁矩方案")
    kb = _compatible_kb(pos, report["motif"], report["system_type"])
    if kb:
        if kb.get("afm"):
            allowed = ({"Ni", "O"}, {"Co", "O"}, {"Mn", "O"}, {"Fe", "O"},
                       {"La", "Fe", "O"}, {"La", "Cr", "O"}, {"Bi", "Fe", "O"}, {"Cr"})
            if set(comp) not in allowed and not (set(comp) == {"Co", "O"} and comp["Co"] * 4 == comp["O"] * 3):
                raise ValueError(f"{kb['name']} 的磁序不能由通用最近邻二染色确定；请明确给出位点磁矩")
        moments = build_moments_from_assignments(pos, report, kb["assignments"], afm=kb.get("afm", []))
        return moments, f"[回退] 结构筛选后的知识库候选：{kb['name']}。{kb.get('note', '')} 仍需比较不同磁序的总能。"
    if len(comp) == 1:
        el = next(iter(comp))
        if el in ("Cr", "Mn"):
            raise ValueError("Cr/Mn 磁序依赖晶型与磁性超胞，请提供明确位点磁矩")
        if report["system_type"] == "slab" and el not in METAL_MOMENT:
            raise ValueError("该单元素表面可能与体相磁性不同，请提供明确磁矩方案")
        m = METAL_MOMENT.get(el, 0.0)
        moments = [m] * pos.n_atoms
        reason = "纯元素的有限自旋种子；晶型与表面可能改变磁性"
    elif set(comp) & ANIONS:
        seeds, resolved = _ionic_seed_moments(comp)
        moments = [seeds[el] for el in pos.symbols]
        reason = ("按常见固定价和电中性得到的离子自旋种子（假设无过氧键、无缺陷电荷）" if resolved
                  else "价态/磁序不能唯一确定，使用元素高自旋同向种子；请另测 AFM/亚铁磁与低自旋候选")
    else:
        moments = [METAL_MOMENT.get(el, 0.0) for el in pos.symbols]
        reason = "合金的金属自旋种子，局域环境可能改变磁矩"
    return moments, "[回退] " + reason + "；初猜不代表磁基态。"


def resolve_ispin(moments: Sequence[float], requested: Optional[int] = None) -> int:
    if requested is not None and (type(requested) is not int or requested not in (1, 2)):
        raise ValueError("ISPIN 只能为整数 1 或 2")
    magnetic = any(abs(_finite_moment(m)) > 1e-9 for m in moments)
    if requested == 1 and magnetic:
        raise ValueError("ISPIN=1 与非零 MAGMOM 矛盾")
    return requested if requested is not None else (2 if magnetic else 1)


def plan_from_llm_args(pos: Poscar, report: Dict, args: Dict) -> Tuple[List[float], List[str]]:
    """校验完整方案后展开，非法/不相容的磁序不得提交。"""
    if not isinstance(args, dict):
        raise ValueError("磁矩方案必须是对象")
    order = str(args.get("magnetic_order") or "auto").lower()
    aliases = {"ferromagnetic": "fm", "ferro": "fm", "antiferromagnetic": "afm",
               "antiferro": "afm", "ferri": "ferrimagnetic", "nm": "nonmagnetic",
               "non-magnetic": "nonmagnetic", "none": "nonmagnetic"}
    order = aliases.get(order, order)
    if order not in ("auto", "fm", "afm", "ferrimagnetic", "nonmagnetic"):
        raise ValueError(f"未知 magnetic_order: {order}")
    afm = args.get("afm_elements") or []
    if afm and order in ("fm", "ferrimagnetic", "nonmagnetic"):
        raise ValueError("afm_elements 与指定 magnetic_order 矛盾")
    moments = build_moments_from_assignments(
        pos, report, args.get("assignments", []), args.get("site_overrides"), afm,
        afm_elementwise=(order == "afm"))
    if order == "nonmagnetic" and any(abs(m) > 1e-9 for m in moments):
        raise ValueError("nonmagnetic 方案必须明确为所有位点赋零磁矩")
    if order == "nonmagnetic" and args.get("ispin") not in (None, 1):
        raise ValueError("nonmagnetic 方案要求 ISPIN=1")
    if order == "fm":
        moments = [abs(m) for m in moments]
    if order in ("afm", "ferrimagnetic") and not (any(m > 1e-9 for m in moments) and any(m < -1e-9 for m in moments)):
        raise ValueError("AFM/亚铁磁方案必须包含正负两类非零磁矩")
    resolve_ispin(moments, args.get("ispin"))
    warnings = validate_moments(moments, pos.n_atoms)
    if order == "afm" and abs(sum(moments)) > 1e-6:
        warnings.append("AFM 候选的初始总磁矩不为零；请检查子晶格数目、磁矩大小和表面不补偿")
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
                            "也可直接写元素符号。自动校验周期磁序，无法确定时返回错误。"
                        ),
                    },
                    "magnetic_order": {
                        "type": "string",
                        "enum": ["auto", "fm", "afm", "ferrimagnetic", "nonmagnetic"],
                        "description": (
                            "整体磁序：fm=铁磁(全部取正)，afm=在共同磁性亚晶格上检查周期相容性并赋号，"
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
                    "ispin": {"type": "integer", "enum": [1, 2], "description": "省略时：全零为1，非零为2"},
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
# 11. 工具运行时
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
            try:
                result = self._apply(args)
            except (ValueError, TypeError, KeyError) as exc:
                return {"ok": False, "error": str(exc)}
            if name == "submit_magmom" and result.get("ok"):
                self.submitted = args
            return result
        return {"ok": False, "error": f"未知工具: {name}"}

    def _apply(self, args: Dict) -> Dict:
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

        return {
            "ok": True,
            "system_type": self.report.get("system_type", "bulk"),
            "per_site_group": pretty,
            "n_atoms": self.pos.n_atoms,
            "warnings": warnings,
            "uncovered_magnetic_sites": [],  # 完整覆盖已由 plan_from_llm_args 强制检查。
            "ispin": resolve_ispin(moments, args.get("ispin")),
            "initial_total_moment": round(sum(moments), 6),
            "note": "warnings 非空时请修正方案后重新调用。",
        }


def element_info(symbol: str) -> Dict:
    symbol = (symbol or "").strip().capitalize()
    ions = MAGNETIC_IONS.get(symbol, {})
    shell = "f" if symbol in {
        "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
        "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
        "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm",
    } else "d"
    ion_table = {
        f"{symbol}{ox}+": {
            "unpaired_electrons_high_spin": hs,
            "unpaired_electrons_low_spin": ls,
            f"{shell}_electrons": d,
        }
        for ox, (d, hs, ls) in ions.items()
    }
    return {
        "symbol": symbol,
        "atomic_mass": ATOMIC_MASS.get(symbol),
        "covalent_radius": RCOV.get(symbol),
        "common_oxidation_states": COMMON_OXIDATION.get(symbol, []),
        "magnetic_ions": ion_table,
        "electron_shell": shell if ions else None,
        "spin_count_convention": (
            "f 壳层按 Hund 自旋计数，磁矩为 2S 的初猜；"
            "不含轨道/SOC，不能视作实验有效磁矩（例如 Eu3+ 的 SOC 基态 J=0）。"
            if shell == "f" else
            "d 壳层 HS/LS 指理想八面体晶体场，初猜为 2S 而非 sqrt(n(n+2))；"
            "平方平面 d8 可为 S=0，须另行判断。"
        ),
        "default_guess": default_element_moment(symbol),
        "nonmagnetic": symbol in NONMAGNETIC,
        "nonmagnetic_scope": "仅为常见闭壳层化合物的兜底，不适用于孤立原子、自由基或缺陷。",
        "metal_moment": METAL_MOMENT.get(symbol),
    }


# ============================================================================
# 12. 用户提示词
# ============================================================================
def build_user_prompt(pos: Poscar, poscar_text: str, hint: str = "") -> str:
    text = poscar_text
    if len(text) > 8000:
        text = text[:8000] + "\n... (POSCAR 已截断，完整信息以 analyze_structure 工具结果为准)"
    prompt = (
        "请为下面这个 POSCAR 设置一致的 ISPIN / MAGMOM 初猜，"
        "重点处理过渡金属的价态与晶体学位点、以及可能的反铁磁/亚铁磁序。\n\n"
        f"POSCAR 内容：\n```\n{text}\n```\n"
    )
    if hint:
        prompt += f"\n用户补充说明：{hint}\n"
    prompt += "\n请先调用 analyze_structure。"
    return prompt


# ============================================================================
# 13. OpenAI 兼容客户端（仅标准库 urllib）
# ============================================================================
class LLMClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = LLM_BASE_URL,
        model: str = LLM_MODEL,
        timeout: int = LLM_TIMEOUT,
        temperature: float = LLM_TEMPERATURE,
        max_retries: int = LLM_MAX_RETRIES,
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
        or LLM_API_KEY
    )
    base_url = (
        args.base_url
        or os.environ.get("LLM_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or LLM_BASE_URL
    )
    model = (
        args.model
        or os.environ.get("LLM_MODEL")
        or LLM_MODEL
    )
    if not api_key:
        raise RuntimeError(
            "未找到 API Key。请设置 LLM_API_KEY / OPENAI_API_KEY / DEEPSEEK_API_KEY，"
            "或用 --api-key 指定；也可以用 --no-llm 走内置启发式。"
        )
    return LLMClient(api_key, base_url=base_url, model=model,
                     temperature=args.temperature, timeout=args.timeout,
                     max_retries=args.max_retries)


# ============================================================================
# 14. Agent 主循环
# ============================================================================
def run_agent(
    client: LLMClient,
    runtime: ToolRuntime,
    user_prompt: str,
    max_steps: int = LLM_MAX_STEPS,
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
                result = runtime.call("submit_magmom", parsed)
                if result.get("ok"):
                    return runtime.submitted
                messages.append({"role": "user", "content": "方案校验失败，请修正：" + str(result.get("error"))})
                continue
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
# 15. 输出 / CLI
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
        ispin = resolve_ispin(moments, plan.get("ispin"))
        rationale = plan.get("rationale", "")
        source = "LLM 工具调用"
    else:
        moments, rationale = heuristic_plan(pos, runtime.report)
        warnings = validate_moments(moments, pos.n_atoms)
        ispin = resolve_ispin(moments)
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
        append_to_incar(block, incar)
        print(f"已更新自旋参数: {incar}")

    return block


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="用 LLM 工具调用为 POSCAR 生成 VASP MAGMOM 初猜"
    )
    ap.add_argument("poscar", nargs="?", default="POSCAR", help="POSCAR 文件路径")
    ap.add_argument("--incar", default="INCAR", help="要更新自旋参数的 INCAR（默认 INCAR）")
    ap.add_argument("--print", dest="do_print", action="store_true", help="只打印不写文件")
    ap.add_argument("--no-llm", action="store_true", help="不调用 LLM，直接用内置启发式")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--temperature", type=float, default=LLM_TEMPERATURE)
    ap.add_argument("--timeout", type=float, default=LLM_TIMEOUT, help="单次 LLM 请求超时秒数")
    ap.add_argument("--max-retries", type=int, default=LLM_MAX_RETRIES)
    ap.add_argument("--max-steps", type=int, default=LLM_MAX_STEPS)
    ap.add_argument("--hint", default="", help="给 LLM 的补充说明（如已知价态/磁性）")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--vacuum-threshold", type=float, default=5.0,
                    help="bulk/slab 判别的真空层阈值 (Å)，默认 5.0")
    args = ap.parse_args(argv)
    if not math.isfinite(args.temperature) or not 0 <= args.temperature <= 2:
        ap.error("--temperature 必须在 0 到 2 之间")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        ap.error("--timeout 必须为正数")
    if args.max_retries < 1 or args.max_steps < 1:
        ap.error("--max-retries 和 --max-steps 必须为正整数")
    if not math.isfinite(args.vacuum_threshold) or args.vacuum_threshold <= 0:
        ap.error("--vacuum-threshold 必须为正数")
    return args


def _main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    global _ACTIVE_VACUUM_THRESHOLD
    _ACTIVE_VACUUM_THRESHOLD = args.vacuum_threshold
    if not os.path.exists(args.poscar):
        print(f"找不到 POSCAR: {args.poscar}", file=sys.stderr)
        return 2

    pos = parse_poscar_full(args.poscar)
    runtime = ToolRuntime(pos)
    with open(args.poscar, "r", encoding="utf-8") as f:
        poscar_text = f.read()
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


def main(argv: Optional[List[str]] = None) -> int:
    try:
        return _main(argv)
    except (ValueError, OSError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    # Windows 的默认代码页可能无法输出 Å/μB；管道和终端统一使用 UTF-8。
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
