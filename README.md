<div align="center">

# 🧲 add-spin.py

**从 POSCAR 一键生成 VASP 磁矩初猜（`ISPIN` / `MAGMOM`）**

*LLM 工具调用 · bulk/slab 判别 · 晶体学位点磁性识别 · 铁磁/亚铁磁/反铁磁*

[![Python](https://img.shields.io/badge/Python-%E2%89%A53.9-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![VASP](https://img.shields.io/badge/VASP-MAGMOM-1f6feb.svg)](https://www.vasp.at/)
[![LLM](https://img.shields.io/badge/LLM-function%20calling-8A2BE2.svg)](#-llm-工具调用)
[![Dependencies](https://img.shields.io/badge/deps-numpy%20only-success.svg)](https://numpy.org/)
[![Single File](https://img.shields.io/badge/single--file-add--spin.py-blue.svg)](#-文件说明)

</div>

---

## ✨ 为什么需要它

`MAGMOM` 初猜直接决定 VASP 能否收敛到正确的磁基态，尤其是**尖晶石/反尖晶石、
钙钛矿、LDH、以及切面 slab**。旧式“一个元素一个固定值”的做法在复杂体系里经常出错：

| 体系 | ❌ 按元素拍脑袋 | ✅ add-spin.py |
|---|---|---|
| 纯金属 **Pt** | 给 3 | 给 **0**（体相非磁） |
| **Fe₃O₄** 反尖晶石 | 所有 Fe 都 +5 | 8a Fe **+5**、16d Fe **−5**（亚铁磁） |
| **NiO** 岩盐 | 所有 Ni 都 +3 | Ni 亚晶格**正负交替**（II 型反铁磁） |
| **MgAl₂O₄** 正尖晶石 | Al 也给 3 | **全 0** |
| **Co₃O₄** | Co 全 +5 | 8a Co²⁺ ±3、16d Co³⁺ 低自旋 0 |
| **NiO(001) slab** | 表面原子照抄体相 | 识别**表面/次表面**，表面 Ni 单独赋值 |
| **ZnS / ZnO** | 乱给磁矩 | 判别闪锌矿/纤锌矿，**非磁 0** |
| **LaFeO₃** 钙钛矿 | 所有原子都给值 | 仅 B 位 Fe 有磁矩，A 位 La = 0 |
| **NiFe-LDH** | 分不清层板/层间 | 仅层板八面体金属有磁矩，层间物种 = 0 |

---

## 📚 目录

- [✨ 为什么需要它](#-为什么需要它)
- [🚀 快速开始](#-快速开始)
- [🧰 命令行](#-命令行)
- [🔍 工作原理](#-工作原理)
  - [体系类型判别 bulk / slab / molecule](#体系类型判别-bulk--slab--molecule)
  - [结构基元识别](#结构基元识别)
  - [反铁磁 / 亚铁磁符号](#反铁磁--亚铁磁符号)
- [🤖 LLM 工具调用](#-llm-工具调用)
- [🐍 作为 Python 模块](#-作为-python-模块)
- [🧪 自检与示例](#-自检与示例)
- [📁 文件说明](#-文件说明)
- [⚠️ 注意事项](#️-注意事项)
- [📄 许可](#-许可)

---

## 🚀 快速开始

```bash
# 查看全部参数
python add-spin.py --help

# 用 LLM 分析 POSCAR，并追加写入 INCAR
export DEEPSEEK_API_KEY=sk-xxxx        # 或 OPENAI_API_KEY / LLM_API_KEY
python add-spin.py POSCAR

# 只打印，不写文件
python add-spin.py POSCAR --print

# 不联网：内置知识库 + 晶体学启发式
python add-spin.py POSCAR --no-llm --print

# 补充已知信息（会拼进提示词）
python add-spin.py POSCAR --hint "Fe 为 +3 价，体系是 G 型反铁磁"

# 任意 OpenAI 兼容服务
python add-spin.py POSCAR --base-url https://api.deepseek.com/v1 --model deepseek-chat

# 内置物理自检 / 导出示例结构
python add-spin.py --self-test
python add-spin.py --make-examples ./examples
```

> **依赖**：Python ≥ 3.9 + `numpy`。HTTP 调用使用标准库，**无需 openai SDK**。

---

## 🧰 命令行

| 参数 | 说明 | 默认 |
|:--|:--|:--|
| `poscar` | POSCAR 路径（位置参数） | `POSCAR` |
| `--incar FILE` | 追加写入的 INCAR | `INCAR` |
| `--print` | 只打印，不写 INCAR | — |
| `--no-llm` | 不调用 LLM，直接用内置启发式 | — |
| `--hint TEXT` | 给 LLM 的补充说明（价态、磁序等） | — |
| `--api-key` / `--base-url` / `--model` / `--temperature` | LLM 配置 | 环境变量 |
| `--max-steps` | LLM 工具调用最大轮数 | `8` |
| `--vacuum-threshold` | bulk/slab 判别的真空层阈值 (Å) | `5.0` |
| `--quiet` | 不打印 LLM 过程 | — |
| `--self-test` | 运行内置自检后退出 | — |
| `--make-examples DIR` | 导出内置示例 POSCAR 后退出 | — |
| `-h, --help` | 显示帮助 | — |

### 环境变量

| 变量 | 说明 | 默认 |
|:--|:--|:--|
| `LLM_API_KEY` / `OPENAI_API_KEY` / `DEEPSEEK_API_KEY` | API Key（任一） | — |
| `LLM_BASE_URL` / `OPENAI_BASE_URL` | OpenAI 兼容地址 | `https://api.deepseek.com/v1` |
| `LLM_MODEL` | 模型名 | `deepseek-chat` |

### 输出示例 · NiO(001) slab

```text
Mag parameter
   ISPIN = 2
   # Ni(1)=2.5  O(1)=0  Ni(1)=-2  O(1)=0  Ni(1)=2  O(1)=0
   MAGMOM =  2.5   0   -2   0   2   0
```

---

## 🔍 工作原理

### 体系类型判别 bulk / slab / molecule

判别算法与 [`mk-KPOINTS`](https://github.com/moyulyy/mk-KPOINTS) 一致：

1. 对 a / b / c 三方向，把分数坐标排序，求**最大周期空隙**（分数）；
2. 空隙 × 晶格长度 = **真空层厚度**（Å）；
3. 真空层 > 阈值（默认 5 Å）即认为该方向存在真空：

| 条件 | 类型 |
|:--|:--|
| a、b、c 都有真空 | `mole` |
| 只有 c 有真空 | `slab` |
| 都无真空 | `bulk` |

对 `slab` 进一步**几何分层**：由最大真空轴求表面法向 → 投影原子 → 聚类成原子层 →
标记每位点的 `role`（`surface_top` / `surface_bottom` / `subsurface` / `interior`），
并用 `cn_deficit`（该元素最大配位 − 本位点配位）标出配位不饱和位点。

### 结构基元识别

对**最大配位数**（而非表面瞬时配位）做判据，因此同一套逻辑对 bulk 与 slab 都成立：

| family | 判据 | 磁性处理要点 |
|:--|:--|:--|
| `metal` | 单元素 | Fe/Co/Ni 铁磁；Cr/Mn 反铁磁；Pt/Pd/Cu/Ag/Au/Al 非磁 |
| `spinel` | A:B:O = 1:2:4 | 按 8a(CN4)/16d(CN6) 分别赋值；正/反尖晶石自动判别 |
| `perovskite` | ABO₃ | B 位(CN6) 有磁矩，A 位(CN≥8) = 0 |
| `double_perovskite` | A₂BB′O₆ | B/B′ 常反平行（亚铁磁），如 Sr₂FeMoO₆ |
| `rocksalt` | AO | MnO/FeO/CoO/NiO 反铁磁 |
| `zincblende` / `wurtzite` | AB，阳离子 CN4 | ZnS/ZnO/GaAs 非磁；MnS/MnSe/MnTe 反铁磁 |
| `ldh` / `hydroxide` | 含 H+O 的层板/水镁石 | 仅层板八面体金属有磁矩，层间物种 = 0 |
| `corundum` / `rutile` | A₂O₃ / MO₂ | Cr₂O₃/Fe₂O₃ 反铁磁；CrO₂ 铁磁 |

### 反铁磁 / 亚铁磁符号

- **二部图晶格**（bcc、金刚石等）：3×3×3 超胞近邻图二染色；
- **非二部图晶格**（fcc 岩盐 NiO、尖晶石八面体亚晶格）：**磁层投影**自动挑选方向，
  按层号奇偶给 `±`（即 NiO 的 (111) 面内铁磁、面间反铁磁）；
- **亚铁磁**（Fe₃O₄、NiFe₂O₄…）：直接给“8a 与 16d 整体反号”，不在 16d 内部交替；
- 每个位点只应用一次符号，避免表面/内部规则互相覆盖。

---

## 🤖 LLM 工具调用

程序把结构分析封装成 5 个 function-calling 工具：

| 工具 | 作用 |
|:--|:--|
| `analyze_structure()` | 体系类型、真空层、slab 分层、每位点配位数/几何/role、结构基元、知识库匹配 |
| `get_element_info(symbol)` | 元素各价态/自旋态的未成对电子数与默认初猜 |
| `lookup_known_material(formula_or_name)` | 内置材料知识库（50+ 条） |
| `preview_magmom(...)` | 试算方案（返回按 `元素(CN,role)` 汇总 + 告警） |
| `submit_magmom(...)` | 提交最终方案 |

模型只需给出「按 **元素 + 配位数 + 位点角色** 的赋值规则」，由脚本展开到每个原子，避免顺序错位：

```jsonc
// 尖晶石 / 双钙钛矿 → 亚铁磁
{
  "assignments": [
    {"element": "Fe", "coordination": 4, "moment":  5.0},   // 8a 四面体
    {"element": "Fe", "coordination": 6, "moment": -5.0},   // 16d 八面体
    {"element": "O",  "moment": 0.0}
  ],
  "magnetic_order": "ferrimagnetic"
}

// 岩盐反铁磁 → 自动正负交替
{ "assignments": [{"element":"Ni","moment":2.0},{"element":"O","moment":0.0}],
  "magnetic_order": "afm" }

// slab：表面积配位不饱和位单独赋值
{
  "assignments": [
    {"element":"Ni","site_role":"surface","moment": 2.5},
    {"element":"Ni","moment": 2.0},
    {"element":"O","moment": 0.0}
  ],
  "magnetic_order": "afm"
}
```

<details>
<summary><b>字段说明</b>（点击展开）</summary>

- `assignments`：`element`（必需）、`coordination`、`site_role`、`moment`、`afm_group`
- `site_role`：`any` / `surface` / `top_surface` / `bottom_surface` / `subsurface` /
  `interior` / `bulk` / `cluster`
- `magnetic_order`：`fm` / `afm` / `ferrimagnetic` / `nonmagnetic`
- `submit_magmom` 还支持 `site_overrides`（单点覆盖）、`ispin`、`rationale`
- **表面回退**：表面低配位会自动继承该元素最接近的体相“母配位数”规则，只写体相 CN 即可

</details>

---

## 🐍 作为 Python 模块

```python
import importlib.util

spec = importlib.util.spec_from_file_location("add_spin", "add-spin.py")
add_spin = importlib.util.module_from_spec(spec)
spec.loader.exec_module(add_spin)

pos = add_spin.parse_poscar_full("POSCAR")      # 完整解析
report = add_spin.analyze_structure(pos)
print(report["system_type"], report["slab"])     # bulk / slab / mole、分层信息
print(report["motif"]["name"], report["site_groups"])

moments, why = add_spin.heuristic_plan(pos, report)
print(add_spin.format_magmom(pos, moments))
```

<details>
<summary><b>旧版 API 仍保留</b>（向后兼容 fast-vasp）</summary>

```python
pairs = add_spin.parse_poscar("POSCAR")   # -> [(元素, 数量), ...]
block = add_spin.build_magmom(pairs)
add_spin.append_magmom("POSCAR", "INCAR")
```

</details>

---

## 🧪 自检与示例

```bash
python add-spin.py --self-test                  # 10 组结构物理自检
python add-spin.py --make-examples ./examples   # 导出示例 POSCAR
```

覆盖：Pt = 0、Fe 铁磁、NiO 反铁磁、NiO(001) slab 判别与表面磁矩、ZnS 闪锌矿非磁、
ZnO 纤锌矿非磁、LaFeO₃ 钙钛矿、NiAl-LDH 层板磁性、Fe₃O₄ 反尖晶石、MgAl₂O₄ 正尖晶石。

---

## 📁 文件说明

```text
add-spin.py    单文件实现（解析 + 分析 + 知识库 + LLM 工具调用 + CLI）
README.md      本文档
LICENSE        MIT
```

---

## ⚠️ 注意事项

- `MAGMOM` 只是**初猜**，最终磁矩由 VASP 自洽收敛决定；目标是帮助收敛到正确磁基态。
- 体/表面判别沿用 `mk-KPOINTS` 的“最大周期空隙”算法；极少数原子晶胞的空隙会偏大，
  可用 `--vacuum-threshold` 调整。
- 表面金属磁矩增强只是经验性建议（约 +10%~30%），氧化物通常保持价态不变。
- 写入 INCAR 为**追加**模式；若已有 `ISPIN/MAGMOM`，程序会提示，请先清理旧参数。
- 非共线磁、SOC、自旋螺旋等需要 `LNONCOLLINEAR/SAXIS`，不在本工具范围内。
- 内置启发式只是 LLM 不可用时的兜底，精度不如 LLM + 知识库，请优先使用 LLM 模式。

---

## 📄 许可

[MIT](LICENSE) © 2026 lyy
