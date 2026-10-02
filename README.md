# add-spin.py

从 POSCAR 生成 VASP 共线自旋计算的 `ISPIN` 与逐原子 `MAGMOM` 初猜。脚本解析晶格、坐标和元素顺序，分析周期配位、真空方向与位点角色，再结合材料知识库、形式价态或可选的 LLM 工具调用生成方案。

`MAGMOM` 是初始化参数，不能保证收敛到磁基态。比较 FM、AFM、亚铁磁和不同自旋态时，应分别开展一致参数下的 DFT 计算并比较收敛能量；LLM 的建议也需要同样验证。

运行依赖为 Python ≥ 3.9 与 NumPy。HTTP 请求使用 Python 标准库，无需安装 OpenAI SDK。

## 快速开始

```bash
# 查看帮助
python add-spin.py --help

# 只分析并打印，不写 INCAR（离线启发式）
python add-spin.py POSCAR --no-llm --print

# 调用 LLM 生成方案并更新 INCAR（需要 API Key）
export LLM_API_KEY=...
python add-spin.py POSCAR --incar INCAR

# 补充已知价态/磁序信息（仅传给 LLM）
python add-spin.py POSCAR --hint "Ni 为 +2 价，反铁磁"
```

## LLM 配置

连接与运行默认配置集中在 `add-spin.py` 头部的 `LLM_*` 常量中，紧随其后的 `SYSTEM_PROMPT` 定义 LLM 的分析与提交约束：

| 常量 | 默认值 | 用途 |
|---|---|---|
| `LLM_API_KEY` | 空字符串 | OpenAI 兼容服务的 API Key |
| `LLM_BASE_URL` | `https://api.deepseek.com/v1` | 服务地址 |
| `LLM_MODEL` | `deepseek-chat` | 模型名称 |
| `LLM_TEMPERATURE` | `0.2` | 采样温度 |
| `LLM_TIMEOUT` | `180` | 单次请求的超时秒数 |
| `LLM_MAX_RETRIES` | `3` | 每轮请求的最大总尝试次数（含首次） |
| `LLM_MAX_STEPS` | `8` | 工具调用最大轮数 |

API Key、服务地址和模型名称的优先级为 **命令行参数 > 环境变量 > 脚本头部常量**。环境变量按下列顺序取第一个非空值：

| 配置 | 环境变量 |
|---|---|
| API Key | `LLM_API_KEY`、`OPENAI_API_KEY`、`DEEPSEEK_API_KEY` |
| 服务地址 | `LLM_BASE_URL`、`OPENAI_BASE_URL` |
| 模型 | `LLM_MODEL` |

温度、超时、重试和最大轮数直接使用头部常量，可由对应命令行参数覆盖。建议用环境变量保存密钥，避免将真实密钥写入版本库。使用 `--no-llm` 时仅运行本地分析；没有可用 API Key 或 LLM 调用失败时，程序尝试本地回退，但无法可靠构造方案时会报错。

LLM 通过 `analyze_structure`、`get_element_info`、`lookup_known_material`、`preview_magmom` 和 `submit_magmom` 调用结构分析与方案校验工具。规则展开和原子顺序校验由脚本完成。

## 命令行参数

命令入口为 `python add-spin.py`，完整帮助可通过 `--help` 查看。

| 参数 | 说明 | 默认值 |
|---|---|---|
| `poscar` | POSCAR 路径，位置参数 | `POSCAR` |
| `--incar FILE` | 要更新的 INCAR 路径 | `INCAR` |
| `--print` | 只打印参数，不写入 INCAR | 关闭 |
| `--no-llm` | 使用本地知识库与启发式 | 关闭 |
| `--hint TEXT` | 传给 LLM 的已知价态、磁序等补充信息 | 空 |
| `--api-key KEY` | 覆盖 API Key | 按配置优先级读取 |
| `--base-url URL` | 覆盖服务地址 | 按配置优先级读取 |
| `--model NAME` | 覆盖模型名称 | 按配置优先级读取 |
| `--temperature FLOAT` | 覆盖采样温度 | `LLM_TEMPERATURE` |
| `--timeout SECONDS` | 覆盖单次请求超时 | `LLM_TIMEOUT` |
| `--max-retries N` | 覆盖每轮请求最大总尝试次数 | `LLM_MAX_RETRIES` |
| `--max-steps N` | 覆盖工具调用最大轮数 | `LLM_MAX_STEPS` |
| `--vacuum-threshold ANGSTROM` | 真空判别阈值，单位 Å | `5.0` |
| `--quiet` | 减少 LLM 调用过程输出 | 关闭 |
| `-h, --help` | 显示帮助 | — |

`--hint` 仅供 LLM 使用，不会自动转成离线规则或强制约束。

## 磁性设置逻辑

- **结构与化学环境**：按 POSCAR 中的原子顺序解析，保留重复元素块。配位分析计入周期镜像；体相、表面和分子的判别以及结构基元识别均为几何启发式，不能替代完整晶体学鉴定。
- **磁矩幅值**：已知材料提供初猜参考。未知化合物仅在候选形式价态满足电荷中性且具有唯一解时使用对应离子自旋计数；价态不能唯一确定时，采用元素经验初猜，并明确磁序与价态尚未确定，不自动宣称 AFM 基态。
- **自旋数据含义**：d 壳层高、低自旋表对应理想八面体晶体场；平方平面等配位需要另行判断。初猜使用自旋磁矩 `2S`，不使用顺磁有效磁矩 `sqrt(n(n+2))`。f 壳层数据是自旋计数，不包含 SOC 与轨道磁矩。
- **位点匹配**：按元素、可选配位数和位点角色严格匹配。配位降低的表面原子不会自动继承某个体相位点；无法匹配时应补充明确规则或逐原子覆盖，避免把低配位八面体误认成四面体。
- **AFM 符号**：自动二染色仅适用于当前周期晶胞的相容近邻图。岩盐 AFM-II 需满足相应磁周期约束。晶胞不能容纳该磁序、近邻图存在挫折或缺少磁序信息时，会要求构造合适超胞或提供明确的位点磁矩，不用任意正负交替冒充目标磁序。
- **亚铁磁**：使用明确位点规则的正负号，保留不同亚晶格的相对方向。Fe₃O₄ 八面体 `4.5 μB` 是 Fe²⁺/Fe³⁺ 的平均初猜，不表示已解析低温电荷有序。
- **分子**：不直接套用体相材料知识库。H₂ 和 O₂ 有专门处理；其他无法可靠确定自旋的分子或团簇需要明确的自旋信息，否则报错。

形式价态模型无法唯一描述共价性、混合价、电荷转移、缺陷、电荷有序或强关联自旋态。几何近邻关系也不能单独决定交换耦合符号。因此局域磁矩与磁序候选仍需根据具体研究对象确认。

## 输出与校验

默认全零磁矩输出 `ISPIN=1`，有非零磁矩输出 `ISPIN=2`；调用接口时也可显式保留全零 `ISPIN=2` 种子。程序检查逐原子磁矩数量、数值有限性、位点索引及规则覆盖，并保持输出与 POSCAR 的原子顺序一致。`ISPIN=1` 与非零磁矩等矛盾设置不会直接写出。

写入时更新已有 `ISPIN` / `MAGMOM`，清理重复的同名设置，保留 INCAR 的其他参数。检测到启用的 `LSORBIT`、`LNONCOLLINEAR` 或 `NUPDOWN` 约束时，拒绝将本工具的共线初猜直接写入，以免改变既有计算的含义。

本工具不生成非共线三分量磁矩、SOC 磁各向异性方案、自旋螺旋或磁性超胞。需要这些计算时，应按目标磁结构单独构造输入。

## 开发验证

回归测试命令：

```bash
python -m unittest discover -s tests -v
```

测试分三部分，均不联网：

- `tests/test_add_spin.py`：POSCAR 解析（缩放、Cartesian/Direct、异常输入）、slab 分层与真空判据、磁矩方案校验、INCAR 更新与反铁磁周期相容性。
- `tests/test_geometry_motif.py`：配位壳层与结构基元识别（bcc、金刚石、岩盐、钙钛矿、刚玉及尖晶石判据等）。
- `tests/test_physics_and_cli.py`：离子自旋计数、分子特例、LLM 配置优先级与命令行失败路径。

主程序为 `add-spin.py`，测试位于 `tests`。旧模块接口 `parse_poscar`、`build_magmom`、`append_magmom` 保留兼容用途；只有元素数量的输入无法提供完整晶体环境，复杂磁性体系应使用完整 POSCAR 分析流程。

许可证：[MIT](LICENSE)。
