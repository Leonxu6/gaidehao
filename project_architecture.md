---
name: ScholAI 技术架构（已审查通过v2）
description: 经过3轮AgentTeam迭代、PM严格审查通过的完整技术架构
type: project
---

架构文档位置：项目目录下的 技术架构.md（完整版）
调研报告位置：项目目录下的 技术调研报告.md

**核心Pipeline（6步）：**
1. 命题提取：Qwen2.5-7B提取核心命题列表（JSON格式，全程追踪）
2. 语义锚点：BGE-M3全文嵌入（强制CPU模式）
3. 概率层重写：Qwen2.5-7B，temperature=0.9/top_p=0.85/min_p=0.05，Mirostat干预
4. 语义验证（三层）：BGE-M3余弦粗筛(≥0.80) → NLI命题精验 → BERTScore辅助
5. 质量控制（两轨分离）：检测规避指标(PPL/Burstiness/TTR) ‖ 语言质量(G-Eval 4维加权)
6. 连贯性修复：概率层重新生成，禁止任何文字编辑

**关键技术选型：**
- 推理框架：llama.cpp + llama-cpp-python（唯一支持Mirostat的轻量框架）
- 主模型：Qwen2.5-7B Q4_K_M（中文首选）
- NLI验证：nli-deberta-v3-xsmall（70MB，CPU，专门检测命题极性反转）
- 嵌入：BAAI/bge-m3（CPU强制运行）

**G-Eval规格（最终版）：**
- 维度权重：忠实度35% + 连贯性25% + 流畅性25% + 可读性15%
- 参数：temperature=0.0，单次采样，overall≥3.5通过
- 实现：复用Qwen2.5实例，零额外内存

**硬件：** Mac Mini M4 16GB（开发机）
- CPU推理方案（无独立GPU）
- Qwen2.5-7B Q4_K_M：CPU推理约8-14 tok/s（M4芯片Metal加速更快，预计20-40 tok/s）
- 内存：模型总占用约10-12GB，16GB可行

**成本：**
- 本地推理：~0.002元/次
- 第三方API备选（Qwen-Plus）：~0.004元/次

**Why:** 架构经过AgentTeam（架构师+PM审查员+调研员）3轮迭代审查通过，所有P0/P1/P2问题已修复。
**How to apply:** 开发时严格按照技术架构.md实现，尤其Step 5两轨指标职能不能混用。
