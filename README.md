# 后训练学习与复现

这个文件夹用于收集后训练相关的开源仓库，并保存自己的代码阅读笔记和小实验。当前从 **ToolRL** 开始。

## 从这里开始

- [ToolRL 学习笔记](notes/ToolRL.md)：原理理解、代码阅读、练习结果和复现记录。
- [学习规划](notes/学习规划.md)：后续候选方向与阶段安排。

## 文件放在哪里

```text
Agent-post-train/
├── README.md                     # 学习入口和项目索引
├── AGENTS.md                     # AI 助手的协作规则
├── .envs/                        # 各 repo 的本地独立环境，按需创建
├── repos/
│   └── ToolRL/                   # 官方代码，独立 Git 仓库
└── notes/
    ├── 学习规划.md                # 整体学习路线
    └── ToolRL.md                 # ToolRL 的全部学习记录
```

`repos/` 放上游代码，`notes/` 下**一个项目只维护一个 MD 文件**。原理理解、代码对应、问题、练习和复现记录都写在同一个文件里，随学习进度持续更新。新增项目时只增加对应的仓库和笔记，不预建多层目录。

## 下载到其他电脑

本仓库保存自己的学习材料，ToolRL 通过 Git 子模块引用官方代码并固定版本。首次下载时，需要一并取回子模块：

```bash
git clone --recurse-submodules https://github.com/Ricardo-Ssy/Agent-post-train.git
```

如果已下载主仓库，或拉取了新的学习进度，在工作区根目录执行下面的命令，让上游代码与记录的版本一致：

```bash
git submodule update --init --recursive
```

本仓库公开，可直接浏览和下载。ToolRL 内部的代码改动需要在其独立仓库中管理；主仓库记录子模块版本，不会自动保存子模块内尚未提交的改动。

## 独立环境

各 repo 的环境统一放在工作区根目录的 `.envs/<项目名>/`，例如 ToolRL 使用 `.envs/toolrl/`。不同 repo 不共用环境，只有实际运行项目时才创建对应环境。当前仅保留目录占位文件，尚未安装训练依赖。

建议使用 Conda 按路径创建环境，便于各 repo 使用不同的 Python 版本。先安装 Conda，并根据上游依赖要求确定 Python 版本；在工作区根目录执行以下命令，将 `X.Y` 替换为所需版本：

```bash
conda create --prefix "$PWD/.envs/toolrl" python=X.Y
conda activate "$PWD/.envs/toolrl"
```

激活后，进入 `repos/ToolRL/`，按上游说明安装依赖；其他 repo 使用各自的环境目录。Python、PyTorch、CUDA 等版本需结合项目要求和设备确认，实际安装命令与验证结果记录在 `notes/<项目名>.md`。

`.envs/` 仅提交 `.gitkeep` 占位文件，实际环境全部由 `.gitignore` 排除，不随仓库上传或下载。依赖声明沿用各 repo 的 `requirements.txt`、`environment.yml` 等文件（如有）；更换机器后按依赖声明和项目笔记重建环境。

## 项目进度

| 项目 | 固定版本 | 当前进度 |
| --- | --- | --- |
| [ToolRL](https://github.com/qiancheng0/ToolRL/tree/8cee13ec0ca72f0461da372a93a6fd8140dbb840) | [8cee13ec](https://github.com/qiancheng0/ToolRL/tree/8cee13ec0ca72f0461da372a93a6fd8140dbb840) | 已阅读核心代码并验证奖励样例；未进行模型推理或训练 |

ToolRL 的 8 个手写示例、评分结果和可运行代码已放在 [ToolRL.md](notes/ToolRL.md) 中。当前尚未进行模型训练。

## 使用约定

- 每个上游仓库保留自己的环境和版本管理，学习内容统一写入 `notes/<项目名>.md`。
- 有意义的实验记录代码版本、配置、运行命令和实际结果；区分手写示例、模型推理和训练结果。
- 大型数据、模型和检查点等到实际使用时再安排存储，不预建空目录，不提交密钥或大型产物。
- AI 助手遵循 [AGENTS.md](AGENTS.md)，协作规则只维护这一份。
