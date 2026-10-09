"""Run hand-written examples through ToolRL's unchanged default reward function.

This is a CPU-only learning exercise, not model inference or a training run.
Uses only Python's standard library; run from any working directory.
"""

import argparse
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


WORKSPACE = Path(__file__).resolve().parents[3]
REPOSITORY = WORKSPACE / "repos" / "ToolRL"
VARIANTS = (
    "WITHLENGTH", "REFINEDREWARD", "COARSEREWARD", "STRICTMATCH",
    "CORRECTMAX1", "MAX1STEP30MAX3", "SCHEDULEREWARD",
    "SCHEDULELENGTH", "INTERMEDIATEREWARD",
)


def tool_output(name="get_weather", parameters=None):
    if parameters is None:
        parameters = {"city": "Beijing"}
    call = json.dumps({"name": name, "parameters": parameters})
    return "<think>Query the requested city.</think>\n<tool_call>\n" + call + "\n</tool_call>"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Save JSON to a new file; existing files are preserved.")
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("Output already exists. Choose another filename or omit --output.")

    sys.dont_write_bytecode = True
    for name in VARIANTS:
        os.environ[name] = "0"
    # The upstream parser selects its chat format from this variable.
    os.environ["EXPERIMENT_NAME"] = "inspect-qwen-default-reward"

    path = REPOSITORY / "verl/utils/reward_score/rlla.py"
    spec = importlib.util.spec_from_file_location("toolrl_reward", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    target = tool_output()
    response_target = "<think>Respond.</think>\n<response>Beijing</response>"
    examples = [
        ("工具和参数完全正确", target, target),
        ("工具正确但城市错误", tool_output(parameters={"city": "Shanghai"}), target),
        ("工具正确但缺少参数", tool_output(parameters={}), target),
        ("选错工具", tool_output(name="get_time"), target),
        ("调用正确但缺少 think 标签", target.split("\n", 1)[1], target),
        ("格式标签正确但 JSON 无效", "<think>Query.</think>\n<tool_call>\nnot-json\n</tool_call>", target),
        ("需要调用工具却直接回答", response_target, target),
        ("纯回答目标但回答内容不同", "<think>Respond.</think>\n<response>Shanghai</response>", response_target),
    ]

    rows = []
    for label, prediction, ground_truth in examples:
        # Upstream prints entire inputs; retain only structured scores here.
        with contextlib.redirect_stdout(io.StringIO()):
            scores = module.compute_score(prediction, ground_truth, step=0)
        rows.append({
            "case": label,
            "prediction": prediction,
            "ground_truth": ground_truth,
            **dict(zip(("total", "format", "correctness", "length"), scores)),
        })

    result = {
        "kind": "handwritten_reward_examples_not_model_evaluation",
        "commit": subprocess.check_output(
            ["git", "-C", str(REPOSITORY), "rev-parse", "HEAD"], text=True
        ).strip(),
        "python": platform.python_version(),
        "settings": {**{name: "0" for name in VARIANTS}, "EXPERIMENT_NAME": os.environ["EXPERIMENT_NAME"]},
        "cases": rows,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

    print("示例 | 总分 | 格式 | 正确性 | 长度")
    for row in rows:
        print(f"{row['case']} | {row['total']:g} | {row['format']:g} | {row['correctness']:g} | {row['length']:g}")


if __name__ == "__main__":
    main()
