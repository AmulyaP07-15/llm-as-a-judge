#!/usr/bin/env python3
"""Standalone entry point for the LLM-as-a-Judge bias project.

Running this file reproduces the results reported in the paper. By default it
runs in analysis-only mode, which reads the committed verdict CSVs and prints
the objective and subjective results tables. Analysis-only mode needs no Groq
API key and spends no tokens, so a grader can confirm the project runs end to
end for free.

Passing --full instead reruns the complete pipeline for each task, from dataset
preparation through generation, judging, and analysis. Full mode calls the Groq
API, so it needs the GROQ_API_KEY environment variable set.

The project keeps each task in its own folder.
    TriviaQA dataset/       the objective task
    IBM-Debater dataset/    the subjective task
Each task script reads its CSVs from the current directory and, for the
subjective task, imports groq_utils from its own folder. So every script is run
with its working directory set to that task's folder. This file lives in the
repository root and drives both.

Usage
-----
    python demo.py                      # analysis only, both tasks (default)
    python demo.py --task objective     # analysis only, objective task
    python demo.py --task subjective    # analysis only, subjective task
    python demo.py --full               # full pipeline, both tasks (needs API key)
    python demo.py --full --task objective
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Repository root is the directory this file sits in. Task folders and scripts
# are resolved relative to this, so demo.py works from any working directory.
REPO_ROOT = Path(__file__).resolve().parent

# Task folder names. These contain spaces, which is fine because scripts are
# launched through a subprocess argument list rather than a shell command.
OBJECTIVE_DIR = "TriviaQA dataset"
SUBJECTIVE_DIR = "IBM-Debater dataset"

# ---------------------------------------------------------------------------
# Pipeline definitions.
#
# Each stage is (script_filename, human_label). Scripts run in list order, each
# with its working directory set to the task folder. The final stage is the
# analysis script, which is the only stage that runs in analysis-only mode
# because it reads CSVs and needs no API.
#
# Filenames match the repository exactly. Linux is case sensitive, so the
# capitalisation here has to line up with the committed files, including the
# lowercase d in IBM-Debater_dataset_argument_generator.py.
# ---------------------------------------------------------------------------
OBJECTIVE_PIPELINE = [
    ("TriviaQA_dataset_preparation.py", "Prepare TriviaQA sample"),
    ("TriviaQA_answer_generator.py", "Generate model answers"),
    ("TriviaQA_pairwise_combinations.py", "Build pairwise comparisons"),
    ("TriviaQA_judge_answers.py", "Judge every pair"),
    ("TriviaQA_analyze_bias.py", "Analyse bias and print tables"),
]

SUBJECTIVE_PIPELINE = [
    ("IBM-Debater_dataset_preparation.py", "Prepare ArgKP topics"),
    ("IBM-Debater_dataset_argument_generator.py", "Generate arguments"),
    ("IBM-Debater_judge_arguments.py", "Judge every pair"),
    ("analyze_bias.py", "Analyse bias and print tables"),
]

# CSVs each analysis stage reads in analysis-only mode. demo.py checks these up
# front so a missing file produces a clear message instead of a stack trace
# from deep inside a task script.
OBJECTIVE_ANALYSIS_INPUTS = ["TriviaQA_judge_verdicts.csv", "TriviaQA_sampled.csv"]
SUBJECTIVE_ANALYSIS_INPUTS = ["judge_verdicts.csv", "argument_outputs.csv"]


def print_banner(text):
    """Print a heading that stands out in a long console log."""
    line = "=" * 70
    print("\n" + line)
    print(text)
    print(line + "\n")


def run_script(task_dir, script_name, label):
    """Run one task script as a subprocess from inside its task folder.

    Returns True on success and False on any failure. The task script's own
    stdout and stderr stream straight to the console, so the results tables it
    prints appear inline. sys.executable keeps the subprocess on the same
    interpreter and virtual environment as demo.py. The working directory is
    the task folder, which is how the script finds its CSVs and its local
    groq_utils import.
    """
    folder = REPO_ROOT / task_dir
    script_path = folder / script_name

    if not folder.exists():
        print("  MISSING FOLDER: " + task_dir)
        print("  Expected at " + str(folder))
        return False

    if not script_path.exists():
        print("  MISSING SCRIPT: " + script_name)
        print("  Expected at " + str(script_path))
        print("  Check the filename and its capitalisation against the repo.")
        return False

    print("  -> " + label + "  [" + task_dir + "/" + script_name + "]")
    result = subprocess.run(
        [sys.executable, script_name],
        cwd=str(folder),
    )
    if result.returncode != 0:
        print("  SCRIPT FAILED: " + script_name
              + " exited with code " + str(result.returncode))
        return False
    return True


def check_inputs(task_dir, required_files):
    """Return the list of required CSVs missing from the task folder."""
    folder = REPO_ROOT / task_dir
    missing = []
    for name in required_files:
        if not (folder / name).exists():
            missing.append(name)
    return missing


def run_analysis_only(task_dir, pipeline, required_inputs, task_label):
    """Run just the analysis stage of a task, which is the last script.

    This is the default path. It needs no API key. If the committed CSVs are
    absent it explains what is missing and points at --full.
    """
    print_banner(task_label + "  (analysis only)")

    missing = check_inputs(task_dir, required_inputs)
    if missing:
        print("  Cannot run the analysis. Missing input CSVs in "
              + task_dir + ":")
        for name in missing:
            print("    - " + name)
        print("  These are produced by the full pipeline.")
        print("  Rerun with --full to regenerate them, which needs GROQ_API_KEY.")
        return False

    analysis_script, analysis_label = pipeline[-1]
    return run_script(task_dir, analysis_script, analysis_label)


def run_full_pipeline(task_dir, pipeline, task_label):
    """Run every stage of a task in order, stopping at the first failure.

    Full mode calls the Groq API during generation and judging, so GROQ_API_KEY
    must be set. Each stage depends on the CSV written by the previous stage, so
    a failure aborts the rest of that task.
    """
    print_banner(task_label + "  (full pipeline)")

    for script_name, label in pipeline:
        if not run_script(task_dir, script_name, label):
            print("  Stopping " + task_label + " pipeline after a failed stage.")
            return False
    return True


def parse_args():
    """Define and parse the command line interface."""
    parser = argparse.ArgumentParser(
        description="Reproduce the LLM-as-a-Judge bias results.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Rerun the complete pipeline from scratch. Needs GROQ_API_KEY. "
             "Default is analysis only on the committed CSVs.",
    )
    parser.add_argument(
        "--task",
        choices=["objective", "subjective", "both"],
        default="both",
        help="Which task to run. Default is both.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Full mode needs a Groq key. Fail early with a clear message rather than
    # letting a task script raise a cryptic auth error mid-run.
    if args.full and not os.environ.get("GROQ_API_KEY"):
        print("ERROR: --full needs the GROQ_API_KEY environment variable set.")
        print("Set it, or drop --full to run analysis on the committed CSVs.")
        return 1

    run_objective = args.task in ("objective", "both")
    run_subjective = args.task in ("subjective", "both")

    print_banner("LLM-as-a-Judge bias project")
    mode = "full pipeline" if args.full else "analysis only"
    print("Mode: " + mode)
    print("Task: " + args.task)
    print("Repo: " + str(REPO_ROOT))

    results = {}

    if run_objective:
        label = "OBJECTIVE TASK  (TriviaQA)"
        if args.full:
            results["objective"] = run_full_pipeline(
                OBJECTIVE_DIR, OBJECTIVE_PIPELINE, label)
        else:
            results["objective"] = run_analysis_only(
                OBJECTIVE_DIR, OBJECTIVE_PIPELINE,
                OBJECTIVE_ANALYSIS_INPUTS, label)

    if run_subjective:
        label = "SUBJECTIVE TASK  (IBM Debater ArgKP)"
        if args.full:
            results["subjective"] = run_full_pipeline(
                SUBJECTIVE_DIR, SUBJECTIVE_PIPELINE, label)
        else:
            results["subjective"] = run_analysis_only(
                SUBJECTIVE_DIR, SUBJECTIVE_PIPELINE,
                SUBJECTIVE_ANALYSIS_INPUTS, label)

    # Summary line so the exit status is obvious at a glance.
    print_banner("SUMMARY")
    all_ok = True
    for task_name, ok in results.items():
        status = "ok" if ok else "FAILED"
        print("  " + task_name + ": " + status)
        all_ok = all_ok and ok

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
