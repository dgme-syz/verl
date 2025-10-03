# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify, ExprExtractionConfig

def normalize_answer_str(s: str) -> str | None:
    if s is None:
        return None
    return s.strip().replace(",", "").replace("$", "")


def extract_solution(solution_str: str, method="strict"):
    if method == "strict":
        m = re.findall(r"\\boxed\{([^}]*)\}", solution_str)
        if not m:
            return None
        return normalize_answer_str(m[-1])
    else:
        m = re.findall(r"(-?[0-9\.,]+)", solution_str)
        if not m:
            return None
        for x in reversed(m):
            x2 = normalize_answer_str(x)
            if x2 not in ["", "."]:
                return x2
        return None


def compute_score(
    solution_str, 
    ground_truth, 
    extra_info=None,
    data_source=None,
    method="strict", 
    score=1.0
):

    other_solutions = extra_info.get("re_generation", None)
    solution_str = [solution_str] 
    if other_solutions is not None:
        solution_str += other_solutions

    step = score / len(solution_str)
    for x in solution_str:
        gold_parsed = parse(f"\\boxed{ground_truth}", extraction_mode="first_match")
        if len(gold_parsed) != 0:
            ans_parsed = parse(
                x,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            boxed="all",
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=True,
                    ),
                    ExprExtractionConfig(),
                ],
                extraction_mode="first_match",
            )

            if len(ans_parsed) != 0:
                try:
                    if verify(gold_parsed, ans_parsed):
                        return score
                except Exception as e:
                    print(f"Verification failed: {e}")
        score -= step
            
        
    return 0.0
    
def eval_score(
    solution_str, 
    ground_truth, 
    extra_info=None,
    data_source=None,
    method="strict", 
    score=1.0
):
    gold_parsed = parse(f"\\boxed{ground_truth}", extraction_mode="first_match")
    if len(gold_parsed) != 0:
        ans_parsed = parse(
            solution_str,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        boxed="all",
                        units=True,
                    ),
                    # Ensures that boxed is tried first
                    boxed_match_priority=0,
                    try_extract_without_anchor=True,
                ),
                ExprExtractionConfig(),
            ],
            extraction_mode="first_match",
        )

        if len(ans_parsed) != 0:
            try:
                if verify(gold_parsed, ans_parsed):
                    return score
            except Exception as e:
                print(f"Verification failed: {e}")
    return 0.0
