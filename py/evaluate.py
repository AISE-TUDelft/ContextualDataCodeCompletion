import glob
import os.path
import sys
from pathlib import Path
import json
from typing import List
import Levenshtein as Levenshtein
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score

model = sys.argv[1]  # codegpt, incoder, unixcoder
single_token = len(sys.argv) > 2 and sys.argv[2] == "single"

model_predictions_folder = os.path.join("./data/predictions", model)


def get_types(folder_name: str):
    types = folder_name.split("-")[0]
    if types == "normal":
        return "OT"
    elif types == "explicit":
        return "AT"
    elif types == "untyped":
        return "NT"

    raise Exception(f"Unknown types {types} (folder name: {folder_name})")


def get_comments(folder_name: str):
    comments = folder_name.split("-")[1]
    if comments == "all":
        return "AC"
    elif comments == "none":
        return "NC"
    elif comments == "single_line":
        return "SL"
    elif comments == "multi_line":
        return "ML"
    elif comments == "docblock":
        return "DB"

    raise Exception(f"Unknown comments {comments} (folder name: {folder_name})")


def empty_evaluation_obj():
    return {
        "n": 0,
        "meteor": 0,
        "bleu": 0,
        "levenshtein": 0,
        "rouge": 0,
        "exact_match": 0
    }


def average_evaluation(evaluation):
    evaluation["meteor"] /= evaluation["n"]
    evaluation["bleu"] /= evaluation["n"]
    evaluation["levenshtein"] /= evaluation["n"]
    evaluation["rouge"] /= evaluation["n"]
    evaluation["exact_match"] /= evaluation["n"]


def round_evaluation(evaluation):
    evaluation["meteor"] = round(evaluation["meteor"] * 100, 2)
    evaluation["bleu"] = round(evaluation["bleu"] * 100, 2)
    evaluation["levenshtein"] = round(evaluation["levenshtein"] * 100, 2)
    evaluation["rouge"] = round(evaluation["rouge"] * 100, 2)
    evaluation["exact_match"] = round(evaluation["exact_match"] * 100, 2)


def update_evaluation(evaluation, meteor, bleu, levenshtein, rouge, exact_match):
    evaluation["levenshtein"] += levenshtein
    evaluation["bleu"] += bleu
    evaluation["rouge"] += rouge
    evaluation["meteor"] += meteor
    evaluation["exact_match"] += exact_match
    evaluation["n"] += 1


def lcs(X: List[str], Y: List[str]) -> int:
    # find the length of the strings
    m = len(X)
    n = len(Y)

    # declaring the array for storing the dp values
    L = [[None] * (n + 1) for i in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])

    return L[m][n]


def rouge_l(line_tokens: List[str], completion_tokens: List[str]) -> float:
    if len(line_tokens) == 0 or len(completion_tokens) == 0:
        return 0
    lcs_length = lcs(line_tokens, completion_tokens)
    precision = lcs_length / len(completion_tokens)
    recall = lcs_length / len(line_tokens)
    if precision + recall > 0:
        return 2 * precision * recall / (precision + recall)
    else:
        return 0.0


# path should be ./data/predictions/<model>/<typed>-<comments>
def evaluate_folder(path: Path):
    result = empty_evaluation_obj()
    postprocessed_file = path / 'postprocessed.txt'

    if single_token:
        metrics_file = path / 'metrics_single_token.txt'
    else:
        metrics_file = path / 'metrics.txt'

    input_token_count = 0

    with postprocessed_file.open() as ppf, metrics_file.open('w') as mf:
        for line in ppf.readlines():
            obj = json.loads(line)
            gt, gt_tokens, prediction, prediction_tokens, input, input_tokens = obj["gt"], obj[
                "gtTokens"], obj["prediction"], obj["predictionTokens"], obj["input"], obj[
                                                                                    "inputTokens"]
            if single_token:
                gt_tokens = gt_tokens[0:1]
                prediction_tokens = prediction_tokens[0:1]
                gt = "".join(gt_tokens)
                prediction = "".join(prediction_tokens)

            if prediction == "":
                mf.write(f"{json.dumps({'empty': True})}\n")
                continue

            m_exact_match = 1 if gt.split() == prediction.split() else 0
            input_token_count += len(input_tokens)
            m_bleu = sentence_bleu([gt_tokens], prediction_tokens,
                                   smoothing_function=SmoothingFunction().method2)
            m_levenshtein = Levenshtein.ratio(gt, prediction)
            m_meteor = meteor_score(references=[gt_tokens], hypothesis=prediction_tokens)
            m_rouge = rouge_l(gt_tokens, prediction_tokens)

            update_evaluation(
                result,
                bleu=m_bleu,
                levenshtein=m_levenshtein,
                meteor=m_meteor,
                rouge=m_rouge,
                exact_match=m_exact_match
            )

            m_obj = {
                "empty": False,
                "meteor": m_meteor,
                "bleu": m_bleu,
                "levenshtein": m_levenshtein,
                "rouge": m_rouge,
                "exact_match": m_exact_match
            }
            mf.write(f"{json.dumps(m_obj)}\n")

    average_evaluation(result)
    round_evaluation(result)

    return result


if __name__ == "__main__":
    folders = glob.glob(model_predictions_folder + "/*")
    print(f"Found {len(folders)} folders: {', '.join(folders)}")
    print(f"Single token: {single_token}")

    table_start = """
\\begin{table}[tb]
\\caption{All Results (Test + Validation)}
\\centering
\\begin{tabular}{lclrrrrr}
\\toprule
Model & Dataset & CMT & EM & ES & B4 & RL & MR \\\\
\\midrule
""".lstrip()

    if single_token:
        table_start = table_start.replace(" & B4 & RL & MR", "")

    table_end = """
\\bottomrule
\\end{tabular}
\\end{table}
""".lstrip()

    rows = []

    for folder in sorted(folders, key=lambda x: x[::-1]):
        folder_name = os.path.basename(folder)
        folder_path = Path(folder)
        result = evaluate_folder(folder_path)
        exact_match = result["exact_match"]
        levenshtein = result["levenshtein"]
        bleu = result["bleu"]
        rouge = result["rouge"]
        meteor = result["meteor"]
        stats = f"{folder_path.name}: EM: {exact_match}, ES: {levenshtein}"
        try:
            row = f"{model} & {get_types(folder_name)} & {get_comments(folder_name)} & \\numprint{{{exact_match:.2f}}} & \\numprint{{{levenshtein:.2f}}}"
            if not single_token:
                stats += f", B4: {bleu}, RL: {rouge}, MT: {meteor}"
                row += f" & \\numprint{{{bleu:.2f}}} & \\numprint{{{rouge:.2f}}} & \\numprint{{{meteor:.2f}}}"
            row += "\\\\"
            rows.append(row)
            print(stats)
        except:
            pass

    rows = sorted(rows)
    rows = "\n".join(rows) + "\n"

    print(table_start + rows + table_end)
