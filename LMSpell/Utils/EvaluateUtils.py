import re
import pandas as pd
from Utils.ConfigUtils import ConfigUtils
from diff_match_patch import diff_match_patch

class EvaluateUtils:
    @staticmethod
    def remove_space_before_period(text):
        return re.sub(r'\s([.])', r'\1', text)

    @staticmethod
    def levenshtein_distance(a, b):
        """
        Returns the character-level Levenshtein distance between two strings `a` and `b`.
        This is the minimal number of single-character edits (insert, delete, substitute)
        required to transform `a` into `b`.
        """
        la, lb = len(a), len(b)
        dp = [[0]*(lb+1) for _ in range(la+1)]
        
        for i in range(la+1):
            dp[i][0] = i
        for j in range(lb+1):
            dp[0][j] = j
        
        for i in range(1, la+1):
            for j in range(1, lb+1):
                cost = 0 if a[i-1] == b[j-1] else 1
                dp[i][j] = min(
                    dp[i-1][j] + 1,    # Delete char from a
                    dp[i][j-1] + 1,    # Insert char into a
                    dp[i-1][j-1] + cost
                )
        return dp[la][lb]

    @staticmethod
    def align_texts(source, target):
        """
        Aligns two sentences at the word level, but uses character-level edit distance
        to decide the cost of substituting one word with another.

        Returns a dict with:
        - 'aligned_source': the source words or +/– notations
        - 'aligned_target': the target words
        """
        source_words = source.split()
        target_words = target.split()

        m, n = len(source_words), len(target_words)

        dp = [[0]*(n+1) for _ in range(m+1)]
        
        def insertion_cost(word):
            return len(word)

        def deletion_cost(word):
            return len(word)

        def substitution_cost(w1, w2):
            return EvaluateUtils.levenshtein_distance(w1, w2)
        
        for i in range(m+1):
            for j in range(n+1):
                if i == 0 and j == 0:
                    dp[i][j] = 0
                elif i == 0: 
                    dp[i][j] = dp[i][j-1] + insertion_cost(target_words[j-1])
                elif j == 0: 
                    dp[i][j] = dp[i-1][j] + deletion_cost(source_words[i-1])
                else:
                    cost_sub = dp[i-1][j-1] + substitution_cost(source_words[i-1], target_words[j-1])
                    cost_del = dp[i-1][j] + deletion_cost(source_words[i-1]) 
                    cost_ins = dp[i][j-1] + insertion_cost(target_words[j-1]) 
                    dp[i][j] = min(cost_sub, cost_del, cost_ins)

        aligned_source = []
        i, j = m, n
        while i > 0 or j > 0:
            if i > 0 and j > 0:
                sub_cost = dp[i-1][j-1] + substitution_cost(source_words[i-1], target_words[j-1])
                if dp[i][j] == sub_cost:
                    if substitution_cost(source_words[i-1], target_words[j-1]) == 0:
                        aligned_source.append(source_words[i-1])
                    else:
                        aligned_source.append(source_words[i-1])
                    i -= 1
                    j -= 1
                    continue
            
            if i > 0:
                del_cost = dp[i-1][j] + deletion_cost(source_words[i-1])
                if dp[i][j] == del_cost:
                    aligned_source.append("".join(f"+{ch}" for ch in source_words[i-1]))
                    i -= 1
                    continue
            
            if j > 0:
                ins_cost = dp[i][j-1] + insertion_cost(target_words[j-1])
                if dp[i][j] == ins_cost:
                    aligned_source.append("".join(f"-{ch}" for ch in target_words[j-1]))
                    j -= 1
                    continue

        aligned_source.reverse()

        return {
            "aligned_source": aligned_source,
            "aligned_target": target_words
        }

    @staticmethod
    def normalize_spaces(text):
        return text.replace("\u00A0", "\u0020")

    @staticmethod
    def mark_insertions_and_deletions(original, gold):
        num_org_words = len(original.split())
        num_gold_words = len(gold.split())
        normalized_original = EvaluateUtils.normalize_spaces(original)
        normalized_gold = EvaluateUtils.normalize_spaces(gold)
        alignment_result = EvaluateUtils.align_texts(normalized_original, normalized_gold)     
        new_original = alignment_result['aligned_source']
        new_gold = alignment_result['aligned_target']
        modified_original = []
        dmp = diff_match_patch()

        i=0
        j=0

        while i<len(new_original) or j<len(new_gold):
            if i >= len(new_original):
                o = ' '
            else:
                o = new_original[i]
            if j >= len(new_gold):
                g = ' '
            else:
                g = new_gold[j]
            if o == g:
                modified_original.append(o)
                modified_original.append(" ")
                i += 1
                j += 1
                num_org_words -= 1
                num_gold_words -= 1
                continue
            elif o[0] == '-':
                modified_original.append(o)
                modified_original.append("- ")
                i += 1
                j += 1
                num_gold_words -= 1
                continue
            elif o[0] == '+':
                modified_original.append(o)
                modified_original.append("+ ")
                i += 1
                num_org_words -= 1
                continue

            num_org_words -= 1
            num_gold_words -= 1
            diff = dmp.diff_main(g, o)

            prev = (0, '')

            for part in diff:
                if part[0] == 0:
                    modified_original.append(part[1])
                elif part[0] == -1:
                    for k in part[1]:
                        modified_original.append('-' + k)
                elif part[0] == 1:
                    for k in part[1]:
                        modified_original.append('+' + k)
                if prev[0] == -1 and part[0] == 1:
                    old = prev[1]
                    new = part[1]
                    if (len(old) <= 2) and (len(new) <= 2):
                        del modified_original[-(len(old)+len(new)):]
                        if len(old) == len(new):
                            modified_original.append(new)
                        if len(old) > len(new):
                            modified_original.append(new)
                            modified_original.append('-' + old[len(new):])
                        if len(old) < len(new):
                            modified_original.append(new[:len(old)])
                            modified_original.append('+' + new[len(old):])

                prev = part
            i += 1
            j += 1
            if (num_org_words > 0) and (num_gold_words > 0):
                modified_original.append(" ")
            elif (not (num_org_words > 0)) and (num_gold_words > 0):
                modified_original.append("- ")
            elif (num_org_words > 0) and (not (num_gold_words > 0)):
                modified_original.append("+ ")

        returning_text = ''.join(modified_original).strip()
        if returning_text[-1] == '+':
            returning_text = returning_text[:-1]
        elif returning_text[-1] == '-':
            returning_text = returning_text[:-1]
            
        return returning_text

    @staticmethod
    def has_excessive_symbols(input_string):
        plus_count = input_string.count('+')
        minus_count = input_string.count('-')
        return plus_count > 20 or minus_count > 20

    @staticmethod
    def evaluate(srcs, preds, refs, replace=False):
        length = len(refs)
        z = 0

        if isinstance(srcs, str):
            srcs = [srcs]
            preds = [preds]
            refs = [refs]
        else:
            srcs = list(srcs)
            preds = list(preds)
            refs = list(refs)

        srcs = [s.replace('\u00A0', ' ') for s in srcs]
        preds = [p.replace('\u00A0', ' ') for p in preds]
        refs = [r.replace('\u00A0', ' ') for r in refs]

        d_tp_agg, d_tn_agg, d_fp_agg, d_fn_agg = 0, 0, 0, 0  # detection
        c_tp_agg, c_tn_agg, c_fp_agg, c_fn_agg = 0, 0, 0, 0  # correction
        detection_recall = 0
        detection_precision = 0
        correction_recall = 0
        correction_precision = 0
        detection_accuracy = 0
        detection_F1 = 0    
        detection_F0_5 = 0
        corrections_accuracy = 0
        corrections_F1 = 0
        corrections_F0_5 = 0
        replacements = 0
        
        for src_single, pred_single, ref_single in zip(srcs, preds, refs):
            src_single = (src_single.strip('.')).strip()
            pred_single = (pred_single.strip('.')).strip()
            ref_single = (ref_single.strip('.')).strip()
            z+=1

            for i in range(3):
                if i == 0:
                    target = src_single
                elif i == 1:
                    target = pred_single
                else:
                    target = ref_single
                prev1 = ''
                prev2 = ''
                new_str = []
                for j in target:
                    if prev1 == 'ෙ' and j == 'ා':
                        new_str.pop()
                        new_str.append('ො')
                    elif prev1 == 'ෙ' and j == 'ෙ':
                        new_str.pop()
                        new_str.append('ෛ')
                    elif prev1 == 'ෙ' and j == 'ෟ':
                        new_str.pop()
                        new_str.append('ෞ')
                    elif prev1 == 'ෙ' and j == '්':
                        new_str.pop()
                        new_str.append('ේ')
                    elif prev1 == 'ෘ' and j == 'ෘ':
                        new_str.pop()
                        new_str.append('ෲ')
                    elif prev2 == 'ෙ' and prev1 == 'ා' and j == '්':
                        new_str.pop()
                        new_str.pop()
                        new_str.append('ෝ')
                    else:
                        new_str.append(j)
                    prev1 = j
                    prev2 = prev1
                
                if i == 0:
                    src_single = ''.join(new_str)
                elif i == 1:
                    pred_single = ''.join(new_str)
                else:
                    ref_single = ''.join(new_str)

            aligned_src = []
            aligned_pred = []
            aligned_ref = []

            def safe_division(n, d):
                return n / d if d > 0 else 0

            pred_edits_ps = aligned_pred
            gold_edits_ps = aligned_ref
            x_ps = aligned_src

            marked_src = EvaluateUtils.mark_insertions_and_deletions(src_single, ref_single)
            marked_pred = EvaluateUtils.mark_insertions_and_deletions(pred_single, ref_single)

            if EvaluateUtils.has_excessive_symbols(marked_pred):
                replacements += 1
                if replace:
                    marked_pred = marked_src

            aligned_pred.extend(marked_pred)
            aligned_ref.extend(ref_single)
            aligned_src.extend(marked_src)

            d_tp, d_tn, d_fp, d_fn = 0, 0, 0, 0  # detection
            c_tp, c_tn, c_fp, c_fn = 0, 0, 0, 0  # correction
                
            i = 0
            j = 0
            k = 0

            while max(i, j, k) < max(len(x_ps), len(gold_edits_ps), len(pred_edits_ps)):
                if i >= len(x_ps):
                    x = ''
                else:
                    x = x_ps[i]
                if j >= len(gold_edits_ps):
                    gold = ''
                else:
                    gold = gold_edits_ps[j]
                if k >= len(pred_edits_ps):   
                    pred = ''
                else:
                    pred = pred_edits_ps[k]

                if x == '+':
                    x += x_ps[i + 1]
                    if pred == '+':
                        pred += pred_edits_ps[k + 1]
                        if x != pred:
                            d_tp += 1
                            c_fp += 1
                        else:
                            d_fn += 1
                            c_fn += 1
                        k += 2
                    elif pred == '-':
                        d_tp += 1
                        c_fp += 1
                    else:
                        d_tp += 1
                        if pred == gold:
                            c_tp += 1
                        else:
                            c_fp += 1
                    i += 2
                elif x == '-':
                    if pred == '-':
                        d_fn += 1
                        c_fn += 1
                        i += 2
                        k += 2
                        j += 1
                    elif pred == '+':
                        d_fp += 1
                        c_fp += 1
                        k += 2
                    else:
                        if pred == gold:
                            d_tp += 1
                            c_tp += 1
                            i += 2
                            k += 1
                            j += 1
                        else:
                            d_tp += 1
                            c_fp += 1
                            i += 2
                            k += 1
                            j += 1
                else:
                    if pred == '+': # Insertion
                        d_fp += 1
                        c_fp += 1
                        k += 2
                    elif pred == '-': # Deletion
                        d_fp += 1
                        c_fp += 1
                        i += 1
                        k += 2
                        j += 1
                    else:
                        if x == gold:   # Correct
                            if pred == gold:
                                d_tn += 1
                                c_tn += 1
                            else:
                                d_fp += 1
                                c_fp += 1
                        else:   # Substitution
                            if pred == gold:
                                d_tp += 1
                                c_tp += 1
                            else:
                                d_fn += 1
                                c_fn += 1
                        i += 1
                        j += 1
                        k += 1
            
            d_tp_agg += d_tp
            d_tn_agg += d_tn
            d_fp_agg += d_fp
            d_fn_agg += d_fn
            c_tp_agg += c_tp
            c_tn_agg += c_tn
            c_fp_agg += c_fp
            c_fn_agg += c_fn
            
            d_recall = safe_division(d_tp, d_tp + d_fn)
            d_prec = safe_division(d_tp, d_fp + d_tp)
            c_recall = safe_division(c_tp, c_tp + c_fn)
            c_prec = safe_division(c_tp, c_fp + c_tp)
    
            if (d_tp == 0) and (d_fn == 0) and (d_fp == 0):
                d_recall = 1
                d_prec = 1
                c_recall = 1
                c_prec = 1

            detection_recall += d_recall
            detection_precision += d_prec
            correction_recall += c_recall
            correction_precision += c_prec
            detection_accuracy += safe_division((d_tn + d_tp), (d_tn + d_tp + d_fn + d_fp)) * 100
            detection_F1 += safe_division((2 * d_recall * d_prec * 100), (d_recall + d_prec))
            detection_F0_5 += safe_division(((1 + 0.5 ** 2) * d_recall * d_prec * 100), (d_recall + (0.5 ** 2) * d_prec))
            corrections_accuracy += safe_division((c_tn + c_tp), (c_tn + c_tp + c_fn + c_fp)) * 100
            corrections_F1 += safe_division((2 * c_recall * c_prec * 100), (c_recall + c_prec))
            corrections_F0_5 += safe_division(((1 + 0.5 ** 2) * c_recall * c_prec * 100), (c_recall + (0.5 ** 2) * c_prec))

        return {
            'Detection Accuracy': detection_accuracy / length,
            'Detection Recall': (detection_recall * 100) / length,
            'Detection Precision': (detection_precision * 100) / length,
            'Detection F1': detection_F1 / length,
            'Detection F0.5': detection_F0_5 / length,
            'Correction Accuracy': corrections_accuracy / length,
            'Correction Recall': (correction_recall * 100) / length,
            'Correction Precision': (correction_precision * 100) / length,
            'Correction F1': corrections_F1 / length,
            'Correction F0.5': corrections_F0_5 / length,
        }
    
    @staticmethod
    def evaluate_from_dataframe(df, output_dir, replace=False):
        """
        Evaluate the model using a DataFrame containing source, predicted, and reference columns.
        """

        result_col_names = ConfigUtils.get_results_columns()

        srcs = df[result_col_names[0]]
        preds = df[result_col_names[1]]
        refs = df[result_col_names[2]]

        EvaluateUtils.process_results(EvaluateUtils.evaluate(srcs, preds, refs, replace), output_dir)
    
    @staticmethod
    def process_results(results, output_dir):
        import os
        """
        Process the results DataFrame and save it to a CSV file.
        """
        
        results_file_path = os.path.join(os.getcwd(), f"{output_dir}/evaluation_results.txt")
        with open(results_file_path, "w", encoding="utf-8") as file:
            for key, value in results.items():
                file.write(f"{key}: {value}\n")
        print(f"Evaluation results saved to {results_file_path}")
        
        print("Evaluation results: ")
        for key, value in results.items():
            print(f"{key}: {value}")

    @staticmethod
    def print_in_tab_seperated_format(df):
        filtered_df = df[df['Without Replacement'].astype(str).str.contains(r"\.")]
        row_data = "\t".join(
            f"{float(wr):.2f}" if wr == wrp and not isinstance(wr, str) and not isinstance(wrp, str)
            else f"{float(wr):.2f}({float(wrp):.2f})" if not isinstance(wr, str) and not isinstance(wrp, str)
            else f"{wr}"  # fallback if values are strings
            for wr, wrp in zip(filtered_df['Without Replacement'], filtered_df['With Replacement'])
        )