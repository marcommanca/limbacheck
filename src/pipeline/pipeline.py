import torch
import re
import jellyfish
from symspellpy import (SymSpell, Verbosity)

class SardinianSpellchecker:
    """
    Hybrid spelling-correction system for the Sardinian language.
    Combines SymSpell for (structural) candidate generation with an MLM BERT for
    contextual ranking.
    """

    def __init__(self, tokenizer, mlm, detector, vocabulary, device,
                 detection_threshold, correction_threshold,
                 max_levenshtein_distance, max_candidates, debug=False):
        """ Initializes the correction pipeline. """
        self.tokenizer = tokenizer
        self.mlm = mlm
        self.detector = detector
        self.vocabulary = vocabulary
        self.device = device
        self.detection_threshold = detection_threshold
        self.correction_threshold = correction_threshold
        self.max_levenshtein_distance = max_levenshtein_distance
        self.max_candidates = max_candidates
        self.debug = debug

        # SymSpell vocabulary initialization (max distance 3 to allow grid search;
        # the actual distance used is still the one passed via the parameter)
        symspell_capacity = max(3, self.max_levenshtein_distance)
        self.sym_spell = SymSpell(max_dictionary_edit_distance=symspell_capacity, prefix_length=15)
        for word in self.vocabulary:
            self.sym_spell.create_dictionary_entry(word, 1)

        # Regex to identify Sardinian words (including internal apostrophes and hyphens)
        self.word_pattern = re.compile(r"['\-]?[a-zA-Zàèìòùáéíóú]+(?:['\-][a-zA-Zàèìòùáéíóú]+)*['\-]?|\d+|[^a-zA-Z0-9àèìòùáéíóú\s]")

    def split_text(self, text):
        """ Splits the text into words and tokens, keeping punctuation intact. """
        return self.word_pattern.findall(text)

    def detect_errors(self, words, threshold):
        """ Token Classification to detect misspelled words in context. """
        inputs = self.tokenizer(words, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=512).to(self.device)

        # Error detection with the spellchecker model (token classification, taking
        # the max value per token), with values:
        # (0: CORRECT, 1: ERROR)
        with torch.no_grad():
            outputs = self.detector(**inputs)
            # Softmax probabilities over the logits
            probs = torch.softmax(outputs.logits, dim=-1)
            # Extract the probability associated with the ERROR class (index 1)
            error_probs = probs[0, :, 1].tolist()

        word_ids = inputs.word_ids()
        error_flags = [False] * len(words)

        # Map predicted probabilities per token back to the source word
        for idx, word_idx in enumerate(word_ids):
            if word_idx is not None and error_probs[idx] > threshold:
                error_flags[word_idx] = True

        return error_flags

    def generate_candidates(self, words, idx, original_word):
        """ Candidate generation via SymSpell (O(1)). """
        original_word = original_word.lower()

        # O(1) lookup with SymSpell, drastically cutting candidate-generation time
        possible_targets = self.sym_spell.lookup(original_word, Verbosity.ALL, max_edit_distance=self.max_levenshtein_distance)

        # Extract the proposed terms and their distance
        candidates_with_dist = [(t.term, t.distance) for t in possible_targets]

        # Sort ascending by Levenshtein distance
        candidates_with_dist.sort(key=lambda x: x[1])

        # Remove the original word (already lowercased) from the candidates, if present
        all_candidates_with_dist = [(word, dist) for word, dist in candidates_with_dist if word != original_word]

        if not all_candidates_with_dist:
            return []

        # Keep only the candidates at the minimum distance found, or at most one more (min_dist + 1)
        min_dist = all_candidates_with_dist[0][1]  # Already sorted, so the first is the minimum

        # Filter by distance
        valid_cands = [
            (word, dist) for word, dist in all_candidates_with_dist
            if dist <= min_dist + 1
        ]

        # Deduplicate and cap the count to preserve the MLM's performance
        seen = set()
        final_candidates = []
        for word, dist in valid_cands:
            if word not in seen:
                seen.add(word)
                final_candidates.append(word)
                if len(final_candidates) >= self.max_candidates:
                    break

        return final_candidates


    def rank_candidates(self, words, error_idx, candidates, original_word):
        """
        Contextual ranking via Masked Language Model.
        Computes the direct log-likelihood for each candidate.
        Returns the candidates ordered by the hybrid logic: (1) Levenshtein
        distance, (2) MLM score.
        """
        if not candidates:
            return float('-inf'), []

        # Build the batch of sentence variants
        all_targets = [original_word] + candidates
        batch_sentences = []
        for target in all_targets:
            temp_words = words.copy()
            temp_words[error_idx] = target
            batch_sentences.append(temp_words)

        # Batch tokenization
        inputs = self.tokenizer(
            batch_sentences,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.mlm(**inputs)
            log_probs = torch.log_softmax(outputs.logits, dim=-1)

        scores = []

        # Extract the score for each sub-token belonging to the candidate word
        for batch_idx, target in enumerate(all_targets):
            word_ids = inputs.word_ids(batch_index=batch_idx)
            token_scores = []

            for i, word_idx in enumerate(word_ids):
                if word_idx == error_idx:
                    token_id = inputs["input_ids"][batch_idx, i]
                    token_scores.append(log_probs[batch_idx, i, token_id].item())

            score = sum(token_scores) / len(token_scores) if token_scores else float('-inf')
            scores.append(score)

        original_score = scores[0]

        # Final hybrid aggregation and ranking
        scored_candidates = []
        for i, cand in enumerate(candidates):
            cand_score = scores[i + 1]
            dist = jellyfish.damerau_levenshtein_distance(original_word.lower(), cand.lower())

            ###
            if self.debug:
                print(f"    * Candidate probability '{cand}' (dist {dist}): {cand_score:.4f}")
            ###

            scored_candidates.append({"word": cand, "score": cand_score, "dist": dist})

        scored_candidates.sort(key=lambda x: (x["dist"], -x["score"]))

        return original_score, scored_candidates

    def process_text(self, input_text):
        """
        Core method: runs the full analysis logic and extracts the complete "chunks".
        Returns a list of dicts that reconstructs the original string exactly.
        """
        # Use finditer to also capture exact indices (start, end)
        matches = list(self.word_pattern.finditer(input_text))
        words = [m.group() for m in matches]

        # Error detection
        error_flags = self.detect_errors(words, threshold=self.detection_threshold)

        analysis = []
        last_end = 0

        for idx, match in enumerate(matches):
            word = match.group()
            start, end = match.span()

            # Capture whitespace/punctuation between tokens
            if start > last_end:
                whitespace_chunk = input_text[last_end:start]
                analysis.append({
                    "word": whitespace_chunk,
                    "is_error": False,
                    "candidates": [],
                    "start": last_end,
                    "end": start
                })

            # Capture the analyzed text chunk
            is_error = error_flags[idx]
            word_data = {
                "word": word,
                "is_error": False,
                "candidates": [],
                "start": start,
                "end": end
            }

            # Basic filter: only correct words made mostly of letters
            if is_error and word.replace("'", "").replace("-", "").isalpha():

                ###
                if self.debug:
                    print(f"\n[DEBUG] Analyzing error on: '{word}'")
                ###

                # Candidate generation
                candidates = self.generate_candidates(words, idx, word)

                # Contextual ranking and final decision via threshold
                if candidates:

                    ###
                    if self.debug:
                        print(f"  - Candidates:    {candidates}")
                    ###

                    original_score, scored_candidates = self.rank_candidates(words, idx, candidates, word)

                    if scored_candidates:
                        best_score = scored_candidates[0]["score"]
                        diff = best_score - original_score

                        ###
                        if self.debug:
                            print(f"  - Final comparison: '{scored_candidates[0]['word']}' ({best_score:.4f}) vs '{word}' ({original_score:.4f})")
                            print(f"  - Difference: {diff:.4f} (Threshold: {self.correction_threshold})")
                        ###

                        # Flag as an error only if the delta exceeds the correction threshold
                        if diff > self.correction_threshold:
                            word_data["is_error"] = True

                            formatted_cands = []
                            for c in scored_candidates[:self.max_candidates]:
                                c_word = c["word"]
                                # Restore original casing
                                if word.istitle():
                                    c_word = c_word.capitalize()
                                elif word.isupper():
                                    c_word = c_word.upper()
                                formatted_cands.append(c_word)

                            word_data["candidates"] = formatted_cands

                            ###
                            if self.debug:
                                print(f"  >>> CORRECTED TO: '{formatted_cands[0]}'\n")
                            ###

                        ###
                        elif self.debug:
                            print(f"  >>> DISCARDED (improvement insufficient)\n")
                        ###

            analysis.append(word_data)
            last_end = end

        # Capture any remaining characters at the end of the string
        if last_end < len(input_text):
            final_chunk = input_text[last_end:]
            analysis.append({
                "word": final_chunk,
                "is_error": False,
                "candidates": [],
                "start": last_end,
                "end": len(input_text)
            })

        return analysis

    def correct_text(self, input_text):
        """ Full text-correction pipeline, returning only the corrected string. """
        analysis = self.process_text(input_text)

        output_text = input_text
        # Process in reverse so replacements don't shift indices when the new word
        # has a different length
        for chunk in reversed(analysis):
            if chunk["is_error"] and chunk["candidates"]:
                best_match = chunk["candidates"][0]
                start = chunk["start"]
                end = chunk["end"]
                output_text = output_text[:start] + best_match + output_text[end:]

        return output_text

    def analyze_text(self, input_text):
        """ Returns the JSON analysis for the GUI, stripping unneeded metadata. """
        analysis = self.process_text(input_text)

        # Drop exact indices, the JS frontend doesn't need them
        for chunk in analysis:
            chunk.pop("start", None)
            chunk.pop("end", None)

        return analysis
