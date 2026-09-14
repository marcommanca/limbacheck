import random
import re
import unicodedata


class SyntheticErrorGenerator:
    """Injects synthetic spelling errors into a clean sentence, for data augmentation."""

    def __init__(self, error_rate=0.15):
        self.error_rate = error_rate

        # Error type probabilities:
        # - phonetic: applies confusion rules based on the official Sardinian graphemic repertoire.
        # - typo:     simulates keyboard typing mistakes.
        # - accent:   drops the accent from the word.
        self.error_types = ['phonetic', 'typo', 'accent']
        self.error_weights = [0.45, 0.35, 0.20]

        # Typos: mapping to keyboard-adjacent letters
        self.typos = {
            'q': ['w', 'a'],                     'w': ['q', 'e', 'a', 's'],
            'e': ['w', 'r', 's', 'd'],           'r': ['e', 't', 'd', 'f'],
            't': ['r', 'y', 'f', 'g'],           'y': ['t', 'u', 'g', 'h', 'j'],
            'u': ['y', 'i', 'h', 'j'],           'i': ['u', 'o', 'j', 'k'],
            'o': ['i', 'p', 'k', 'l'],           'p': ['o', 'l'],
            'a': ['q', 'w', 's', 'z'],           's': ['w', 'e', 'a', 'd', 'z', 'x'],
            'd': ['e', 'r', 's', 'f', 'x', 'c'], 'f': ['r', 't', 'd', 'g', 'c', 'v'],
            'g': ['t', 'y', 'f', 'h', 'v', 'b'], 'h': ['y', 'u', 'g', 'j', 'b', 'n'],
            'j': ['u', 'i', 'h', 'k', 'n', 'm'], 'k': ['i', 'o', 'j', 'l', 'm'],
            'l': ['i', 'o', 'p', 'k'],           'z': ['a', 's', 'x'],
            'x': ['s', 'd', 'z', 'c'],           'c': ['d', 'f', 'x', 'v'],
            'v': ['f', 'g', 'c', 'b'],           'b': ['g', 'h', 'v', 'n'],
            'n': ['h', 'j', 'b', 'm'],           'm': ['j', 'k', 'n'],

            'à': ['a'],                          'è': ['e'],
            'é': ['e'],                          'ì': ['i'],
            'ò': ['o'],                          'ù': ['u'],
        }

        # Confusion sets based on the official Sardinian graphemic repertoire
        self.phonetic_rules = {
            # Degemination / hypercorrection (also lenition)
            'bb': ['b'],        'b': ['bb', 'v'],       'v':   ['b'],
            'cc': ['c'],
            'dd': ['d'],        'd': ['r', 't', 'dd'],
            'ff': ['f'],        'f': ['ff', 'v'],
            'gg': ['g'],        'g': ['gh', 'gg'],
            'll': ['l'],
            'mm': ['m'],
            'nn': ['n'],
            'pp': ['p'],        'p': ['pp', 'b'],
            'rr': ['r'],        'r': ['l', 'rr'],
            'ss': ['s'],        's': ['ss', 'z'],
            'tt': ['t'],        't': ['d', 'tt', 'th'], 'th':  ['t', 'z', 's'],

            # Nasals and palatals
            'mb':  ['mm'],        'nd':  ['nn'],
            'gi':  ['j', 'ghi'],  'gn':  ['nn', 'ni'],
            'j':   ['i', 'gi'],

            # Affricates: a lot of variation in Sardinian orthography
            'tz': ['z', 'ss', 'c'], 'z':  ['tz', 's'],

            # Italian-Sardinian conflicts from speech-to-writing transcription
            'xe':  ['sce', 'ge', 'se'],   'xi':  ['sci', 'gi', 'si'],
            'sce': ['xe', 'se', 'ce'],    'sci': ['xi', 'si', 'ci'],
            'ci':  ['sci', 'chi'],        'ce':  ['sce', 'che'],
            'che': ['ce', 'ke'],          'chi': ['ci', 'ki'],
            'ghe': ['ge'],                'ghi': ['gi'],
        }

    def strip_accents(self, word: str) -> str:
        """Removes accents from a word."""
        nfkd = unicodedata.normalize('NFKD', word)
        return "".join([char for char in nfkd if not unicodedata.combining(char)])

    def inject_phonetic_error(self, word: str) -> str:
        """Replaces a substring using rules based on the graphemic repertoire."""

        # Prefer longer patterns first (e.g. a word containing 'tz' also contains
        # 'z', so 'tz' should take priority)
        possible_targets = sorted(
            [t for t in self.phonetic_rules.keys() if t in word],
            key=len, reverse=True
        )
        if not possible_targets:
            return word

        weights = [len(t)**2 for t in possible_targets]
        chosen_target = random.choices(possible_targets, weights=weights, k=1)[0]
        replacement = random.choice(self.phonetic_rules[chosen_target])
        return word.replace(chosen_target, replacement, 1)

    def inject_keyboard_typo(self, word: str) -> str:
        """Substitutes, inserts, deletes, or transposes a character, simulating a keyboard typo."""

        oper = random.choice(['sub', 'ins', 'del', 'inv'])
        idx = random.randint(0, len(word) - 1)

        char = word[idx]

        if oper == 'sub' and char in self.typos:
            typo_char = random.choice(self.typos[char])
            return word[:idx] + typo_char + word[idx+1:]

        elif oper == 'ins' and char in self.typos:
            typo_char = random.choice(self.typos[char])
            return word[:idx] + typo_char + word[idx:]

        elif oper == 'del' and len(word) > 3:
            return word[:idx] + word[idx+1:]

        elif oper == 'inv' and idx < len(word) - 1:
            return word[:idx] + word[idx+1] + word[idx] + word[idx+2:]

        return word

    def corrupt_word(self, word: str) -> str:
        """Applies one of the possible error types to a single word."""

        choice = random.choices(
            self.error_types,
            weights=self.error_weights,
            k=1
        )[0]

        if choice == 'phonetic':
            return self.inject_phonetic_error(word)
        elif choice == 'typo':
            return self.inject_keyboard_typo(word)
        elif choice == 'accent':
            return self.strip_accents(word)

        return word

    def corrupt_sentence(self, sentence: str) -> dict:
        """
        Processes a whole sentence, returning the corrupted version, the
        tokenized word list, and the labels for the Token Classification task.
        """
        pattern = re.compile(r"['\-]?[a-zA-Zàèìòùáéíóú]+(?:['\-][a-zA-Zàèìòùáéíóú]+)*['\-]?|\d+|[^a-zA-Z0-9àèìòùáéíóú\s]")

        corrupted_words = []
        labels = []
        modifications = []

        for match in pattern.finditer(sentence):
            word = match.group(0)
            is_word = word.replace("'", "").replace("-", "").isalpha()

            noisy_word = word
            label = 0

            if is_word and len(word) > 2 and random.random() < self.error_rate:
                # Retry up to 5 times to try to guarantee the word actually gets corrupted
                for _ in range(5):
                    candidate = self.corrupt_word(word)
                    if candidate != word:
                        noisy_word = candidate
                        break

                if noisy_word != word:
                    label = 1
                    modifications.append({
                        "start": match.start(),
                        "end": match.end(),
                        "new_word": noisy_word
                    })

            corrupted_words.append(noisy_word)
            labels.append(label)

        # Rebuild the string using the original indices (in reverse)
        corrupted_text = sentence
        for mod in reversed(modifications):
            start = mod["start"]
            end = mod["end"]
            new_word = mod["new_word"]
            corrupted_text = corrupted_text[:start] + new_word + corrupted_text[end:]

        return {
            "original_text": sentence,
            "corrupted_text": corrupted_text,
            "corrupted_words_list": corrupted_words,
            "error_labels": labels
        }
