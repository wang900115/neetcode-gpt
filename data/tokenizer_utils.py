from typing import List, Dict

class Solution:
    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        # Tokenize each number using greedy left-to-right longest match.
        # Return a list of token lists showing how each number gets split.
         # Sort tokens by length (longest first).
        tokens = sorted(vocab.keys(), key=len, reverse=True)

        result = []

        for number in numbers:
            text = str(number)
            position = 0
            number_tokens = []

            while position < len(text):
                matched = False

                for token in tokens:
                    if text.startswith(token,position):
                        number_tokens.append(token)
                        position += len(token)
                        matched = True
                        break
                if not matched:
                    raise ValueError(f"No matching token at position {position}:{text}")
            result.append(number_tokens)
        return result

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # Count how many tokens the text uses with greedy tokenization.
        # Use greedy left-to-right longest match.
        tokens = sorted(vocab.keys(), key=len, reverse=True)

        count = 0
        position = 0
        while position < len(text):
            matched = False
            for token in tokens:
                if text.startswith(token, position):
                    count += 1
                    position += len(token)
                    matched = True
                    break
            if not matched:
                raise ValueError(f"No matching token at position {position}:{word}")
        return count


    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        # Higher = more expensive and less efficient.
        # Round to 4 decimal places.
        words = text.split()
        
        total = self.count_tokens(text, vocab)

        return round(total /len(words),4)
