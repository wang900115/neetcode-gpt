from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed
        tokens = list(corpus)
        merges = []

        for _ in range(num_merges):
            pairs = Counter(
                (tokens[i], tokens[i+1])
                for i in range(len(tokens)-1)
            )
        
            if not pairs: 
                break
            
            best_pair = min(
                pairs,
                key = lambda pair: (-pairs[pair], pair)
            )

            a, b = best_pair
            merges.append([a,b])

            merged_tokens = []
            i = 0
            while i < len(tokens):
                if (
                    i + 1 <len(tokens)
                    and tokens[i] == a
                    and tokens[i+1] == b 
                ):
                    merged_tokens.append(a+b)
                    i += 2
                else:
                    merged_tokens.append(tokens[i])
                    i += 1
            
            tokens = merged_tokens

        return merges