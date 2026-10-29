from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        # 3. Return the list of merges performed
        tokens = list(corpus)
        merges = []

        for _ in range(num_merges):
        #    a. Count frequency of all adjacent token pairs
            pairs = Counter(
                (tokens[i], tokens[i+1])
                for i in range(len(tokens)-1)
            )
        
            if not pairs: 
                break
            
        #    b. Find the most frequent pair (break ties lexicographically)
            best_pair = min(
                pairs,
                key = lambda pair: (-pairs[pair], pair)
            )

            a, b = best_pair
        #    d. Record the merge as [token_a, token_b]
            merges.append([a,b])

        #    c. Merge all non-overlapping occurrences left to right
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