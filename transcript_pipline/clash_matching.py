from sentence_transformers import SentenceTransformer, util

#essentially, here, even though the argument and what the debator says are different in words, it anaylyses whether the meaning and intention are the same.

embedder = SentenceTransformer('all-MiniLM-L6-v2') # the embedders converts meaning that humans understand into computable numbers on a co-ordinated graph/vector
