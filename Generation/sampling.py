# Re-run code after kernel reset

import numpy as np
import faiss
import random
from sentence_transformers import SentenceTransformer
from collections import Counter
from typing import List, Dict, Any, Tuple
from itertools import islice, chain


import numpy as np
import faiss
import random
from sentence_transformers import SentenceTransformer
from collections import Counter
from typing import List, Dict, Any, Tuple
from itertools import islice, chain


class SampleManager:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name, device='cuda')  # Use GPU for acceleration
        self.samples = []
        self.encoded_samples = None
        self.index = None
        self.checked_pairs = set()
        self.gpu_res = faiss.StandardGpuResources()
        self._index_initialized = False
        self.dimension = None

    def add_samples(self, new_samples: List[str]):
        if not new_samples:
            return

        print("start encoding new samples")
        new_encoded = self.model.encode(
            new_samples,
            convert_to_numpy=True,
            batch_size=64,
            show_progress_bar= True,
            normalize_embeddings=True  
        ).astype('float32')
        print("end encoding new samples")
        start_idx = len(self.samples)
        self.samples.extend(new_samples)

        if self.encoded_samples is None:
            self.encoded_samples = new_encoded
            self.dimension = new_encoded.shape[1]
            self._initialize_index()
        else:
            self.encoded_samples = np.vstack((self.encoded_samples, new_encoded))
            self._add_to_index(new_encoded)

        self._update_distance_matrix_incrementally(new_encoded, start_idx)

    def _initialize_index(self):
        if not self._index_initialized and self.dimension is not None:
            d = self.dimension
            cpu_index = faiss.IndexFlatIP(d)
            self.index = faiss.index_cpu_to_gpu(self.gpu_res, 0, cpu_index)
            
            n = len(self.encoded_samples)
            self.distance_matrix = np.zeros((n, n), dtype=np.float32)
            np.fill_diagonal(self.distance_matrix, -np.inf)
            
            faiss.normalize_L2(self.encoded_samples)
            self.index.add(self.encoded_samples)
            
            self.distance_matrix = 1 - np.dot(self.encoded_samples, self.encoded_samples.T)
            np.fill_diagonal(self.distance_matrix, -np.inf)
            
            self._index_initialized = True

    def _add_to_index(self, new_encoded):
        if self.index is not None:
            faiss.normalize_L2(new_encoded)
            self.index.add(new_encoded)

    def _update_distance_matrix_incrementally(self, new_encoded, start_idx):
        if len(self.samples) <= 1:
            return
            
        n_old = start_idx
        n_new = len(new_encoded)
        n_total = n_old + n_new
        
        if n_old == 0:
            return
            
        new_distance_matrix = np.zeros((n_total, n_total), dtype=np.float32)
        new_distance_matrix[:n_old, :n_old] = self.distance_matrix
        

        new_old_dist = 1 - np.dot(new_encoded, self.encoded_samples[:n_old].T)
        new_distance_matrix[n_old:, :n_old] = new_old_dist
        new_distance_matrix[:n_old, n_old:] = new_old_dist.T
        
        new_new_dist = 1 - np.dot(new_encoded, new_encoded.T)
        new_distance_matrix[n_old:, n_old:] = new_new_dist
        
        np.fill_diagonal(new_distance_matrix, -np.inf)
        
        self.distance_matrix = new_distance_matrix

    def find_least_similar_pair(self, exclude_checked=False):
        if self.encoded_samples is None or len(self.samples) < 2:
            return None, None

        n = len(self.samples)
        min_ip = float('inf')
        least_similar_pair = None

        flat_indices = np.unravel_index(np.argmax(self.distance_matrix), self.distance_matrix.shape)
        i, j = flat_indices
        
        pair = tuple(sorted((i, j)))
        if exclude_checked and pair in self.checked_pairs:
            temp_matrix = self.distance_matrix.copy()
            temp_matrix[i, j] = -np.inf
            temp_matrix[j, i] = -np.inf
            flat_indices = np.unravel_index(np.argmax(temp_matrix), temp_matrix.shape)
            i, j = flat_indices
            pair = tuple(sorted((i, j)))
        
        self.checked_pairs.add(pair)
        cosine_distance = self.distance_matrix[i, j]
        return pair, cosine_distance

    def get_sample(self, index):
        return self.samples[index]

    def get_random_sample(self):
        if not self.samples:
            raise IndexError("Cannot get a random sample from an empty set.")
        return random.choice(self.samples)

    def calculate_diversity_scores(self, n_gram_range: Tuple[int, int] = (1, 3)) -> dict:
        aps_score = self._calculate_aps()
        ingf_score = self._calculate_ingf(n_gram_range)
        return {
            "APS": aps_score,
            "INGF": ingf_score
        }

    def _calculate_aps(self) -> float:
        if len(self.encoded_samples) < 2:
            return 0.0

        similarity_matrix = 1 - self.distance_matrix  
        np.fill_diagonal(similarity_matrix, 0)
        total_similarity = np.sum(np.triu(similarity_matrix, 1))
        pair_count = len(self.encoded_samples) * (len(self.encoded_samples) - 1) / 2
        return total_similarity / pair_count if pair_count > 0 else 0.0

    def _calculate_ingf(self, n_gram_range: Tuple[int, int]) -> float:
        def ngrams(seq, n):
            return zip(*(islice(seq, i, None) for i in range(n)))

        def get_ngrams(sample, n_gram_range):
            tokens = sample.split()
            return chain.from_iterable(ngrams(tokens, n) for n in range(n_gram_range[0], n_gram_range[1] + 1))

        all_ngrams = list(chain.from_iterable(get_ngrams(sample, n_gram_range) for sample in self.samples))
        ngram_counts = Counter(all_ngrams)
        total_ngrams = sum(ngram_counts.values())
        if total_ngrams == 0:
            return 0.0
        count_array = np.array(list(ngram_counts.values()))
        ingf_score = np.sum((count_array / total_ngrams) ** 2)
        return ingf_score
