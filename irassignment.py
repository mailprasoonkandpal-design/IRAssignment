import csv
import io
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import get_close_matches
from typing import Dict, List, Optional, Set, Tuple

import streamlit as st


STOPWORDS = {
	"a",
	"an",
	"and",
	"are",
	"as",
	"at",
	"be",
	"by",
	"for",
	"from",
	"has",
	"he",
	"in",
	"is",
	"it",
	"its",
	"of",
	"on",
	"that",
	"the",
	"to",
	"was",
	"were",
	"will",
	"with",
	"this",
	"these",
	"those",
	"or",
	"not",
	"but",
	"if",
	"then",
	"than",
	"into",
	"out",
	"about",
	"over",
	"under",
}


def basic_tokenize(text: str) -> List[str]:
	return re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", text)


def handle_hyphen(token: str, mode: str) -> List[str]:
	if "-" not in token:
		return [token]
	if mode == "keep":
		return [token]
	if mode == "split":
		return [part for part in token.split("-") if part]
	if mode == "both":
		parts = [part for part in token.split("-") if part]
		compact = token.replace("-", "")
		return parts + ([compact] if compact else [])
	return [token]


def simple_stem(word: str) -> str:
	suffixes = ["ingly", "edly", "ing", "edly", "ed", "ies", "es", "s"]
	w = word
	for suffix in suffixes:
		if len(w) > len(suffix) + 2 and w.endswith(suffix):
			if suffix == "ies":
				return w[:-3] + "y"
			return w[: -len(suffix)]
	return w


def simple_lemmatize(word: str) -> str:
	irregular = {
		"children": "child",
		"mice": "mouse",
		"geese": "goose",
		"men": "man",
		"women": "woman",
		"better": "good",
		"best": "good",
		"worse": "bad",
		"went": "go",
	}
	if word in irregular:
		return irregular[word]
	if word.endswith("ies") and len(word) > 4:
		return word[:-3] + "y"
	if word.endswith("ves") and len(word) > 4:
		return word[:-3] + "f"
	if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
		return word[:-1]
	return word


def preprocess_text(
	text: str,
	lower: bool,
	remove_stop: bool,
	hyphen_mode: str,
	normalize: str,
) -> List[str]:
	raw_tokens = basic_tokenize(text)
	out = []
	for token in raw_tokens:
		t = token.lower() if lower else token
		expanded = handle_hyphen(t, hyphen_mode)
		for e in expanded:
			if remove_stop and e in STOPWORDS:
				continue
			if normalize == "stemming":
				e = simple_stem(e)
			elif normalize == "lemmatization":
				e = simple_lemmatize(e)
			out.append(e)
	return out


def build_inverted_index(tokenized_docs: List[List[str]]) -> Dict[str, Set[int]]:
	inv = defaultdict(set)
	for doc_id, tokens in enumerate(tokenized_docs):
		for tok in tokens:
			inv[tok].add(doc_id)
	return dict(inv)


def build_positional_index(tokenized_docs: List[List[str]]) -> Dict[str, Dict[int, List[int]]]:
	pos = defaultdict(lambda: defaultdict(list))
	for doc_id, tokens in enumerate(tokenized_docs):
		for idx, tok in enumerate(tokens):
			pos[tok][doc_id].append(idx)
	return {term: dict(doc_map) for term, doc_map in pos.items()}


def build_biword_index(tokenized_docs: List[List[str]]) -> Dict[str, Set[int]]:
	biword = defaultdict(set)
	for doc_id, tokens in enumerate(tokenized_docs):
		for i in range(len(tokens) - 1):
			bi = f"{tokens[i]} {tokens[i + 1]}"
			biword[bi].add(doc_id)
	return dict(biword)


def phrase_query_biword(phrase_tokens: List[str], biword_index: Dict[str, Set[int]]) -> Set[int]:
	if len(phrase_tokens) < 2:
		return set()
	pairs = [f"{phrase_tokens[i]} {phrase_tokens[i + 1]}" for i in range(len(phrase_tokens) - 1)]
	docs = None
	for pair in pairs:
		current = biword_index.get(pair, set())
		docs = current if docs is None else docs.intersection(current)
	return docs if docs is not None else set()


def phrase_query_positional(
	phrase_tokens: List[str],
	positional_index: Dict[str, Dict[int, List[int]]],
) -> Set[int]:
	if not phrase_tokens:
		return set()
	candidate_docs = None
	for t in phrase_tokens:
		docs = set(positional_index.get(t, {}).keys())
		candidate_docs = docs if candidate_docs is None else candidate_docs.intersection(docs)
	if not candidate_docs:
		return set()

	matched = set()
	for doc_id in candidate_docs:
		first_positions = positional_index[phrase_tokens[0]][doc_id]
		for p in first_positions:
			ok = True
			for offset, term in enumerate(phrase_tokens[1:], start=1):
				if (p + offset) not in positional_index[term][doc_id]:
					ok = False
					break
			if ok:
				matched.add(doc_id)
				break
	return matched


@dataclass
class BSTNode:
	key: str
	postings: Set[int]
	left: Optional["BSTNode"] = None
	right: Optional["BSTNode"] = None


class BinarySearchTree:
	def __init__(self):
		self.root: Optional[BSTNode] = None

	def insert(self, key: str, postings: Set[int]) -> None:
		if self.root is None:
			self.root = BSTNode(key, postings)
			return
		cur = self.root
		while True:
			if key < cur.key:
				if cur.left is None:
					cur.left = BSTNode(key, postings)
					return
				cur = cur.left
			elif key > cur.key:
				if cur.right is None:
					cur.right = BSTNode(key, postings)
					return
				cur = cur.right
			else:
				cur.postings = postings
				return

	def search(self, key: str) -> Optional[Set[int]]:
		cur = self.root
		while cur is not None:
			if key < cur.key:
				cur = cur.left
			elif key > cur.key:
				cur = cur.right
			else:
				return cur.postings
		return None


class BTreeNode:
	def __init__(self, leaf: bool):
		self.leaf = leaf
		self.keys: List[str] = []
		self.values: List[Set[int]] = []
		self.children: List["BTreeNode"] = []


class BTree:
	def __init__(self, t: int = 3):
		self.t = t
		self.root = BTreeNode(True)

	def search(self, key: str, node: Optional[BTreeNode] = None) -> Optional[Set[int]]:
		if node is None:
			node = self.root
		i = 0
		while i < len(node.keys) and key > node.keys[i]:
			i += 1
		if i < len(node.keys) and key == node.keys[i]:
			return node.values[i]
		if node.leaf:
			return None
		return self.search(key, node.children[i])

	def split_child(self, parent: BTreeNode, i: int) -> None:
		t = self.t
		y = parent.children[i]
		z = BTreeNode(y.leaf)

		mid_key = y.keys[t - 1]
		mid_value = y.values[t - 1]

		z.keys = y.keys[t:]
		z.values = y.values[t:]
		y.keys = y.keys[: t - 1]
		y.values = y.values[: t - 1]

		if not y.leaf:
			z.children = y.children[t:]
			y.children = y.children[:t]

		parent.children.insert(i + 1, z)
		parent.keys.insert(i, mid_key)
		parent.values.insert(i, mid_value)

	def insert(self, key: str, value: Set[int]) -> None:
		root = self.root
		if len(root.keys) == (2 * self.t) - 1:
			new_root = BTreeNode(False)
			new_root.children.append(root)
			self.split_child(new_root, 0)
			self.root = new_root
			self._insert_non_full(new_root, key, value)
		else:
			self._insert_non_full(root, key, value)

	def _insert_non_full(self, node: BTreeNode, key: str, value: Set[int]) -> None:
		i = len(node.keys) - 1
		if node.leaf:
			while i >= 0 and key < node.keys[i]:
				i -= 1
			if i >= 0 and node.keys[i] == key:
				node.values[i] = value
				return
			node.keys.insert(i + 1, key)
			node.values.insert(i + 1, value)
			return

		while i >= 0 and key < node.keys[i]:
			i -= 1
		i += 1

		if i < len(node.keys) and node.keys[i] == key:
			node.values[i] = value
			return

		if len(node.children[i].keys) == (2 * self.t) - 1:
			self.split_child(node, i)
			if key > node.keys[i]:
				i += 1
			elif key == node.keys[i]:
				node.values[i] = value
				return
		self._insert_non_full(node.children[i], key, value)


def boolean_and_query(query: str, inv_index: Dict[str, Set[int]], preprocess_fn) -> Set[int]:
	terms = preprocess_fn(query)
	if not terms:
		return set()
	result = None
	for t in terms:
		docs = inv_index.get(t, set())
		result = docs if result is None else result.intersection(docs)
	return result if result is not None else set()


def precision_at_k(retrieved: List[int], relevant: Set[int], k: int = 5) -> float:
	top = retrieved[:k]
	if not top:
		return 0.0
	good = sum(1 for d in top if d in relevant)
	return good / len(top)


def edit_distance(a: str, b: str) -> int:
	if a == b:
		return 0
	if not a:
		return len(b)
	if not b:
		return len(a)
	dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
	for i in range(len(a) + 1):
		dp[i][0] = i
	for j in range(len(b) + 1):
		dp[0][j] = j
	for i in range(1, len(a) + 1):
		for j in range(1, len(b) + 1):
			cost = 0 if a[i - 1] == b[j - 1] else 1
			dp[i][j] = min(
				dp[i - 1][j] + 1,
				dp[i][j - 1] + 1,
				dp[i - 1][j - 1] + cost,
			)
	return dp[-1][-1]


def soundex(term: str) -> str:
	term = term.upper()
	if not term:
		return ""
	first = term[0]
	mapping = {
		"B": "1",
		"F": "1",
		"P": "1",
		"V": "1",
		"C": "2",
		"G": "2",
		"J": "2",
		"K": "2",
		"Q": "2",
		"S": "2",
		"X": "2",
		"Z": "2",
		"D": "3",
		"T": "3",
		"L": "4",
		"M": "5",
		"N": "5",
		"R": "6",
	}
	digits = [mapping.get(ch, "0") for ch in term[1:]]
	compressed = []
	prev = ""
	for d in digits:
		if d != prev:
			compressed.append(d)
		prev = d
	filtered = [d for d in compressed if d != "0"]
	code = first + "".join(filtered)
	code = (code + "000")[:4]
	return code


def build_kgram_index(vocab: Set[str], k: int = 3) -> Dict[str, Set[str]]:
	idx = defaultdict(set)
	for term in vocab:
		padded = f"${term}$"
		for i in range(len(padded) - k + 1):
			gram = padded[i : i + k]
			idx[gram].add(term)
	return dict(idx)


def kgram_candidates(term: str, kgram_idx: Dict[str, Set[str]], k: int = 3) -> List[Tuple[str, float]]:
	grams = set()
	padded = f"${term}$"
	for i in range(len(padded) - k + 1):
		grams.add(padded[i : i + k])
	candidates = Counter()
	for g in grams:
		for t in kgram_idx.get(g, set()):
			candidates[t] += 1
	scored = []
	for cand, overlap in candidates.items():
		cand_grams = {f"${cand}$"[i : i + k] for i in range(len(cand) + 2 - k + 1)}
		union = len(grams.union(cand_grams))
		jaccard = overlap / union if union else 0.0
		scored.append((cand, jaccard))
	scored.sort(key=lambda x: x[1], reverse=True)
	return scored[:10]


def read_uploaded_docs(uploaded_files) -> List[str]:
	docs = []
	for f in uploaded_files:
		name = f.name.lower()
		if name.endswith(".txt"):
			text = f.read().decode("utf-8", errors="ignore")
			parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
			if parts:
				docs.extend(parts)
		elif name.endswith(".csv"):
			text = f.read().decode("utf-8", errors="ignore")
			reader = csv.reader(io.StringIO(text))
			rows = [" ".join(col.strip() for col in row if col.strip()) for row in reader]
			docs.extend([r for r in rows if r])
	return docs


def main() -> None:
	st.set_page_config(page_title="IR Assignment 1", layout="wide")
	st.title("Information Retrieval System - Streamlit Workflow")
	st.caption("Upload documents, preprocess, index, query, and compare retrieval strategies.")

	st.sidebar.header("Configuration")
	lower = st.sidebar.checkbox("Lowercasing", value=True)
	remove_stop = st.sidebar.checkbox("Stop word removal", value=True)
	hyphen_mode = st.sidebar.selectbox("Hyphen handling", ["keep", "split", "both"], index=2)
	normalize = st.sidebar.selectbox("Normalization", ["none", "stemming", "lemmatization"], index=0)

	uploaded = st.file_uploader(
		"Upload dataset (.txt and/or .csv)",
		type=["txt", "csv"],
		accept_multiple_files=True,
	)

	if not uploaded:
		st.info("Upload one or more files to start the end-to-end IR workflow.")
		return

	docs = read_uploaded_docs(uploaded)
	if not docs:
		st.warning("No valid documents were parsed. Please check the uploaded file format.")
		return

	def preprocess_for_mode(text: str, mode: str) -> List[str]:
		return preprocess_text(
			text,
			lower=lower,
			remove_stop=remove_stop,
			hyphen_mode=hyphen_mode,
			normalize=mode,
		)

	tokenized_docs = [preprocess_for_mode(d, normalize) for d in docs]
	inv_index = build_inverted_index(tokenized_docs)
	pos_index = build_positional_index(tokenized_docs)
	biword_index = build_biword_index(tokenized_docs)
	vocab = set(inv_index.keys())

	tabs = st.tabs(
		[
			"A) Workflow",
			"B) Preprocessing",
			"C) Phrase Queries",
			"D) BST vs B-Tree",
			"E) Tolerant Retrieval",
			"G) Inference",
		]
	)

	with tabs[0]:
		st.subheader("Uploaded Documents")
		st.write(f"Total parsed documents: {len(docs)}")
		max_preview = st.slider("Preview first N documents", min_value=1, max_value=min(20, len(docs)), value=min(5, len(docs)))
		for i, doc in enumerate(docs[:max_preview], start=1):
			st.markdown(f"**Doc {i}**")
			st.write(doc[:700] + ("..." if len(doc) > 700 else ""))

		st.subheader("Front-end Query Execution")
		q = st.text_input("Enter query")
		if st.button("Run Boolean AND Query"):
			hits = boolean_and_query(
				q,
				inv_index,
				lambda txt: preprocess_text(txt, lower, remove_stop, hyphen_mode, normalize),
			)
			st.write(f"Matched document IDs (0-based): {sorted(hits)}")
			for doc_id in sorted(hits)[:10]:
				st.write(f"Doc {doc_id}: {docs[doc_id][:250]}")

	with tabs[1]:
		st.subheader("Preprocessing Effects")
		sample_id = st.number_input("Document ID for before/after view", min_value=0, max_value=len(docs) - 1, value=0)
		raw_tokens = basic_tokenize(docs[sample_id])
		final_tokens = tokenized_docs[sample_id]
		st.write("Raw tokens (first 40):", raw_tokens[:40])
		st.write("Preprocessed tokens (first 40):", final_tokens[:40])

		st.subheader("Inverted Index Snapshot")
		top_terms = sorted(inv_index.keys())[:20]
		for t in top_terms:
			st.write(f"{t}: {sorted(inv_index[t])}")

		st.subheader("Stemming vs Lemmatization Comparison")
		eval_queries_txt = st.text_area(
			"Evaluation queries (one per line)",
			value="information retrieval\nsearch engine\ntext processing",
		)
		eval_queries = [x.strip() for x in eval_queries_txt.splitlines() if x.strip()]

		stem_docs = [preprocess_for_mode(d, "stemming") for d in docs]
		lem_docs = [preprocess_for_mode(d, "lemmatization") for d in docs]
		stem_inv = build_inverted_index(stem_docs)
		lem_inv = build_inverted_index(lem_docs)

		rows = []
		for query in eval_queries:
			q_stem = preprocess_for_mode(query, "stemming")
			q_lem = preprocess_for_mode(query, "lemmatization")

			rel = set()
			for i, d in enumerate(docs):
				low = d.lower()
				if any(tok in low for tok in basic_tokenize(query.lower())):
					rel.add(i)

			stem_hits = []
			if q_stem:
				sets = [stem_inv.get(t, set()) for t in q_stem]
				stem_set = sets[0].copy() if sets else set()
				for s in sets[1:]:
					stem_set &= s
				stem_hits = sorted(stem_set)

			lem_hits = []
			if q_lem:
				sets = [lem_inv.get(t, set()) for t in q_lem]
				lem_set = sets[0].copy() if sets else set()
				for s in sets[1:]:
					lem_set &= s
				lem_hits = sorted(lem_set)

			p_stem = precision_at_k(stem_hits, rel, k=5)
			p_lem = precision_at_k(lem_hits, rel, k=5)
			rows.append(
				{
					"query": query,
					"P@5_stemming": round(p_stem, 3),
					"P@5_lemmatization": round(p_lem, 3),
					"stem_hits": len(stem_hits),
					"lemma_hits": len(lem_hits),
				}
			)

		st.dataframe(rows, use_container_width=True)
		if rows:
			avg_stem = sum(r["P@5_stemming"] for r in rows) / len(rows)
			avg_lem = sum(r["P@5_lemmatization"] for r in rows) / len(rows)
			better = "Stemming" if avg_stem > avg_lem else "Lemmatization" if avg_lem > avg_stem else "Tie"
			st.success(
				f"Average P@5 - Stemming: {avg_stem:.3f}, Lemmatization: {avg_lem:.3f}. Better: {better}."
			)

	with tabs[2]:
		st.subheader("Phrase Query: Biword vs Positional Index")
		phrase = st.text_input("Phrase query", value="information retrieval system")
		phrase_tokens = preprocess_text(phrase, lower, remove_stop, hyphen_mode, normalize)

		st.write("Biword index sample:")
		for key in sorted(list(biword_index.keys()))[:12]:
			st.write(f"{key}: {sorted(biword_index[key])}")

		st.write("Positional index sample:")
		for key in sorted(list(pos_index.keys()))[:8]:
			st.write(f"{key}: {pos_index[key]}")

		bi_hits = phrase_query_biword(phrase_tokens, biword_index)
		pos_hits = phrase_query_positional(phrase_tokens, pos_index)

		st.markdown("**Query result using biword index**")
		st.write(sorted(bi_hits))
		st.markdown("**Query result using positional index**")
		st.write(sorted(pos_hits))

		false_positive_docs = sorted(bi_hits - pos_hits)
		st.markdown("**Potential false positives from biword index**")
		st.write(false_positive_docs if false_positive_docs else "None detected for this phrase.")

		st.info(
			"Biword index matches adjacent pairs and can return documents where phrase pairs appear separately. "
			"Positional index enforces exact consecutive token positions, so phrase matching is more accurate."
		)

	with tabs[3]:
		st.subheader("Dictionary Search: BST vs B-Tree")

		bst = BinarySearchTree()
		btree = BTree(t=3)
		for term, postings in inv_index.items():
			bst.insert(term, postings)
			btree.insert(term, postings)

		query_terms_txt = st.text_area(
			"Dictionary terms to test (one per line)",
			value="information\nretrieval\nsystem\nsearch\nnonexistent",
		)
		query_terms = [q.strip() for q in query_terms_txt.splitlines() if q.strip()]

		perf_rows = []
		for term in query_terms:
			t0 = time.perf_counter()
			bst_res = bst.search(term)
			t1 = time.perf_counter()
			bst_lookup = (t1 - t0) * 1e6

			t2 = time.perf_counter()
			bst_docs = list(bst_res) if bst_res else []
			t3 = time.perf_counter()
			bst_retrieval = (t3 - t2) * 1e6

			t4 = time.perf_counter()
			btree_res = btree.search(term)
			t5 = time.perf_counter()
			btree_lookup = (t5 - t4) * 1e6

			t6 = time.perf_counter()
			btree_docs = list(btree_res) if btree_res else []
			t7 = time.perf_counter()
			btree_retrieval = (t7 - t6) * 1e6

			perf_rows.append(
				{
					"term": term,
					"bst_query_us": round(bst_lookup, 3),
					"bst_retrieval_us": round(bst_retrieval, 3),
					"btree_query_us": round(btree_lookup, 3),
					"btree_retrieval_us": round(btree_retrieval, 3),
					"bst_docs": len(bst_docs),
					"btree_docs": len(btree_docs),
				}
			)

		st.dataframe(perf_rows, use_container_width=True)
		if perf_rows:
			avg_bst_q = sum(r["bst_query_us"] for r in perf_rows) / len(perf_rows)
			avg_btree_q = sum(r["btree_query_us"] for r in perf_rows) / len(perf_rows)
			faster = "BST" if avg_bst_q < avg_btree_q else "B-Tree"
			st.success(
				f"Average query time -> BST: {avg_bst_q:.3f} us, B-Tree: {avg_btree_q:.3f} us. Faster: {faster}."
			)

	with tabs[4]:
		st.subheader("Tolerant Retrieval")
		misspelled = st.text_input("Imperfect query term", value="retrival")
		mode = st.selectbox(
			"Choose tolerant retrieval method",
			["Wildcard", "Spelling correction", "Edit distance", "K-gram", "Phonetic"],
		)

		if mode == "Wildcard":
			pattern = misspelled.replace("*", ".*")
			regex = re.compile(f"^{pattern}$")
			matched = sorted([t for t in vocab if regex.match(t)])[:30]
			st.write("Wildcard-matched terms:", matched)
			docs_hit = sorted(set().union(*(inv_index.get(t, set()) for t in matched))) if matched else []
			st.write("Matched document IDs:", docs_hit)

		elif mode == "Spelling correction":
			candidates = get_close_matches(misspelled, list(vocab), n=5, cutoff=0.6)
			st.write("Closest terms:", candidates)
			if candidates:
				st.write("Top candidate docs:", sorted(inv_index.get(candidates[0], set())))

		elif mode == "Edit distance":
			ranked = sorted([(term, edit_distance(misspelled, term)) for term in vocab], key=lambda x: x[1])[:10]
			st.write("Nearest terms by edit distance:", ranked)
			if ranked:
				st.write("Top candidate docs:", sorted(inv_index.get(ranked[0][0], set())))

		elif mode == "K-gram":
			k_idx = build_kgram_index(vocab, k=3)
			ranked = kgram_candidates(misspelled, k_idx, k=3)
			st.write("Top K-gram candidates (term, Jaccard score):", ranked)
			if ranked:
				st.write("Top candidate docs:", sorted(inv_index.get(ranked[0][0], set())))

		elif mode == "Phonetic":
			code = soundex(misspelled)
			matches = [term for term in vocab if soundex(term) == code]
			st.write(f"Soundex code: {code}")
			st.write("Phonetic matches:", matches[:20])
			docs_hit = sorted(set().union(*(inv_index.get(t, set()) for t in matches))) if matches else []
			st.write("Matched document IDs:", docs_hit)

	with tabs[5]:
		st.subheader("Inference and Discussion")

		stem_docs = [preprocess_for_mode(d, "stemming") for d in docs]
		lem_docs = [preprocess_for_mode(d, "lemmatization") for d in docs]
		stem_inv = build_inverted_index(stem_docs)
		lem_inv = build_inverted_index(lem_docs)

		sample_queries = ["information retrieval", "search system", "text mining"]
		stem_scores = []
		lem_scores = []
		for query in sample_queries:
			q_st = preprocess_for_mode(query, "stemming")
			q_le = preprocess_for_mode(query, "lemmatization")

			rel = set(i for i, d in enumerate(docs) if any(tok in d.lower() for tok in basic_tokenize(query.lower())))
			st_hits = set.intersection(*(stem_inv.get(t, set()) for t in q_st)) if q_st else set()
			le_hits = set.intersection(*(lem_inv.get(t, set()) for t in q_le)) if q_le else set()
			stem_scores.append(precision_at_k(sorted(st_hits), rel, 5))
			lem_scores.append(precision_at_k(sorted(le_hits), rel, 5))

		avg_st = sum(stem_scores) / len(stem_scores) if stem_scores else 0.0
		avg_le = sum(lem_scores) / len(lem_scores) if lem_scores else 0.0

		phrase = "information retrieval system"
		p_tokens = preprocess_text(phrase, lower, remove_stop, hyphen_mode, normalize)
		bi_hits = phrase_query_biword(p_tokens, biword_index)
		po_hits = phrase_query_positional(p_tokens, pos_index)

		bst = BinarySearchTree()
		btree = BTree(t=3)
		for term, postings in inv_index.items():
			bst.insert(term, postings)
			btree.insert(term, postings)

		test_terms = ["information", "retrieval", "system", "search", "model"]
		bst_time = 0.0
		bt_time = 0.0
		for t in test_terms:
			t0 = time.perf_counter()
			bst.search(t)
			t1 = time.perf_counter()
			bst_time += t1 - t0

			t2 = time.perf_counter()
			btree.search(t)
			t3 = time.perf_counter()
			bt_time += t3 - t2

		preprocessing_best = "Stop word removal + normalization" if remove_stop or normalize != "none" else "Tokenization only"
		stem_vs_lemma = "Stemming" if avg_st > avg_le else "Lemmatization" if avg_le > avg_st else "Both performed similarly"
		phrase_best = "Positional index" if len(po_hits) <= len(bi_hits) else "Biword index"
		tree_faster = "BST" if bst_time < bt_time else "B-Tree"

		st.markdown("1. **Which preprocessing technique improved retrieval quality?**")
		st.write(f"{preprocessing_best} improved precision by reducing noisy terms and term variants.")

		st.markdown("2. **Was stemming or lemmatization better for the dataset?**")
		st.write(f"Based on average P@5, {stem_vs_lemma} is better for this dataset.")

		st.markdown("3. **Which phrase query index was more accurate?**")
		st.write(
			f"{phrase_best} is more accurate; biword had {len(bi_hits - po_hits)} potential false positives in this run."
		)

		st.markdown("4. **Which tree structure was faster?**")
		st.write(f"{tree_faster} had lower average query latency on this dataset size.")

		st.markdown("5. **How tolerant was the retrieval model?**")
		st.write(
			"The system handled imperfect input via wildcard matching, nearest-term correction, edit-distance ranking, "
			"k-gram candidate generation, and phonetic matching."
		)

		st.markdown("6. **What are the limitations of the system?**")
		st.write(
			"This is an in-memory prototype, uses lightweight stemming/lemmatization heuristics, and does not include advanced ranking "
			"models like BM25 or semantic embeddings."
		)

		st.markdown("7. **How can the system be improved?**")
		st.write(
			"Add true linguistic NLP tools, scalable persistent indexes, BM25/vector ranking, larger evaluation sets with relevance labels, "
			"and richer query parsing (OR/NOT, proximity, fuzzy operators)."
		)


if __name__ == "__main__":
	main()
