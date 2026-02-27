"""
AI Risk Database Hybrid Search Service
Hybrid approach: lexical pre-filtering + semantic re-ranking.
Uses AI_risk_database_v4.csv as the source dataset.
"""

import csv
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from openai import OpenAI


# ====== ENVIRONMENT SETUP ======
load_dotenv('./.secrets')

API_GATEWAY_KEY = os.getenv('API_GATEWAY_KEY')
if not API_GATEWAY_KEY:
	raise ValueError("API_GATEWAY_KEY not found in .secrets file")

OPENAI_BASE_URL = 'https://k7uffyg03f.execute-api.us-east-1.amazonaws.com/prod/openai/v1'

client = OpenAI(
	default_headers={"x-api-key": API_GATEWAY_KEY},
	base_url=OPENAI_BASE_URL
)


# ====== PATHS ======
BASE_DIR = os.path.dirname(__file__)
DATASET_PATH = os.path.join(BASE_DIR, "AI_risk_database_v4.csv")
CACHE_DIR = os.path.join(BASE_DIR, ".cache")
EMBED_CACHE_PATH = os.path.join(CACHE_DIR, "ai_risk_embeddings.json")


# ====== CONFIG ======
EMBEDDING_MODEL = "text-embedding-3-small"
MAX_EMBED_CHARS = 4000


def _ensure_cache_dir() -> None:
	if not os.path.exists(CACHE_DIR):
		os.makedirs(CACHE_DIR, exist_ok=True)


def _dataset_sha256(file_path: str) -> str:
	sha256 = hashlib.sha256()
	with open(file_path, "rb") as f:
		for chunk in iter(lambda: f.read(8192), b""):
			sha256.update(chunk)
	return sha256.hexdigest()


def _find_header_row(file_path: str) -> Tuple[int, List[str]]:
	with open(file_path, "r", encoding="utf-8", newline="") as f:
		reader = csv.reader(f)
		for i, row in enumerate(reader):
			if not row:
				continue
			if row[0].strip() == "Title" and "QuickRef" in row:
				return i, row
	raise ValueError("Could not find header row in AI_risk_database_v4.csv")


def load_ai_risk_records(file_path: str = DATASET_PATH) -> List[Dict[str, str]]:
	header_index, header_row = _find_header_row(file_path)

	records: List[Dict[str, str]] = []
	with open(file_path, "r", encoding="utf-8", newline="") as f:
		for _ in range(header_index):
			next(f)

		reader = csv.DictReader(f, fieldnames=header_row)
		for row in reader:
			cleaned = {k: (v.strip() if isinstance(v, str) else "") for k, v in row.items()}
			title = cleaned.get("Title", "")
			description = cleaned.get("Description", "")
			if not title and not description:
				continue
			records.append(cleaned)

	return records


def _truncate_text(text: str, max_chars: int = MAX_EMBED_CHARS) -> str:
	text = text.strip()
	if len(text) <= max_chars:
		return text
	return text[:max_chars] + "..."


def _build_document_text(record: Dict[str, str]) -> str:
	fields = [
		"Title",
		"Risk category",
		"Risk subcategory",
		"Description",
		"Additional ev.",
		"Category level",
		"Entity",
		"Intent",
		"Timing",
		"Domain",
		"Sub-domain",
		"QuickRef",
		"Ev_ID",
	]

	parts = []
	for field in fields:
		value = record.get(field, "")
		if value:
			parts.append(f"{field}: {value}")

	return "\n".join(parts)


def _tokenize(text: str) -> List[str]:
	text = text.lower()
	text = re.sub(r"[^a-z0-9\s]", " ", text)
	return [t for t in text.split() if t]


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
	if not vec_a or not vec_b or len(vec_a) != len(vec_b):
		return 0.0
	dot = 0.0
	norm_a = 0.0
	norm_b = 0.0
	for a, b in zip(vec_a, vec_b):
		dot += a * b
		norm_a += a * a
		norm_b += b * b
	if norm_a == 0.0 or norm_b == 0.0:
		return 0.0
	return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


@dataclass
class SearchResult:
	title: str
	risk_category: str
	risk_subcategory: str
	description: str
	quickref: str
	ev_id: str
	domain: str
	sub_domain: str
	lexical_score: float
	semantic_score: float
	combined_score: float

	def to_dict(self) -> Dict[str, object]:
		return {
			"title": self.title,
			"risk_category": self.risk_category,
			"risk_subcategory": self.risk_subcategory,
			"description": self.description,
			"quickref": self.quickref,
			"ev_id": self.ev_id,
			"domain": self.domain,
			"sub_domain": self.sub_domain,
			"lexical_score": round(self.lexical_score, 4),
			"semantic_score": round(self.semantic_score, 4),
			"combined_score": round(self.combined_score, 4),
		}


class HybridRiskSearchService:
	def __init__(self, embedding_model: str = EMBEDDING_MODEL):
		self.embedding_model = embedding_model
		self.records = load_ai_risk_records()
		self.doc_texts = [_build_document_text(r) for r in self.records]
		self.doc_tokens = [set(_tokenize(text)) for text in self.doc_texts]
		self.dataset_sha = _dataset_sha256(DATASET_PATH)
		self._embedding_cache: Dict[str, List[float]] = {}
		self._load_cache()

	def _load_cache(self) -> None:
		_ensure_cache_dir()
		if not os.path.exists(EMBED_CACHE_PATH):
			return
		try:
			with open(EMBED_CACHE_PATH, "r", encoding="utf-8") as f:
				data = json.load(f)
			if data.get("dataset_sha") != self.dataset_sha:
				return
			if data.get("model") != self.embedding_model:
				return
			embeddings = data.get("embeddings", {})
			if isinstance(embeddings, dict):
				self._embedding_cache = embeddings
		except Exception:
			self._embedding_cache = {}

	def _save_cache(self) -> None:
		_ensure_cache_dir()
		payload = {
			"dataset_sha": self.dataset_sha,
			"model": self.embedding_model,
			"embeddings": self._embedding_cache,
		}
		with open(EMBED_CACHE_PATH, "w", encoding="utf-8") as f:
			json.dump(payload, f)

	def _embed_text(self, text: str) -> List[float]:
		text = _truncate_text(text)
		response = client.embeddings.create(
			model=self.embedding_model,
			input=text
		)
		return response.data[0].embedding

	def _get_doc_embedding(self, index: int) -> List[float]:
		key = str(index)
		cached = self._embedding_cache.get(key)
		if cached:
			return cached
		embedding = self._embed_text(self.doc_texts[index])
		self._embedding_cache[key] = embedding
		self._save_cache()
		return embedding

	def _lexical_scores(self, query_tokens: List[str]) -> List[float]:
		if not query_tokens:
			return [0.0 for _ in self.doc_tokens]
		scores = []
		query_set = set(query_tokens)
		for tokens in self.doc_tokens:
			overlap = len(query_set.intersection(tokens))
			score = overlap / max(len(query_set), 1)
			scores.append(score)
		return scores

	def search(
		self,
		query: str,
		top_k: int = 5,
		lexical_k: int = 50,
		alpha: float = 0.35
	) -> List[Dict[str, object]]:
		if not query or not query.strip():
			return []

		query_tokens = _tokenize(query)
		lexical_scores = self._lexical_scores(query_tokens)

		ranked_indices = sorted(
			range(len(lexical_scores)),
			key=lambda i: lexical_scores[i],
			reverse=True
		)

		candidates = ranked_indices[:max(lexical_k, top_k)]

		query_embedding = self._embed_text(query)
		results: List[SearchResult] = []

		max_lexical = max([lexical_scores[i] for i in candidates], default=0.0)

		for idx in candidates:
			semantic_score = _cosine_similarity(query_embedding, self._get_doc_embedding(idx))
			lexical_score = lexical_scores[idx]

			lexical_norm = (lexical_score / max_lexical) if max_lexical > 0 else 0.0
			semantic_norm = (semantic_score + 1) / 2  # normalize from [-1,1] to [0,1]

			combined = (alpha * lexical_norm) + ((1 - alpha) * semantic_norm)

			record = self.records[idx]
			results.append(
				SearchResult(
					title=record.get("Title", ""),
					risk_category=record.get("Risk category", ""),
					risk_subcategory=record.get("Risk subcategory", ""),
					description=record.get("Description", ""),
					quickref=record.get("QuickRef", ""),
					ev_id=record.get("Ev_ID", ""),
					domain=record.get("Domain", ""),
					sub_domain=record.get("Sub-domain", ""),
					lexical_score=lexical_score,
					semantic_score=semantic_score,
					combined_score=combined,
				)
			)

		results.sort(key=lambda r: r.combined_score, reverse=True)
		return [r.to_dict() for r in results[:top_k]]


_SERVICE_INSTANCE: HybridRiskSearchService = None


def get_hybrid_search_service() -> HybridRiskSearchService:
	global _SERVICE_INSTANCE
	if _SERVICE_INSTANCE is None:
		_SERVICE_INSTANCE = HybridRiskSearchService()
	return _SERVICE_INSTANCE


def hybrid_search(query: str, top_k: int = 5, lexical_k: int = 50, alpha: float = 0.35) -> List[Dict[str, object]]:
	return get_hybrid_search_service().search(query, top_k=top_k, lexical_k=lexical_k, alpha=alpha)


def format_search_results_markdown(results: List[Dict[str, object]], query: str) -> str:
	if not results:
		return f"❌ No results found for: **{query}**"

	lines = [f"## 🔎 Results for: **{query}**", ""]
	for i, item in enumerate(results, start=1):
		title = item.get("title", "Unknown")
		category = item.get("risk_category", "")
		subcategory = item.get("risk_subcategory", "")
		description = item.get("description", "")
		ev_id = item.get("ev_id", "")
		domain = item.get("domain", "")
		sub_domain = item.get("sub_domain", "")

		lines.append(f"### {i}. {title}")
		if category or subcategory:
			lines.append(f"- **Category**: {category} / {subcategory}")
		if domain or sub_domain:
			lines.append(f"- **Domain**: {domain} / {sub_domain}")
		if ev_id:
			lines.append(f"- **Evidence ID**: {ev_id}")
		if description:
			lines.append(f"- **Description**: {description}")
		lines.append("")

	lines.append("---")
	lines.append("_Hybrid ranking: lexical pre-filtering + semantic similarity re-ranking._")
	return "\n".join(lines)


print("✅ AI Risk Database hybrid search service initialized")
