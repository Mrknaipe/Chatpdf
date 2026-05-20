from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time

def semantic_quality(self, generated: str, reference: str) -> dict:
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=True )
    scores = scorer.score(reference, generated)
    return {
        "rouge1": round(scores['rouge1'].fmeasure, 3),
        "rouge2": round(scores['rouge2'].fmeasure, 3),
        "rougeL": round(scores['rougeL'].fmeasure, 3),
    }

def semantic_similarity(self, generated: str, reference: str) -> float:
    vec1 = self.model.encode([generated])   # vecteur de la réponse générée
    vec2 = self.model.encode([reference])   # vecteur de la réponse de référence
    score = cosine_similarity(vec1, vec2)[0][0]
    return round(float(score), 3)

def technical_performance(self, answer: str, elapsed: float) -> dict:
    return {
        "response_time_s": round(elapsed, 2),
        "refused": "cannot find" in answer.lower(),
        "response_length_words": len(answer.split()),
    }