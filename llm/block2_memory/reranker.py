"""
    Реранкер на основе локальной LLM для получения наиболее релевантных ответов модели

"""

import re
from typing import List, Dict, Any, Optional

class LocalLLMReranker:
    def __init__(self, llm, batch_size: int = 3):
        self.llm = llm
        self.batch_size = batch_size
        
    def rerank(self, query: str, documents: List[Dict], top_k: int = 3) -> List[Dict]:
        if not documents:
            return documents

        batches = [documents[i:i + self.batch_size] for i in range(0, len(documents), self.batch_size)]
        all_results = []
        
        for batch in batches:
            batch_results = self._rank_batch(query, batch)
            all_results.extend(batch_results)
        
        all_results.sort(key=lambda x: x.get('combined_score', x.get('relevance_score', 0)), reverse=True)
        return all_results[:top_k]
    
    def _rank_batch(self, query: str, documents: List[Dict]) -> List[Dict]:
        if len(documents) == 1:
            return [self._rank_single(query, documents[0])]
        
        prompt = self._create_batch_prompt(query, documents)
        
        try:
            response = self.llm(
                prompt,
                max_tokens=150,
                temperature=0.1,
                top_p=0.9,
                echo=False,
                stop=["\n\n", "###"]
            )
            
            answer = response['choices'][0]['text'].strip()
            scores = self._parse_scores(answer, len(documents))
            results = []
            for i, (doc, score) in enumerate(zip(documents, scores)):
                doc_with_score = doc.copy()
                doc_with_score['llm_score'] = score
                original_score = doc.get('relevance_score', 0.5)
                doc_with_score['combined_score'] = round(
                    0.6 * score + 0.4 * original_score, 4
                )
                results.append(doc_with_score)
            
            return results
            
        except Exception as e:
            print(f"Ошибка при реранкинге батча: {e}")
            return documents
    
    def _rank_single(self, query: str, document: Dict) -> Dict:
        prompt = self._create_single_prompt(query, document['text'])
        
        try:
            response = self.llm(
                prompt,
                max_tokens=50,
                temperature=0.1,
                top_p=0.9,
                echo=False,
                stop=["\n"]
            )
            
            answer = response['choices'][0]['text'].strip()
            score = self._parse_single_score(answer)
            
            doc_with_score = document.copy()
            doc_with_score['llm_score'] = score
            original_score = document.get('relevance_score', 0.5)
            doc_with_score['combined_score'] = round(
                0.6 * score + 0.4 * original_score, 4
            )
            
            return doc_with_score
            
        except Exception as e:
            print(f"Ошибка при реранкинге документа: {e}")
            return document
    
    def _create_single_prompt(self, query: str, document: str) -> str:

        if len(document) > 900:
            document = document[:900] + "..."
        
        prompt = f"""<|user|>Оцени релевантность документа к вопросу по шкале от 0 до 10.
        0 - совершенно нерелевантно
        5 - частично релевантно
        10 - идеально релевантно, содержит прямой ответ
        Вопрос: {query}
        Документ: {document}
        Твоя оценка (только число от 0 до 10):<|end|><|assistant|>"""
        
        return prompt
    
    def _create_batch_prompt(self, query: str, documents: List[Dict]) -> str:
        prompt = f"""<|user|>Оцени релевантность каждого документа к вопросу по шкале от 0 до 10.
        Ответ дай в формате: "Оценки: X, Y, Z" где X,Y,Z - числа в том же порядке, что и документы.
        Вопрос: {query}
        Документы:"""
        
        for i, doc in enumerate(documents, 1):
            text = doc['text']
            if len(text) > 750:
                text = text[:750] + "..."
            prompt += f"\n{i}. {text}\n"
        
        prompt += """\nТвои оценки (только числа через запятую):<|end|><|assistant|>Оценки:"""
        
        return prompt
    
    def _parse_single_score(self, answer: str) -> float:
        numbers = re.findall(r'\d+\.?\d*', answer)
        if numbers:
            try:
                score = float(numbers[0])
                if score > 1:
                    score = score / 10
                return min(1.0, max(0.0, score))
            except:
                pass
        return 0.5 
    
    def _parse_scores(self, answer: str, expected_count: int) -> List[float]:
        numbers = re.findall(r'\d+\.?\d*', answer)
        
        scores = []
        for num in numbers[:expected_count]:
            try:
                score = float(num)
                if score > 1:
                    score = score / 10
                scores.append(min(1.0, max(0.0, score)))
            except:
                scores.append(0.5)

        while len(scores) < expected_count:
            scores.append(0.5)
        
        return scores