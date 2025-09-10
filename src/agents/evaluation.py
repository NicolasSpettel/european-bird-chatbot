import json
import time
import logging
from typing import List, Dict, Any
from langsmith import Client
from src.agents.bird_qa_agent import BirdQAAgent

logger = logging.getLogger(__name__)

class BirdQAEvaluator:
    """Evaluation framework for Bird QA Agent"""
    
    def __init__(self):
        self.agent = BirdQAAgent()
        self.langsmith_client = self.agent.langsmith_client
        
        self.test_cases = [
            {"query": "tell me about robins", "expected_tool": "bird_query"},
            {"query": "what does a sparrow look like", "expected_tool": "bird_query"},
            {"query": "european eagle description", "expected_tool": "bird_query"},
            {"query": "what is a blue jay's habitat", "expected_tool": "bird_query"},
            {"query": "what do flamingos eat", "expected_tool": "bird_query"},
            {"query": "tell me about the common starling", "expected_tool": "bird_query"},
            {"query": "how big is a hummingbird", "expected_tool": "bird_query"},

            {"query": "birding community", "expected_tool": "youtube_query"},
            {"query": "birdwatching tips for beginners", "expected_tool": "youtube_query"},
            {"query": "how to identify birds", "expected_tool": "youtube_query"},
            {"query": "best binoculars for birdwatching", "expected_tool": "youtube_query"},
            {"query": "bird behavior patterns", "expected_tool": "youtube_query"},
            {"query": "how to attract birds to my garden", "expected_tool": "youtube_query"},
            {"query": "what bird calls sound like a wolf", "expected_tool": "youtube_query"},
            {"query": "best bird feeders", "expected_tool": "youtube_query"},
        ]
    
    def evaluate_tool_selection(self, response_text: str, expected_tool: str) -> bool:
        if expected_tool == "bird_query":
            return ("species" in response_text.lower() or "bird" in response_text.lower() or
                    "image:" in response_text.lower() or "http" in response_text)
        else:
            return ("video" in response_text.lower() or "youtube.com" in response_text or
                    "watch" in response_text.lower())
    
    def evaluate_response_quality(self, query: str, response: str, expected_tool: str) -> Dict[str, float]:
        metrics = {}
        length = len(response)
        metrics["length_score"] = 1.0 if 50 <= length <= 500 else max(0, 1 - abs(length - 275) / 275)
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        overlap = len(query_words.intersection(response_words))
        metrics["relevance_score"] = min(1.0, overlap / max(1, len(query_words)))
        
        if expected_tool == "bird_query":
            metrics["has_species_info"] = 1.0 if any(word in response.lower() for word in ["species", "bird", "description"]) else 0.0
            metrics["has_image"] = 1.0 if "image:" in response.lower() or "http" in response else 0.0
        else:
            metrics["has_practical_info"] = 1.0 if any(word in response.lower() for word in ["tip", "guide", "how", "way", "method"]) else 0.0
            metrics["has_source"] = 1.0 if "youtube" in response.lower() or "video" in response.lower() else 0.0
        
        return metrics
    
    def run_evaluation(self) -> Dict[str, Any]:
        results = {"results": [], "summary": {}}
        
        for i, test_case in enumerate(self.test_cases):
            print(f"Testing {i+1}/{len(self.test_cases)}: {test_case['query']}")
            start_time = time.time()
            response = self.agent.ask(test_case["query"])
            response_time = time.time() - start_time
            
            correct_tool = self.evaluate_tool_selection(response["answer"], test_case["expected_tool"])
            quality_metrics = self.evaluate_response_quality(
                test_case["query"], response["answer"], test_case["expected_tool"]
            )
            quality_score = sum(quality_metrics.values()) / len(quality_metrics)
            
            test_result = {
                "query": test_case["query"],
                "expected_tool": test_case["expected_tool"],
                "response": response["answer"],
                "response_time": response_time,
                "correct_tool": correct_tool,
                "quality_score": quality_score,
                "has_error": response.get("error", False),
                "tool_used": response.get("tool_used", "N/A")
            }
            results["results"].append(test_result)
        
        return results
    
    def write_evaluation_report_to_file(self, results: Dict[str, Any], filename: str = "evaluation_results.txt"):
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("Evaluation Report\n")
            f.write("---------------------\n\n")
            f.write("Quality Score Calculation: The quality score is a custom metric that averages four checks: "
                    "response length, keyword relevance, and the presence of category-specific keywords (e.g., "
                    "image links for bird facts or 'YouTube' for tips).\n\n")
            f.write("✅ indicates a correct tool usage, while ❌ indicates a wrong tool usage\n\n")
            
            for result in results["results"]:
                status = "✅" if result["correct_tool"] else "❌"
                
                f.write(f"Question: {result['query']} (Expected Tool: {result['expected_tool']})\n")
                f.write(f"Bot's Answer: {result['response']}\n")
                f.write(f"{status} {result['query']}\n")
                f.write(f"   Tool: {result['expected_tool']} | Quality: {result['quality_score']:.2f} | Time: {result['response_time']:.2f}s\n")
                if result["has_error"]:
                    f.write("   ⚠️  Error in response\n")
                f.write("-" * 20 + "\n\n")
        print(f"Results saved to {filename}")

if __name__ == "__main__":
    evaluator = BirdQAEvaluator()
    results = evaluator.run_evaluation()
    evaluator.write_evaluation_report_to_file(results)