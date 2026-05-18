import json
import time
from typing import Optional
from datetime import datetime
from pathlib import Path
import pandas as pd
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    balanced_accuracy_score,
    classification_report
)
import hashlib


class PromptTester:
    def __init__(
        self,
        client,
        model: str = 'openai/gpt-4-turbo:free',
        cache_dir: Path = None,
        max_samples: Optional[int] = None,
    ):
        self.client = client
        self.model = model
        self.cache_dir = cache_dir or Path('.prompt_cache')
        self.cache_dir.mkdir(exist_ok=True)
        self.max_samples = max_samples
        self.results = {}
        self.valid_labels = {'up', 'down', 'none'}
    
    def _get_cache_key(self, prompt_path: Path, row_idx: int) -> str:
        """Generate unique cache key"""
        key_str = f"{prompt_path.name}_{row_idx}_{self.model}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[str]:
        """Load response from cache if it exists"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f).get('response')
        return None
    
    def _save_to_cache(self, cache_key: str, response: str):
        """Save response to cache"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        with open(cache_file, 'w') as f:
            json.dump({'response': response, 'timestamp': datetime.now().isoformat()}, f)
    
    def _format_prompt(self, template: str, row: pd.Series) -> str:
        """Format prompt by passing all row fields as kwargs"""
        try:
            # Simply pass all fields from row as kwargs
            return template.format(**row.to_dict())
        except KeyError as e:
            print(f"⚠️ Formatting error: missing field '{e}'")
            print(f"   Available fields: {list(row.index)}")
            raise
        except Exception as e:
            print(f"⚠️ Formatting error: {e}")
            print(f"   Template: {template[:200]}")
            print(f"   Data: {row.to_dict()}")
            raise
    
    def _parse_response(self, response_text: str) -> Optional[str]:
        """
        Parse LLM response, search for up/down/none labels.
        Normalize to one of the valid classes.
        """
        text = response_text.lower().strip()
        
        # Look for explicit answer: "Answer: down", "Label: up", etc.
        for line in text.split('\n'):
            if any(prefix in line for prefix in ['answer:', 'label:', 'result:', 'classification:', 'prediction:']):
                # Extract part after colon
                answer_part = line.split(':', 1)[-1].strip()
                
                # Look for valid class in this part
                for valid_label in self.valid_labels:
                    if valid_label in answer_part.lower():
                        return valid_label
        
        # Look for class in first sentence
        first_sentence = text.split('.')[0] if '.' in text else text.split('\n')[0]
        for valid_label in self.valid_labels:
            if valid_label in first_sentence:
                return valid_label
        
        # Last attempt: look for class in last sentence
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if sentences:
            for valid_label in self.valid_labels:
                if valid_label in sentences[-1].lower():
                    return valid_label
        
        # If not found at all - just search for occurrence
        for valid_label in self.valid_labels:
            if valid_label in text:
                return valid_label
        
        return None
    
    def test_prompt(
        self,
        prompt_path: Path,
        df: pd.DataFrame,
        label_column: str = 'label'
    ) -> dict:
        """Test one prompt on the dataset"""
        
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt not found: {prompt_path}")
        
        prompt_template = prompt_path.read_text(encoding='utf-8')
        predictions = []
        ground_truth = []
        responses = []
        response_times = []
        errors = []
        
        test_df = df.head(self.max_samples) if self.max_samples else df
        total = len(test_df)
        
        print(f"\n🧪 Testing prompt: {prompt_path.name}")
        print(f"📝 Samples: {total}")
        
        for idx, (row_idx, row) in enumerate(test_df.iterrows(), 1):
            cache_key = self._get_cache_key(prompt_path, row_idx)
            
            # Check cache
            cached_response = self._load_from_cache(cache_key)
            if cached_response:
                response_text = cached_response
                print(f"  [{idx:3d}/{total}] ✅ from cache", end='\r')
            else:
                try:
                    formatted_prompt = self._format_prompt(prompt_template, row)
                    
                    start_time = time.time()
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{'role': 'user', 'content': formatted_prompt}],
                        temperature=0.3,
                        max_tokens=300,
                    )
                    elapsed = time.time() - start_time
                    
                    response_text = response.choices[0].message.content
                    response_times.append(elapsed)
                    
                    self._save_to_cache(cache_key, response_text)
                    print(f"  [{idx:3d}/{total}] ⏱️  {elapsed:.1f}s", end='\r')
                    
                    time.sleep(0.05)
                    
                except Exception as e:
                    error_msg = f"Sample {row_idx}: {str(e)}"
                    errors.append(error_msg)
                    print(f"  [{idx:3d}/{total}] ❌ error", end='\r')
                    predictions.append('ERROR')
                    ground_truth.append(row[label_column])
                    responses.append('')
                    continue
            
            # Parse response
            parsed = self._parse_response(response_text)
            predictions.append(parsed or 'UNPARSED')
            ground_truth.append(row[label_column])
            responses.append(response_text[:300])
        
        print()
        
        # Compute metrics
        metrics = self._compute_metrics(predictions, ground_truth)
        
        return {
            'prompt_name': prompt_path.name,
            'predictions': predictions,
            'ground_truth': ground_truth,
            'responses': responses,
            'metrics': metrics,
            'response_times': response_times,
            'errors': errors,
        }
    
    def _compute_metrics(self, predictions: list, ground_truth: list) -> dict:
        """Compute all metrics for 3-class classification problem"""
        
        # Parsing statistics
        errors_count = sum(1 for p in predictions if p == 'ERROR')
        unparsed_count = sum(1 for p in predictions if p == 'UNPARSED')
        
        # Filter errors and unparseable examples
        valid_mask = [
            p != 'ERROR' and p != 'UNPARSED' 
            for p in predictions
        ]
        
        if not any(valid_mask):
            return {
                'accuracy': 0,
                'error': 'No valid predictions',
                'errors': errors_count,
                'unparsed': unparsed_count,
            }
        
        valid_pred = [p for p, m in zip(predictions, valid_mask) if m]
        valid_gt = [g for g, m in zip(ground_truth, valid_mask) if m]
        valid_count = len(valid_pred)
        total_count = len(predictions)
        
        # Simple accuracy
        accuracy = sum(p == g for p, g in zip(valid_pred, valid_gt)) / valid_count if valid_count > 0 else 0
        
        try:
            # Weighted metrics (average per class with weights)
            precision, recall, f1, support = precision_recall_fscore_support(
                valid_gt, valid_pred, average='weighted', zero_division=0
            )
            
            # Macro metrics (simple average)
            macro_f1 = precision_recall_fscore_support(
                valid_gt, valid_pred, average='macro', zero_division=0
            )[2]
            
            # Balanced accuracy
            balanced_acc = balanced_accuracy_score(valid_gt, valid_pred)
            
            # Confusion matrix
            cm = confusion_matrix(valid_gt, valid_pred, labels=['up', 'down', 'none'])
            
            # Per-class metrics
            class_report = classification_report(
                valid_gt, valid_pred, 
                labels=['up', 'down', 'none'],
                output_dict=True, 
                zero_division=0
            )
            
            return {
                'accuracy': round(accuracy, 4),
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1_weighted': round(f1, 4),
                'f1_macro': round(macro_f1, 4),
                'balanced_accuracy': round(balanced_acc, 4),
                'valid_samples': valid_count,
                'total_samples': total_count,
                'error_rate': round(1 - valid_count / total_count, 4),
                'errors': errors_count,
                'unparsed': unparsed_count,
                'confusion_matrix': cm.tolist(),
                'cm_labels': ['up', 'down', 'none'],
                'class_report': class_report,
            }
        
        except Exception as e:
            return {
                'accuracy': accuracy,
                'error': str(e)
            }
    
    def compare_prompts(self) -> pd.DataFrame:
        """Compare all tested prompts"""
        
        comparison_data = []
        
        for prompt_name, result in self.results.items():
            metrics = result['metrics']
            comparison_data.append({
                'Prompt': prompt_name,
                'Accuracy': metrics.get('accuracy', 0),
                'F1 (weighted)': metrics.get('f1_weighted', 0),
                'F1 (macro)': metrics.get('f1_macro', 0),
                'Precision': metrics.get('precision', 0),
                'Recall': metrics.get('recall', 0),
                'Balanced Acc': metrics.get('balanced_accuracy', 0),
                'Valid Samples': metrics.get('valid_samples', 0),
                'Errors': metrics.get('errors', 0),
                'Unparseable': metrics.get('unparsed', 0),
            })
        
        df_comparison = pd.DataFrame(comparison_data).sort_values('F1 (macro)', ascending=False)
        return df_comparison
    
    def print_detailed_results(self, prompt_name: str):
        """Print detailed results for a prompt"""
        
        if prompt_name not in self.results:
            print(f"❌ Results for '{prompt_name}' not found")
            return
        
        result = self.results[prompt_name]
        metrics = result['metrics']
        
        print(f"\n{'='*70}")
        print(f"📊 DETAILED RESULTS: {prompt_name}")
        print(f"{'='*70}")
        
        print(f"\n📈 Main metrics:")
        print(f"  Accuracy:         {metrics.get('accuracy', 0):.4f}")
        print(f"  F1 (weighted):    {metrics.get('f1_weighted', 0):.4f}")
        print(f"  F1 (macro):       {metrics.get('f1_macro', 0):.4f}")
        print(f"  Precision:        {metrics.get('precision', 0):.4f}")
        print(f"  Recall:           {metrics.get('recall', 0):.4f}")
        print(f"  Balanced Acc:     {metrics.get('balanced_accuracy', 0):.4f}")
        
        print(f"\n📦 Sample processing quality:")
        print(f"  Total:            {metrics.get('total_samples', 0)}")
        print(f"  Valid:            {metrics.get('valid_samples', 0)}")
        print(f"  Parsing errors:   {metrics.get('errors', 0)}")
        print(f"  Unparseable:      {metrics.get('unparsed', 0)}")
        
        if result['response_times']:
            avg_time = sum(result['response_times']) / len(result['response_times'])
            print(f"  Average time:     {avg_time:.2f}s")
        
        if result['errors']:
            print(f"\n⚠️  Error examples:")
            for err in result['errors'][:3]:
                print(f"   • {err}")
        
        # Confusion Matrix
        if 'confusion_matrix' in metrics:
            print(f"\n🔀 Confusion Matrix:")
            cm_df = pd.DataFrame(
                metrics['confusion_matrix'],
                index=metrics['cm_labels'],
                columns=metrics['cm_labels']
            )
            print(cm_df)
        
        # Per-class metrics
        if 'class_report' in metrics:
            print(f"\n📋 Per-class metrics:")
            cr = pd.DataFrame(metrics['class_report']).T
            print(cr[['precision', 'recall', 'f1-score', 'support']].round(4))
    
    def save_analysis(self, output_dir: Path = Path('prompt_analysis')):
        """Save all analysis to files"""
        output_dir.mkdir(exist_ok=True)
        
        # Prompt comparison
        comparison_df = self.compare_prompts()
        comparison_df.to_csv(output_dir / 'comparison.csv', index=False)
        print(f"✅ Comparison saved: {output_dir / 'comparison.csv'}")
        
        # Detailed results for each prompt
        for prompt_name, result in self.results.items():
            prompt_dir = output_dir / prompt_name.replace('.txt', '')
            prompt_dir.mkdir(exist_ok=True)
            
            # Predictions
            results_df = pd.DataFrame({
                'ground_truth': result['ground_truth'],
                'prediction': result['predictions'],
                'response_sample': [r[:100] for r in result['responses']],
            })
            results_df.to_csv(prompt_dir / 'predictions.csv', index=False)
            
            # Metrics summary
            with open(prompt_dir / 'metrics.json', 'w') as f:
                metrics_clean = {
                    k: v for k, v in result['metrics'].items() 
                    if k not in ['class_report', 'confusion_matrix']
                }
                json.dump(metrics_clean, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Results for '{prompt_name}' saved to {prompt_dir}")