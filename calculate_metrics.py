import pandas as pd
from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score, hamming_loss
import numpy as np
from typing import Set, List, Dict, Any, Optional, Callable
import ast
class SetMetricsCalculator:
    """
    A flexible class for calculating various metrics when comparing actual vs predicted sets.
    """
    
    def __init__(self):
        self.results = {}
    
    @staticmethod
    def safe_json_parse(x) -> Set[str]:
        """
        Safely parse JSON-like strings to sets.
        """
        try:
            if isinstance(x, str):
                # Handle single quotes in JSON-like strings
                result = set(ast.literal_eval(x.replace("'", "\"")))
            elif isinstance(x, (list, set)):
                result = set(x)
            else:
                result = set()
            return result
        except Exception as e:
            print(f"Error parsing value: {x}, Error: {e}")
            return set()
    
    @staticmethod
    def filter_by_prefix(x_set: Set[str], prefix: str = "F") -> Set[str]:
        """
        Filter set elements by prefix.
        """
        return {x for x in x_set if str(x).startswith(prefix)}
    
    @staticmethod
    def extract_higher_level_codes(x_set: Set[str], delimiter: str = ".") -> Set[str]:
        """
        Extract higher-level codes by splitting on delimiter and taking first part.
        """
        return {str(k).split(delimiter)[0] for k in x_set}
    
    def prepare_data(self, 
                     df: pd.DataFrame, 
                     actual_col: str, 
                     predicted_col: str,
                     actual_preprocessor: Optional[Callable] = None,
                     predicted_preprocessor: Optional[Callable] = None,
                     filter_prefix: Optional[str] = None,
                     extract_higher_level: bool = False,
                     delimiter: str = ".") -> pd.DataFrame:
        """
        Prepare data for metrics calculation.
        
        Args:
            df: Input DataFrame
            actual_col: Column name containing actual values
            predicted_col: Column name containing predicted values
            actual_preprocessor: Function to preprocess actual values
            predicted_preprocessor: Function to preprocess predicted values
            filter_prefix: Prefix to filter elements (e.g., "F" for F-codes)
            extract_higher_level: Whether to extract higher-level codes
            delimiter: Delimiter for higher-level code extraction
            
        Returns:
            DataFrame with processed actual and predicted sets
        """
        df_copy = df.copy()
        
        # Apply preprocessors or default parsing
        if actual_preprocessor:
            df_copy["actual_set"] = df_copy[actual_col].apply(actual_preprocessor)
        else:
            df_copy["actual_set"] = df_copy[actual_col].apply(self.safe_json_parse)
            
        if predicted_preprocessor:
            df_copy["predicted_set"] = df_copy[predicted_col].apply(predicted_preprocessor)
        else:
            df_copy["predicted_set"] = df_copy[predicted_col].apply(self.safe_json_parse)
        
        # Filter by prefix if specified
        if filter_prefix:
            df_copy["actual_set"] = df_copy["actual_set"].apply(
                lambda x: self.filter_by_prefix(x, filter_prefix)
            )
            df_copy["predicted_set"] = df_copy["predicted_set"].apply(
                lambda x: self.filter_by_prefix(x, filter_prefix)
            )
        
        # Extract higher-level codes if specified
        if extract_higher_level:
            df_copy["actual_set"] = df_copy["actual_set"].apply(
                lambda x: self.extract_higher_level_codes(x, delimiter)
            )
            df_copy["predicted_set"] = df_copy["predicted_set"].apply(
                lambda x: self.extract_higher_level_codes(x, delimiter)
            )
        
        return df_copy
    
    def calculate_metrics(self, actual_sets: List[Set], predicted_sets: List[Set]) -> Dict[str, float]:
        """
        Calculate various metrics for set comparison.
        
        Args:
            actual_sets: List of actual sets
            predicted_sets: List of predicted sets
            
        Returns:
            Dictionary containing calculated metrics
        """
        # Get all unique labels across all sets
        all_labels = sorted(set.union(*actual_sets, *predicted_sets))
        
        def set_to_binary_vector(s: Set, all_labels: List) -> List[int]:
            return [1 if label in s else 0 for label in all_labels]

        # Convert sets to binary vectors
        actual_binary = np.array([set_to_binary_vector(s, all_labels) for s in actual_sets])
        predicted_binary = np.array([set_to_binary_vector(s, all_labels) for s in predicted_sets])

        # Calculate metrics
        metrics = {}
        
        # Jaccard Similarity (per sample, then averaged)
        jaccard_scores = [jaccard_score(a, p, average='micro', zero_division=0) 
                         for a, p in zip(actual_binary, predicted_binary)]
        metrics["Jaccard Similarity"] = np.mean(jaccard_scores)

        # Precision, Recall, F1 Score (per sample, then averaged)
        precision_scores = [precision_score(a, p, average='micro', zero_division=0) 
                           for a, p in zip(actual_binary, predicted_binary)]
        recall_scores = [recall_score(a, p, average='micro', zero_division=0) 
                        for a, p in zip(actual_binary, predicted_binary)]
        f1_scores = [f1_score(a, p, average='micro', zero_division=0) 
                    for a, p in zip(actual_binary, predicted_binary)]
        
        metrics["Precision"] = np.mean(precision_scores)
        metrics["Recall"] = np.mean(recall_scores)
        metrics["F1 Score"] = np.mean(f1_scores)

        # Hamming Loss
        metrics["Hamming Loss"] = hamming_loss(actual_binary, predicted_binary)

        # Set Accuracy (exact match)
        exact_matches = [1 if np.array_equal(a, p) else 0 
                        for a, p in zip(actual_binary, predicted_binary)]
        metrics["Set Accuracy"] = np.mean(exact_matches)
        
        return metrics
    
    def calculate_set_comparisons(self, actual_sets: List[Set], predicted_sets: List[Set]) -> Dict[str, int]:
        """
        Calculate set comparison statistics.
        
        Args:
            actual_sets: List of actual sets
            predicted_sets: List of predicted sets
            
        Returns:
            Dictionary containing comparison counts
        """
        exactly_equal = 0
        at_least_one_common = 0
        completely_different = 0
        total_samples = len(actual_sets)

        for actual, predicted in zip(actual_sets, predicted_sets):
            if actual == predicted:
                exactly_equal += 1
            elif actual & predicted:  # Intersection is not empty
                at_least_one_common += 1
            else:
                completely_different += 1

        return {
            "Total Samples": total_samples,
            "Exactly Equal": exactly_equal,
            "At Least One Common Element": at_least_one_common,
            "Completely Different": completely_different,
            "Exactly Equal (%)": (exactly_equal / total_samples) * 100 if total_samples > 0 else 0,
            "At Least One Common (%)": (at_least_one_common / total_samples) * 100 if total_samples > 0 else 0,
            "Completely Different (%)": (completely_different / total_samples) * 100 if total_samples > 0 else 0,
        }
    
    def run_evaluation(self, 
                      df: pd.DataFrame,
                      actual_col: str,
                      predicted_col: str,
                      output_file: Optional[str] = None,
                      **kwargs) -> Dict[str, Any]:
        """
        Run complete evaluation pipeline.
        
        Args:
            df: Input DataFrame
            actual_col: Column name containing actual values
            predicted_col: Column name containing predicted values
            output_file: Optional output file path for processed DataFrame
            **kwargs: Additional arguments for data preparation
            
        Returns:
            Dictionary containing all results
        """
        # Prepare data
        processed_df = self.prepare_data(df, actual_col, predicted_col, **kwargs)
        
        # Calculate metrics
        metrics = self.calculate_metrics(
            processed_df["actual_set"].tolist(),
            processed_df["predicted_set"].tolist()
        )
        
        # Calculate set comparisons
        comparisons = self.calculate_set_comparisons(
            processed_df["actual_set"].tolist(),
            processed_df["predicted_set"].tolist()
        )
        
        # Store results
        self.results = {
            "metrics": metrics,
            "comparisons": comparisons,
            "processed_data": processed_df
        }
        
        # Save processed data if output file specified
        if output_file:
            processed_df.to_csv(output_file, index=False, sep='\t')
            print(f"Processed data saved to: {output_file}")
        
        return self.results
    
    def print_results(self):
        """Print formatted results."""
        if not self.results:
            print("No results to display. Run evaluation first.")
            return
        
        print("=" * 50)
        print("METRICS")
        print("=" * 50)
        for metric, value in self.results["metrics"].items():
            print(f"{metric}: {value:.4f}")
        
        print("\n" + "=" * 50)
        print("SET COMPARISONS")
        print("=" * 50)
        for comparison, count in self.results["comparisons"].items():
            if "%" in comparison:
                print(f"{comparison}: {count:.2f}")
            else:
                print(f"{comparison}: {count}")

# Example usage functions for different data types

def evaluate_clinical_codes(df: pd.DataFrame, 
                           actual_col: str = "DIAGNOSES", 
                           predicted_col: str = "codes",
                           output_file: Optional[str] = None):
    """Example for clinical coding evaluation."""
    calculator = SetMetricsCalculator()
    
    # Custom preprocessors for clinical data
    def preprocess_actual(x):
        return set(str(x).split(" ")) if pd.notna(x) else set()
    
    results = calculator.run_evaluation(
        df=df,
        actual_col=actual_col,
        predicted_col=predicted_col,
        actual_preprocessor=preprocess_actual,
        predicted_preprocessor=calculator.safe_json_parse,
        filter_prefix="F",  # Only F-codes
        extract_higher_level=True,  # Use higher-level codes
        output_file=output_file
    )
    
    calculator.print_results()
    return results

def evaluate_json_predictions(df: pd.DataFrame,
                             actual_col: str = "ground_truth",
                             predicted_col: str = "prediction",
                             output_file: Optional[str] = None):
    """Example for JSON prediction evaluation."""
    calculator = SetMetricsCalculator()
    
    # Custom preprocessor for ground truth that might be space-separated
    def preprocess_ground_truth(x):
        if pd.isna(x):
            return set()
        # Handle "ICD-10-CM codes: F32.2 F32.A F33.2 R45.851" format
        if ":" in str(x):
            codes_part = str(x).split(":", 1)[1].strip()
            return set(codes_part.split())
        return set(str(x).split())
    
    results = calculator.run_evaluation(
        df=df,
        actual_col=actual_col,
        predicted_col=predicted_col,
        actual_preprocessor=preprocess_ground_truth,
        filter_prefix="F",  # Only F-codes
        output_file=output_file
    )
    
    calculator.print_results()
    return results

# Main execution example
if __name__ == "__main__":
    df = pd.read_csv("llama_finetuning_results_comparison.csv")
    calculator = SetMetricsCalculator()
    results = calculator.run_evaluation(
        df=df,
        actual_col="ground_truth_labels",
        predicted_col="predictions_after",
        filter_prefix="F",
        extract_higher_level=True,
    )
    calculator.print_results()