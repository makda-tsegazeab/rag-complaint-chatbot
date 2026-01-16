import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
import logging
from sklearn.model_selection import train_test_split
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StratifiedSampler:
    """Create stratified sample of complaints for embedding pipeline"""
    
    def __init__(self, sample_size: int = 15000, random_state: int = 42):
        """
        Initialize sampler
        
        Args:
            sample_size: Number of complaints to sample (10K-15K as per requirements)
            random_state: Random seed for reproducibility
        """
        self.sample_size = sample_size
        self.random_state = random_state
        logger.info(f"Initialized sampler with sample_size={sample_size}, random_state={random_state}")
    
    def load_processed_data(self, filepath: str = '../data/processed/filtered_complaints.csv') -> pd.DataFrame:
        """Load the processed complaints from Task 1"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Processed data not found at {filepath}. Run Task 1 first.")
        
        logger.info(f"Loading processed data from {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"✓ Loaded {len(df):,} processed complaints")
        
        # Check required columns
        required_cols = ['product_category', 'consumer_complaint_narrative']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        return df
    
    def create_stratified_sample(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create stratified sample ensuring proportional product representation
        
        Args:
            df: Processed complaints dataframe from Task 1
            
        Returns:
            Stratified sample dataframe
        """
        logger.info("Creating stratified sample...")
        
        # Check if we have enough data
        if len(df) < self.sample_size:
            logger.warning(f"Dataset size ({len(df):,}) < sample size ({self.sample_size:,}). Using entire dataset.")
            return df.copy()
        
        # Ensure we have product_category column
        if 'product_category' not in df.columns:
            raise ValueError("Dataframe must contain 'product_category' column for stratified sampling")
        
        # Get product distribution
        product_distribution = df['product_category'].value_counts(normalize=True)
        logger.info(f"Original product distribution:")
        for product, proportion in product_distribution.items():
            count = len(df[df['product_category'] == product])
            logger.info(f"  {product}: {count:,} ({proportion*100:.1f}%)")
        
        # Calculate samples per category
        samples_per_category = {}
        for product, proportion in product_distribution.items():
            samples_per_category[product] = int(self.sample_size * proportion)
        
        # Adjust for rounding errors
        total_sampled = sum(samples_per_category.values())
        if total_sampled < self.sample_size:
            # Add remaining samples to largest category
            largest_category = max(samples_per_category, key=samples_per_category.get)
            samples_per_category[largest_category] += self.sample_size - total_sampled
            logger.info(f"Added {self.sample_size - total_sampled} samples to {largest_category}")
        
        logger.info(f"\nTarget sample distribution:")
        for product, n_samples in samples_per_category.items():
            logger.info(f"  {product}: {n_samples:,} samples")
        
        # Perform stratified sampling
        sampled_dfs = []
        for product, n_samples in samples_per_category.items():
            product_df = df[df['product_category'] == product]
            
            if len(product_df) <= n_samples:
                # Take all if category has fewer samples than needed
                sampled_dfs.append(product_df)
                logger.info(f"  {product}: took all {len(product_df):,} samples (less than target)")
            else:
                # Sample without replacement
                product_sample = product_df.sample(
                    n=n_samples,
                    random_state=self.random_state,
                    replace=False
                )
                sampled_dfs.append(product_sample)
                logger.info(f"  {product}: sampled {n_samples:,} of {len(product_df):,}")
        
        # Combine all samples
        stratified_sample = pd.concat(sampled_dfs, ignore_index=True)
        
        # Shuffle the sample
        stratified_sample = stratified_sample.sample(
            frac=1,
            random_state=self.random_state
        ).reset_index(drop=True)
        
        logger.info(f"\n✓ Created stratified sample of {len(stratified_sample):,} complaints")
        
        return stratified_sample
    
    def validate_sample(self, original_df: pd.DataFrame, sample_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate that sample maintains original distribution
        
        Args:
            original_df: Original dataframe
            sample_df: Sampled dataframe
            
        Returns:
            Dictionary with validation metrics
        """
        validation_results = {}
        
        # 1. Compare product distributions
        original_dist = original_df['product_category'].value_counts(normalize=True).sort_index()
        sample_dist = sample_df['product_category'].value_counts(normalize=True).sort_index()
        
        distribution_diff = (sample_dist - original_dist).abs().mean()
        
        validation_results['product_distribution'] = {
            'original': original_dist.to_dict(),
            'sample': sample_dist.to_dict(),
            'average_difference': distribution_diff,
            'max_difference': (sample_dist - original_dist).abs().max()
        }
        
        # 2. Compare narrative lengths
        if 'narrative_length' in original_df.columns and 'narrative_length' in sample_df.columns:
            validation_results['narrative_lengths'] = {
                'original_mean': original_df['narrative_length'].mean(),
                'sample_mean': sample_df['narrative_length'].mean(),
                'original_median': original_df['narrative_length'].median(),
                'sample_median': sample_df['narrative_length'].median(),
                'length_difference': abs(original_df['narrative_length'].mean() - sample_df['narrative_length'].mean())
            }
        
        # 3. Check for data completeness
        validation_results['completeness'] = {
            'original_size': len(original_df),
            'sample_size': len(sample_df),
            'sampling_ratio': len(sample_df) / len(original_df),
            'missing_narratives_original': original_df['consumer_complaint_narrative'].isna().sum(),
            'missing_narratives_sample': sample_df['consumer_complaint_narrative'].isna().sum()
        }
        
        return validation_results
    
    def print_validation_summary(self, validation_results: Dict[str, Any]):
        """Print comprehensive validation summary"""
        print("\n" + "="*70)
        print("STRATIFIED SAMPLING VALIDATION REPORT")
        print("="*70)
        
        # Product distribution comparison
        print("\n📊 PRODUCT DISTRIBUTION COMPARISON:")
        print("-" * 60)
        
        orig_dist = validation_results['product_distribution']['original']
        sample_dist = validation_results['product_distribution']['sample']
        
        print(f"{'Product Category':<25} {'Original %':<12} {'Sample %':<12} {'Difference':<12}")
        print("-" * 60)
        
        for product in sorted(set(orig_dist.keys()) | set(sample_dist.keys())):
            orig_pct = orig_dist.get(product, 0) * 100
            sample_pct = sample_dist.get(product, 0) * 100
            diff = sample_pct - orig_pct
            
            diff_symbol = "✓" if abs(diff) < 2 else "⚠" if abs(diff) < 5 else "✗"
            print(f"{product:<25} {orig_pct:>10.1f}% {sample_pct:>10.1f}% {diff:>+10.1f}% {diff_symbol}")
        
        print(f"\nAverage distribution difference: {validation_results['product_distribution']['average_difference']*100:.2f}%")
        print(f"Maximum distribution difference: {validation_results['product_distribution']['max_difference']*100:.2f}%")
        
        # Narrative length comparison
        if 'narrative_lengths' in validation_results:
            print("\n📝 NARRATIVE LENGTH COMPARISON:")
            print("-" * 60)
            
            lengths = validation_results['narrative_lengths']
            print(f"{'Metric':<20} {'Original':<12} {'Sample':<12} {'Difference':<12}")
            print("-" * 60)
            print(f"{'Mean length':<20} {lengths['original_mean']:>10.1f} {lengths['sample_mean']:>10.1f} {lengths['length_difference']:>10.1f}")
            print(f"{'Median length':<20} {lengths['original_median']:>10.1f} {lengths['sample_median']:>10.1f} "
                  f"{abs(lengths['original_median'] - lengths['sample_median']):>10.1f}")
        
        # Completeness check
        print("\n📈 SAMPLING COMPLETENESS:")
        print("-" * 60)
        
        completeness = validation_results['completeness']
        print(f"Original dataset size:    {completeness['original_size']:,}")
        print(f"Sample size:              {completeness['sample_size']:,}")
        print(f"Sampling ratio:           {completeness['sampling_ratio']:.2%}")
        print(f"Missing narratives (orig): {completeness['missing_narratives_original']:,}")
        print(f"Missing narratives (sample): {completeness['missing_narratives_sample']:,}")
        
        # Overall assessment
        print("\n" + "="*70)
        print("SAMPLING QUALITY ASSESSMENT:")
        print("-" * 70)
        
        # Check criteria
        distribution_good = validation_results['product_distribution']['average_difference'] < 0.03  # < 3%
        size_good = completeness['sample_size'] >= 10000 and completeness['sample_size'] <= 15000
        
        if distribution_good and size_good:
            print("✅ EXCELLENT: Sample meets all quality criteria!")
            print("   - Product distribution well-preserved")
            print("   - Sample size within target range (10K-15K)")
        elif distribution_good:
            print("⚠️  GOOD: Product distribution preserved, but sample size may need adjustment")
        elif size_good:
            print("⚠️  ACCEPTABLE: Sample size good, but product distribution has some bias")
        else:
            print("❌ NEEDS IMPROVEMENT: Both distribution and size criteria not met")
        
        print("=" * 70)
    
    def run_sampling_pipeline(self, 
                            input_path: str = '../data/processed/filtered_complaints.csv',
                            output_path: str = '../data/sampled/complaints_sample.csv') -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Run complete sampling pipeline
        
        Args:
            input_path: Path to processed complaints from Task 1
            output_path: Path to save sampled complaints
            
        Returns:
            Tuple of (sample_df, validation_results)
        """
        logger.info("="*70)
        logger.info("STARTING STRATIFIED SAMPLING PIPELINE")
        logger.info("="*70)
        
        try:
            # 1. Load processed data
            df = self.load_processed_data(input_path)
            
            # 2. Create stratified sample
            sample_df = self.create_stratified_sample(df)
            
            # 3. Validate sample
            validation_results = self.validate_sample(df, sample_df)
            
            # 4. Save sample
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            sample_df.to_csv(output_path, index=False)
            logger.info(f"\n✓ Saved stratified sample to {output_path}")
            
            # 5. Print validation summary
            self.print_validation_summary(validation_results)
            
            return sample_df, validation_results
            
        except Exception as e:
            logger.error(f"Sampling pipeline failed: {e}")
            raise

# Example usage
if __name__ == "__main__":
    print("Testing Stratified Sampling Pipeline...")
    print("="*70)
    
    # Initialize sampler with 12,500 samples (middle of 10K-15K range)
    sampler = StratifiedSampler(sample_size=12500, random_state=42)
    
    # Run pipeline
    sample_df, validation = sampler.run_sampling_pipeline(
        input_path='../data/processed/filtered_complaints.csv',
        output_path='../data/sampled/complaints_sample.csv'
    )
    
    print("\n" + "="*70)
    print("SAMPLING COMPLETE - READY FOR TASK 2B: CHUNKING & EMBEDDING")
    print("="*70)