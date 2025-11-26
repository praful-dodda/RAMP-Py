import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial import cKDTree
import os
import warnings
from preprocess import get_ozone_file
from evaluate_models import (
    preprocess_to_long_format, 
    add_tags_to_toar, 
    collocate_data,
    calculate_performance_metrics,
    get_region, 
    get_season
)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class CorrelationAnalyzer:
    def __init__(self, year=2017, output_dir="./correlation_analysis"):
        """
        Initialize the correlation analyzer for M3fusion, NJML, and TOAR-II datasets.
        
        Parameters:
        -----------
        year : int
            Year to analyze (must be within overlap period: 2003-2019)
        output_dir : str
            Directory to save analysis results
        """
        self.year = year
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Validate year is in overlap period for all datasets
        if not (2003 <= year <= 2019):
            raise ValueError("Year must be between 2003-2019 for all three datasets to have data")
        
        print(f"Initializing correlation analysis for year {year}")
        
    def load_and_prepare_data(self):
        """Load and prepare all three datasets for analysis."""
        print("Loading datasets...")
        
        # Load TOAR-II observations
        try:
            toar_file = get_ozone_file("TOAR-II", self.year)
            self.toar_raw = pd.read_csv(toar_file)
            self.toar_long = preprocess_to_long_format(self.toar_raw, is_toar=True)
            self.toar_tagged = add_tags_to_toar(self.toar_long)
            print(f"✓ Loaded TOAR-II: {len(self.toar_raw)} stations, {len(self.toar_long)} obs points")
        except Exception as e:
            raise Exception(f"Failed to load TOAR-II data: {e}")
        
        # Load M3fusion (soft data 1)
        try:
            m3fusion_file = get_ozone_file("M3fusion", self.year)
            self.m3fusion_raw = pd.read_csv(m3fusion_file)
            self.m3fusion_long = preprocess_to_long_format(self.m3fusion_raw, is_toar=False)
            print(f"✓ Loaded M3fusion: {len(self.m3fusion_raw)} grid points, {len(self.m3fusion_long)} data points")
        except Exception as e:
            raise Exception(f"Failed to load M3fusion data: {e}")
        
        # Load NJML (soft data 2)
        try:
            njml_file = get_ozone_file("NJML", self.year)
            self.njml_raw = pd.read_csv(njml_file)
            self.njml_long = preprocess_to_long_format(self.njml_raw, is_toar=False)
            print(f"✓ Loaded NJML: {len(self.njml_raw)} grid points, {len(self.njml_long)} data points")
        except Exception as e:
            raise Exception(f"Failed to load NJML data: {e}")
            
    def collocate_datasets(self):
        """Collocate all datasets to common observation points."""
        print("\nCollocating datasets...")
        
        # Collocate M3fusion with TOAR-II
        self.m3fusion_collocated = collocate_data(self.toar_tagged, self.m3fusion_long)
        self.m3fusion_collocated['model_source'] = 'M3fusion'
        print(f"✓ M3fusion-TOAR collocated: {len(self.m3fusion_collocated)} matched points")
        
        # Collocate NJML with TOAR-II  
        self.njml_collocated = collocate_data(self.toar_tagged, self.njml_long)
        self.njml_collocated['model_source'] = 'NJML'
        print(f"✓ NJML-TOAR collocated: {len(self.njml_collocated)} matched points")
        
        # For M3fusion vs NJML comparison, we need to find common grid points
        self.compare_soft_data()
        
    def compare_soft_data(self):
        """Compare M3fusion and NJML at common grid points."""
        print("Comparing soft datasets at common grid points...")
        
        # Find common lat/lon grid points between M3fusion and NJML
        m3_coords = self.m3fusion_raw[['lat', 'lon']].round(2)  # Round to avoid floating point issues
        njml_coords = self.njml_raw[['lat', 'lon']].round(2)
        
        # Merge on coordinates to find common points
        common_coords = pd.merge(
            m3_coords.reset_index().rename(columns={'index': 'm3_idx'}),
            njml_coords.reset_index().rename(columns={'index': 'njml_idx'}),
            on=['lat', 'lon'],
            how='inner'
        )
        
        print(f"Found {len(common_coords)} common grid points between M3fusion and NJML")
        
        if len(common_coords) == 0:
            print("Warning: No common grid points found between M3fusion and NJML")
            self.soft_comparison = pd.DataFrame()
            return
        
        # Extract data for common points
        m3_common = self.m3fusion_raw.iloc[common_coords['m3_idx']].reset_index(drop=True)
        njml_common = self.njml_raw.iloc[common_coords['njml_idx']].reset_index(drop=True)
        
        # Convert to long format and combine
        m3_long = preprocess_to_long_format(m3_common, is_toar=False)
        njml_long = preprocess_to_long_format(njml_common, is_toar=False)
        
        # Merge on coordinates and month
        self.soft_comparison = pd.merge(
            m3_long, njml_long,
            on=['lat', 'lon', 'month'],
            suffixes=('_m3fusion', '_njml'),
            how='inner'
        )
        
        print(f"✓ Soft data comparison dataset: {len(self.soft_comparison)} matched points")
        
    def calculate_correlations(self):
        """Calculate correlation statistics for all dataset pairs."""
        print("\nCalculating correlations...")
        
        correlations = {}
        
        # 1. M3fusion vs TOAR-II (soft data 1 vs observations)
        if len(self.m3fusion_collocated) > 0:
            correlations['M3fusion_vs_TOAR'] = self._compute_correlation_stats(
                self.m3fusion_collocated['model_ozone'], 
                self.m3fusion_collocated['observed_ozone'],
                'M3fusion vs TOAR-II'
            )
        
        # 2. NJML vs TOAR-II (soft data 2 vs observations)  
        if len(self.njml_collocated) > 0:
            correlations['NJML_vs_TOAR'] = self._compute_correlation_stats(
                self.njml_collocated['model_ozone'],
                self.njml_collocated['observed_ozone'], 
                'NJML vs TOAR-II'
            )
        
        # 3. M3fusion vs NJML (soft data 1 vs soft data 2)
        if len(self.soft_comparison) > 0:
            correlations['M3fusion_vs_NJML'] = self._compute_correlation_stats(
                self.soft_comparison['ozone_m3fusion'],
                self.soft_comparison['ozone_njml'],
                'M3fusion vs NJML'
            )
        
        self.correlations = correlations
        self._print_correlation_summary()
        
    def _compute_correlation_stats(self, x, y, pair_name):
        """Compute comprehensive correlation statistics."""
        # Remove NaN values
        valid_mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[valid_mask]
        y_clean = y[valid_mask]
        
        if len(x_clean) < 2:
            return {'error': 'Insufficient data points'}
        
        stats_dict = {
            'n_points': len(x_clean),
            'pearson_r': stats.pearsonr(x_clean, y_clean)[0],
            'pearson_p': stats.pearsonr(x_clean, y_clean)[1],
            'spearman_r': stats.spearmanr(x_clean, y_clean)[0],
            'spearman_p': stats.spearmanr(x_clean, y_clean)[1],
            'kendall_tau': stats.kendalltau(x_clean, y_clean)[0],
            'kendall_p': stats.kendalltau(x_clean, y_clean)[1],
            'r_squared': stats.pearsonr(x_clean, y_clean)[0] ** 2,
            'rmse': np.sqrt(np.mean((x_clean - y_clean) ** 2)),
            'mae': np.mean(np.abs(x_clean - y_clean)),
            'bias': np.mean(x_clean - y_clean),
            'x_mean': np.mean(x_clean),
            'y_mean': np.mean(y_clean),
            'x_std': np.std(x_clean),
            'y_std': np.std(y_clean)
        }
        
        return stats_dict
    
    def _print_correlation_summary(self):
        """Print a summary of correlation results."""
        print("\n" + "="*60)
        print("CORRELATION ANALYSIS SUMMARY")
        print("="*60)
        
        for pair_name, stats in self.correlations.items():
            if 'error' in stats:
                print(f"\n{pair_name}: {stats['error']}")
                continue
                
            print(f"\n{pair_name}:")
            print(f"  Sample size: {stats['n_points']:,}")
            print(f"  Pearson correlation: {stats['pearson_r']:.4f} (p={stats['pearson_p']:.2e})")
            print(f"  Spearman correlation: {stats['spearman_r']:.4f} (p={stats['spearman_p']:.2e})")
            print(f"  R-squared: {stats['r_squared']:.4f}")
            print(f"  RMSE: {stats['rmse']:.3f}")
            print(f"  Bias: {stats['bias']:.3f}")
            
    def create_visualization_plots(self):
        """Create comprehensive visualization plots."""
        print("\nGenerating visualization plots...")
        
        # 1. Scatter plots for all pairs
        self._create_scatter_plots()
        
        # 2. Regional correlation analysis
        self._create_regional_analysis()
        
        # 3. Monthly correlation patterns
        self._create_monthly_analysis()
        
        # 4. Distribution comparisons
        self._create_distribution_plots()
        
        # 5. Correlation heatmap
        self._create_correlation_heatmap()
        
    def _create_scatter_plots(self):
        """Create scatter plots for all dataset pairs."""
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # M3fusion vs TOAR-II
        if len(self.m3fusion_collocated) > 0:
            ax = axes[0]
            x = self.m3fusion_collocated['model_ozone']
            y = self.m3fusion_collocated['observed_ozone']
            ax.scatter(x, y, alpha=0.5, s=10)
            ax.plot([0, 100], [0, 100], 'r--', linewidth=2, label='1:1 line')
            
            # Add statistics
            stats = self.correlations['M3fusion_vs_TOAR']
            ax.text(0.05, 0.95, f"R = {stats['pearson_r']:.3f}\nR² = {stats['r_squared']:.3f}\nRMSE = {stats['rmse']:.2f}\nBias = {stats['bias']:.2f}", 
                   transform=ax.transAxes, verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_xlabel('M3fusion Ozone (ppb)')
            ax.set_ylabel('TOAR-II Observed Ozone (ppb)')
            ax.set_title('M3fusion vs TOAR-II Observations')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # NJML vs TOAR-II
        if len(self.njml_collocated) > 0:
            ax = axes[1]
            x = self.njml_collocated['model_ozone']
            y = self.njml_collocated['observed_ozone']
            ax.scatter(x, y, alpha=0.5, s=10)
            ax.plot([0, 100], [0, 100], 'r--', linewidth=2, label='1:1 line')
            
            # Add statistics
            stats = self.correlations['NJML_vs_TOAR']
            ax.text(0.05, 0.95, f"R = {stats['pearson_r']:.3f}\nR² = {stats['r_squared']:.3f}\nRMSE = {stats['rmse']:.2f}\nBias = {stats['bias']:.2f}", 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_xlabel('NJML Ozone (ppb)')
            ax.set_ylabel('TOAR-II Observed Ozone (ppb)')
            ax.set_title('NJML vs TOAR-II Observations')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # M3fusion vs NJML
        if len(self.soft_comparison) > 0:
            ax = axes[2]
            x = self.soft_comparison['ozone_m3fusion']
            y = self.soft_comparison['ozone_njml']
            ax.scatter(x, y, alpha=0.5, s=10)
            ax.plot([0, 100], [0, 100], 'r--', linewidth=2, label='1:1 line')
            
            # Add statistics
            stats = self.correlations['M3fusion_vs_NJML']
            ax.text(0.05, 0.95, f"R = {stats['pearson_r']:.3f}\nR² = {stats['r_squared']:.3f}\nRMSE = {stats['rmse']:.2f}\nBias = {stats['bias']:.2f}", 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_xlabel('M3fusion Ozone (ppb)')
            ax.set_ylabel('NJML Ozone (ppb)')
            ax.set_title('M3fusion vs NJML')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'scatter_plots_{self.year}.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
    def _create_regional_analysis(self):
        """Analyze correlations by region."""
        print("Creating regional correlation analysis...")
        
        if len(self.m3fusion_collocated) == 0 and len(self.njml_collocated) == 0:
            return
            
        regional_stats = {}
        
        # M3fusion regional analysis
        if len(self.m3fusion_collocated) > 0:
            regional_stats['M3fusion'] = self.m3fusion_collocated.groupby('region').apply(
                lambda x: calculate_performance_metrics(x, 'observed_ozone', 'model_ozone')
            ).reset_index()
            
        # NJML regional analysis  
        if len(self.njml_collocated) > 0:
            regional_stats['NJML'] = self.njml_collocated.groupby('region').apply(
                lambda x: calculate_performance_metrics(x, 'observed_ozone', 'model_ozone')
            ).reset_index()
        
        # Plot regional comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        metrics = ['correlation', 'RMSE', 'ME', 'MAE']
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            
            x_pos = np.arange(len(regional_stats.get('M3fusion', pd.DataFrame()).get('region', [])))
            
            if 'M3fusion' in regional_stats and not regional_stats['M3fusion'].empty:
                ax.bar(x_pos - 0.2, regional_stats['M3fusion'][metric], 0.4, 
                      label='M3fusion', alpha=0.8)
                      
            if 'NJML' in regional_stats and not regional_stats['NJML'].empty:
                ax.bar(x_pos + 0.2, regional_stats['NJML'][metric], 0.4, 
                      label='NJML', alpha=0.8)
            
            if len(x_pos) > 0:
                ax.set_xticks(x_pos)
                ax.set_xticklabels(regional_stats[list(regional_stats.keys())[0]]['region'], rotation=45)
            
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} by Region')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'regional_analysis_{self.year}.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
    def _create_monthly_analysis(self):
        """Analyze correlations by month."""
        print("Creating monthly correlation analysis...")
        
        monthly_correlations = {}
        
        # M3fusion monthly correlations
        if len(self.m3fusion_collocated) > 0:
            monthly_m3 = []
            for month in range(1, 13):
                month_data = self.m3fusion_collocated[self.m3fusion_collocated['month'] == month]
                if len(month_data) > 10:  # Minimum sample size
                    corr = stats.pearsonr(month_data['model_ozone'], month_data['observed_ozone'])[0]
                    monthly_m3.append(corr)
                else:
                    monthly_m3.append(np.nan)
            monthly_correlations['M3fusion'] = monthly_m3
        
        # NJML monthly correlations
        if len(self.njml_collocated) > 0:
            monthly_njml = []
            for month in range(1, 13):
                month_data = self.njml_collocated[self.njml_collocated['month'] == month]
                if len(month_data) > 10:  # Minimum sample size
                    corr = stats.pearsonr(month_data['model_ozone'], month_data['observed_ozone'])[0]
                    monthly_njml.append(corr)
                else:
                    monthly_njml.append(np.nan)
            monthly_correlations['NJML'] = monthly_njml
        
        # M3fusion vs NJML monthly correlations
        if len(self.soft_comparison) > 0:
            monthly_soft = []
            for month in range(1, 13):
                month_data = self.soft_comparison[self.soft_comparison['month'] == month]
                if len(month_data) > 10:
                    corr = stats.pearsonr(month_data['ozone_m3fusion'], month_data['ozone_njml'])[0]
                    monthly_soft.append(corr)
                else:
                    monthly_soft.append(np.nan)
            monthly_correlations['M3fusion_vs_NJML'] = monthly_soft
        
        # Plot monthly correlations
        plt.figure(figsize=(12, 8))
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        for dataset, correlations in monthly_correlations.items():
            plt.plot(months, correlations, 'o-', linewidth=2, markersize=8, label=dataset)
        
        plt.xlabel('Month')
        plt.ylabel('Pearson Correlation')
        plt.title(f'Monthly Correlation Patterns ({self.year})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'monthly_correlations_{self.year}.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
    def _create_distribution_plots(self):
        """Create distribution comparison plots."""
        print("Creating distribution comparison plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Distribution of all datasets
        ax = axes[0, 0]
        if len(self.m3fusion_collocated) > 0:
            ax.hist(self.m3fusion_collocated['model_ozone'], bins=50, alpha=0.7, label='M3fusion', density=True)
            ax.hist(self.m3fusion_collocated['observed_ozone'], bins=50, alpha=0.7, label='TOAR-II', density=True)
        if len(self.njml_collocated) > 0:
            ax.hist(self.njml_collocated['model_ozone'], bins=50, alpha=0.7, label='NJML', density=True)
        ax.set_xlabel('Ozone (ppb)')
        ax.set_ylabel('Density')
        ax.set_title('Distribution Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Q-Q plots
        ax = axes[0, 1]
        if len(self.m3fusion_collocated) > 0:
            stats.probplot(self.m3fusion_collocated['model_ozone'], dist="norm", plot=ax)
            ax.set_title('Q-Q Plot: M3fusion vs Normal')
        
        ax = axes[1, 0]
        if len(self.njml_collocated) > 0:
            stats.probplot(self.njml_collocated['model_ozone'], dist="norm", plot=ax)
            ax.set_title('Q-Q Plot: NJML vs Normal')
        
        ax = axes[1, 1]
        if len(self.m3fusion_collocated) > 0:
            stats.probplot(self.m3fusion_collocated['observed_ozone'], dist="norm", plot=ax)
            ax.set_title('Q-Q Plot: TOAR-II vs Normal')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'distributions_{self.year}.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
    def _create_correlation_heatmap(self):
        """Create a correlation matrix heatmap."""
        print("Creating correlation heatmap...")
        
        # Prepare correlation matrix
        corr_matrix = pd.DataFrame(index=['M3fusion', 'NJML', 'TOAR-II'], 
                                  columns=['M3fusion', 'NJML', 'TOAR-II'])
        corr_matrix.loc[:, :] = np.nan
        
        # Fill diagonal with 1s
        np.fill_diagonal(corr_matrix.values, 1.0)
        
        # Fill in correlations
        if 'M3fusion_vs_TOAR' in self.correlations:
            corr_matrix.loc['M3fusion', 'TOAR-II'] = self.correlations['M3fusion_vs_TOAR']['pearson_r']
            corr_matrix.loc['TOAR-II', 'M3fusion'] = self.correlations['M3fusion_vs_TOAR']['pearson_r']
            
        if 'NJML_vs_TOAR' in self.correlations:
            corr_matrix.loc['NJML', 'TOAR-II'] = self.correlations['NJML_vs_TOAR']['pearson_r']
            corr_matrix.loc['TOAR-II', 'NJML'] = self.correlations['NJML_vs_TOAR']['pearson_r']
            
        if 'M3fusion_vs_NJML' in self.correlations:
            corr_matrix.loc['M3fusion', 'NJML'] = self.correlations['M3fusion_vs_NJML']['pearson_r']
            corr_matrix.loc['NJML', 'M3fusion'] = self.correlations['M3fusion_vs_NJML']['pearson_r']
        
        # Create heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix.astype(float), annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title(f'Dataset Correlation Matrix ({self.year})')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'correlation_heatmap_{self.year}.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
    def save_results(self):
        """Save correlation results to files."""
        print(f"\nSaving results to {self.output_dir}...")
        
        # Save correlation statistics
        correlation_df = pd.DataFrame(self.correlations).T
        correlation_df.to_csv(os.path.join(self.output_dir, f'correlation_stats_{self.year}.csv'))
        
        # Save collocated datasets
        if len(self.m3fusion_collocated) > 0:
            self.m3fusion_collocated.to_csv(os.path.join(self.output_dir, f'm3fusion_collocated_{self.year}.csv'), index=False)
            
        if len(self.njml_collocated) > 0:
            self.njml_collocated.to_csv(os.path.join(self.output_dir, f'njml_collocated_{self.year}.csv'), index=False)
            
        if len(self.soft_comparison) > 0:
            self.soft_comparison.to_csv(os.path.join(self.output_dir, f'soft_data_comparison_{self.year}.csv'), index=False)
        
        print("✓ Results saved successfully!")
        
    def run_full_analysis(self):
        """Run the complete correlation analysis pipeline."""
        print(f"Starting comprehensive correlation analysis for {self.year}")
        print("="*60)
        
        try:
            self.load_and_prepare_data()
            self.collocate_datasets() 
            self.calculate_correlations()
            self.create_visualization_plots()
            self.save_results()
            
            print("\n" + "="*60)
            print("CORRELATION ANALYSIS COMPLETE!")
            print(f"Results saved to: {self.output_dir}")
            print("="*60)
            
        except Exception as e:
            print(f"Error during analysis: {e}")
            raise


# Example usage
if __name__ == "__main__":
    # Run analysis for a specific year
    year = 2017  # Choose a year between 2003-2019
    
    analyzer = CorrelationAnalyzer(year=year, output_dir=f"./correlation_analysis_{year}")
    analyzer.run_full_analysis()
    
    # To run for multiple years:
    # for year in range(2003, 2020):
    #     try:
    #         analyzer = CorrelationAnalyzer(year=year, output_dir=f"./correlation_analysis_{year}")
    #         analyzer.run_full_analysis()
    #     except Exception as e:
    #         print(f"Failed for year {year}: {e}")