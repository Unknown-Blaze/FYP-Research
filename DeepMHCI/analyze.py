#!/usr/bin/env python3
"""
DeepMHCI Results Analysis and Visualization Script
Generates comprehensive visualizations for model performance analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DeepMHCIAnalyzer:
    def __init__(self, results_dir="results"):
        self.results_dir = Path(results_dir)
        self.output_dir = Path("analysis_plots")
        self.output_dir.mkdir(exist_ok=True)
        
        # Load all results
        self.load_results()
        
    def load_results(self):
        """Load all result files"""
        print("Loading DeepMHCI results...")
        
        # Main 5CV results
        self.cv_results = pd.read_csv(self.results_dir / "DeepMHCI-5CV.csv")
        
        # HPV vaccine test results
        self.hpv_results = pd.read_csv(self.results_dir / "DeepMHCI-ba_data-HPV_vaccine.csv")
        
        # Length-specific results
        self.length_results = {}
        for length in [8, 9, 10, 11, 12]:
            try:
                csv_file = self.results_dir / f"DeepMHCI-ba_data-HPV_vaccine-{length}length.csv"
                if csv_file.exists():
                    self.length_results[length] = pd.read_csv(csv_file)
            except:
                continue
        
        print(f"Loaded {len(self.cv_results)} CV results")
        print(f"Loaded {len(self.hpv_results)} HPV test results")
        print(f"Loaded {len(self.length_results)} length-specific results")
    
    def plot_overall_performance_summary(self):
        """Plot overall performance metrics summary"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('DeepMHCI Overall Performance Summary', fontsize=16, fontweight='bold')
        
        # AUC distribution
        axes[0, 0].hist(self.cv_results['AUC'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(self.cv_results['AUC'].mean(), color='red', linestyle='--', 
                           label=f'Mean: {self.cv_results["AUC"].mean():.3f}')
        axes[0, 0].set_xlabel('AUC Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('AUC Distribution Across Alleles')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # AUPR distribution
        axes[0, 1].hist(self.cv_results['AUPR'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].axvline(self.cv_results['AUPR'].mean(), color='red', linestyle='--',
                           label=f'Mean: {self.cv_results["AUPR"].mean():.3f}')
        axes[0, 1].set_xlabel('AUPR Score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('AUPR Distribution Across Alleles')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Sample size vs Performance
        axes[1, 0].scatter(self.cv_results['total'], self.cv_results['AUC'], alpha=0.6, color='orange')
        axes[1, 0].set_xlabel('Number of Samples')
        axes[1, 0].set_ylabel('AUC Score')
        axes[1, 0].set_title('Sample Size vs AUC Performance')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Positive ratio vs Performance
        positive_ratio = self.cv_results['positive'] / self.cv_results['total']
        axes[1, 1].scatter(positive_ratio, self.cv_results['AUC'], alpha=0.6, color='purple')
        axes[1, 1].set_xlabel('Positive Sample Ratio')
        axes[1, 1].set_ylabel('AUC Score')
        axes[1, 1].set_title('Class Balance vs AUC Performance')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'overall_performance_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_allele_performance_heatmap(self):
        """Plot performance heatmap across alleles"""
        # Select top performing alleles for better visualization
        top_alleles = self.cv_results.nlargest(50, 'AUC')
        
        # Create pivot table for heatmap
        heatmap_data = top_alleles[['allele', 'AUC', 'AUPR', 'PCC', 'SRCC']].set_index('allele')
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                   cbar_kws={'label': 'Performance Score'})
        plt.title('Top 50 Alleles Performance Heatmap', fontsize=16, fontweight='bold')
        plt.xlabel('Performance Metrics')
        plt.ylabel('MHC Alleles')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'allele_performance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_allele_family_analysis(self):
        """Analyze performance by allele family"""
        # Extract allele family (e.g., HLA-A, HLA-B, etc.)
        self.cv_results['allele_family'] = self.cv_results['allele'].str.extract(r'([A-Z]+-\w+)')[0]
        
        # Group by family and calculate mean performance
        family_performance = self.cv_results.groupby('allele_family').agg({
            'AUC': ['mean', 'std', 'count'],
            'AUPR': ['mean', 'std'],
            'PCC': ['mean', 'std'],
            'SRCC': ['mean', 'std']
        }).round(3)
        
        # Flatten column names
        family_performance.columns = ['_'.join(col).strip() for col in family_performance.columns]
        
        # Plot family performance
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Performance by Allele Family', fontsize=16, fontweight='bold')
        
        # AUC by family
        family_auc = self.cv_results.groupby('allele_family')['AUC'].mean().sort_values(ascending=False)
        axes[0, 0].bar(range(len(family_auc)), family_auc.values, color='skyblue', alpha=0.8)
        axes[0, 0].set_xticks(range(len(family_auc)))
        axes[0, 0].set_xticklabels(family_auc.index, rotation=45, ha='right')
        axes[0, 0].set_ylabel('Mean AUC')
        axes[0, 0].set_title('Mean AUC by Allele Family')
        axes[0, 0].grid(True, alpha=0.3)
        
        # AUPR by family
        family_aupr = self.cv_results.groupby('allele_family')['AUPR'].mean().sort_values(ascending=False)
        axes[0, 1].bar(range(len(family_aupr)), family_aupr.values, color='lightgreen', alpha=0.8)
        axes[0, 1].set_xticks(range(len(family_aupr)))
        axes[0, 1].set_xticklabels(family_aupr.index, rotation=45, ha='right')
        axes[0, 1].set_ylabel('Mean AUPR')
        axes[0, 1].set_title('Mean AUPR by Allele Family')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Sample count by family
        family_count = self.cv_results.groupby('allele_family')['total'].sum().sort_values(ascending=False)
        axes[1, 0].bar(range(len(family_count)), family_count.values, color='orange', alpha=0.8)
        axes[1, 0].set_xticks(range(len(family_count)))
        axes[1, 0].set_xticklabels(family_count.index, rotation=45, ha='right')
        axes[1, 0].set_ylabel('Total Samples')
        axes[1, 0].set_title('Sample Count by Allele Family')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Performance correlation
        axes[1, 1].scatter(self.cv_results['total'], self.cv_results['AUC'], alpha=0.6, color='purple')
        axes[1, 1].set_xlabel('Number of Samples')
        axes[1, 1].set_ylabel('AUC Score')
        axes[1, 1].set_title('Sample Size vs AUC Correlation')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'allele_family_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save family performance table
        family_performance.to_csv(self.output_dir / 'allele_family_performance.csv')
        print("Family performance analysis saved to CSV")
    
    def plot_length_performance_analysis(self):
        """Analyze performance by peptide length"""
        if not self.length_results:
            print("No length-specific results found")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Performance by Peptide Length', fontsize=16, fontweight='bold')
        
        lengths = sorted(self.length_results.keys())
        metrics = ['AUC', 'AUPR', 'PCC', 'SRCC']
        
        for i, metric in enumerate(metrics):
            row, col = i // 2, i % 2
            values = []
            for length in lengths:
                if metric in self.length_results[length].columns:
                    values.append(self.length_results[length][metric].mean())
                else:
                    values.append(np.nan)
            
            axes[row, col].bar(lengths, values, color=f'C{i}', alpha=0.8, edgecolor='black')
            axes[row, col].set_xlabel('Peptide Length')
            axes[row, col].set_ylabel(f'Mean {metric}')
            axes[row, col].set_title(f'{metric} by Peptide Length')
            axes[row, col].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for j, v in enumerate(values):
                if not np.isnan(v):
                    axes[row, col].text(lengths[j], v + 0.01, f'{v:.3f}', 
                                      ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'length_performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_top_performers(self):
        """Plot top performing alleles"""
        # Top 20 by AUC
        top_auc = self.cv_results.nlargest(20, 'AUC')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Top AUC performers
        bars1 = ax1.barh(range(len(top_auc)), top_auc['AUC'], color='skyblue', alpha=0.8)
        ax1.set_yticks(range(len(top_auc)))
        ax1.set_yticklabels(top_auc['allele'])
        ax1.set_xlabel('AUC Score')
        ax1.set_title('Top 20 Alleles by AUC', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for i, bar in enumerate(bars1):
            width = bar.get_width()
            ax1.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center', fontweight='bold')
        
        # Top AUPR performers
        top_aupr = self.cv_results.nlargest(20, 'AUPR')
        bars2 = ax2.barh(range(len(top_aupr)), top_aupr['AUPR'], color='lightgreen', alpha=0.8)
        ax2.set_yticks(range(len(top_aupr)))
        ax2.set_yticklabels(top_aupr['allele'])
        ax2.set_xlabel('AUPR Score')
        ax2.set_title('Top 20 Alleles by AUPR', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for i, bar in enumerate(bars2):
            width = bar.get_width()
            ax2.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'top_performers.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_performance_correlations(self):
        """Plot correlations between different metrics"""
        metrics = ['AUC', 'AUPR', 'PCC', 'SRCC', 'total', 'positive']
        correlation_data = self.cv_results[metrics].corr()
        
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(correlation_data, dtype=bool))
        sns.heatmap(correlation_data, mask=mask, annot=True, fmt='.3f', 
                   cmap='RdYlBu_r', center=0, square=True)
        plt.title('Performance Metrics Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_correlations.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_sample_size_analysis(self):
        """Analyze performance vs sample size"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Performance vs Sample Size Analysis', fontsize=16, fontweight='bold')
        
        # Sample size distribution
        axes[0, 0].hist(self.cv_results['total'], bins=30, alpha=0.7, color='lightblue', edgecolor='black')
        axes[0, 0].set_xlabel('Number of Samples')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Sample Size Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # AUC vs sample size (log scale)
        axes[0, 1].scatter(np.log10(self.cv_results['total']), self.cv_results['AUC'], alpha=0.6, color='orange')
        axes[0, 1].set_xlabel('Log10(Number of Samples)')
        axes[0, 1].set_ylabel('AUC Score')
        axes[0, 1].set_title('AUC vs Sample Size (Log Scale)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Positive ratio vs sample size
        positive_ratio = self.cv_results['positive'] / self.cv_results['total']
        axes[1, 0].scatter(self.cv_results['total'], positive_ratio, alpha=0.6, color='green')
        axes[1, 0].set_xlabel('Number of Samples')
        axes[1, 0].set_ylabel('Positive Sample Ratio')
        axes[1, 0].set_title('Class Balance vs Sample Size')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Performance stability (std vs sample size)
        # Group by sample size ranges and calculate std
        sample_bins = pd.cut(self.cv_results['total'], bins=10)
        stability = self.cv_results.groupby(sample_bins)['AUC'].std()
        axes[1, 1].bar(range(len(stability)), stability.values, alpha=0.7, color='red')
        axes[1, 1].set_xlabel('Sample Size Bins')
        axes[1, 1].set_ylabel('AUC Standard Deviation')
        axes[1, 1].set_title('Performance Stability vs Sample Size')
        axes[1, 1].set_xticks(range(len(stability)))
        axes[1, 1].set_xticklabels([f'{int(bin_.left)}-{int(bin_.right)}' for bin_ in stability.index], 
                                   rotation=45, ha='right')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'sample_size_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        report = []
        report.append("=" * 60)
        report.append("DEEPMHCI MODEL PERFORMANCE SUMMARY REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Overall statistics
        report.append("OVERALL PERFORMANCE STATISTICS:")
        report.append("-" * 40)
        report.append(f"Total alleles tested: {len(self.cv_results)}")
        report.append(f"Mean AUC: {self.cv_results['AUC'].mean():.3f} ± {self.cv_results['AUC'].std():.3f}")
        report.append(f"Mean AUPR: {self.cv_results['AUPR'].mean():.3f} ± {self.cv_results['AUPR'].std():.3f}")
        report.append(f"Mean PCC: {self.cv_results['PCC'].mean():.3f} ± {self.cv_results['PCC'].std():.3f}")
        report.append(f"Mean SRCC: {self.cv_results['SRCC'].mean():.3f} ± {self.cv_results['SRCC'].std():.3f}")
        report.append("")
        
        # Top performers
        report.append("TOP 10 PERFORMERS BY AUC:")
        report.append("-" * 40)
        top_auc = self.cv_results.nlargest(10, 'AUC')[['allele', 'AUC', 'AUPR', 'total', 'positive']]
        for _, row in top_auc.iterrows():
            report.append(f"{row['allele']:15} | AUC: {row['AUC']:.3f} | AUPR: {row['AUPR']:.3f} | "
                        f"Samples: {row['total']} | Positives: {row['positive']}")
        report.append("")
        
        # Sample size analysis
        report.append("SAMPLE SIZE ANALYSIS:")
        report.append("-" * 40)
        report.append(f"Total samples across all alleles: {self.cv_results['total'].sum():,}")
        report.append(f"Mean samples per allele: {self.cv_results['total'].mean():.1f}")
        report.append(f"Median samples per allele: {self.cv_results['total'].median():.1f}")
        report.append(f"Min samples: {self.cv_results['total'].min()}")
        report.append(f"Max samples: {self.cv_results['total'].max():,}")
        report.append("")
        
        # Performance ranges
        report.append("PERFORMANCE RANGES:")
        report.append("-" * 40)
        report.append(f"AUC range: {self.cv_results['AUC'].min():.3f} - {self.cv_results['AUC'].max():.3f}")
        report.append(f"AUPR range: {self.cv_results['AUPR'].min():.3f} - {self.cv_results['AUPR'].max():.3f}")
        report.append(f"PCC range: {self.cv_results['PCC'].min():.3f} - {self.cv_results['PCC'].max():.3f}")
        report.append(f"SRCC range: {self.cv_results['SRCC'].min():.3f} - {self.cv_results['SRCC'].max():.3f}")
        report.append("")
        
        # Save report
        with open(self.output_dir / 'performance_summary_report.txt', 'w') as f:
            f.write('\n'.join(report))
        
        # Print report
        print('\n'.join(report))
        print(f"\nDetailed report saved to: {self.output_dir / 'performance_summary_report.txt'}")
    
    def run_all_analyses(self):
        """Run all analysis functions"""
        print("Starting DeepMHCI comprehensive analysis...")
        print(f"Results will be saved to: {self.output_dir}")
        
        try:
            self.plot_overall_performance_summary()
            print("✓ Overall performance summary generated")
            
            self.plot_allele_performance_heatmap()
            print("✓ Allele performance heatmap generated")
            
            self.plot_allele_family_analysis()
            print("✓ Allele family analysis generated")
            
            self.plot_length_performance_analysis()
            print("✓ Length performance analysis generated")
            
            self.plot_top_performers()
            print("✓ Top performers analysis generated")
            
            self.plot_performance_correlations()
            print("✓ Performance correlations generated")
            
            self.plot_sample_size_analysis()
            print("✓ Sample size analysis generated")
            
            self.generate_summary_report()
            print("✓ Summary report generated")
            
            print(f"\n🎉 All analyses completed! Check {self.output_dir} for results.")
            
        except Exception as e:
            print(f"Error during analysis: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    # Create analyzer and run all analyses
    analyzer = DeepMHCIAnalyzer()
    analyzer.run_all_analyses()
