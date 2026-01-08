#!/bin/bash
#
# Run complete model performance summary analysis
# This script generates all performance summaries and analysis outputs
#
# Usage: ./run_model_summary_analysis.sh
#

set -e  # Exit on error

PYTHON=${PYTHON:-"python"}

echo "=============================================================="
echo "RAMP Model Performance Summary & Analysis"
echo "=============================================================="
echo ""
echo "This script will:"
echo "  1. Generate performance summaries for all model-year combinations"
echo "  2. Create combined summaries per model"
echo "  3. Generate master summary across all models"
echo "  4. Analyze results and create visualizations"
echo "  5. Generate model selection recommendations"
echo ""
echo "Output directories:"
echo "  • ./model_summaries/ - Detailed CSV summaries"
echo "  • ./model_analysis/  - Analysis results and plots"
echo ""
echo "=============================================================="
echo ""

# Check if ramp_data directory exists
if [ ! -d "ramp_data" ]; then
    echo "❌ ERROR: ramp_data directory not found!"
    echo ""
    echo "Please ensure you have run RAMP analysis first using:"
    echo "  python ramp_analysis_parallel.py --year YEAR --model MODEL --cores N"
    echo ""
    exit 1
fi

# Check if there are any collocated data files
collocated_count=$(find ramp_data -name "collocated_data_*" -type f 2>/dev/null | wc -l)
if [ "$collocated_count" -eq 0 ]; then
    echo "❌ ERROR: No collocated data files found in ramp_data/"
    echo ""
    echo "Please run RAMP analysis first to generate required data files."
    echo ""
    exit 1
fi

echo "✓ Found $collocated_count model-year combinations in ramp_data/"
echo ""

# Step 1: Generate summaries
echo "=============================================================="
echo "STEP 1: Generating Performance Summaries"
echo "=============================================================="
echo ""

$PYTHON generate_model_performance_summaries.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ ERROR: Summary generation failed!"
    echo "Please check the error messages above."
    exit 1
fi

echo ""
echo "✅ Summary generation completed successfully"
echo ""

# Step 2: Analyze results
echo "=============================================================="
echo "STEP 2: Analyzing Results and Creating Visualizations"
echo "=============================================================="
echo ""

$PYTHON analyze_model_summaries.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ ERROR: Analysis failed!"
    echo "Please check the error messages above."
    exit 1
fi

echo ""
echo "✅ Analysis completed successfully"
echo ""

# Summary of outputs
echo "=============================================================="
echo "COMPLETE! All outputs generated successfully"
echo "=============================================================="
echo ""
echo "📁 Output directories:"
echo "  • ./model_summaries/ - Detailed performance summaries"
echo "  • ./model_analysis/  - Analysis and visualizations"
echo ""

# Count files
summary_files=$(find model_summaries -type f -name "*.csv" 2>/dev/null | wc -l)
analysis_files=$(find model_analysis -type f 2>/dev/null | wc -l)

echo "📊 Generated files:"
echo "  • Summary CSVs: $summary_files"
echo "  • Analysis outputs: $analysis_files"
echo ""

# Show key outputs
echo "🔑 Key files to review:"
echo ""
echo "  Quick Start:"
echo "    cat model_analysis/best_model_per_year.csv"
echo "    cat model_analysis/model_selection_recommendations.csv"
echo ""
echo "  Master Summary:"
echo "    cat model_summaries/ALL_MODELS_master_summary.csv"
echo "    cat model_summaries/ALL_MODELS_average_by_model.csv"
echo ""
echo "  Visualizations:"
echo "    ls -lh model_analysis/*.png"
echo ""

# Check if we can display images
if command -v display &> /dev/null; then
    echo "To view plots:"
    echo "  display model_analysis/model_comparison_heatmap.png"
elif command -v open &> /dev/null; then
    echo "To view plots:"
    echo "  open model_analysis/*.png"
elif command -v xdg-open &> /dev/null; then
    echo "To view plots:"
    echo "  xdg-open model_analysis/*.png"
fi

echo ""
echo "=============================================================="
echo "For detailed documentation, see: README_MODEL_SUMMARIES.md"
echo "=============================================================="
echo ""

exit 0
