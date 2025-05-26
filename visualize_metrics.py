import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

def plot_metrics(csv_file_path: Path, output_dir: Path):
    """
    Reads hit metrics CSV and generates basic visualizations.
    """
    if not csv_file_path.exists():
        print(f"Error: CSV file not found at {csv_file_path}")
        return

    try:
        df = pd.read_csv(csv_file_path)
    except Exception as e:
        print(f"Error reading CSV file {csv_file_path}: {e}")
        return

    if df.empty:
        print(f"Warning: CSV file {csv_file_path} is empty")
        return

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    base_filename = csv_file_path.stem.replace('_hit_metrics', '')

    # Set style for seaborn plots
    sns.set_theme(style="whitegrid")

    # --- Histograms ---
    metrics_to_histogram = [
        'time_on_target_s', 'time_between_hits_s',
        'flick_time_s', 'adjustment_time_s',
        'flick_distance_px', 'norm_flick_time_s_per_px',
        'avg_flick_speed_px_s', 'speed_at_flick_end_px_f'
    ]

    print("\nGenerating histograms")
    for metric in metrics_to_histogram:
        if metric in df.columns and df[metric].notna().any(): # Check if column exists and has non-NaN data
            plt.figure(figsize=(10, 6))
            sns.histplot(df[metric].dropna(), kde=True, bins=20)
            plt.title(f'Distribution of {metric}')
            plt.xlabel(metric)
            plt.ylabel('Frequency')
            plt.tight_layout()
            plt.savefig(output_dir / f"{base_filename}_hist_{metric}.png")
            plt.close()
        else:
            print(f"Skipping histogram for '{metric}': column not found or all NaN.")


    # --- Scatter Plots ---
    scatter_plot_pairs = [
        ('flick_distance_px', 'flick_time_s'),
        ('flick_distance_px', 'avg_flick_speed_px_s'),
        ('speed_at_flick_end_px_f', 'adjustment_time_s'),
        ('flick_time_s', 'adjustment_time_s')
    ]

    print("\nGenerating scatter plots")
    for x_metric, y_metric in scatter_plot_pairs:
        if x_metric in df.columns and y_metric in df.columns and \
           df[x_metric].notna().any() and df[y_metric].notna().any():
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=df.dropna(subset=[x_metric, y_metric]), x=x_metric, y=y_metric, alpha=0.6)
            sns.regplot(data=df.dropna(subset=[x_metric, y_metric]), x=x_metric, y=y_metric, scatter=False, color='red')
            plt.title(f'{y_metric} vs. {x_metric}')
            plt.xlabel(x_metric)
            plt.ylabel(y_metric)
            plt.tight_layout()
            plt.savefig(output_dir / f"{base_filename}_scatter_{y_metric}_vs_{x_metric}.png")
            plt.close()
        else:
            print(f"Skipping scatter plot for '{y_metric}' vs '{x_metric}': one or both columns not found or all NaN.")

    # --- Line Plots - Trends over 'hit_number' ---
    metrics_for_line_plot = [
        'flick_time_s', 'adjustment_time_s',
        'norm_flick_time_s_per_px', 'avg_flick_speed_px_s'
    ]
    if 'hit_number' in df.columns:
        print("\nGenerating line plots over hit number")
        for metric in metrics_for_line_plot:
            if metric in df.columns and df[metric].notna().any():
                plt.figure(figsize=(12, 6))
                sns.lineplot(data=df.dropna(subset=[metric]), x='hit_number', y=metric, marker='o')
                if len(df[metric].dropna()) >= 5: # Need >= 5 points for rolling avg
                    rolling_avg = df[metric].dropna().rolling(window=5, center=True).mean()
                    plt.plot(df['hit_number'][df[metric].notna()], rolling_avg, color='red', linestyle='--', label='5-hit Rolling Avg')
                    plt.legend()

                plt.title(f'{metric} over Hit Number')
                plt.xlabel('Hit Number')
                plt.ylabel(metric)
                plt.tight_layout()
                plt.savefig(output_dir / f"{base_filename}_line_{metric}_over_hits.png")
                plt.close()
            else:
                print(f"Skipping line plot for '{metric}': column not found or all NaN.")
    else:
        print("Skipping line plots: 'hit_number' column not found.")

    print(f"\nVisualizations saved to: {output_dir.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize hit metrics from a CSV file.")
    parser.add_argument("csv_file", type=str, help="Path to the _hit_metrics.csv file.")
    parser.add_argument("-o", "--output_directory", type=str, default="metric_visualizations",
                        help="Directory to save the generated plots (default: metric_visualizations).")

    args = parser.parse_args()

    csv_path = Path(args.csv_file)
    output_path = Path(args.output_directory)

    plot_metrics(csv_path, output_path)