#!/usr/bin/env python3
"""
Visualization utilities for CPU/GPU analysis results.
"""

import json
import matplotlib.pyplot as plt
from typing import Dict, List
import argparse


def plot_cpu_utilization(measurements: List[Dict], output_file: str = None):
    """Plot CPU utilization over time."""
    timestamps = range(len(measurements))
    cpu_percentages = [m['cpu']['cpu_percent_avg'] for m in measurements]
    
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, cpu_percentages, 'b-', linewidth=2, label='CPU Usage')
    plt.xlabel('Measurement Number')
    plt.ylabel('CPU Utilization (%)')
    plt.title('CPU Utilization Over Time')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(0, 100)
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"CPU plot saved to {output_file}")
    else:
        plt.show()
    
    plt.close()


def plot_memory_utilization(measurements: List[Dict], output_file: str = None):
    """Plot memory utilization over time."""
    timestamps = range(len(measurements))
    memory_percentages = [m['memory']['memory_percent'] for m in measurements]
    
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, memory_percentages, 'r-', linewidth=2, label='Memory Usage')
    plt.xlabel('Measurement Number')
    plt.ylabel('Memory Utilization (%)')
    plt.title('Memory Utilization Over Time')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(0, 100)
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Memory plot saved to {output_file}")
    else:
        plt.show()
    
    plt.close()


def plot_combined_metrics(measurements: List[Dict], output_file: str = None):
    """Plot CPU and memory utilization in subplots."""
    timestamps = range(len(measurements))
    cpu_percentages = [m['cpu']['cpu_percent_avg'] for m in measurements]
    memory_percentages = [m['memory']['memory_percent'] for m in measurements]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # CPU subplot
    ax1.plot(timestamps, cpu_percentages, 'b-', linewidth=2, label='CPU Usage')
    ax1.set_xlabel('Measurement Number')
    ax1.set_ylabel('CPU Utilization (%)')
    ax1.set_title('CPU Utilization Over Time')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(0, 100)
    
    # Memory subplot
    ax2.plot(timestamps, memory_percentages, 'r-', linewidth=2, label='Memory Usage')
    ax2.set_xlabel('Measurement Number')
    ax2.set_ylabel('Memory Utilization (%)')
    ax2.set_title('Memory Utilization Over Time')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim(0, 100)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Combined plot saved to {output_file}")
    else:
        plt.show()
    
    plt.close()


def visualize_results(results_file: str, plot_type: str = 'combined', output_file: str = None):
    """
    Load and visualize analysis results.
    
    Args:
        results_file: Path to JSON results file
        plot_type: Type of plot ('cpu', 'memory', or 'combined')
        output_file: Optional output file for saving the plot
    """
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        print(f"Error: Results file '{results_file}' not found")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in results file: {e}")
        return
    
    measurements = results.get('measurements', [])
    
    if not measurements:
        print("No measurements found in results file")
        return
    
    if plot_type == 'cpu':
        plot_cpu_utilization(measurements, output_file)
    elif plot_type == 'memory':
        plot_memory_utilization(measurements, output_file)
    else:
        plot_combined_metrics(measurements, output_file)


def main():
    """Main entry point for visualization."""
    parser = argparse.ArgumentParser(
        description='Visualize CPU/GPU analysis results',
    )
    parser.add_argument(
        'results_file',
        type=str,
        help='Path to JSON results file'
    )
    parser.add_argument(
        '-t', '--type',
        type=str,
        choices=['cpu', 'memory', 'combined'],
        default='combined',
        help='Type of plot to generate (default: combined)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Output file for saving the plot'
    )
    
    args = parser.parse_args()
    
    visualize_results(args.results_file, args.type, args.output)


if __name__ == '__main__':
    main()
