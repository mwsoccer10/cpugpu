#!/usr/bin/env python3
"""
Example usage of the CPU/GPU Resource Analysis Tool.

This script demonstrates how to use the analyzer module to collect
and analyze system resource metrics.
"""

from analyzer import ResourceAnalyzer
import time


def example_basic_analysis():
    """Example: Basic resource analysis."""
    print("Example 1: Basic Resource Analysis")
    print("-" * 50)
    
    analyzer = ResourceAnalyzer()
    
    # Run analysis for 5 seconds with 1-second intervals
    analyzer.run_analysis(duration=5, interval=1)
    
    # Print summary
    analyzer.print_summary()
    
    # Save results
    analyzer.save_results('example_results.json')
    
    print("\n")


def example_custom_monitoring():
    """Example: Custom monitoring with manual control."""
    print("Example 2: Custom Monitoring")
    print("-" * 50)
    
    analyzer = ResourceAnalyzer()
    
    # Collect metrics manually
    print("Collecting metrics...")
    for i in range(3):
        print(f"  Measurement {i+1}/3")
        metrics = analyzer.collect_all_metrics()
        print(f"    CPU: {metrics['cpu']['cpu_percent_avg']:.2f}%")
        print(f"    Memory: {metrics['memory']['memory_percent']:.2f}%")
        time.sleep(1)
    
    # Analyze collected data
    analysis = analyzer.analyze_measurements()
    print("\nAnalysis Results:")
    print(f"  Average CPU: {analysis['cpu_analysis']['avg_utilization']:.2f}%")
    print(f"  Average Memory: {analysis['memory_analysis']['avg_utilization']:.2f}%")
    
    print("\n")


def example_single_snapshot():
    """Example: Single snapshot of current metrics."""
    print("Example 3: Single Snapshot")
    print("-" * 50)
    
    analyzer = ResourceAnalyzer()
    
    # Get current CPU metrics
    cpu_metrics = analyzer.collect_cpu_metrics()
    print("Current CPU Metrics:")
    print(f"  Average Utilization: {cpu_metrics['cpu_percent_avg']:.2f}%")
    print(f"  CPU Cores: {cpu_metrics['cpu_count']} (logical), {cpu_metrics['cpu_count_physical']} (physical)")
    if 'cpu_freq_current' in cpu_metrics:
        print(f"  Current Frequency: {cpu_metrics['cpu_freq_current']:.2f} MHz")
    
    # Get current memory metrics
    memory_metrics = analyzer.collect_memory_metrics()
    print("\nCurrent Memory Metrics:")
    print(f"  Utilization: {memory_metrics['memory_percent']:.2f}%")
    print(f"  Used: {memory_metrics['memory_used'] / (1024**3):.2f} GB")
    print(f"  Available: {memory_metrics['memory_available'] / (1024**3):.2f} GB")
    print(f"  Total: {memory_metrics['memory_total'] / (1024**3):.2f} GB")
    
    # Get current disk metrics
    disk_metrics = analyzer.collect_disk_metrics()
    print("\nCurrent Disk Metrics:")
    print(f"  Utilization: {disk_metrics['disk_percent']:.2f}%")
    print(f"  Used: {disk_metrics['disk_used'] / (1024**3):.2f} GB")
    print(f"  Free: {disk_metrics['disk_free'] / (1024**3):.2f} GB")
    print(f"  Total: {disk_metrics['disk_total'] / (1024**3):.2f} GB")
    
    print("\n")


def main():
    """Run all examples."""
    print("\n" + "="*50)
    print("CPU/GPU Resource Analysis Tool - Examples")
    print("="*50 + "\n")
    
    # Run examples
    example_single_snapshot()
    example_custom_monitoring()
    example_basic_analysis()
    
    print("="*50)
    print("All examples completed!")
    print("="*50 + "\n")


if __name__ == '__main__':
    main()
