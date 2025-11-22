#!/usr/bin/env python3
"""
CPU/GPU Resource Analysis Tool

This module provides functionality for analyzing CPU and GPU performance metrics,
including utilization, memory usage, and performance trends.
"""

import psutil
import time
import json
from datetime import datetime
from typing import Dict, List, Optional
import argparse


class ResourceAnalyzer:
    """Analyzes CPU and system resource utilization."""
    
    def __init__(self):
        self.measurements = []
    
    def collect_cpu_metrics(self) -> Dict:
        """Collect current CPU metrics."""
        cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
        cpu_freq = psutil.cpu_freq(percpu=False)
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent_per_core': cpu_percent,
            'cpu_percent_avg': sum(cpu_percent) / len(cpu_percent),
            'cpu_count': psutil.cpu_count(logical=True),
            'cpu_count_physical': psutil.cpu_count(logical=False),
        }
        
        if cpu_freq:
            metrics['cpu_freq_current'] = cpu_freq.current
            metrics['cpu_freq_min'] = cpu_freq.min
            metrics['cpu_freq_max'] = cpu_freq.max
        
        return metrics
    
    def collect_memory_metrics(self) -> Dict:
        """Collect current memory metrics."""
        virtual_mem = psutil.virtual_memory()
        swap_mem = psutil.swap_memory()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'memory_total': virtual_mem.total,
            'memory_available': virtual_mem.available,
            'memory_used': virtual_mem.used,
            'memory_percent': virtual_mem.percent,
            'swap_total': swap_mem.total,
            'swap_used': swap_mem.used,
            'swap_percent': swap_mem.percent,
        }
    
    def collect_disk_metrics(self) -> Dict:
        """Collect current disk I/O metrics."""
        disk_usage = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'disk_total': disk_usage.total,
            'disk_used': disk_usage.used,
            'disk_free': disk_usage.free,
            'disk_percent': disk_usage.percent,
        }
        
        if disk_io:
            metrics['disk_read_count'] = disk_io.read_count
            metrics['disk_write_count'] = disk_io.write_count
            metrics['disk_read_bytes'] = disk_io.read_bytes
            metrics['disk_write_bytes'] = disk_io.write_bytes
        
        return metrics
    
    def collect_all_metrics(self) -> Dict:
        """Collect all available metrics."""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'cpu': self.collect_cpu_metrics(),
            'memory': self.collect_memory_metrics(),
            'disk': self.collect_disk_metrics(),
        }
        
        self.measurements.append(metrics)
        return metrics
    
    def analyze_measurements(self) -> Dict:
        """Analyze collected measurements and generate statistics."""
        if not self.measurements:
            return {'error': 'No measurements collected'}
        
        cpu_percentages = [m['cpu']['cpu_percent_avg'] for m in self.measurements]
        memory_percentages = [m['memory']['memory_percent'] for m in self.measurements]
        
        analysis = {
            'total_measurements': len(self.measurements),
            'time_range': {
                'start': self.measurements[0]['timestamp'],
                'end': self.measurements[-1]['timestamp'],
            },
            'cpu_analysis': {
                'avg_utilization': sum(cpu_percentages) / len(cpu_percentages),
                'max_utilization': max(cpu_percentages),
                'min_utilization': min(cpu_percentages),
            },
            'memory_analysis': {
                'avg_utilization': sum(memory_percentages) / len(memory_percentages),
                'max_utilization': max(memory_percentages),
                'min_utilization': min(memory_percentages),
            },
        }
        
        return analysis
    
    def run_analysis(self, duration: int = 10, interval: int = 1) -> Dict:
        """
        Run continuous analysis for specified duration.
        
        Args:
            duration: Total duration in seconds
            interval: Sampling interval in seconds
        
        Returns:
            Dictionary containing analysis results
        """
        print(f"Starting resource analysis for {duration} seconds...")
        start_time = time.time()
        
        while time.time() - start_time < duration:
            self.collect_all_metrics()
            time.sleep(interval)
        
        print(f"Analysis complete. Collected {len(self.measurements)} measurements.")
        return self.analyze_measurements()
    
    def save_results(self, filename: str):
        """Save measurements and analysis to a JSON file."""
        results = {
            'measurements': self.measurements,
            'analysis': self.analyze_measurements(),
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {filename}")
    
    def print_summary(self):
        """Print a summary of the analysis."""
        analysis = self.analyze_measurements()
        
        if 'error' in analysis:
            print(f"Error: {analysis['error']}")
            return
        
        print("\n" + "="*50)
        print("RESOURCE ANALYSIS SUMMARY")
        print("="*50)
        print(f"\nTotal Measurements: {analysis['total_measurements']}")
        print(f"Time Range: {analysis['time_range']['start']} to {analysis['time_range']['end']}")
        
        print("\nCPU Analysis:")
        print(f"  Average Utilization: {analysis['cpu_analysis']['avg_utilization']:.2f}%")
        print(f"  Maximum Utilization: {analysis['cpu_analysis']['max_utilization']:.2f}%")
        print(f"  Minimum Utilization: {analysis['cpu_analysis']['min_utilization']:.2f}%")
        
        print("\nMemory Analysis:")
        print(f"  Average Utilization: {analysis['memory_analysis']['avg_utilization']:.2f}%")
        print(f"  Maximum Utilization: {analysis['memory_analysis']['max_utilization']:.2f}%")
        print(f"  Minimum Utilization: {analysis['memory_analysis']['min_utilization']:.2f}%")
        print("="*50 + "\n")


def main():
    """Main entry point for the analyzer."""
    parser = argparse.ArgumentParser(
        description='CPU/GPU Resource Analysis Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '-d', '--duration',
        type=int,
        default=10,
        help='Duration of analysis in seconds (default: 10)'
    )
    parser.add_argument(
        '-i', '--interval',
        type=int,
        default=1,
        help='Sampling interval in seconds (default: 1)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Output file for results (JSON format)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress summary output'
    )
    
    args = parser.parse_args()
    
    # Create analyzer and run analysis
    analyzer = ResourceAnalyzer()
    analyzer.run_analysis(duration=args.duration, interval=args.interval)
    
    # Print summary unless quiet mode
    if not args.quiet:
        analyzer.print_summary()
    
    # Save results if output file specified
    if args.output:
        analyzer.save_results(args.output)


if __name__ == '__main__':
    main()
