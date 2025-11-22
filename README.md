# cpugpu

Resource Control System (RCS) - CPU/GPU Performance Analysis Tool

## Overview

`cpugpu` is a Python-based resource analysis tool for monitoring and analyzing CPU and system resource utilization. It provides real-time monitoring, statistical analysis, and visualization capabilities for system performance metrics.

## Features

- **Real-time Monitoring**: Collect CPU, memory, and disk I/O metrics in real-time
- **Statistical Analysis**: Compute average, min, and max utilization over time
- **Visualization**: Generate plots and charts for performance metrics
- **Flexible Duration**: Run analysis for custom time periods with configurable intervals
- **Data Export**: Save results to JSON format for further analysis
- **Command-line Interface**: Easy-to-use CLI for quick analysis

## Installation

1. Clone the repository:
```bash
git clone https://github.com/mwsoccer10/cpugpu.git
cd cpugpu
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Analysis

Run a basic 10-second analysis:
```bash
python analyzer.py
```

### Custom Duration and Interval

Run analysis for 30 seconds with 2-second sampling intervals:
```bash
python analyzer.py --duration 30 --interval 2
```

### Save Results to File

Save analysis results to a JSON file:
```bash
python analyzer.py --duration 20 --output results.json
```

### Visualize Results

Generate visualization from saved results:
```bash
python visualize.py results.json
```

Generate specific plot types:
```bash
# CPU utilization only
python visualize.py results.json --type cpu --output cpu_plot.png

# Memory utilization only
python visualize.py results.json --type memory --output memory_plot.png

# Combined plot (default)
python visualize.py results.json --type combined --output combined_plot.png
```

### Run Examples

See example usage patterns:
```bash
python example.py
```

## Command-line Options

### analyzer.py

- `-d, --duration`: Duration of analysis in seconds (default: 10)
- `-i, --interval`: Sampling interval in seconds (default: 1)
- `-o, --output`: Output file for results (JSON format)
- `--quiet`: Suppress summary output

### visualize.py

- `results_file`: Path to JSON results file (required)
- `-t, --type`: Type of plot ('cpu', 'memory', or 'combined')
- `-o, --output`: Output file for saving the plot

## Output Format

The analyzer generates JSON output with the following structure:

```json
{
  "measurements": [
    {
      "timestamp": "2025-11-22T10:00:00.000000",
      "cpu": {
        "cpu_percent_avg": 25.5,
        "cpu_count": 8,
        "cpu_freq_current": 2400.0
      },
      "memory": {
        "memory_percent": 45.2,
        "memory_total": 17179869184,
        "memory_used": 7766597632
      },
      "disk": {
        "disk_percent": 60.5,
        "disk_total": 1000000000000
      }
    }
  ],
  "analysis": {
    "cpu_analysis": {
      "avg_utilization": 25.5,
      "max_utilization": 30.2,
      "min_utilization": 20.1
    },
    "memory_analysis": {
      "avg_utilization": 45.2,
      "max_utilization": 50.0,
      "min_utilization": 40.5
    }
  }
}
```

## Examples

### Example 1: Quick System Check
```python
from analyzer import ResourceAnalyzer

analyzer = ResourceAnalyzer()
metrics = analyzer.collect_all_metrics()
print(f"CPU Usage: {metrics['cpu']['cpu_percent_avg']:.2f}%")
print(f"Memory Usage: {metrics['memory']['memory_percent']:.2f}%")
```

### Example 2: Continuous Monitoring
```python
from analyzer import ResourceAnalyzer

analyzer = ResourceAnalyzer()
analyzer.run_analysis(duration=60, interval=5)
analyzer.print_summary()
analyzer.save_results('monitoring_results.json')
```

## Requirements

- Python 3.6+
- numpy >= 1.21.0
- pandas >= 1.3.0
- matplotlib >= 3.4.0
- psutil >= 5.8.0

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
