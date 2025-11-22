#!/usr/bin/env python3
"""
Unit tests for the CPU/GPU Resource Analysis Tool.
"""

import unittest
import json
import os
import time
from analyzer import ResourceAnalyzer


class TestResourceAnalyzer(unittest.TestCase):
    """Test cases for ResourceAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = ResourceAnalyzer()
    
    def tearDown(self):
        """Clean up after tests."""
        # Clean up any test output files
        test_files = ['test_output.json', 'test_analysis.json']
        for file in test_files:
            if os.path.exists(file):
                os.remove(file)
    
    def test_collect_cpu_metrics(self):
        """Test CPU metrics collection."""
        metrics = self.analyzer.collect_cpu_metrics()
        
        # Verify required fields exist
        self.assertIn('timestamp', metrics)
        self.assertIn('cpu_percent_per_core', metrics)
        self.assertIn('cpu_percent_avg', metrics)
        self.assertIn('cpu_count', metrics)
        self.assertIn('cpu_count_physical', metrics)
        
        # Verify data types
        self.assertIsInstance(metrics['cpu_percent_per_core'], list)
        self.assertIsInstance(metrics['cpu_percent_avg'], (int, float))
        self.assertIsInstance(metrics['cpu_count'], int)
        
        # Verify valid ranges
        self.assertGreaterEqual(metrics['cpu_percent_avg'], 0)
        self.assertLessEqual(metrics['cpu_percent_avg'], 100)
        self.assertGreater(metrics['cpu_count'], 0)
    
    def test_collect_memory_metrics(self):
        """Test memory metrics collection."""
        metrics = self.analyzer.collect_memory_metrics()
        
        # Verify required fields exist
        self.assertIn('timestamp', metrics)
        self.assertIn('memory_total', metrics)
        self.assertIn('memory_available', metrics)
        self.assertIn('memory_used', metrics)
        self.assertIn('memory_percent', metrics)
        
        # Verify valid ranges
        self.assertGreaterEqual(metrics['memory_percent'], 0)
        self.assertLessEqual(metrics['memory_percent'], 100)
        self.assertGreater(metrics['memory_total'], 0)
        self.assertGreaterEqual(metrics['memory_used'], 0)
        self.assertGreaterEqual(metrics['memory_available'], 0)
    
    def test_collect_disk_metrics(self):
        """Test disk metrics collection."""
        metrics = self.analyzer.collect_disk_metrics()
        
        # Verify required fields exist
        self.assertIn('timestamp', metrics)
        self.assertIn('disk_total', metrics)
        self.assertIn('disk_used', metrics)
        self.assertIn('disk_free', metrics)
        self.assertIn('disk_percent', metrics)
        
        # Verify valid ranges
        self.assertGreaterEqual(metrics['disk_percent'], 0)
        self.assertLessEqual(metrics['disk_percent'], 100)
        self.assertGreater(metrics['disk_total'], 0)
    
    def test_collect_all_metrics(self):
        """Test collection of all metrics."""
        metrics = self.analyzer.collect_all_metrics()
        
        # Verify structure
        self.assertIn('timestamp', metrics)
        self.assertIn('cpu', metrics)
        self.assertIn('memory', metrics)
        self.assertIn('disk', metrics)
        
        # Verify measurements were stored
        self.assertEqual(len(self.analyzer.measurements), 1)
        self.assertEqual(self.analyzer.measurements[0], metrics)
    
    def test_analyze_measurements_empty(self):
        """Test analysis with no measurements."""
        analysis = self.analyzer.analyze_measurements()
        
        self.assertIn('error', analysis)
        self.assertEqual(analysis['error'], 'No measurements collected')
    
    def test_analyze_measurements_with_data(self):
        """Test analysis with collected measurements."""
        # Collect some measurements
        for _ in range(3):
            self.analyzer.collect_all_metrics()
            time.sleep(0.1)
        
        analysis = self.analyzer.analyze_measurements()
        
        # Verify structure
        self.assertIn('total_measurements', analysis)
        self.assertIn('time_range', analysis)
        self.assertIn('cpu_analysis', analysis)
        self.assertIn('memory_analysis', analysis)
        
        # Verify counts
        self.assertEqual(analysis['total_measurements'], 3)
        
        # Verify CPU analysis
        cpu_analysis = analysis['cpu_analysis']
        self.assertIn('avg_utilization', cpu_analysis)
        self.assertIn('max_utilization', cpu_analysis)
        self.assertIn('min_utilization', cpu_analysis)
        self.assertGreaterEqual(cpu_analysis['avg_utilization'], 0)
        self.assertLessEqual(cpu_analysis['avg_utilization'], 100)
        
        # Verify memory analysis
        memory_analysis = analysis['memory_analysis']
        self.assertIn('avg_utilization', memory_analysis)
        self.assertIn('max_utilization', memory_analysis)
        self.assertIn('min_utilization', memory_analysis)
        self.assertGreaterEqual(memory_analysis['avg_utilization'], 0)
        self.assertLessEqual(memory_analysis['avg_utilization'], 100)
    
    def test_run_analysis(self):
        """Test running a complete analysis."""
        analysis = self.analyzer.run_analysis(duration=2, interval=1)
        
        # Verify we collected measurements
        self.assertGreater(len(self.analyzer.measurements), 0)
        
        # Verify analysis structure
        self.assertIn('total_measurements', analysis)
        self.assertIn('cpu_analysis', analysis)
        self.assertIn('memory_analysis', analysis)
    
    def test_save_results(self):
        """Test saving results to a file."""
        # Collect some measurements
        self.analyzer.collect_all_metrics()
        self.analyzer.collect_all_metrics()
        
        # Save results
        output_file = 'test_output.json'
        self.analyzer.save_results(output_file)
        
        # Verify file exists
        self.assertTrue(os.path.exists(output_file))
        
        # Verify file contents
        with open(output_file, 'r') as f:
            results = json.load(f)
        
        self.assertIn('measurements', results)
        self.assertIn('analysis', results)
        self.assertEqual(len(results['measurements']), 2)
    
    def test_print_summary(self):
        """Test printing summary (just verify it doesn't crash)."""
        # Collect some measurements
        self.analyzer.collect_all_metrics()
        
        # This should not raise an exception
        try:
            self.analyzer.print_summary()
            success = True
        except Exception:
            success = False
        
        self.assertTrue(success)
    
    def test_measurements_accumulation(self):
        """Test that measurements accumulate correctly."""
        initial_count = len(self.analyzer.measurements)
        
        self.analyzer.collect_all_metrics()
        self.assertEqual(len(self.analyzer.measurements), initial_count + 1)
        
        self.analyzer.collect_all_metrics()
        self.assertEqual(len(self.analyzer.measurements), initial_count + 2)


if __name__ == '__main__':
    unittest.main()
