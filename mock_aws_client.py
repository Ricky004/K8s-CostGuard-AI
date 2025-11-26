"""
Mock AWS clients for development/testing without real AWS credentials
"""
import json
import random
from typing import Dict, Any


class MockDataLoader:
    """Loads mock data from JSON file"""
    
    def __init__(self, mock_file: str = "mock_data.json"):
        with open(mock_file, 'r') as f:
            self.data = json.load(f)
    
    def get(self, *keys):
        """Navigate nested dict using keys"""
        result = self.data
        for key in keys:
            result = result[key]
        return result


class MockCostExplorerClient:
    """Mock boto3 Cost Explorer client"""
    
    def __init__(self):
        self.loader = MockDataLoader()
    
    def get_cost_and_usage(self, TimePeriod, Granularity, Metrics, Filter=None):
        """Mock get_cost_and_usage response"""
        
        if Filter and 'SERVICE' in str(Filter):
            # Storage anomaly request
            return {
                'ResultsByTime': [
                    {'Total': {'UnblendedCost': {'Amount': '10.50'}}},
                    {'Total': {'UnblendedCost': {'Amount': '31.50'}}}
                ]
            }
        
        # Daily spend request
        daily_data = self.loader.get('cost_and_usage', 'daily_spend')
        
        return {
            'ResultsByTime': [
                {'Total': {'UnblendedCost': {'Amount': str(daily_data['yesterday'])}}},
                {'Total': {'UnblendedCost': {'Amount': str(daily_data['today'])}}}
            ]
        }


class MockCloudWatchClient:
    """Mock boto3 CloudWatch client"""
    
    def __init__(self):
        self.loader = MockDataLoader()
    
    def get_metric_statistics(self, Namespace, MetricName, Dimensions, 
                              StartTime, EndTime, Period, Statistics):
        """Mock get_metric_statistics response"""
        
        pod_name = next((d['Value'] for d in Dimensions if d['Name'] == 'PodName'), None)
        namespace = next((d['Value'] for d in Dimensions if d['Name'] == 'Namespace'), None)
        
        # Try to find matching pod in mock data
        pods = self.loader.get('cloudwatch_metrics', 'pod_utilization')
        
        for pod in pods:
            if pod['pod_name'] == pod_name and pod['namespace'] == namespace:
                return {
                    'Datapoints': [
                        {'Average': pod['cpu_percent']}
                    ]
                }
        
        # Random fallback if not found
        return {
            'Datapoints': [
                {'Average': random.uniform(2.0, 85.0)}
            ]
        }


class MockPrometheusConnect:
    """Mock Prometheus client"""
    
    def __init__(self, url, disable_ssl=True):
        self.url = url
        self.loader = MockDataLoader()
    
    def custom_query(self, query):
        """Mock Prometheus query response"""
        
        # Extract pod name from query (simplified parsing)
        if 'pod="' in query:
            start = query.index('pod="') + 5
            end = query.index('"', start)
            pod_name = query[start:end]
            
            historical_data = self.loader.get('prometheus_metrics', 'pod_historical_usage')
            
            if pod_name in historical_data:
                pod_data = historical_data[pod_name]
                
                if 'cpu' in query:
                    return [{'value': [None, pod_data['peak_cpu_cores']]}]
                elif 'memory' in query:
                    # Convert MiB to bytes
                    bytes_value = pod_data['peak_memory_mib'] * 1024 * 1024
                    return [{'value': [None, bytes_value]}]
        
        # Fallback
        return [{'value': [None, 0.5]}]
