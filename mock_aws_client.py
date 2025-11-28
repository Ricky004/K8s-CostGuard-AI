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
    
    def get_cost_and_usage(self, TimePeriod, Granularity, Metrics, Filter=None, GroupBy=None):
        """Mock get_cost_and_usage response"""
        
        # Cost by service breakdown
        if GroupBy and GroupBy[0].get('Key') == 'SERVICE':
            return {
                'ResultsByTime': [
                    {
                        'Groups': [
                            {'Keys': ['Amazon Elastic Compute Cloud - Compute'], 'Metrics': {'UnblendedCost': {'Amount': '125.50'}}},
                            {'Keys': ['Amazon Simple Storage Service'], 'Metrics': {'UnblendedCost': {'Amount': '45.30'}}},
                            {'Keys': ['Amazon Relational Database Service'], 'Metrics': {'UnblendedCost': {'Amount': '89.20'}}},
                            {'Keys': ['Amazon CloudWatch'], 'Metrics': {'UnblendedCost': {'Amount': '12.75'}}},
                            {'Keys': ['AWS Lambda'], 'Metrics': {'UnblendedCost': {'Amount': '8.40'}}},
                        ]
                    }
                ]
            }
        
        # Storage anomaly request (14 days of data)
        if Filter and 'SERVICE' in str(Filter):
            return {
                'ResultsByTime': [
                    {'Total': {'UnblendedCost': {'Amount': '8.50'}}},
                    {'Total': {'UnblendedCost': {'Amount': '9.20'}}},
                    {'Total': {'UnblendedCost': {'Amount': '8.80'}}},
                    {'Total': {'UnblendedCost': {'Amount': '9.50'}}},
                    {'Total': {'UnblendedCost': {'Amount': '8.90'}}},
                    {'Total': {'UnblendedCost': {'Amount': '9.10'}}},
                    {'Total': {'UnblendedCost': {'Amount': '9.30'}}},
                    {'Total': {'UnblendedCost': {'Amount': '15.20'}}},
                    {'Total': {'UnblendedCost': {'Amount': '18.50'}}},
                    {'Total': {'UnblendedCost': {'Amount': '19.80'}}},
                    {'Total': {'UnblendedCost': {'Amount': '21.30'}}},
                    {'Total': {'UnblendedCost': {'Amount': '22.10'}}},
                    {'Total': {'UnblendedCost': {'Amount': '23.50'}}},
                    {'Total': {'UnblendedCost': {'Amount': '24.20'}}}
                ]
            }
        
        # Daily spend request (7 days)
        daily_data = self.loader.get('cost_and_usage', 'daily_spend')
        
        return {
            'ResultsByTime': [
                {'Total': {'UnblendedCost': {'Amount': '42.10'}}},
                {'Total': {'UnblendedCost': {'Amount': '43.50'}}},
                {'Total': {'UnblendedCost': {'Amount': '44.20'}}},
                {'Total': {'UnblendedCost': {'Amount': '45.80'}}},
                {'Total': {'UnblendedCost': {'Amount': '46.50'}}},
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
                cpu_val = pod['cpu_percent']
                # Return both Average and Maximum if requested
                result = {'Datapoints': [{}]}
                if 'Average' in Statistics:
                    result['Datapoints'][0]['Average'] = cpu_val
                if 'Maximum' in Statistics:
                    result['Datapoints'][0]['Maximum'] = min(cpu_val + 15, 100)  # Max is slightly higher
                
                # Add memory data if it's a memory metric
                if 'memory' in MetricName.lower():
                    mem_val = cpu_val * 0.8  # Mock: memory usually lower than CPU
                    result['Datapoints'][0]['Average'] = mem_val
                    if 'Maximum' in Statistics:
                        result['Datapoints'][0]['Maximum'] = min(mem_val + 10, 100)
                
                return result
        
        # Random fallback if not found
        cpu_val = random.uniform(2.0, 85.0)
        result = {'Datapoints': [{}]}
        if 'Average' in Statistics:
            result['Datapoints'][0]['Average'] = cpu_val
        if 'Maximum' in Statistics:
            result['Datapoints'][0]['Maximum'] = min(cpu_val + 15, 100)
        return result


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
