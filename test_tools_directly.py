"""Test the CostAnalyzer tools directly without the agent"""
from main import CostAnalyzerAgent

def test_tools():
    print("🧪 Testing CostAnalyzer Tools Directly\n")
    
    agent = CostAnalyzerAgent()
    
    print("="*60)
    print("1. get_daily_spend_and_trend()")
    print("="*60)
    result = agent.get_daily_spend_and_trend()
    print(result)
    print()
    
    print("="*60)
    print("2. check_storage_anomaly()")
    print("="*60)
    result = agent.check_storage_anomaly()
    print(result)
    print()
    
    print("="*60)
    print("3. check_resource_utilization('production', 'api-server')")
    print("="*60)
    result = agent.check_resource_utilization('production', 'api-server')
    print(result)
    print()
    
    print("="*60)
    print("4. check_resource_utilization('production', 'worker')")
    print("="*60)
    result = agent.check_resource_utilization('production', 'worker')
    print(result)
    print()
    
    print("="*60)
    print("5. get_cost_by_service(7)")
    print("="*60)
    result = agent.get_cost_by_service(7)
    print(result)
    print()
    
    print("✅ All tools executed successfully!")

if __name__ == "__main__":
    test_tools()
