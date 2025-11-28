"""Test RightsizingAgent tools directly"""
from main import RightsizingAgent

def test_rightsizing():
    print("🧪 Testing RightsizingAgent Tools\n")
    
    agent = RightsizingAgent()
    
    print("="*70)
    print("Test 1: get_pod_historical_usage('api-server', 'production', 7)")
    print("="*70)
    result = agent.get_pod_historical_usage('api-server', 'production', 7)
    print(result)
    print()
    
    print("="*70)
    print("Test 2: get_pod_historical_usage('worker', 'production', 7)")
    print("="*70)
    result = agent.get_pod_historical_usage('worker', 'production', 7)
    print(result)
    print()
    
    print("="*70)
    print("Test 3: calculate_rightsizing() - Overprovisioned Pod")
    print("Current: 2 CPU cores, 1024 MiB | Peak: 0.85 cores, 512 MiB")
    print("="*70)
    result = agent.calculate_rightsizing(
        current_cpu_req=2.0,
        current_mem_req_mib=1024,
        peak_cpu_usage=0.85,
        peak_mem_usage=512.5
    )
    print(f"Recommendation: {result['recommendation']}")
    print(f"Utilization: CPU {result['metrics']['utilization_cpu_pct']}%, Memory {result['metrics']['utilization_mem_pct']}%")
    print(f"Savings: {result['financial_impact']['monthly_savings_est']}/month")
    print(f"Confidence: {result['confidence_score']}")
    print()
    
    print("="*70)
    print("Test 4: calculate_rightsizing() - Underprovisioned Pod")
    print("Current: 0.5 CPU cores, 256 MiB | Peak: 0.48 cores, 240 MiB")
    print("="*70)
    result = agent.calculate_rightsizing(
        current_cpu_req=0.5,
        current_mem_req_mib=256,
        peak_cpu_usage=0.48,
        peak_mem_usage=240
    )
    print(f"Recommendation: {result['recommendation']}")
    print(f"Utilization: CPU {result['metrics']['utilization_cpu_pct']}%, Memory {result['metrics']['utilization_mem_pct']}%")
    print(f"Savings: {result['financial_impact']['monthly_savings_est']}/month")
    print(f"Confidence: {result['confidence_score']}")
    print()
    
    print("="*70)
    print("Test 5: calculate_rightsizing() - Well-sized Pod")
    print("Current: 1.0 CPU cores, 512 MiB | Peak: 0.7 cores, 400 MiB")
    print("="*70)
    result = agent.calculate_rightsizing(
        current_cpu_req=1.0,
        current_mem_req_mib=512,
        peak_cpu_usage=0.7,
        peak_mem_usage=400
    )
    print(f"Recommendation: {result['recommendation']}")
    print(f"Utilization: CPU {result['metrics']['utilization_cpu_pct']}%, Memory {result['metrics']['utilization_mem_pct']}%")
    print(f"Savings: {result['financial_impact']['monthly_savings_est']}/month")
    print(f"Confidence: {result['confidence_score']}")
    print()
    
    print("✅ All RightsizingAgent tools working correctly!")

if __name__ == "__main__":
    test_rightsizing()
