"""Test the improved CostAnalyzer agent"""
import asyncio
from main import AgentOrchestrator

async def test():
    system = AgentOrchestrator()
    
    print("\n" + "="*60)
    print("Test 1: Daily spend trend")
    print("="*60)
    response = await system.dispatch("What's my daily spend?")
    print(response)
    
    print("\n" + "="*60)
    print("Test 2: Storage anomaly detection")
    print("="*60)
    response = await system.dispatch("Check for storage cost anomalies")
    print(response)
    
    print("\n" + "="*60)
    print("Test 3: Pod resource utilization")
    print("="*60)
    response = await system.dispatch("Check utilization for api-server pod in production")
    print(response)
    
    print("\n" + "="*60)
    print("Test 4: Cost breakdown by service")
    print("="*60)
    response = await system.dispatch("Show me cost breakdown by service")
    print(response)

if __name__ == "__main__":
    asyncio.run(test())
