"""Test coordination between CostAnalyzer and RightsizingAgent"""
import asyncio
from main import AgentOrchestrator

async def test_coordination():
    print("🧪 Testing Multi-Agent Coordination\n")
    
    system = AgentOrchestrator()
    
    # Test 1: Cost-related query
    print("="*70)
    print("Test 1: Cost-related query (should route to CostAnalyzer)")
    print("Query: 'What are my AWS costs this week?'")
    print("="*70)
    response = await system.dispatch("What are my AWS costs this week?")
    print(f"Response: {response[:200]}...")
    print()
    
    # Test 2: Technical/pod query
    print("="*70)
    print("Test 2: Technical query (should route to RightsizingOptimizer)")
    print("Query: 'Analyze pod resource usage'")
    print("="*70)
    response = await system.dispatch("Analyze pod resource usage")
    print(f"Response: {response[:200]}...")
    print()
    
    # Test 3: Ambiguous query
    print("="*70)
    print("Test 3: Ambiguous query (should default to CostAnalyzer)")
    print("Query: 'Help me optimize my infrastructure'")
    print("="*70)
    response = await system.dispatch("Help me optimize my infrastructure")
    print(f"Response: {response[:200]}...")
    print()
    
    # Test 4: Mixed query (cost + technical)
    print("="*70)
    print("Test 4: Mixed query - cost savings from rightsizing")
    print("Query: 'How much money can I save by rightsizing my pods?'")
    print("="*70)
    response = await system.dispatch("How much money can I save by rightsizing my pods?")
    print(f"Response: {response[:200]}...")
    print()
    
    print("✅ Coordination tests complete!")

if __name__ == "__main__":
    asyncio.run(test_coordination())
