import os
import asyncio
import uuid
from dotenv import load_dotenv
from abc import ABC, abstractmethod
from typing import List, Optional, Callable, Any, Dict
from datetime import datetime, timedelta

import boto3
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search
from google.genai import types
from prometheus_api_client import PrometheusConnect
from prometheus_api_client.utils import parse_datetime

load_dotenv()

# Load Google API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504]
)

class BaseAgent(ABC):

    def __init__(
        self,
        name: str,
        model: str = "gemini-2.5-flash-lite",
        instruction: Optional[str] = None,
        tools: Optional[List] = None
    ):
        self.name = name
        self.model_name = model
        self.tools = tools or []
        self.instruction = instruction
        self.agent = None
        self.session_id = str(uuid.uuid4())  # Generate unique session ID
        self.user_id = "default_user"
        self._initialize_agent()

        self.session_service = InMemorySessionService()
        # Don't create session here - do it in chat() method
        
        self.runner = Runner(
            agent=self.agent,
            app_name="agents",
            session_service=self.session_service
        )
        self.session_initialized = False  # Track if session is created

    def _initialize_agent(self):
        """Initialize the ADK agent properly."""

        self.agent = LlmAgent(
            name=self.name,
            model=Gemini(
                model=self.model_name,
                api_key=GOOGLE_API_KEY,
                retry_options=retry_config
            ),
            description=self._get_description(),
            instruction=self.instruction or self._get_default_instruction(),
            tools=self.tools,
        )

    @abstractmethod
    def _get_default_instruction(self) -> str:
        """Override to provide default instructions"""
        pass

    @abstractmethod
    def _get_description(self) -> str:
        """Override to provide agent description"""
        pass

    def get_agent(self) -> LlmAgent:
        """Return the ADK Agent instance."""
        return self.agent

    def add_tool(self, tool: Callable):
        """Add a tool and refresh the agent config."""
        self.tools.append(tool)
        self._initialize_agent()

    async def chat(self, user_query: str):
        """Handle chat and return complete response"""
        
        # Create session on first use
        if not self.session_initialized:
            await self.session_service.create_session(
                app_name="agents",
                user_id=self.user_id,
                session_id=self.session_id
            )
            self.session_initialized = True
        
        user_message = types.Content(role='user', parts=[types.Part(text=user_query)])
        
        async for event in self.runner.run_async(
            user_id=self.user_id, 
            session_id=self.session_id,
            new_message=user_message 
        ):
            if event.is_final_response():
                return event.content.parts[0].text
    
        return "No response received."


class CostAnalyzerAgent(BaseAgent):

    def __init__(self):
        self.ce_client = boto3.client("ce", region_name="us-east-1")
        self.cw_client = boto3.client("cloudwatch", region_name="us-east-1")

        tools = [
            self.get_daily_spend_and_trend,
            self.check_storage_anomaly,
            self.check_resource_utilization
        ]

        super().__init__(
            name="CostAnalyzer",
            tools=tools
        )

    def _get_default_instruction(self):
        return ( 
                "You are a real-time Cloud Cost Analyzer Agent.\n"
                "Your responsibilities:\n"
                "- Monitor cloud costs (hourly/daily)\n"
                "- Detect anomalies in spend trends\n"
                "- Track CPU/Memory/Storage utilization\n"
                "- Predict spikes and overshoot of budget\n"
                "- Trigger alerts when abnormal patterns appear\n"
                "Use available tools to fetch metrics and billing data."
        )
        
    def _get_description(self):
        return "Real-time cost monitoring and anomaly detection agent"

    def get_daily_spend_and_trend(self):

        now = datetime.utcnow()
        start_date = (now - timedelta(days=7)).strftime('%Y-%m-%d')
        end_date = now.strftime('%Y-%m-%d')

        try:
            response = self.ce_client.get_cost_and_usage(
                TimePeriod={'Start': start_date, 'End': end_date},
                Granularity='DAILY',
                Metrics=['UnblendedCost']
            )

            results = response['ResultsByTime']
            if len(results) < 2:
                return "Insufficient data for trend analysis."

            yesterday_data = results[-2]
            today_data = results[-1]

            cost_yesterday = float(yesterday_data['Total']['UnblendedCost']['Amount'])
            cost_today = float(today_data['Total']['UnblendedCost']['Amount'])
            
            if cost_yesterday > 0:
                trend_pct = ((cost_today - cost_yesterday) / cost_yesterday) * 100
            else:
                trend_pct = 100.0

            alert_emoji = "⚠️" if trend_pct > 20 else "💸"

            return {
                "status": f"{alert_emoji} Daily spend: ${cost_today:.2f} ({trend_pct:+.1f}% from yersterday)",
                "raw_data": {"today": cost_today, "yesterday": cost_yesterday}
            }

        except Exception as e:
            return f"Error fetching cost: {str(e)}"

        
    def check_storage_anomaly(self):

        now = datetime.utcnow()
        start_date = (now - timedelta(days=7)).strftime('%Y-%m-%d')
        end_date = now.strftime('%Y-%m-%d')

        response = self.ce_client.get_cost_and_usage(
            TimePeriod={'Start': start_date, 'End': end_date},
            Granularity='DAILY',
            Metrics=['UnblendedCost'],
            Filter={
                'Dimensions': {
                    'Key': 'SERVICE',
                    'Values': ['Amazon Simple Storage Service', 'EC2 - Other']
                }
            }
        )  
        
        # Returning a mock alert
        return "📊 Storage costs up 200% this week"  
    
    def check_resource_utilization(self, namespace, pod_name):

        try:
            response = self.cw_client.get_metric_statistics(
                Namespace='ContainerInsights', 
                MetricName='pod_cpu_utilization',
                Dimensions=[
                    {'Name': 'PodName', 'Value': pod_name},
                    {'Name': 'Namespace', 'Value': namespace},
                    {'Name': 'ClusterName', 'Value': 'my-cluster-name'}
                ],
                StartTime=datetime.utcnow() - timedelta(hours=1),
                EndTime=datetime.utcnow(),
                Period=3600,
                Statistics=['Average'] 
            )

            if response['Datapoints']:
                avg_cpu = response['Datapoints'][0]['Average']
                if avg_cpu < 5.0:
                    return f"⚠️ Pod '{pod_name}' is 95% idle"
                
            return "✅ Utilization normal"

        except Exception as e:
            return f"Error checking metrics: {str(e)}"

        
class RightsizingAgent(BaseAgent):

    def __init__(self, prometheus_url: str = "http://prometheus-server.monitoring.svc.cluster.local"):
        self.prom = PrometheusConnect(url=prometheus_url, disable_ssl=True)

        tools = [
            self.get_pod_historical_usage,
            self.calculate_rightsizing
        ]

        super().__init__(
            name="RightsizingOptimizer",
            tools=tools
        )

    def _get_default_instruction(self):
        return (
            "You are a Kubernetes Rightsizing Expert.\n"
            "Your Goal: Reduce waste without causing OOM (Out of Memory) errors.\n"
            "1. Fetch historical usage for a specific pod (look back 7 days).\n"
            "2. Compare 'Requested' resources vs 'Actual' usage (Max & Avg).\n"
            "3. Apply a safety buffer (usually +20% over Peak usage).\n"
            "4. Calculate potential cost savings.\n"
            "5. Output a recommendation with a confidence score."
        )

    def _get_description(self):
        return "Analyzes container usage vs requests to recommend optimized resource limits."

    def get_pod_historical_usage(self, pod_name: str, namespace: str, days: int = 7) -> Dict[str, float]:

        start_time = parse_datetime(f"{days}d")
        end_time = parse_datetime("now")
        step = "1h" # Granularity

        try:
            cpu_query = f'max_over_time(rate(container_cpu_usage_seconds_total{{pod="{pod_name}", namespace="{namespace}"}}[5m])[{days}d:1h])'  
        
            mem_query = f'max_over_time(container_memory_working_set_bytes{{pod="{pod_name}", namespace="{namespace}"}}[{days}d:1h])'

            cpu_data = self.prom.custom_query(query=cpu_query)
            mem_data = self.prom.custom_query(query=mem_query)

            if not cpu_data or not mem_data:
              return {"error": "No metrics found. Check pod name or Prometheus retention."}

            max_cpu_usage = float(cpu_data[0]['value'][1])
            max_mem_bytes = float(mem_data[0]['value'][1])

            max_mem_mib = max_mem_bytes / (1024 * 1024)

            return {
                "pod": pod_name,
                "period": f"{days} days",
                "peak_cpu_cores": round(max_cpu_usage, 4),
                "peak_memory_mib": round(max_mem_mib, 2)
            }

        except Exception as e:
            return {"error": f"Prometheus query failed: {str(e)}"}

    def calculate_rightsizing(
        self, current_cpu_req: float, current_mem_req_mib: float,
        peak_cpu_usage: float, peak_mem_usage: float
        ) -> Dict[str, Any]:

        """
        Mathematically calculates the new recommended limits with a safety buffer.
        """
        # Configuration
        SAFETY_BUFFER = 0.20  # 20% headroom
        MIN_CPU = 0.1         # Minimum 100m CPU
        MIN_MEM = 128         # Minimum 128Mi Memory
        
        # Cost constants (Approximation: $30/vCPU/mo, $4/GB/mo)
        COST_PER_CPU_CORE = 30.0
        COST_PER_GB_MEM = 4.0

        # 1. Calculate Recommended Request (Peak * 1.2)
        rec_cpu = max(MIN_CPU, peak_cpu_usage * (1 + SAFETY_BUFFER))
        rec_mem = max(MIN_MEM, peak_mem_usage * (1 + SAFETY_BUFFER))

        # Rounding for cleanliness (CPU to 1 decimal, Mem to nearest 10Mi)
        rec_cpu = round(rec_cpu, 1)
        rec_mem = round(rec_mem, -1) # Rounds to nearest 10

        # 2. Calculate Savings
        cpu_saved = max(0, current_cpu_req - rec_cpu)
        mem_saved_gb = max(0, (current_mem_req_mib - rec_mem) / 1024)
        
        est_savings = (cpu_saved * COST_PER_CPU_CORE) + (mem_saved_gb * COST_PER_GB_MEM)

        # 3. Determine Confidence Score
        # High confidence if usage is consistent (metric variance is low - simplified here)
        # Lower confidence if peak usage is extremely close to current limit (risk of throttling)
        confidence = "High"
        if peak_cpu_usage > (current_cpu_req * 0.9):
            confidence = "Low (Risk of Throttling)"
        
        return {
            "recommendation": {
                "cpu_request": f"{rec_cpu} cores",
                "memory_request": f"{rec_mem} Mi"
            },
            "metrics": {
                "utilization_cpu_pct": round((peak_cpu_usage / current_cpu_req) * 100, 1),
                "utilization_mem_pct": round((peak_mem_usage / current_mem_req_mib) * 100, 1)
            },
            "financial_impact": {
                "monthly_savings_est": f"${round(est_savings, 2)}",
                "description": "Based on avg AWS pricing"
            },
            "confidence_score": confidence
        }

class AgentOrchestrator:
    def __init__(self):
        print("🚀 Booting Multi-Agent System...")
        self.cost_agent = CostAnalyzerAgent()
        self.sizing_agent = RightsizingAgent()
        print("✅ Agents Online: CostAnalyzer, RightsizingOptimizer")

    async def dispatch(self, user_query: str):
        """
        Simple intent detection to route the query to the right specialist.
        """
        query_lower = user_query.lower()
        
        # KEYWORD ROUTING LOGIC
        cost_keywords = ['cost', 'spend', 'bill', 'price', 'budget', 'money']
        tech_keywords = ['pod', 'cpu', 'memory', 'ram', 'rightsiz', 'limit', 'request']

        if any(k in query_lower for k in cost_keywords):
            print(f"\n[🔄 Router] Handoff to: {self.cost_agent.name}")
            response = await self.cost_agent.chat(user_query)
            return f"💰 {self.cost_agent.name}: {response}"

        elif any(k in query_lower for k in tech_keywords):
            print(f"\n[🔄 Router] Handoff to: {self.sizing_agent.name}")
            response = await self.sizing_agent.chat(user_query)
            return f"🛠️ {self.sizing_agent.name}: {response}"

        else:
            # Fallback: Send to both or ask for clarification? 
            # Let's default to Cost for general inquiries
            print(f"\n[🔄 Router] Ambiguous intent. Defaulting to CostAgent.")
            response = await self.cost_agent.chat(user_query)
            return response

async def main():
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ Error: GOOGLE_API_KEY is missing.")
        sys.exit(1)

    system = AgentOrchestrator()

    print("\n💬 Multi-Agent CLI (Type 'quit' to exit)")
    print("---------------------------------------------")
    
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ['quit', 'exit']:
                break
            
            response = await system.dispatch(user_input)
            print(response)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__": 
    asyncio.run(main())

    