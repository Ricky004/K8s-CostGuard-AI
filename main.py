import os
import asyncio
import uuid
from dotenv import load_dotenv
from abc import ABC, abstractmethod
from typing import List, Optional, Callable, Any, Dict
from datetime import datetime, timedelta, timezone

import boto3
from google.adk.agents import LlmAgent, LoopAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search
from google.genai import types
from prometheus_api_client import PrometheusConnect
from prometheus_api_client.utils import parse_datetime

# Import mock clients for dev mode
from mock_aws_client import MockCostExplorerClient, MockCloudWatchClient, MockPrometheusConnect

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
        tools: Optional[List] = None,
        agent_type: str = "llm"
    ):
        self.name = name
        self.model_name = model
        self.tools = tools or []
        self.instruction = instruction
        self.agent = None
        self.agent_type = agent_type
        self.session_id = str(uuid.uuid4()) 
        self.user_id = "default_user"
        self._initialize_agent()

        self.session_service = InMemorySessionService()
        
        self.runner = Runner(
            agent=self.agent,
            app_name="agents",
            session_service=self.session_service
        )
        self.session_initialized = False  

    def _initialize_agent(self):
        """Initialize the ADK agent properly."""

        model = Gemini(
            model=self.model_name,
            api_key=GOOGLE_API_KEY,
            retry_options=retry_config
        )
        
        # Choose agent type based on parameter
        if self.agent_type == "loop":
            cost_analyzer_agent = LlmAgent(
                name=self.name,
                model=model,
                description=self._get_description(),
                instruction=self.instruction or self._get_default_instruction(),
                tools=self.tools,
            )

            self.agent = LoopAgent(
                name="cost_analyze_loop",
                sub_agents=[cost_analyzer_agent],
                max_iterations=5,
            )
        else:  # default to LlmAgent
            self.agent = LlmAgent(
                name=self.name,
                model=model,
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
                if event.content and event.content.parts and len(event.content.parts) > 0:
                    return event.content.parts[0].text
                return "Received empty response from agent."
    
        return "No response received."


class CostAnalyzerAgent(BaseAgent):

    def __init__(self):
        # Check if AWS credentials are available
        has_aws_creds = os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY")
        
        if has_aws_creds:
            self.ce_client = boto3.client("ce", region_name="us-east-1")
            self.cw_client = boto3.client("cloudwatch", region_name="us-east-1")
        else:
            print("⚠️  Running in DEV MODE (no AWS credentials) - using mock data")
            self.ce_client = MockCostExplorerClient()
            self.cw_client = MockCloudWatchClient()

        tools = [
            self.get_daily_spend_and_trend,
            self.check_storage_anomaly,
            self.check_resource_utilization,
            self.get_cost_by_service
        ]

        super().__init__(
            name="CostAnalyzer",
            tools=tools,
            agent_type="loop"
        )

    def _get_default_instruction(self):
        return ( 
                "You are a Cloud Cost Analyzer Agent specialized in AWS cost optimization.\n\n"
                "Your responsibilities:\n"
                "1. Monitor daily AWS spending trends and identify cost increases >20%\n"
                "2. Detect storage cost anomalies (S3, EBS) by comparing recent vs baseline costs\n"
                "3. Analyze pod/container resource utilization (CPU, Memory) to identify:\n"
                "   - Underutilized resources (<5% CPU) that can be downsized\n"
                "   - Overutilized resources (>90% CPU/Memory) that need scaling\n"
                "4. Provide actionable recommendations with cost impact estimates\n\n"
                "When analyzing costs:\n"
                "- Always call the appropriate tool to get real data\n"
                "- Compare current metrics against baselines/thresholds\n"
                "- Explain the business impact (e.g., 'This 30% increase costs $X extra per month')\n"
                "- Suggest specific actions (e.g., 'Reduce pod CPU request from 2 cores to 0.5 cores')\n\n"
                "Be concise but informative. Use emojis for visual clarity."
        )
        
    def _get_description(self):
        return "Real-time cost monitoring and anomaly detection agent"

    def get_daily_spend_and_trend(self):
        """
        Fetches the last 7 days of AWS costs and calculates the day-over-day trend.
        Returns daily spend amount and percentage change from yesterday.
        Alerts if increase is >20%.
        """
        now = datetime.now(timezone.utc)
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
                "status": f"{alert_emoji} Daily spend: ${cost_today:.2f} ({trend_pct:+.1f}% from yesterday)",
                "raw_data": {"today": cost_today, "yesterday": cost_yesterday}
            }

        except Exception as e:
            return f"Error fetching cost: {str(e)}"

        
    def check_storage_anomaly(self):
        """Detect storage cost anomalies by comparing recent vs historical costs"""
        now = datetime.now(timezone.utc)
        start_date = (now - timedelta(days=14)).strftime('%Y-%m-%d')  # 2 weeks for comparison
        end_date = now.strftime('%Y-%m-%d')

        try:
            response = self.ce_client.get_cost_and_usage(
                TimePeriod={'Start': start_date, 'End': end_date},
                Granularity='DAILY',
                Metrics=['UnblendedCost'],
                Filter={
                    'Dimensions': {
                        'Key': 'SERVICE',
                        'Values': ['Amazon Simple Storage Service', 'Amazon Elastic Compute Cloud - Compute']
                    }
                }
            )  
            
            results = response['ResultsByTime']
            if len(results) < 7:
                return "Insufficient data for storage anomaly detection"
            
            # Compare last 3 days avg vs previous week avg
            recent_costs = [float(r['Total']['UnblendedCost']['Amount']) for r in results[-3:]]
            previous_costs = [float(r['Total']['UnblendedCost']['Amount']) for r in results[-10:-3]]
            
            recent_avg = sum(recent_costs) / len(recent_costs)
            previous_avg = sum(previous_costs) / len(previous_costs)
            
            if previous_avg > 0:
                change_pct = ((recent_avg - previous_avg) / previous_avg) * 100
            else:
                change_pct = 100.0
            
            if change_pct > 50:
                return f"🚨 Storage anomaly detected! Costs up {change_pct:.1f}% (${recent_avg:.2f}/day vs ${previous_avg:.2f}/day baseline)"
            elif change_pct > 20:
                return f"⚠️ Storage costs increasing: +{change_pct:.1f}% (${recent_avg:.2f}/day vs ${previous_avg:.2f}/day)"
            else:
                return f"✅ Storage costs normal: ${recent_avg:.2f}/day (baseline: ${previous_avg:.2f}/day)"
                
        except Exception as e:
            return f"Error checking storage: {str(e)}"  
    
    def check_resource_utilization(self, namespace: str, pod_name: str, cluster_name: str = None):
        """Check CPU and Memory utilization for a specific pod in EKS/ECS"""
        if not cluster_name:
            cluster_name = os.getenv('EKS_CLUSTER_NAME', 'my-cluster-name')
            
        try:
            # Get CPU metrics
            cpu_response = self.cw_client.get_metric_statistics(
                Namespace='ContainerInsights', 
                MetricName='pod_cpu_utilization',
                Dimensions=[
                    {'Name': 'PodName', 'Value': pod_name},
                    {'Name': 'Namespace', 'Value': namespace},
                    {'Name': 'ClusterName', 'Value': cluster_name}
                ],
                StartTime=datetime.now(timezone.utc) - timedelta(hours=1),
                EndTime=datetime.now(timezone.utc),
                Period=3600,
                Statistics=['Average', 'Maximum'] 
            )
            
            # Get Memory metrics
            mem_response = self.cw_client.get_metric_statistics(
                Namespace='ContainerInsights',
                MetricName='pod_memory_utilization',
                Dimensions=[
                    {'Name': 'PodName', 'Value': pod_name},
                    {'Name': 'Namespace', 'Value': namespace},
                    {'Name': 'ClusterName', 'Value': cluster_name}
                ],
                StartTime=datetime.now(timezone.utc) - timedelta(hours=1),
                EndTime=datetime.now(timezone.utc),
                Period=3600,
                Statistics=['Average', 'Maximum']
            )

            if cpu_response['Datapoints'] and mem_response['Datapoints']:
                avg_cpu = cpu_response['Datapoints'][0]['Average']
                max_cpu = cpu_response['Datapoints'][0]['Maximum']
                avg_mem = mem_response['Datapoints'][0]['Average']
                max_mem = mem_response['Datapoints'][0]['Maximum']
                
                # Determine status
                if avg_cpu < 5.0 and avg_mem < 10.0:
                    return f"⚠️ Pod '{pod_name}' is severely underutilized - CPU: {avg_cpu:.1f}%, Memory: {avg_mem:.1f}% (Consider downsizing)"
                elif max_cpu > 90.0 or max_mem > 90.0:
                    return f"🚨 Pod '{pod_name}' is resource-constrained - CPU: {avg_cpu:.1f}% (max: {max_cpu:.1f}%), Memory: {avg_mem:.1f}% (max: {max_mem:.1f}%) (Consider upsizing)"
                elif avg_cpu > 70.0 or avg_mem > 70.0:
                    return f"⚠️ Pod '{pod_name}' running hot - CPU: {avg_cpu:.1f}%, Memory: {avg_mem:.1f}%"
                else:
                    return f"✅ Pod '{pod_name}' utilization healthy - CPU: {avg_cpu:.1f}%, Memory: {avg_mem:.1f}%"
            
            return f"⚠️ No metrics found for pod '{pod_name}' in namespace '{namespace}'. Check if Container Insights is enabled."

        except Exception as e:
            return f"Error checking metrics: {str(e)}"
    
    def get_cost_by_service(self, days: int = 7):
        """
        Get cost breakdown by AWS service for the last N days.
        Helps identify which services are driving costs.
        """
        now = datetime.now(timezone.utc)
        start_date = (now - timedelta(days=days)).strftime('%Y-%m-%d')
        end_date = now.strftime('%Y-%m-%d')
        
        try:
            response = self.ce_client.get_cost_and_usage(
                TimePeriod={'Start': start_date, 'End': end_date},
                Granularity='DAILY',
                Metrics=['UnblendedCost'],
                GroupBy=[{'Type': 'DIMENSION', 'Key': 'SERVICE'}]
            )
            
            # Aggregate costs by service
            service_costs = {}
            for result in response['ResultsByTime']:
                for group in result.get('Groups', []):
                    service = group['Keys'][0]
                    cost = float(group['Metrics']['UnblendedCost']['Amount'])
                    service_costs[service] = service_costs.get(service, 0) + cost
            
            # Sort by cost descending
            sorted_services = sorted(service_costs.items(), key=lambda x: x[1], reverse=True)
            
            # Format top 5 services
            total_cost = sum(service_costs.values())
            top_services = sorted_services[:5]
            
            result = f"💰 Total cost (last {days} days): ${total_cost:.2f}\n\nTop services:\n"
            for service, cost in top_services:
                pct = (cost / total_cost * 100) if total_cost > 0 else 0
                result += f"  • {service}: ${cost:.2f} ({pct:.1f}%)\n"
            
            return result
            
        except Exception as e:
            return f"Error fetching cost by service: {str(e)}"

        
class RightsizingAgent(BaseAgent):

    def __init__(self, prometheus_url: str = "http://prometheus-server.monitoring.svc.cluster.local"):
        # Use mock Prometheus if no real connection available
        has_aws_creds = os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY")
        
        if has_aws_creds:
            self.prom = PrometheusConnect(url=prometheus_url, disable_ssl=True)
        else:
            self.prom = MockPrometheusConnect(url=prometheus_url, disable_ssl=True)

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
            "You are a Kubernetes Rightsizing Expert specialized in optimizing pod resource requests.\n\n"
            "Your Goal: Reduce waste without causing OOM (Out of Memory) errors or CPU throttling.\n\n"
            "Workflow:\n"
            "1. Use get_pod_historical_usage() to fetch 7-day peak CPU and memory usage\n"
            "2. Ask the user for current resource requests if not provided\n"
            "3. Use calculate_rightsizing() to compute optimal requests with 20% safety buffer\n"
            "4. Present recommendations with:\n"
            "   - Current vs recommended resource requests\n"
            "   - Utilization percentages\n"
            "   - Monthly cost savings estimate\n"
            "   - Confidence score (Low if pod is near limits, High otherwise)\n\n"
            "Interpretation Guidelines:\n"
            "- Utilization <50%: Significantly overprovisioned, high savings potential\n"
            "- Utilization 50-80%: Well-sized, minor optimization possible\n"
            "- Utilization >90%: Risk of throttling, recommend upsizing instead\n\n"
            "Always explain the business impact and provide kubectl commands for implementation."
        )

    def _get_description(self):
        return "Analyzes container usage vs requests to recommend optimized resource limits."

    def get_pod_historical_usage(self, pod_name: str, namespace: str, days: int = 7) -> Dict[str, float]:
        """
        Fetches historical peak CPU and memory usage for a Kubernetes pod from Prometheus.
        Returns peak values over the specified time period (default 7 days).
        
        Args:
            pod_name: Name of the pod to analyze
            namespace: Kubernetes namespace where the pod runs
            days: Number of days to look back (default: 7)
        
        Returns:
            Dict with peak_cpu_cores and peak_memory_mib
        """
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
        Calculates optimal resource requests based on historical peak usage with 20% safety buffer.
        
        Args:
            current_cpu_req: Current CPU request in cores (e.g., 2.0 = 2 cores)
            current_mem_req_mib: Current memory request in MiB (e.g., 1024 = 1 GiB)
            peak_cpu_usage: Peak CPU usage observed in cores
            peak_mem_usage: Peak memory usage observed in MiB
        
        Returns:
            Dict with recommendation, utilization metrics, cost savings, and confidence score
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
        Intent detection to route queries to the appropriate specialist agent.
        Supports single-agent and multi-agent coordination for complex queries.
        """
        query_lower = user_query.lower()
        
        # KEYWORD ROUTING LOGIC
        cost_keywords = ['cost', 'spend', 'bill', 'price', 'budget', 'money', 'expensive', 'savings', 'save']
        tech_keywords = ['pod', 'cpu', 'memory', 'ram', 'rightsize', 'limit', 'request', 'container', 'resource']
        optimization_keywords = ['optimize', 'reduce', 'improve', 'efficiency']
        
        has_cost_intent = any(k in query_lower for k in cost_keywords)
        has_tech_intent = any(k in query_lower for k in tech_keywords)
        has_optimization_intent = any(k in query_lower for k in optimization_keywords)
        
        # Complex query: needs both agents (e.g., "save money by rightsizing")
        if (has_cost_intent and has_tech_intent) or (has_optimization_intent and (has_cost_intent or has_tech_intent)):
            print(f"\n[🔃 Reouter] Complex query detected - coordinating both agents")
            
            # First, get technical analysis
            print(f"  └─ Step 1: {self.sizing_agent.name} analyzing resources...")
            tech_response = await self.sizing_agent.chat(
                f"Analyze resource usage and provide rightsizing recommendations. Original query: {user_query}"
            )
            
            # Then, get cost analysis
            print(f"  └─ Step 2: {self.cost_agent.name} analyzing costs...")
            cost_response = await self.cost_agent.chat(
                f"Analyze current costs and spending trends. Original query: {user_query}"
            )
            
            # Combine insights
            return (
                f"🤝 Multi-Agent Analysis:\n\n"
                f"💰 Cost Perspective:\n{cost_response}\n\n"
                f"🛠️ Technical Perspective:\n{tech_response}\n\n"
                f"💡 Recommendation: Review the technical rightsizing suggestions above to achieve the cost savings identified."
            )
        
        # Single agent routing
        elif has_cost_intent:
            print(f"\n[🔄 Router] Handoff to: {self.cost_agent.name}")
            response = await self.cost_agent.chat(user_query)
            return f"💰 {self.cost_agent.name}: {response}"

        elif has_tech_intent:
            print(f"\n[🔄 Router] Handoff to: {self.sizing_agent.name}")
            response = await self.sizing_agent.chat(user_query)
            return f"🛠️ {self.sizing_agent.name}: {response}"

        else:
            # Fallback: Default to Cost for general inquiries
            print(f"\n[🔄 Router] Ambiguous intent. Defaulting to {self.cost_agent.name}.")
            response = await self.cost_agent.chat(user_query)
            return f"💰 {self.cost_agent.name}: {response}"

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

    