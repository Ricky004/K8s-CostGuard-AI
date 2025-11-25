import os
from abc import ABC, abstractmethod
from typing import List, Optional, Callable
from datetime import datetime, timedelta

import boto3
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.tools import google_search
from google.genai import types

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
        self._initialize_agent()

    def _initialize_agent(self):
        """Initialize the ADK agent properly."""

        self.agent = Agent(
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

    def get_agent(self) -> Agent:
        """Return the ADK Agent instance."""
        return self.agent

    def add_tool(self, tool: Callable):
        """Add a tool and refresh the agent config."""
        self.tools.append(tool)
        self._initialize_agent()


class CostAnalyzerAgent(BaseAgent):

    def __init__(self):
        self.ce_client = boto3.client('ce')
        self.cw_client = boto3.client('cloudwatch')

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

        
    
