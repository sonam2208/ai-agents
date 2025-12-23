from azure.ai.agents import AgentsClient
from azure.identity import DefaultAzureCredential
import os
from dotenv import load_dotenv

load_dotenv()

client = AgentsClient(
    endpoint=os.getenv("PROJECT_ENDPOINT"),
    credential=DefaultAzureCredential(
        exclude_environment_credential=True,
        exclude_managed_identity_credential=True
    ),
)

agent = client.create_agent(
    model=os.getenv("MODEL_DEPLOYMENT_NAME"),
    name="test-agent",
    instructions="Say hello"
)

print("Agent created:", agent.id)
client.delete_agent(agent.id)
