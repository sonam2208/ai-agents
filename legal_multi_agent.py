import os
from dotenv import load_dotenv

# Add references
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import (
    ConnectedAgentTool,
    MessageRole,
    ListSortOrder
)
from azure.identity import DefaultAzureCredential

# Clear the console
os.system('cls' if os.name == 'nt' else 'clear')

# Load environment variables
load_dotenv()
project_endpoint = os.getenv("PROJECT_ENDPOINT")
model_deployment = os.getenv("MODEL_DEPLOYMENT_NAME")

# Connect to the agents client
agents_client = AgentsClient(
    endpoint=project_endpoint,
    credential=DefaultAzureCredential(
        exclude_environment_credential=True,
        exclude_managed_identity_credential=True
    ),
)

with agents_client:

    # -----------------------------
    # Agent 1: Compliance Risk Agent
    # -----------------------------
    compliance_agent = agents_client.create_agent(
        model=model_deployment,
        name="compliance_risk_agent",
        instructions="""
Assess legal compliance risk in the given document.
Respond with:
- Risk Level: High / Medium / Low
- Brief explanation
Only assess risk, do not summarize.
"""
    )

    # ----------------------------------
    # Agent 2: Clause Classification Agent
    # ----------------------------------
    clause_agent = agents_client.create_agent(
        model=model_deployment,
        name="clause_classification_agent",
        instructions="""
Identify and classify clauses in the document.
Use categories such as:
- Termination
- Liability
- Data Privacy
- Confidentiality
- Payment
Return a short list with clause type and brief description.
"""
    )

    # -------------------------------
    # Agent 3: Legal Complexity Agent
    # -------------------------------
    complexity_agent = agents_client.create_agent(
        model=model_deployment,
        name="legal_complexity_agent",
        instructions="""
Estimate the legal review complexity of this document.
Use:
- Low: Standard contract
- Medium: Multiple obligations or jurisdictions
- High: Regulatory, cross-border, or ambiguous clauses
Provide brief justification.
"""
    )

    # ----------------------------------
    # Connected Agent Tools
    # ----------------------------------
    compliance_tool = ConnectedAgentTool(
        id=compliance_agent.id,
        name="compliance_risk_agent",
        description="Evaluates compliance risk"
    )

    clause_tool = ConnectedAgentTool(
        id=clause_agent.id,
        name="clause_classification_agent",
        description="Classifies legal clauses"
    )

    complexity_tool = ConnectedAgentTool(
        id=complexity_agent.id,
        name="legal_complexity_agent",
        description="Estimates legal complexity"
    )

    # ----------------------------------
    # Orchestrator Agent (Primary)
    # ----------------------------------
    legal_orchestrator = agents_client.create_agent(
        model=model_deployment,
        name="legal_orchestrator_agent",
        instructions="""
Review the provided legal document.

Use the connected tools to:
1. Identify clause types
2. Assess compliance risk
3. Estimate legal complexity

Then produce a grounded legal summary.
Do NOT invent facts or laws.
Base all findings strictly on the document content.
""",
        tools=[
            compliance_tool.definitions[0],
            clause_tool.definitions[0],
            complexity_tool.definitions[0]
        ]
    )

    # ----------------------------------
    # Run the Legal Review
    # ----------------------------------
    print("Creating agent thread...")
    thread = agents_client.threads.create()

    legal_text = input("\nPaste the legal document text:\n\n")

    agents_client.messages.create(
        thread_id=thread.id,
        role=MessageRole.USER,
        content=legal_text
    )

    print("\nProcessing legal document. Please wait...\n")

    run = agents_client.runs.create_and_process(
        thread_id=thread.id,
        agent_id=legal_orchestrator.id
    )

    if run.status == "failed":
        print(f"Run failed: {run.last_error}")

    messages = agents_client.messages.list(
        thread_id=thread.id,
        order=ListSortOrder.ASCENDING
    )

    for message in messages:
        if message.role == MessageRole.AGENT and message.text_messages:
            print("LEGAL REVIEW OUTPUT:\n")
            print(message.text_messages[-1].text.value)

    # ----------------------------------
    # Clean up
    # ----------------------------------
    agents_client.delete_agent(legal_orchestrator.id)
    agents_client.delete_agent(compliance_agent.id)
    agents_client.delete_agent(clause_agent.id)
    agents_client.delete_agent(complexity_agent.id)

    print("\nAgents cleaned up successfully.")
