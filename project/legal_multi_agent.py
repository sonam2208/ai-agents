import os
from dotenv import load_dotenv

# Azure Agent SDK imports
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import (
    ConnectedAgentTool,
    MessageRole,
    ListSortOrder
)
from azure.identity import DefaultAzureCredential

# Clear console
os.system('cls' if os.name == 'nt' else 'clear')

# Load environment variables
load_dotenv()
project_endpoint = os.getenv("PROJECT_ENDPOINT")
model_deployment = os.getenv("MODEL_DEPLOYMENT_NAME")

# -----------------------------
# File-based RAG helper
# -----------------------------
def load_legal_references():
    with open("legal_refs.txt", "r", encoding="utf-8") as f:
        return f.read()

# Connect to Azure AI Foundry Agents Client
agents_client = AgentsClient(
    endpoint=project_endpoint,
    credential=DefaultAzureCredential(
        exclude_environment_credential=True,
        exclude_managed_identity_credential=True
    ),
)

with agents_client:

    # -----------------------------
    # Agent 1: Clause Classification
    # -----------------------------
    clause_agent = agents_client.create_agent(
        model=model_deployment,
        name="clause_classification_agent",
        instructions="""
Identify and classify clauses in the legal document.
Examples: Termination, Liability, Data Privacy, Confidentiality.
Return concise bullet points.
"""
    )

    # -----------------------------
    # Agent 2: Compliance Risk
    # -----------------------------
    compliance_agent = agents_client.create_agent(
        model=model_deployment,
        name="compliance_risk_agent",
        instructions="""
Assess legal compliance risk in the document.
Return:
- Risk Level (High / Medium / Low)
- Short explanation
"""
    )

    # -----------------------------
    # Agent 3: Legal Complexity
    # -----------------------------
    complexity_agent = agents_client.create_agent(
        model=model_deployment,
        name="legal_complexity_agent",
        instructions="""
Estimate legal review complexity.
Use:
Low / Medium / High
Provide brief justification.
"""
    )

    # -----------------------------
    # Agent 4: File-Based RAG Agent
    # -----------------------------
    rag_agent = agents_client.create_agent(
        model=model_deployment,
        name="legal_rag_agent",
        instructions="""
You are a retrieval agent.
Use the provided legal references to support analysis.
Return only relevant reference text.
Do not interpret or summarize.
"""
    )

    # -----------------------------
    # Connected Agent Tools
    # -----------------------------
    clause_tool = ConnectedAgentTool(
        id=clause_agent.id,
        name="clause_classification_agent",
        description="Identifies legal clauses"
    )

    compliance_tool = ConnectedAgentTool(
        id=compliance_agent.id,
        name="compliance_risk_agent",
        description="Evaluates compliance risk"
    )

    complexity_tool = ConnectedAgentTool(
        id=complexity_agent.id,
        name="legal_complexity_agent",
        description="Estimates legal complexity"
    )

    rag_tool = ConnectedAgentTool(
        id=rag_agent.id,
        name="legal_rag_agent",
        description="Retrieves legal references from file"
    )

    # -----------------------------
    # Orchestrator Agent
    # -----------------------------
    legal_orchestrator = agents_client.create_agent(
        model=model_deployment,
        name="legal_orchestrator_agent",
        instructions="""
Review the legal document.

Steps:
1. Identify clause types
2. Assess compliance risk
3. Estimate legal complexity
4. Use retrieved legal references
5. Produce a grounded legal summary

Rules:
- Base conclusions strictly on document + references
- Do not invent laws or clauses
""",
        tools=[
            clause_tool.definitions[0],
            compliance_tool.definitions[0],
            complexity_tool.definitions[0],
            rag_tool.definitions[0]
        ]
    )

    # -----------------------------
    # Run the Review
    # -----------------------------
    print("Creating agent thread...")
    thread = agents_client.threads.create()

    with open("legal_document.txt", "r", encoding="utf-8") as f:
        legal_doc = f.read()

    legal_refs = load_legal_references()

    agents_client.messages.create(
        thread_id=thread.id,
        role=MessageRole.USER,
        content=f"""
Legal Document:
{legal_doc}

Retrieved Legal References:
{legal_refs}
"""
    )

    print("\nProcessing legal document with file-based RAG...\n")

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
            print("LEGAL COMPLIANCE OUTPUT:\n")
            print(message.text_messages[-1].text.value)

    # -----------------------------
    # Clean up
    # -----------------------------
    agents_client.delete_agent(legal_orchestrator.id)
    agents_client.delete_agent(clause_agent.id)
    agents_client.delete_agent(compliance_agent.id)
    agents_client.delete_agent(complexity_agent.id)
    agents_client.delete_agent(rag_agent.id)

    print("\nAll agents cleaned up.")
