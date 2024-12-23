from services.agent_instruction import AgentInstruction
from services.agent_registry import AgentRegistry
from services.tool_selection import ToolRegistry
from services.workflow_engine import WorkFlowEngine
from services.result_aggregator import ResultAggregationHTML

from models.agent_instruction import *
from models.agent_registry import *
from models.tool_registry import *
from models.workflow_service import *
from models.result_aggregation import *

from models import *


from util.utils import *
from fastapi import FastAPI


# Replace with LiteLLM
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")

app = FastAPI()


@app.post("/toxicity-detection-manual", response_model=Result)
def toxicity_detection_manual(args: UserPrompt):
    print("toxicity_detection_manual not implemented")


@app.post("/toxicity-detection-automated", response_model=Result)
def toxicity_detection_automated(args: PromptLibrary):
    print("toxicity_detection_automated not implemented")


@app.post("/bias-detection-manual", response_model=Result)
def bias_detection_manual(args: UserPrompt):
    print("bias_detection_manual not implemented")


@app.post("/hallucination-detection-workflow", response_model=Result)
def hallucianation_detection_workflow(args: UserPrompt):
    print("hallucianation_detection_workflow not implemented")


@app.get("/")
def read_root():
    return {"service": "online"}
