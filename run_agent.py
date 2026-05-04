import os
import sys
import warnings
import shutil
import uuid

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from core.schema import DatasetProfile, ColumnProfile
from core.graph import app
from core.context_builder import generate_profile
from core.config import ORIGINAL_DATA_PATH, WORKSPACE_DIR, WORKING_DATA_PATH

warnings.filterwarnings("ignore", message="Deserializing unregistered type")
load_dotenv()


def init_workspace():
    """Initialize sandbox: protect originals, create working copy."""
    if not os.path.exists(WORKSPACE_DIR):
        os.makedirs(WORKSPACE_DIR)

    if not os.path.exists(ORIGINAL_DATA_PATH):
        raise FileNotFoundError(f"Original data not found: {ORIGINAL_DATA_PATH}")

    shutil.copy(ORIGINAL_DATA_PATH, WORKING_DATA_PATH)
    print(f"[Sandbox ready]: isolated copy at {WORKING_DATA_PATH}\n")


def main():
    print("Starting data analysis agent...\n")

    session_id = f"task_{uuid.uuid4().hex[:8]}"
    user_workspace = os.path.join("workspaces", session_id)
    os.makedirs(user_workspace, exist_ok=True)

    for filename in os.listdir(user_workspace):
        file_path = os.path.join(user_workspace, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Failed to clear sandbox: {e}")

    original_data = "data/class_survey.xls"
    ext = os.path.splitext(original_data)[1]
    working_data = os.path.join(user_workspace, f"working_data{ext}")
    shutil.copy(original_data, working_data)
    print(f"[Session sandbox ready]: {working_data}\n")

    profile = generate_profile(working_data)
    user_request = (
        "I'd like to explore GPA distribution, factors that affect it, and build a regression model."
    )

    initial_state = {
        "user_request": user_request,
        "dataset_profile": profile,
        "workspace_dir": user_workspace,
        "observations": [],
        "current_step": 0,
        "max_steps": 5,
        "current_context_text": "",
        "current_action": None,
        "current_verification": None,
        "current_execution": None
    }

    config = {"configurable": {"thread_id": session_id}}

    print(f"[User Request]: {user_request}\n")
    print("-" * 40)

    state_input = initial_state

    while True:
        for event in app.stream(state_input, config):
            for node_name, state_updates in event.items():
                print(f"\n--- [node finished: {node_name}] ---")
                if node_name == "supervisor" and "current_action" in state_updates:
                    action = state_updates["current_action"]
                    print(f"[proposed action]: [{action.action_type}] Tool: {action.tool_name}")
                    print(f"[reasoning]: {action.reasoning_summary}")
                elif node_name == "verify" and "current_verification" in state_updates:
                    vr = state_updates["current_verification"]
                    print(f"[verify result]: {vr.status}")

        state = app.get_state(config)
        if not state.next:
            break

        if state.next[0] == "human_review":
            action = state.values["current_action"]
            print(f"\nWARNING [human review required]")
            print(f"Agent requests high-risk action: [{action.tool_name}]")
            print(f"Reasoning: {action.reasoning_summary}")

            user_choice = input("\nAuthorize execution? (y=yes / n=no): ")

            from core.schema import VerificationResult

            if user_choice.strip().lower() == 'y':
                print("Approved; continuing...")
                app.update_state(config, {
                    "current_verification": VerificationResult(action_id=action.action_id, status="allowed")})
            else:
                print("Rejected; sending back to supervisor...")
                app.update_state(config, {"current_verification": VerificationResult(action_id=action.action_id,
                                                                                     status="rejected_recoverable",
                                                                                     feedback="User expressly denied this operation.")})

            state_input = None


if __name__ == "__main__":
    main()