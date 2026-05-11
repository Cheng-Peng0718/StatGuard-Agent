from core.deliverables.contracts import TaskContract, normalize_task_contract
from core.deliverables.evidence import (
    extract_final_answer_content_from_state,
    get_deliverable_evidence,
    get_satisfied_criterion_names,
    get_satisfied_deliverable_names,
    criterion_satisfied_by_final_answer_text,
)
from core.deliverables.gate import (
    DeliverableGateResult,
    evaluate_deliverable_gate_state,
)