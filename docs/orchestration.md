# การประสานงาน

`AgentTree.run(Task)` ประสานเฟสแบบซิงโครนัสที่มีอยู่ดังนี้:

1. คัดแยก Task และค้นหา Manager ที่ตรงกัน
2. แยกงานและมอบหมาย Specialist ที่อยู่ภายใต้ Manager
3. ให้ Specialist ทำงานผ่าน provider binding ภายนอก
4. ตรวจทานโดย Manager และแก้ไขงานของ Specialist ภายในขีดจำกัด
5. ตรวจทานขั้นสุดท้ายโดย Root และพิจารณาใหม่โดย Manager ภายในขีดจำกัด
6. ส่งคืน `FinalResult` และ `ExecutionTrace` แบบสะสม

engine ยังคงเป็นผู้ควบคุมการเลือก ลำดับการทำงาน การตัดสินใจตรวจทาน ขีดจำกัดการแก้ไข การรวมผล และเหตุการณ์แต่ละเฟส แบ็กเอนด์เรียกการดำเนินการเหล่านี้แทนการพัฒนาซ้ำ

## แบ็กเอนด์

`SequentialOrchestrationBackend` เริ่มต้นไม่ต้องใช้แพ็กเกจเสริม หากต้องการใช้อะแดปเตอร์แบบกราฟ:

```sh
python -m pip install "agenttree[langgraph]"
```

```python
from agenttree.orchestration.backends import LangGraphOrchestrationBackend

framework = AgentTree(
    # ใช้ Root และกลยุทธ์การตัดสินใจเดียวกับการกำหนดค่าแบบ sequential
    root_agent=root,
    triage=triage,
    decomposer=decomposer,
    manager_reviewer=manager_reviewer,
    final_reviewer=final_reviewer,
    orchestration_backend=LangGraphOrchestrationBackend(),
)
```

อะแดปเตอร์ LangGraph ใช้กราฟแบบซิงโครนัสในหน่วยความจำโดยไม่มี checkpointer และคง semantics สาธารณะของ `WorkflowState`, `FinalResult` และ trace ไว้ การ import AgentTree หรือรันแบ็กเอนด์ sequential ไม่จำเป็นต้องมี LangGraph

`AgentTreeConfig` ควบคุมเฉพาะพฤติกรรมการจับคู่และการแก้ไขในปัจจุบัน ได้แก่ `manager_match_all`, `specialist_match_all`, `max_manager_revisions` และ `max_final_revisions` โดยเป็น immutable และปฏิเสธค่าที่ไม่ถูกต้อง
# Root output and runtime result

The synchronous runtime has two paths. An optional `BaseRootPlanner` can return
`RootPlan(delegate=False, direct_output=...)` for a request the Root can answer
itself. This path skips triage, Managers, and Specialists. Without a planner,
the existing triage and delegation path remains the default.

For delegated work, the runtime runs triage, Manager delegation, Specialist
execution, Manager review and bounded revision, Root final review, then Root
final synthesis. `ProviderRootSynthesizer` calls any `BaseProvider` with the
original request and a compact context containing accepted Specialist outputs,
review decisions, revision counts, and material gaps. Its response is
`FinalResult.final_output`. The default `ExtractiveRootSynthesizer` joins
accepted outputs for deterministic offline compatibility; applications that
need a newly written answer should inject `ProviderRootSynthesizer`.

`FinalResult.content` remains the legacy structured Manager result. The
`orchestration` property aliases it for diagnostic callers. Display
`final_output` to users. `usage` contains reported provider token totals and
per-call entries; counts remain `None` when a provider does not report them.
`error` has a safe type and message on returned failures. The task ID matches
the trace ID, and `last_state` retains each completed phase result.

`max_manager_revisions` and `max_final_revisions` are independent bounds. A
review PASS over a failed Specialist call does not convert that call into a
successful subtask. Partial Manager work remains `FinalStatus.PARTIAL`;
`success=True` on that status means a usable Root answer was produced, not that
every delegated objective completed. A Root
synthesis failure returns a failed result while preserving reviewed work and
the trace. Completed runs end in `root.synthesis.completed`, then
`execution.completed`; a failed synthesis ends in `root.synthesis.failed`,
then `execution.failed`.

Manager and Specialist execution is sequential. A future Manager exchange
can be added at the Root coordination boundary with structured, policy checked
messages. Tool assignment is currently exposed for Specialists; Root and
Manager tool policy remains future work. Model adapters remain behind
`BaseProvider`.
