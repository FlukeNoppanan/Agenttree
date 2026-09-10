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
