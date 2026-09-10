# การเริ่มต้นใช้งาน

AgentTree ต้องใช้ Python 3.10 ขึ้นไป ติดตั้งแกนหลักที่ไม่มี dependency สำหรับการพัฒนาในเครื่องด้วยคำสั่ง:

```sh
python -m venv .venv
python -m pip install -e ".[dev]"
```

## กำหนดค่า SDK

แอปพลิเคชันต้องระบุตัวตน Root และกลยุทธ์การตัดสินใจทั้งสี่รายการ กลยุทธ์แบบ static และ `MockProvider` เหมาะสำหรับเริ่มต้นแบบออฟไลน์:

```python
from agenttree import AgentTree, ManagerAgent, RootAgent, SpecialistAgent, Task
from agenttree.core import (
    RuleBasedTaskTriage,
    StaticFinalReviewer,
    StaticManagerReviewer,
    StaticTaskDecomposer,
)
from agenttree.models import SubtaskTemplate
from agenttree.providers import MockProvider

framework = AgentTree(
    root_agent=RootAgent(name="Root"),
    triage=RuleBasedTaskTriage({}, fallback_capabilities=("analysis",)),
    decomposer=StaticTaskDecomposer((
        SubtaskTemplate("Summarize", ("summarize",)),
    )),
    manager_reviewer=StaticManagerReviewer(),
    final_reviewer=StaticFinalReviewer(),
)
manager = ManagerAgent(name="Manager", capabilities=("analysis",))
specialist = SpecialistAgent(name="Specialist", capabilities=("summarize",))
provider = MockProvider(response_content="Offline result")

framework.register_manager(manager)
framework.register_specialist(manager, specialist)
framework.register_provider(provider)
framework.bind_provider(specialist, provider)

result = framework.run(Task(objective="Prepare a summary"))
assert result.success
print(result.content)
print(result.trace.events)
```

การลงทะเบียนใช้ object อย่างสม่ำเสมอ Specialist หนึ่งตัวอยู่ภายใต้ Manager หนึ่งตัว ส่วน provider/tool binding จะอยู่ภายนอกเอเจนต์ คุณสมบัติแบบ snapshot เช่น `framework.managers` และ `framework.providers` จะไม่เปิดเผย dictionary ของ registry ที่แก้ไขได้

## เครื่องมือและสถานะ

ลงทะเบียนเครื่องมือด้วย `register_tool()` และอนุญาตให้ Specialist ใช้งานด้วย `bind_tool()` การเรียกเครื่องมือยังต้องทำอย่างชัดเจนผ่าน `ToolExecutor`; `run()` จะไม่เลือกเครื่องมือโดยอัตโนมัติ หลังจบการรันให้ตรวจสอบ `result.trace` และ `framework.last_state` ระบบจะเก็บเฉพาะสถานะการรันล่าสุดไว้ในหน่วยความจำ

ข้อผิดพลาดในการกำหนดค่าจะยก exception ส่วนผลลัพธ์ของลำดับงานตามปกติที่ engine รองรับอยู่แล้ว เช่น ไม่พบรายการที่ตรงหรือการตรวจทานไม่ผ่าน จะส่งคืน `FinalResult` ที่ไม่สำเร็จ ดู [ข้อผิดพลาด](errors.md)

เมื่อกำหนดการคัดแยกหรือการแยกงานที่ขับเคลื่อนด้วย provider แล้ว `AgentTree` จะส่งความสามารถที่ลงทะเบียนและมีสิทธิ์ให้โดยอัตโนมัติ แอปพลิเคชันไม่ต้องคัดลอกรายการความสามารถไปยัง constructor ของกลยุทธ์ หากคำตอบจาก provider ระบุความสามารถของ Manager ที่ไม่มีอยู่ หรือความสามารถที่อยู่นอก Specialist ภายใต้ Manager ที่เลือก ระบบจะยก `DecisionOutputError` ก่อนกำหนดเส้นทาง
