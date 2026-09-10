# ขอบเขตที่ครอบคลุม

ตารางนี้จับคู่ขอบเขตที่ตั้งใจไว้ของเฟรมเวิร์ก AgentTree กับหลักฐานการพัฒนาในเวอร์ชัน 0.2.1 โดยบันทึกความครอบคลุมของซอฟต์แวร์ ไม่ใช่การกล่าวอ้างเกี่ยวกับความฉลาดของโมเดลหรือขีดความสามารถระดับ production

| ข้อกำหนด | การพัฒนา | หลักฐาน |
|---|---|---|
| เอเจนต์แบบลำดับชั้น | `RootAgent`, `ManagerAgent`, `SpecialistAgent` และความเป็นเจ้าของของ Manager | `tests/test_agents.py`, `examples/thesis_demo.py` |
| Contract ของงาน/context | `Task`, `TaskContext`, dataclass ของ lifecycle/ผลลัพธ์ | `tests/test_models.py` |
| การคัดแยกงาน | กลยุทธ์การคัดแยกแบบใช้กฎและขับเคลื่อนด้วย provider | `tests/test_triage.py`, `tests/test_provider_decisions.py` |
| การค้นหาความสามารถ | `CapabilityRegistry` ตามลำดับ พร้อมการจับคู่ ANY/ALL | `tests/test_registry.py`, ผลการกำหนดเส้นทางจากเดโมวิทยานิพนธ์ |
| การแยกงานโดย Manager | การแยกงานย่อยแบบ static และขับเคลื่อนด้วย provider | `tests/test_delegation.py` |
| การทำงานของ Specialist | provider binding ภายนอกและ `AgentResult` ที่ผ่าน normalization | `tests/test_execution.py` |
| การตรวจทาน/แก้ไขโดย Manager | PASS/REVISE/FAIL และการประมวลผลซ้ำที่มีขีดจำกัด | `tests/test_review.py`, `thesis_demo.py --revision` |
| การตรวจทานขั้นสุดท้ายโดย Root | การตัดสินขั้นสุดท้าย การพิจารณาใหม่ และ `FinalResult` | `tests/test_final_review.py` |
| SDK ระดับสูง | เมธอดลงทะเบียน/binding และ `AgentTree.run(Task)` | `tests/test_framework.py`, `tests/test_release.py` |
| Abstraction ของ provider | contract ทั่วไป, Mock, OpenAI, Gemini, Ollama | `tests/test_providers.py`, `tests/test_provider_adapters.py` |
| การตัดสินใจด้วย provider | JSON ที่ผ่านการตรวจสอบสำหรับการคัดแยก การแยกงาน และการตรวจทาน | `tests/test_provider_decisions.py` |
| เครื่องมือฟังก์ชัน | registry, binding และ `ToolExecutor` แบบเรียกอย่างชัดเจน | `tests/test_tools.py`, ตัวอย่างโดเมน IT |
| การเชื่อมต่อเครื่องมือ/ข้อมูลด้วย MCP | client แบบ Mock และ stdio/HTTP จริง พร้อม `MCPToolLoader` | `tests/test_mcp.py`, `tests/test_mcp_transports.py`, เดโม MCP ภายในเครื่อง |
| สถานะและ trace | `WorkflowState` แบบป้องกันการแก้ไข และ `ExecutionTrace` ตามลำดับ | `tests/test_state_and_trace.py`, สรุป trace จากเดโมวิทยานิพนธ์ |
| แบ็กเอนด์การประสานงาน | Sequential และ LangGraph แบบซิงโครนัสที่เป็นตัวเลือก | `tests/test_backends.py`, โหมด LangGraph ของเดโมวิทยานิพนธ์ |
| การใช้ซ้ำหลายโดเมน | การกำหนดค่า IT, HR และผลิตภัณฑ์ไว้นอกแกนหลัก | `examples/domains/`, `evaluation/results.json` |
| Packaging/การใช้งานสำหรับนักพัฒนา | distribution เวอร์ชัน 0.2.1, typed marker และเอกสาร | `tests/test_release.py`, `pyproject.toml`, `docs/` |

## ข้อจำกัดที่ทราบ

- สถานะลำดับงานและ trace อยู่ในหน่วยความจำ ไม่มี persistence หรือฐานข้อมูล
- SDK สาธารณะทำงานแบบซิงโครนัสและไม่ประสานเอเจนต์แบบกระจาย
- ไม่มีโปรโตคอล A2A หรือการสื่อสารระหว่างเอเจนต์แบบ peer-to-peer อย่างอิสระ
- ไม่มี provider fallback, retry/การกำหนดเส้นทางตามต้นทุน และการวางแผนเครื่องมืออัตโนมัติ
- การเรียกเครื่องมือและ MCP ต้องให้แอปพลิเคชันเลือกและสั่งทำงานอย่างชัดเจน
- provider จริงและระบบ MCP ภายนอกต้องผ่านการตรวจสอบให้เหมาะกับแต่ละ environment; การตรวจสอบขั้นสุดท้ายใช้ mock และ fixture ภายในเครื่อง
- ยังไม่ได้ประเมินความสามารถในการปรับขนาดระดับ production การปฏิบัติการด้านความปลอดภัย และความถูกต้องครอบคลุมทุกโดเมน/โมเดล
