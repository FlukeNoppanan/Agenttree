# เอเจนต์และความสามารถ

AgentTree มีบทบาทเอเจนต์เชิงโครงสร้างสามแบบ:

- `RootAgent` ระบุตัวตนของผู้มีอำนาจตรวจทานขั้นสุดท้าย
- `ManagerAgent` ประกาศความสามารถระดับ Manager และเป็นเจ้าของ Specialist
- `SpecialistAgent` ประกาศความสามารถที่ใช้มอบหมายงานย่อย

เอเจนต์เก็บตัวตนและการกำหนดค่า แต่ไม่ได้เรียก provider หรือเครื่องมือด้วยตนเอง ID จะคงที่หลังสร้าง object และระบบจะสร้าง UUID เมื่อแอปพลิเคชันไม่ได้ระบุ ID

```python
from agenttree import ManagerAgent, SpecialistAgent

manager = ManagerAgent(
    id="analysis-manager",
    name="Analysis Manager",
    capabilities=("analysis",),
)
specialist = SpecialistAgent(
    id="summary-specialist",
    name="Summary Specialist",
    capabilities=("summarize", "extract"),
)

framework.register_manager(manager)
framework.register_specialist(manager, specialist)
```

API ระดับสูงจะตรวจสอบตัวตนทั้งหมดที่ได้รับผลกระทบก่อนเปลี่ยนแปลง เพื่อให้ความเป็นเจ้าของของ Manager และสมาชิกใน `CapabilityRegistry` สอดคล้องกัน ID ซ้ำ การเป็นสมาชิกข้าม Manager และ object ที่ขัดแย้งกันถือเป็นข้อผิดพลาด

การจับคู่ความสามารถจะตัดช่องว่างรอบข้อความและเทียบป้ายแบบตรงตัวโดยไม่แยกตัวพิมพ์เล็ก-ใหญ่ การค้นหาคงลำดับการลงทะเบียนและรองรับการจับคู่แบบ ANY เป็นค่าเริ่มต้น หรือแบบ ALL ผ่าน `AgentTreeConfig` ระบบจะไม่ใช้ความคล้ายคลึงเชิงความหมาย การจัดอันดับ การให้คะแนน หรือการอนุมานความสามารถโดยอัตโนมัติ

ระหว่างการเรียก `AgentTree.run()` ตามปกติ `ProviderTaskTriage` จะได้รับความสามารถแบบ normalized ที่มีอยู่จาก Manager ซึ่งลงทะเบียนอยู่ ส่วน `ProviderTaskDecomposer` จะได้รับเฉพาะความสามารถจาก Specialist ที่ลงทะเบียนและอยู่ภายใต้ Manager ที่กำลังแยกงาน ผลลัพธ์จาก provider จะถูก casefold, ตัดรายการซ้ำตามลำดับคำตอบ, แปลงเป็นค่ามาตรฐานที่ลงทะเบียนไว้ และปฏิเสธหากอยู่นอกชุดที่ใช้ได้ หากชุดที่ใช้ได้ว่าง `required_capabilities` ต้องเป็นรายการว่างเท่านั้น
