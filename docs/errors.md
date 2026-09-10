# ข้อผิดพลาดและผลลัพธ์ความล้มเหลว

AgentTree แยกข้อผิดพลาดด้านการเขียนโปรแกรม/การกำหนดค่าออกจากผลลัพธ์ของลำดับงานที่คาดหมาย

## Exception

- การกำหนดค่าเอเจนต์, registry, binding, Task หรือกลยุทธ์ที่ไม่ถูกต้องจะยก `TypeError`, `ValueError` หรือ `KeyError` พร้อมระบุส่วนประกอบที่ได้รับผลกระทบเมื่อทำได้
- `ProviderDependencyError`, `ProviderConfigurationError` และ `ProviderRuntimeError` ใช้แยกความล้มเหลวด้าน SDK เสริม, request/การกำหนดค่า และการสร้างผลลัพธ์/normalization
- `DecisionParseError` หมายถึงผลลัพธ์จาก provider ไม่ใช่ JSON object เดียวที่ตีความได้ชัดเจน ส่วน `DecisionOutputError` หมายถึง JSON ที่ parse แล้วละเมิด contract ของการคัดแยก การแยกงาน หรือการตรวจทาน ซึ่งรวมถึงการเลือกความสามารถที่อยู่นอกตัวเลือกของ Manager ที่ลงทะเบียน หรืออยู่นอกตัวเลือก Specialist ภายใต้ Manager ปัจจุบัน
- `MCPDependencyError`, `MCPConfigurationError`, `MCPConnectionError` และ `MCPRuntimeError` ใช้แยกความพร้อมของ SDK, lifecycle/การกำหนดค่า, การเชื่อมต่อ/ค้นหา และความล้มเหลวจากการเรียกเครื่องมือ
- `BackendDependencyError` และ `BackendConfigurationError` ระบุว่าไม่มีแบ็กเอนด์เสริมหรือ contract ของแบ็กเอนด์ไม่ถูกต้อง

exception จาก provider ระหว่างการสร้างผลลัพธ์ของ Specialist จะถูกทำ normalization เป็นค่า `AgentResult` ที่ล้มเหลว เพื่อให้ reviewer ที่กำหนดไว้ตัดสินผลลัพธ์ของลำดับงานได้ ส่วน exception จาก provider ที่กลยุทธ์ตัดสินใจใช้จะถูกส่งต่อ เพราะเฟรมเวิร์กไม่สามารถสร้างการตัดสินใจแทนได้อย่างปลอดภัย

## ความล้มเหลวแบบมีโครงสร้าง

เมื่อไม่พบ Manager ที่ตรง ไม่มีงานของ Specialist ที่เรียกได้ มีการตัดสินขั้นสุดท้ายเป็น FAIL หรือใช้ขีดจำกัดการตรวจทานครบ ระบบจะส่งคืน `FinalResult` ที่ไม่สำเร็จตาม contract ของลำดับงานที่มีอยู่ ให้ตรวจสอบ `result.status`, `result.final_review` และ `result.trace` หากเกิด exception ระบบจะบันทึก `framework.last_state` ล่าสุดที่มีสถานะล้มเหลวก่อนยก exception ซ้ำ

ข้อความสาธารณะจะไม่เปิดเผยข้อมูลรับรองและ authorization header ส่วนสาเหตุโดยละเอียดจาก SDK หรือ transport ยังคงตรวจสอบได้ผ่าน exception chaining ของ Python
