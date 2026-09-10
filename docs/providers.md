# ผู้ให้บริการโมเดล

ขอบเขตของ provider ทำงานแบบซิงโครนัสและไม่ผูกกับผู้ให้บริการรายใด:

```python
response = provider.generate(ProviderRequest(prompt="Summarize this input"))
```

`ProviderRequest` บรรจุ prompt, system prompt ที่เป็นตัวเลือก, context แบบมีโครงสร้าง, metadata, โมเดล, temperature และขีดจำกัด token ส่วน `ProviderResponse` ทำ normalization ให้เนื้อหา ตัวตน provider/โมเดล usage ที่เป็นตัวเลือก metadata และข้อมูลดิบฉบับคัดลอกที่เป็นตัวเลือก `ProviderConfig` ตั้งใจไม่รวมข้อมูลรับรอง เพราะอาจถูกนำไปใช้ในการวินิจฉัย

## การพัฒนาแบบออฟไลน์

`MockProvider` บันทึก request และส่งคืน response ที่ให้ผลแน่นอนโดยไม่เรียก SDK หรือเครือข่าย:

```python
from agenttree.providers import MockProvider, ProviderConfig

provider = MockProvider(
    ProviderConfig(provider_name="offline-worker", model="offline-fixture"),
    response_content="Fixed output",
)
```

## การตัดสินใจกำหนดเส้นทางด้วย provider

`ProviderTaskTriage(provider)` และ `ProviderTaskDecomposer(provider)` ยังคงเป็นรูปแบบ constructor ที่ใช้งานได้ ในลำดับงานระดับสูง ระบบประสานงานจะใส่ตัวเลือกการกำหนดเส้นทางปัจจุบันใน request ของ provider แต่ละรายการโดยอัตโนมัติ:

- การคัดแยกได้รับความสามารถแบบ normalized จาก Manager ที่ลงทะเบียน
- การแยกงานได้รับความสามารถแบบ normalized จาก Specialist ที่ลงทะเบียนและอยู่ภายใต้ Manager ที่เลือก

system prompt และ context แบบมีโครงสร้างของ request จะแสดงตัวเลือกเหล่านี้อย่างตรงตัว ค่าความสามารถที่ส่งกลับมาจะถูกตัดช่องว่าง, casefold, ตัดรายการซ้ำ และจับคู่กับค่ามาตรฐานที่มีอยู่ ค่าที่ไม่รู้จักจะทำให้ยก `DecisionOutputError` และจะไม่ถูกส่งไปค้นหาใน registry ผู้เรียกใช้งานโดยตรงขั้นสูงสามารถส่ง `available_capabilities=` อย่างชัดเจนได้ หากละไว้จะยังคงพฤติกรรมการเรียกโดยตรงแบบไม่จำกัดเดิมเพื่อความเข้ากันได้

## อะแดปเตอร์เสริม

ติดตั้งอะแดปเตอร์หนึ่งรายการหรือทั้งสามรายการ:

```sh
python -m pip install "agenttree[openai]"
python -m pip install "agenttree[gemini]"
python -m pip install "agenttree[ollama]"
python -m pip install "agenttree[providers]"
```

`OpenAIProvider`, `GeminiProvider` และ `OllamaProvider` รับ `ProviderConfig` ที่ไม่มีข้อมูลลับ และอาจรับ SDK client ที่ inject เข้ามา ไม่มีชื่อโมเดลใดถูกกำหนดตายตัว ให้เลือกโมเดลในการกำหนดค่าของแอปพลิเคชัน ปล่อยให้ SDK ที่เกี่ยวข้องอ่าน environment เช่น `OPENAI_API_KEY`, `GOOGLE_API_KEY` หรือ `OLLAMA_HOST` หรือสร้างและ inject client ที่กำหนดค่าแล้ว ห้ามใส่ข้อมูลลับใน `ProviderConfig.metadata`, metadata ของ Task, ผลลัพธ์ หรือ trace

อะแดปเตอร์ทำ normalization ให้ response และการใช้ token ของ SDK โดยไม่เปิดเผยคลาส SDK แก่ระบบประสานงาน ระบบไม่มีการเลือก provider, fallback, retry, การกำหนดเส้นทางตามต้นทุน หรือการตรวจสถานะแบบสดโดยอัตโนมัติ
