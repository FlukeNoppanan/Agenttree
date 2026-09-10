# เครื่องมือและ MCP

เครื่องมือใช้ contract แบบซิงโครนัสร่วมกัน `FunctionTool` ห่อ Python callable ที่ระบุอย่างชัดเจน ส่วน `MCPTool` ปรับเครื่องมือหนึ่งรายการที่ค้นพบจากเซิร์ฟเวอร์ MCP ทั้งสองใช้ `ToolRegistry`, การกำหนด `ToolBindingRegistry` ภายนอก และ `ToolExecutor`

```python
from agenttree.tools import FunctionTool, ToolBindingRegistry, ToolExecutor, ToolRegistry

def add(left: int, right: int) -> int:
    return left + right

registry = ToolRegistry()
bindings = ToolBindingRegistry()
tool = FunctionTool(name="add", function=add)
registry.register(tool)
bindings.assign(specialist.id, tool.id)
result = ToolExecutor(registry=registry, bindings=bindings).execute(
    specialist=specialist,
    tool_id=tool.id,
    arguments={"left": 2, "right": 3},
)
```

API ระดับสูงที่เทียบเท่าคือ `framework.register_tool(tool)` และ `framework.bind_tool(specialist, tool)` การเรียกใช้ยังต้องทำอย่างชัดเจน

## MCP client

- `MockMCPClient` ใช้ค้นหาและเรียกแบบให้ผลแน่นอนสำหรับการทดสอบออฟไลน์
- `StdioMCPClient` เปิดคำสั่งภายในเครื่องหนึ่งคำสั่งที่กำหนดไว้อย่างชัดเจน
- `StreamableHttpMCPClient` เชื่อมต่อ endpoint HTTP(S) หนึ่งแห่งที่ระบุไว้
- `MCPToolLoader` ค้นหา definition ของ MCP ปรับเป็น `MCPTool` และลงทะเบียนทั้งชุดแบบ atomic ได้

ติดตั้ง transport จริงด้วย `python -m pip install "agenttree[mcp]"` เชื่อมต่อและปิด client อย่างชัดเจน หรือใช้ synchronous context manager ระบบจะไม่ดาวน์โหลดหรือค้นหาเซิร์ฟเวอร์โดยอัตโนมัติ หลีกเลี่ยงการใส่ข้อมูลรับรองใน URL, argument ของคำสั่ง, metadata หรือ trace; HTTP header และค่า environment ของ process ที่ระบุจะอยู่ภายในการกำหนดค่า transport

MCP คือโปรโตคอลเชื่อมต่อเครื่องมือ/ข้อมูลใน AgentTree ไม่ได้เปิดใช้งาน A2A หรือการสื่อสารระหว่างเอเจนต์แบบ peer-to-peer และ `AgentTree.run()` จะไม่วางแผนหรือเรียกเครื่องมือ MCP โดยอัตโนมัติ
