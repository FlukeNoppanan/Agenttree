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

API ระดับสูงคือ `framework.register_tool(tool)` และ `framework.bind_tool(agent, tool)` โดย `agent` เป็น Root, Manager หรือ Specialist ได้ การผูกเครื่องมือแยกจากการผูก provider และอ้างอิง Tool ID ที่ลงทะเบียนแล้ว

## Unified model tool runtime

During `AgentTree.run()`, provider backed Root, Manager, and Specialist calls receive definitions for their assigned, enabled tools. The model may request a call; Python checks the registered ID, exact agent assignment, enabled state, argument shape and types, call budget, and timeout before invoking `BaseTool`. Results return to the same model call history; the model produces the agent's answer or decision. Tool output never directly becomes `FinalResult.final_output`.

```python
from agenttree import AgentTreeConfig
from agenttree.tools import FunctionTool

def project_context() -> dict:
    return {"status": "ready"}

framework.register_tool(FunctionTool(name="project_context", function=project_context))
framework.bind_tool(framework.root_agent, framework.tools[-1])
# Root also needs a provider binding for model driven tool use.
```

`AgentTreeConfig` controls `max_tool_rounds` (3 per provider call), `max_tool_calls` (8 per agent per run), `tool_timeout` (5 seconds), `max_tool_argument_bytes` (16,384), and `max_tool_result_bytes` (65,536). Tool calls execute sequentially. Invalid, unassigned, disabled, failed, oversized, and timed out calls return named safe errors to the model. Trace events include role, Tool ID, call ID, status, and duration; they omit arguments, outputs, tool metadata, and raw exceptions. `FinalResult.metadata["tool_metrics"]` records tool counts separately from token usage when tool events occur.

The runtime redacts common credential keys and token patterns in successful tool outputs before sending them to a provider. Applications must still define tools that never return secrets or permit model arguments to override credentials or configured destinations. A synchronous local function that times out returns promptly, but its daemon thread cannot be stopped safely and may finish later; tools with side effects should use their own interruptible I/O timeout. Provider/model tool capability is checked when assigned tools are offered. The current native Gemini adapter and OpenAI compatible adapters translate tool calls; actual model support varies by selected model.

`ToolExecutor.execute(specialist=...)` remains available as a host controlled manual compatibility API. Model driven calls use the shared `ToolSession` policy. For future Studio integration, the Agent Inspector needs an ordered list of assigned Tool IDs for each Root, Manager, and Specialist, and Live View can render the role aware tool trace events and separate metrics. No Manager to Manager messaging is part of this contract.

## MCP client

- `MockMCPClient` ใช้ค้นหาและเรียกแบบให้ผลแน่นอนสำหรับการทดสอบออฟไลน์
- `StdioMCPClient` เปิดคำสั่งภายในเครื่องหนึ่งคำสั่งที่กำหนดไว้อย่างชัดเจน
- `StreamableHttpMCPClient` เชื่อมต่อ endpoint HTTP(S) หนึ่งแห่งที่ระบุไว้
- `MCPToolLoader` ค้นหา definition ของ MCP ปรับเป็น `MCPTool` และลงทะเบียนทั้งชุดแบบ atomic ได้

ติดตั้ง transport จริงด้วย `python -m pip install "agenttree[mcp]"` เชื่อมต่อและปิด client อย่างชัดเจน หรือใช้ synchronous context manager ระบบจะไม่ดาวน์โหลดหรือค้นหาเซิร์ฟเวอร์โดยอัตโนมัติ หลีกเลี่ยงการใส่ข้อมูลรับรองใน URL, argument ของคำสั่ง, metadata หรือ trace; HTTP header และค่า environment ของ process ที่ระบุจะอยู่ภายในการกำหนดค่า transport

MCP คือโปรโตคอลเชื่อมต่อเครื่องมือ/ข้อมูลใน AgentTree ไม่ได้เปิดใช้งาน A2A หรือการสื่อสารระหว่างเอเจนต์แบบ peer-to-peer เครื่องมือ MCP ที่ลงทะเบียนและผูกกับเอเจนต์จะถูกเสนอให้โมเดลเช่นเดียวกับ `FunctionTool`; การเรียกยังขึ้นกับคำขอจากโมเดลและนโยบาย runtime
