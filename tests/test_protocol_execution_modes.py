"""Shared protocol and execution-mode policy, with real orchestration phases."""
import json
import pytest
from agenttree import ManagerAgent, SpecialistAgent, Task
from agenttree.models import ExecutionMode
from agenttree.core import ProviderRootPlanner
from agenttree.core.structured_output import normalize_decision_output, parse_decision_output
from agenttree.core.execution_codec import dumps, loads
from agenttree.exceptions import DecisionOutputError
from agenttree.providers import MockProvider, ProviderResponse
from tests.test_root_runtime import make_tree, bind_workers
from tests.test_structured_decision_repair import SequenceProvider

@pytest.mark.parametrize("value", [
 {"delegate": True}, '{"delegate":true}', '```json\n{"delegate":true}\n```',
 json.dumps('{"delegate":true}'), {"result":{"delegate": True}},
 {"content":'{"delegate":true}'}, {"output":{"decision":{"delegate":True}}},
])
def test_equivalent_protocol_representations(value):
 assert normalize_decision_output(value)[0] == {"delegate": True}
 provider = SequenceProvider([value])
 result = ProviderRootPlanner(provider).plan(Task("Compare"), make_tree()[0].root_agent)
 assert result.delegate is True
 assert len(provider.requests) == 1

@pytest.mark.parametrize("bad", ['{bad', '{"delegate":true,"delegate":false}',
 '{"value":NaN}', [1], '{"wrapper":{"delegate":true}}', {"delegate":"true"}])
def test_invalid_protocol_not_invented(bad):
 provider = SequenceProvider([bad,bad])
 with pytest.raises(DecisionOutputError):
  ProviderRootPlanner(provider).plan(Task("Compare"), make_tree()[0].root_agent)
 assert len(provider.requests) == 2

def test_native_explicit_content_does_not_mine_raw_envelope():
 class Native(SequenceProvider):
  def generate(self, request):
   return ProviderResponse(content='', provider=self.name,
     structured_content={"delegate":True}, raw_response={"secret":"not inspected"})
 assert ProviderRootPlanner(Native([])).plan(Task("Compare"), make_tree()[0].root_agent).delegate

@pytest.mark.parametrize("mode,delegate", [("fast",False),("fast",True),("deep",False),("deep",True)])
def test_mode_enforced_after_root_plan(mode,delegate):
 planner=ProviderRootPlanner(MockProvider(response_content=json.dumps({"delegate":delegate,"direct_output":"direct"})))
 tree,workers=make_tree(planner=planner)
 bind_workers(tree,workers,MockProvider(response_content="hierarchical answer"))
 result=tree.run(Task("Explain",execution_mode=mode))
 assert result.success
 direct=any(e.event_type=="root.direct_response" for e in result.trace.events)
 assert direct == (mode=="fast" and not delegate)
 assert bool(result.manager_results) != direct
 assert next(e for e in result.trace.events if e.event_type=="execution.started").metadata["execution_mode"]==mode
 if mode=="deep":
  assert all(e.event_type!="root.direct_response" for e in result.trace.events)

@pytest.mark.parametrize("managers,specialists",[(0,0),(1,0)])
def test_deep_impossible_hierarchy_explicit_failure(managers,specialists):
 planner=ProviderRootPlanner(MockProvider(response_content='{"delegate":false,"direct_output":"unsafe"}'))
 tree,_=make_tree(planner=planner,managers=managers,specialists=specialists)
 result=tree.run(Task("Explain",execution_mode="deep"))
 assert not result.success and "Deep execution requires" in result.error["message"]
 assert not planner._provider.requests

def test_deep_preserves_capability_routing_and_owned_specialists():
 tree,workers=make_tree(planner=ProviderRootPlanner(MockProvider(response_content='{"delegate":false,"direct_output":"unsafe"}')))
 other=ManagerAgent(name="Unrelated",capabilities=("security",))
 tree.register_manager(other)
 tree.register_specialist(other,SpecialistAgent(name="Unused",capabilities=("security",)))
 # A nonmatching Specialist beneath the selected Manager also must stay idle.
 manager=tree.managers[0]
 tree.register_specialist(manager,SpecialistAgent(name="Other task",capabilities=("other",)))
 provider=MockProvider(response_content="work")
 bind_workers(tree,workers,provider)
 result=tree.run(Task("Explain",execution_mode="deep"))
 assert result.success and len(provider.requests)==1
 assert len(result.manager_results)==1

def test_mode_durable_codec_and_legacy_default():
 task=Task("Explain",execution_mode="deep")
 assert loads(dumps(task)).execution_mode is ExecutionMode.DEEP
 assert Task("Old caller").execution_mode is ExecutionMode.FAST
 with pytest.raises(ValueError): Task("Explain",execution_mode="invented")

@pytest.mark.parametrize("scenario",["success","manager_revision","final_revision","manager_limit","revision_limit","final_fail","no_manager","no_specialist"])
def test_deep_shared_backends_preserve_review_limits(scenario, fake_api):
 from tests.test_backends import make_framework
 from agenttree.orchestration.backends import SequentialOrchestrationBackend, LangGraphOrchestrationBackend
 for backend in (SequentialOrchestrationBackend(),LangGraphOrchestrationBackend()):
  tree=make_framework(backend,scenario)
  result=tree.run(Task("Work",execution_mode="deep"))
  assert not any(e.event_type=="root.direct_response" for e in result.trace.events)
  if scenario in {"success","manager_revision","final_revision"}: assert result.success
  else: assert not result.success

from tests.test_backends import fake_api  # noqa: E402,F401

@pytest.mark.parametrize("bad",[{1:"not a JSON key"},{"delegate":(True,)},{"value":float("nan")},{"content":{"content":{"content":{"content":{"content":{"delegate":True}}}}}}])
def test_native_and_wrapper_bounds_reject_invalid_representations(bad):
 with pytest.raises(DecisionOutputError): normalize_decision_output(bad)

def test_presentation_wrapper_does_not_change_collaboration_content():
 value={"content":'{"decision":"pass","feedback":"a work product"}'}
 assert parse_decision_output(value)==value

def test_adapter_explicit_native_object_reaches_same_root_plan():
 from agenttree.providers._adapter import response
 from agenttree.providers import ProviderRequest
 class Native(SequenceProvider):
  def generate(self, request):
   return response({"delegate":True},"fixture",self.name,None,request,None,False)
  @property
  def capabilities(self):
   from agenttree.providers import ProviderCapabilities
   return ProviderCapabilities(structured_output=True)
 assert ProviderRootPlanner(Native([])).plan(Task("Compare"),make_tree()[0].root_agent).delegate

def test_deep_cooperative_cancellation_does_not_become_direct_success():
 from threading import Event
 from agenttree.core.execution_control import ExecutionControl, ExecutionCancelled
 tree,_=make_tree()
 cancelled=Event();cancelled.set()
 with pytest.raises(ExecutionCancelled):
  tree.run(Task("Explain",execution_mode="deep"),_control=ExecutionControl(cancelled))
