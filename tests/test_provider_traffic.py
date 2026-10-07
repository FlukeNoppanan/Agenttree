"""Behavioral coverage for shared traffic admission; all Providers are synthetic."""
from dataclasses import replace
from threading import Event, Thread
from types import SimpleNamespace
import json
import pytest

from agenttree.core.execution_control import ExecutionCancelled
from agenttree.providers import BaseProvider, ProviderConfig, ProviderRequest, ProviderResponse, ProviderUsage, ProviderStreamChunk, ProviderCapabilities
from agenttree.providers.exceptions import normalize_provider_error, ProviderRateLimitError, ProviderAuthenticationError, ProviderUnavailableError
from agenttree.providers.traffic import ProviderRequestGovernor, GovernedProvider, traffic_context
from agenttree.providers.traffic_failure import RateLimitSignal, failure_from_exception, seconds, safe_headers


class Provider(BaseProvider):
    def __init__(self,action=None):
        super().__init__(ProviderConfig('test',model='m'));self.calls=[];self.action=action
    @property
    def capabilities(self):return ProviderCapabilities(streaming=True)
    def generate(self,request):
        self.calls.append(request)
        return self.action(request) if self.action else ProviderResponse('OK','test','m',ProviderUsage(total_tokens=4))
    def generate_stream(self,request):
        yield ProviderStreamChunk(delta_text='OK')
        if self.action:self.action(request)
        yield ProviderStreamChunk(response=ProviderResponse('OK','test','m'))


def test_same_scope_queue_waits_and_other_scope_is_independent():
    governor=ProviderRequestGovernor(max_wait=2);entered=Event();release=Event();second=Event()
    p=GovernedProvider(Provider(lambda r:(entered.set(),release.wait(2),ProviderResponse('OK','test'))[-1]),key='shared',governor=governor)
    q=GovernedProvider(Provider(lambda r:(second.set(),ProviderResponse('OK','test'))[-1]),key='shared',governor=governor)
    t=Thread(target=lambda:p.generate(ProviderRequest('a')));t.start();assert entered.wait(1)
    u=Thread(target=lambda:q.generate(ProviderRequest('b')));u.start()
    independent=GovernedProvider(Provider(),key='other',governor=governor)
    assert independent.generate(ProviderRequest('c')).content=='OK'
    assert not second.is_set();release.set();t.join(2);u.join(2)
    assert second.is_set() and not t.is_alive() and not u.is_alive()
    assert all(scope.active==0 for scope in governor._scopes.values())


def test_queued_cancellation_never_dispatches_later():
    g=ProviderRequestGovernor(max_wait=2);cancel=Event();queued=Event();done=Event();errors=[];delegate=Provider()
    p=GovernedProvider(delegate,key='shared',governor=g)
    def check():
        if cancel.is_set():raise ExecutionCancelled('Stopped')
    def run():
        try:
            with traffic_context(check=check,observer=lambda e,d:queued.set() if e=='request.queued' else None):p.generate(ProviderRequest('x'))
        except Exception as e:errors.append(e)
        finally:done.set()
    with g.admission('shared','m'):
        t=Thread(target=run);t.start();assert queued.wait(1);cancel.set();assert done.wait(1)
    t.join();assert isinstance(errors[0],ExecutionCancelled)
    assert not delegate.calls and not g._queue


def test_model_cooldown_does_not_block_unrelated_model():
    now=[0.];g=ProviderRequestGovernor(clock=lambda:now[0],jitter=lambda a,b:0)
    g.cooldown('key','limited',RateLimitSignal(scope='model',retry_after=3600))
    p=GovernedProvider(Provider(),key='key',governor=g)
    assert p.generate(ProviderRequest('x',model='other')).content=='OK'
    with pytest.raises(ProviderRateLimitError):p.generate(ProviderRequest('x',model='limited'))
    assert len(p.delegate.calls)==1


def test_long_retry_is_not_shortened_or_retried():
    g=ProviderRequestGovernor(jitter=lambda a,b:0);events=[]
    def fail(r):raise ProviderRateLimitError('safe',retry_after=18824,failure_scope='model')
    p=GovernedProvider(Provider(fail),key='key',governor=g)
    with traffic_context(observer=lambda e,d:events.append((e,d))):
        with pytest.raises(ProviderRateLimitError) as caught:p.generate(ProviderRequest('secret'))
    assert caught.value.retry_after==18824 and len(p.delegate.calls)==1
    assert 'secret' not in json.dumps(events) and not g._queue
    assert g._scopes[('key','m')].cooldown-g.clock()>18820


def test_known_request_budget_waits_then_resumes_with_actual_usage():
    now=[0.];g=ProviderRequestGovernor(clock=lambda:now[0])
    response=ProviderResponse('OK','test',usage=ProviderUsage(total_tokens=7),metadata={'traffic_headers':{'x-ratelimit-remaining-requests':'0','x-ratelimit-reset-requests':'60s'}})
    g.observe('key','m',response)
    p=GovernedProvider(Provider(),key='key',governor=g)
    with pytest.raises(ProviderRateLimitError):p.generate(ProviderRequest('x'))
    assert not p.delegate.calls
    now[0]=61;assert p.generate(ProviderRequest('x')).content=='OK'


def test_token_usage_reconciliation():
    g=ProviderRequestGovernor();scope=g._scope('key');scope.budgets['tokens']=(70,g.clock()+60)
    g.observe('key','m',ProviderResponse('x','p',usage=ProviderUsage(total_tokens=12)),reserved=20)
    assert scope.budgets['tokens'][0]==78


def test_fifo_between_eligible_requests():
    g=ProviderRequestGovernor();order=[];queued=[Event(),Event()];threads=[]
    with g.admission('key','m'):
        for i in range(2):
            def run(i=i):
                with traffic_context(observer=lambda e,d:queued[i].set() if e=='request.queued' else None):
                    with g.admission('key','m'):order.append(i)
            t=Thread(target=run);threads.append(t);t.start();assert queued[i].wait(1)
    for t in threads:t.join(2)
    assert order==[0,1]


def test_stream_is_not_replayed_after_a_delta():
    def fail(r):raise ProviderRateLimitError('safe',retry_after=.01)
    g=ProviderRequestGovernor();p=GovernedProvider(Provider(fail),key='key',governor=g)
    stream=p.generate_stream(ProviderRequest('x'));assert next(stream).delta_text=='OK'
    with pytest.raises(ProviderRateLimitError):next(stream)
    assert g._scopes['key'].active==0


def test_stream_close_releases_slot():
    g=ProviderRequestGovernor();p=GovernedProvider(Provider(),key='key',governor=g)
    stream=p.generate_stream(ProviderRequest('x'));next(stream);stream.close()
    assert g._scopes['key'].active==0


@pytest.mark.parametrize('status,category',[(401,'authentication'),(403,'authentication'),(404,'model_unavailable'),(400,'invalid_request'),(422,'invalid_request'),(429,'rate_limit'),(503,'capacity'),(408,'timeout')])
def test_failure_categories(status,category):
    result=normalize_provider_error(SimpleNamespace(status_code=status))
    assert failure_from_exception(result).category==category


@pytest.mark.parametrize('body,scope,kind',[
 ({},'unknown','unknown'),
 ({'error':{'message':'Rate limit reached for tokens per minute (TPM)'}},'unknown','tokens_per_minute'),
 ({'error':{'message':'requests per minute (RPM)'}},'unknown','requests_per_minute'),
 ({'error':{'details':[{'@type':'type.googleapis.com/google.rpc.QuotaFailure','violations':[{'quotaDimensions':{'model':'m'},'quotaId':'GenerateRequestsPerDayPerModel'}]}]}},'model','daily_quota'),
 ({'error':{'details':[{'@type':'type.googleapis.com/google.rpc.QuotaFailure','violations':[{'quotaDimensions':{'project':'p'},'quotaId':'RequestsPerMinute'}]}]}},'provider','requests_per_minute')])
def test_quota_kind_and_scope_require_evidence(body,scope,kind):
    result=normalize_provider_error(SimpleNamespace(code=429,body=body))
    assert result.signal.scope==scope and result.signal.kind==kind


def test_retryinfo_seconds_nanos_and_safe_request_id():
    result=normalize_provider_error(SimpleNamespace(code='429',headers={'x-request-id':'safe-request-1','Authorization':'SECRET'},details={'error':{'details':[{'@type':'google.rpc.RetryInfo','retryDelay':{'seconds':'7','nanos':500000000}}]}}))
    assert result.retry_after==7.5 and result.signal.request_id=='safe-request-1'
    assert 'SECRET' not in json.dumps(failure_from_exception(result).diagnostic())


def test_resource_exhausted_sdk_class():
    class ResourceExhausted(Exception):pass
    assert isinstance(normalize_provider_error(ResourceExhausted()),ProviderRateLimitError)


@pytest.mark.parametrize('value,expected',[('1m2s',62),('250ms',.25),('1.5s',1.5),('nan',None),('SECRET',None)])
def test_reset_durations(value,expected):assert seconds(value)==expected


def test_transport_dns_tls_connect():
    import socket,ssl
    for error,stage in [(socket.gaierror(-2,'SECRET'),'dns'),(ssl.SSLCertVerificationError('SECRET'),'tls'),(ConnectionRefusedError('SECRET'),'connect')]:
        failure=failure_from_exception(normalize_provider_error(error))
        assert failure.category=='network' and failure.transport_stage==stage
        assert 'SECRET' not in json.dumps(failure.diagnostic())


def test_no_auth_retry_no_model_fallback():
    def fail(r):raise ProviderAuthenticationError('safe')
    p=GovernedProvider(Provider(fail),key='key',governor=ProviderRequestGovernor())
    request=ProviderRequest('x',model='actual')
    with pytest.raises(ProviderAuthenticationError):p.generate(request)
    assert p.delegate.calls==[request]


def test_active_transport_time_is_not_quota_wait_time():
    now=[0.0];calls=[];events=[]
    g=ProviderRequestGovernor(clock=lambda:now[0],max_wait=2,jitter=lambda a,b:0)
    def action(request):
        calls.append(request);now[0]+=100
        if len(calls)==1:raise ProviderRateLimitError("safe",retry_after=1)
        return ProviderResponse("OK","test")
    def observe(event,data):
        events.append(event)
        if event=="retry.scheduled":now[0]+=1
    p=GovernedProvider(Provider(action),key="key",governor=g)
    request=ProviderRequest("x",model="actual",timeout=10)
    with traffic_context(observer=observe):assert p.generate(request).content=="OK"
    assert calls==[request,request] and 'retry.scheduled' in events


def test_bounded_retry_and_no_cooldown_busy_loop():
    now=[0.0];g=ProviderRequestGovernor(clock=lambda:now[0],max_wait=20,jitter=lambda a,b:0)
    def action(request):raise ProviderRateLimitError("safe",retry_after=1)
    p=GovernedProvider(Provider(action),key="key",governor=g)
    def observe(event,data):
        if event=='retry.scheduled':now[0]+=1
    with traffic_context(observer=observe):
        with pytest.raises(ProviderRateLimitError):p.generate(ProviderRequest("x"))
    assert len(p.delegate.calls)==3


def test_zero_known_tokens_blocks_unknown_request_size():
    now=[0.0];g=ProviderRequestGovernor(clock=lambda:now[0],max_wait=2)
    g.observe('key','m',ProviderResponse('x','test',metadata={'traffic_headers':{'x-ratelimit-remaining-tokens':'0','x-ratelimit-reset-tokens':'1m'}}))
    p=GovernedProvider(Provider(),key='key',governor=g)
    with pytest.raises(ProviderRateLimitError):p.generate(ProviderRequest('x'))
    assert not p.delegate.calls


def test_queue_bound_is_enforced():
    g=ProviderRequestGovernor(max_queue=1)
    g._queue.append(('ticket','key','m'))
    from agenttree.providers.exceptions import ProviderConfigurationError
    with pytest.raises(ProviderConfigurationError):
        with g.admission('key','m'):pass


def test_exhausted_remaining_header_reset_is_not_shortened():
    result=normalize_provider_error(SimpleNamespace(code=429,headers={
        "Retry-After":"2","x-ratelimit-remaining-tokens":"0",
        "x-ratelimit-reset-tokens":"1m","x-ratelimit-limit-tokens":"8000"}))
    assert result.retry_after==60 and result.signal.limits['tokens']==8000
    assert result.signal.remaining['tokens']==0 and result.signal.resets['tokens']==60
    assert result.signal.kind=='unknown' and result.signal.scope=='unknown'
