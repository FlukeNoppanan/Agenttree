"""Process-local Provider traffic control shared by runtime and qualification.

No routing or prompt mutation. FIFO admission among eligible scopes, cooperative
cancellation, learned budgets, bounded retry, and safe lifecycle diagnostics.
"""
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from hashlib import sha256
from random import uniform
from threading import Condition
from time import monotonic
from uuid import uuid4
import math

from agenttree.providers.base import BaseProvider
from agenttree.providers.traffic_failure import failure_from_exception, safe_headers, seconds
from agenttree.providers.exceptions import ProviderRateLimitError, ProviderConfigurationError, ProviderUnavailableError, ProviderTimeoutError


@dataclass
class _Scope:
    active: int = 0
    cooldown: float = 0
    signal: object = None
    failure: object = None
    # Remaining/reset observations are Provider-reported, never guessed limits.
    budgets: dict = field(default_factory=dict)
    updated: float = 0


_traffic_context = ContextVar('agenttree_traffic_context', default={})


@contextmanager
def traffic_context(**values):
    token = _traffic_context.set({"workload": "runtime", **_traffic_context.get(), **values})
    try:
        yield
    finally:
        _traffic_context.reset(token)


def _check():
    from agenttree.core.execution_control import check_execution
    check_execution()
    from agenttree.tools.runtime import _active_tool_cancellation
    cancellation = _active_tool_cancellation.get()
    if cancellation is not None and cancellation.is_set():
        from agenttree.core.execution_control import ExecutionCancelled
        raise ExecutionCancelled("Provider request cancelled")
    check = _traffic_context.get().get('check')
    if check is not None: check()


def _emit(event, **metadata):
    context = _traffic_context.get()
    data = {**metadata, **{k:context[k] for k in ('agent_id','strategy','workload','provider','traffic_request_id') if k in context}}
    callback = context.get('observer')
    if callback is not None: callback(event, data)
    from agenttree.core.operation_journal import current_journal
    journal = current_journal()
    if journal is not None:
        from agenttree.models import ExecutionEvent
        journal.store.append_event(journal.execution_id, ExecutionEvent(
            task_id=journal.execution_id, actor_id=context.get('agent_id'),
            event_type='provider.'+event, metadata=data))


class ProviderRequestGovernor:
    """One process, one shared queue. Unknown quota waits are conservatively
    coordinated at credential scope, while their diagnostic scope stays unknown.
    Model-proven cooldowns do not block unrelated models.
    """
    def __init__(self, *, concurrency=1, max_queue=256, max_scopes=1024,
                 retries=2, max_wait=45.0, clock=monotonic, jitter=uniform):
        if concurrency < 1 or max_queue < 1 or max_scopes < 1 or retries < 0 or max_wait < 0:
            raise ValueError('Invalid governor bounds')
        self.concurrency=concurrency; self.max_queue=max_queue;self.max_scopes=max_scopes
        self.retries=retries;self.max_wait=max_wait;self.clock=clock;self.jitter=jitter
        self._condition=Condition();self._scopes={};self._queue=deque()

    def _scope(self, key):
        if key not in self._scopes:
            if len(self._scopes)>=self.max_scopes:
                now=self.clock()
                stale=next((k for k,v in self._scopes.items() if not v.active and v.cooldown<=now and
                            not any(t[1]==k or (t[1],t[2])==k for t in self._queue) and
                            all(b[1]<=now for b in v.budgets.values())),None)
                if stale is None:raise ProviderConfigurationError('Provider scope capacity reached')
                del self._scopes[stale]
            self._scopes[key]=_Scope(updated=self.clock())
        return self._scopes[key]

    def _delay(self, scope, reserve):
        now=self.clock();delay=max(0,scope.cooldown-now)
        for unit,(remaining,reset) in list(scope.budgets.items()):
            if reset<=now:
                del scope.budgets[unit];continue
            required=1 if unit=='requests' else reserve
            if remaining <= 0 or (required and remaining<required):delay=max(delay,reset-now)
        return delay

    @contextmanager
    def admission(self, key, model, *, reserve=0, max_wait=None):
        ticket=(uuid4().hex,key,model);start=self.clock();waited=False;acquired=False
        bound=self.max_wait if max_wait is None else max_wait
        _check()
        with self._condition:
            if len(self._queue)>=self.max_queue:raise ProviderConfigurationError('Provider queue is full')
            self._queue.append(ticket)
        try:
            _emit('request.queued', model=model, queue_depth=len(self._queue))
            while True:
                _check()
                with self._condition:
                    base=self._scope(key);specific=self._scope((key,model))
                    delay=max(self._delay(base,reserve),self._delay(specific,reserve))
                    earlier=False
                    for prior in self._queue:
                        if prior==ticket:break
                        if prior[1]==key and self._delay(self._scope((key,prior[2])),0)<=0:
                            earlier=True;break
                    if not delay and not earlier and base.active<self.concurrency:
                        _check();self._queue.remove(ticket);base.active+=1;acquired=True
                        for scope in (base,specific):
                            for unit,(remaining,reset) in list(scope.budgets.items()):
                                scope.budgets[unit]=(max(0,remaining-(1 if unit=='requests' else reserve)),reset)
                        break
                    elapsed=self.clock()-start
                    if elapsed>=bound or delay>bound-elapsed:
                        _emit("wait.deferred",model=model,wait_seconds=round(delay,3))
                        if not delay:raise ProviderTimeoutError("Provider queue wait budget exceeded")
                        waiting_scope = specific if specific.cooldown > base.cooldown else base
                        if waiting_scope.failure is not None and waiting_scope.failure.category not in {"rate_limit", "quota_exhausted"}:
                            error = ProviderUnavailableError("Provider retry waiting; try later")
                            error.failure = waiting_scope.failure
                            error.traffic_governed = True
                            raise error
                        from dataclasses import replace
                        from agenttree.providers.traffic_failure import RateLimitSignal
                        signal = replace(waiting_scope.signal, retry_after=delay) if waiting_scope.signal is not None else RateLimitSignal(retry_after=delay)
                        error = ProviderRateLimitError('Provider traffic waiting; retry later',
                            retry_after=delay, failure_scope=signal.scope, quota_exhausted=signal.quota_exhausted, signal=signal)
                        error.traffic_governed = True
                        raise error
                if not waited:
                    waited=True;_emit('wait.started',model=model,reason='cooldown' if delay else 'queue',wait_seconds=round(delay,3))
                with self._condition:self._condition.wait(min(.2,max(.01,bound-(self.clock()-start))))
            if waited:_emit('wait.resumed',model=model,wait_seconds=round(self.clock()-start,3))
            yield self.clock()-start
        except BaseException:
            with self._condition:
                if ticket in self._queue:self._queue.remove(ticket)
                self._condition.notify_all()
            raise
        finally:
            if acquired:self.release(key)

    def release(self,key):
        with self._condition:
            self._scope(key).active-=1
            self._condition.notify_all()

    def cooldown(self,key,model,signal,attempt=0, failure=None):
        delay=signal.retry_after
        if delay is None:delay=min(30.0,2.0**(attempt+1))+self.jitter(0,.5)
        scope_key=(key,model) if signal.scope=='model' else key
        with self._condition:
            scope=self._scope(scope_key);scope.cooldown=max(scope.cooldown,self.clock()+delay)
            scope.signal=signal;scope.failure=failure
            self._condition.notify_all()
        return delay

    def observe(self,key,model,response,reserved=0):
        headers=safe_headers(response.metadata.get('traffic_headers',{}))
        with self._condition:
            # Header scope is unspecified; coordinate credential-wide without
            # claiming the vendor's quota scope. Model scope requires evidence.
            scope=self._scope(key);now=self.clock()
            for unit in ('requests','tokens'):
                remaining=headers.get('x-ratelimit-remaining-'+unit)
                reset=seconds(headers.get('x-ratelimit-reset-'+unit))
                try:value=float(remaining)
                except (TypeError,ValueError):continue
                if reset is not None and math.isfinite(value) and value>=0:
                    prior=scope.budgets.get(unit)
                    scope.budgets[unit]=(min(value,prior[0]),max(now+reset,prior[1])) if prior and prior[1]>now else (value,now+reset)
            usage=response.usage
            actual=getattr(usage,'total_tokens',None) if usage is not None else None
            if actual is not None and 'tokens' in scope.budgets and 'x-ratelimit-remaining-tokens' not in headers:
                remaining,reset=scope.budgets['tokens'];scope.budgets['tokens']=(max(0,remaining+reserved-actual),reset)
            self._condition.notify_all()
        _emit('usage.reconciled',model=model,total_tokens=actual,reported_usage=actual is not None)


shared_governor=ProviderRequestGovernor(concurrency=2)


class GovernedProvider(BaseProvider):
    """Adapter decorator. Same request/model, no fallback. Streaming is never
    replayed after visible output; quota slots live until the iterator closes.
    """
    traffic_managed = True

    def __init__(self,delegate,*,key=None,governor=None):
        super().__init__(delegate.config)
        self.delegate=delegate
        self.scope_key=key or sha256((delegate.provider_type+':'+delegate.name).encode()).hexdigest()
        self.governor=governor or shared_governor

    @property
    def provider_type(self):return self.delegate.provider_type
    @property
    def capabilities(self):return self.delegate.capabilities
    def list_models(self,**kwargs):return self.delegate.list_models(**kwargs)
    def validate_connection(self,**kwargs):return self.delegate.validate_connection(**kwargs)

    def _attempts(self,request):
        model=request.model or self.config.model
        qualifying=_traffic_context.get().get('workload')=='qualification' or request.metadata.get('purpose')=='studio_model_qualification'
        wait=5.0 if qualifying else self.governor.max_wait
        # Reserve the declared output bound or an explicit host total estimate.
        # Unknown input tokenizer cost is not represented as a measured value.
        reserve=request.metadata.get('estimated_total_tokens',request.max_tokens or self.config.max_tokens or 0)
        if not isinstance(reserve,int) or isinstance(reserve,bool) or reserve<0:reserve=0
        return model,wait,reserve

    def _generate(self,request):
        model,wait,reserve=self._attempts(request);waited_total=0.0
        for attempt in range(self.governor.retries+1):
            try:
                with self.governor.admission(self.scope_key,model,reserve=reserve,max_wait=max(0,wait-waited_total)) as waited:
                    waited_total+=waited
                    _emit('request.dispatched',model=model,attempt=attempt+1)
                    _check()
                    response=self.delegate.generate(request)
                    from agenttree.providers.models import ProviderResponse
                    if isinstance(response, ProviderResponse):
                        self.governor.observe(self.scope_key,model,response,reserve)
                    _check()
                    _emit('request.completed',model=model,attempt=attempt+1)
                    return response
            except ProviderRateLimitError as error:
                failure=failure_from_exception(error);signal=failure.rate_limit
                delay=self.governor.cooldown(self.scope_key,model,signal,attempt,failure)
                _emit('rate_limited',model=model,attempt=attempt+1,**failure.diagnostic())
                error.traffic_governed=True
                if signal.quota_exhausted or attempt>=self.governor.retries or delay>wait-waited_total:raise
                _emit('retry.scheduled',model=model,attempt=attempt+2,wait_seconds=delay)
            except (ProviderUnavailableError, ProviderTimeoutError) as error:
                failure=failure_from_exception(error)
                _emit('request.failed',model=model,**failure.diagnostic())
                error.traffic_governed=True
                if not failure.retryable:raise
                from agenttree.providers.traffic_failure import RateLimitSignal
                delay=self.governor.cooldown(self.scope_key,model,RateLimitSignal(),attempt,failure)
                if attempt>=self.governor.retries or delay>wait-waited_total:raise
                _emit('retry.scheduled',model=model,attempt=attempt+2,wait_seconds=delay)
            except Exception as error:
                _emit('request.failed',model=model,**failure_from_exception(error).diagnostic())
                raise

    def _generate_stream(self,request):
        model,wait,reserve=self._attempts(request);emitted=False;waited_total=0.0
        for attempt in range(self.governor.retries+1):
            try:
                with self.governor.admission(self.scope_key,model,reserve=reserve,max_wait=max(0,wait-waited_total)) as waited:
                    waited_total+=waited
                    _emit('request.dispatched',model=model,attempt=attempt+1,streaming=True)
                    _check()
                    for chunk in self.delegate.generate_stream(request):
                        if chunk.response is not None:self.governor.observe(self.scope_key,model,chunk.response,reserve)
                        _check();emitted=True
                        yield chunk
                    _emit('request.completed',model=model,streaming=True)
                    return
            except ProviderRateLimitError as error:
                failure=failure_from_exception(error);delay=self.governor.cooldown(self.scope_key,model,failure.rate_limit,attempt,failure)
                _emit('rate_limited',model=model,**failure.diagnostic())
                if emitted or failure.rate_limit.quota_exhausted or attempt>=self.governor.retries or delay>wait-waited_total:raise
                _emit('retry.scheduled',model=model,attempt=attempt+2,wait_seconds=delay)
            except Exception as error:
                _emit('request.failed',model=model,**failure_from_exception(error).diagnostic());raise


    def generate(self, request):
        with traffic_context(provider=self.provider_type, traffic_request_id=uuid4().hex):
            return self._generate(request)

    def generate_stream(self, request):
        with traffic_context(provider=self.provider_type, traffic_request_id=uuid4().hex):
            yield from self._generate_stream(request)


def governed(provider,*,key=None):
    return provider if getattr(provider,"traffic_managed",False) else GovernedProvider(provider,key=key)
