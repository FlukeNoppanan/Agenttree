"""Safe, SDK-neutral traffic diagnostics; never retain raw provider payloads."""
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import math
import re


@dataclass(frozen=True)
class RateLimitSignal:
    kind: str = "unknown"
    scope: str = "unknown"
    retry_after: float | None = None
    quota_exhausted: bool = False
    request_id: str | None = None
    http_status: int | None = None
    limits: dict = field(default_factory=dict)
    remaining: dict = field(default_factory=dict)
    resets: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderFailure:
    category: str
    retryable: bool = False
    http_status: int | None = None
    rate_limit: RateLimitSignal | None = None
    transport_stage: str | None = None

    def diagnostic(self):
        return asdict(self)


def seconds(value):
    """Numeric seconds, HTTP date or vendor reset duration, bounded by finiteness."""
    if isinstance(value, (int, float, str)):
        try:
            number = float(value)
            if math.isfinite(number):
                return max(0.0, number)
        except (ValueError, OverflowError):
            pass
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r'(?:(\d+(?:\.\d+)?)h)?(?:(\d+(?:\.\d+)?)m)?(?:(\d+(?:\.\d+)?)s)?(?:(\d+(?:\.\d+)?)ms)?', value.strip())
    if match and any(match.groups()):
        return sum(float(v or 0) * scale for v, scale in zip(match.groups(), (3600, 60, 1, .001)))
    try:
        return max(0.0, (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds())
    except (ValueError, TypeError, OverflowError):
        return None


def safe_headers(headers):
    allowed = {'retry-after', 'x-request-id', 'request-id', 'x-goog-request-id'}
    allowed |= {f'x-ratelimit-{part}-{unit}' for part in ('limit','remaining','reset') for unit in ('requests','tokens')}
    if not hasattr(headers, 'items'):
        return {}
    return {str(k).lower(): str(v)[:160] for k,v in headers.items() if str(k).lower() in allowed}


def rate_limit_signal(error, status=None):
    headers = safe_headers(getattr(error, 'headers', None) or getattr(getattr(error,'response',None),'headers',None))
    delay = seconds(headers.get('retry-after'))
    payload = getattr(error, 'body', None)
    if not isinstance(payload, dict):
        payload = getattr(error, 'details', None)
    if not isinstance(payload, (dict,list)):
        response = getattr(error, 'response', None)
        try:
            if callable(getattr(response,'json',None)): payload = response.json()
            elif callable(getattr(error,'read',None)): payload = json.loads(error.read(65_537))
        except (ValueError, TypeError, OSError):
            payload = None
    root = payload.get('error',payload) if isinstance(payload,dict) else {}
    details = root.get('details',[]) if isinstance(root,dict) else []
    if isinstance(payload,list): details = payload
    violations=[]
    for item in details if isinstance(details,list) else []:
        if not isinstance(item,dict):continue
        if str(item.get('@type','')).endswith('RetryInfo'):
            rd = item.get('retryDelay'); d=seconds(rd)
            if isinstance(rd,dict):
                try: d=float(rd.get('seconds',0))+float(rd.get('nanos',0))/1e9
                except (ValueError,TypeError):d=None
            if d is not None and math.isfinite(d):delay=max(delay or 0,d)
        if str(item.get('@type','')).endswith('QuotaFailure'):
            violations.extend(v for v in item.get('violations',[]) if isinstance(v,dict))
    scoped=bool(violations) and all(isinstance(v.get('quotaDimensions'),dict) and v['quotaDimensions'].get('model') for v in violations)
    provider_scope=bool(violations) and all(isinstance(v.get('quotaDimensions'),dict) and v['quotaDimensions'].get('project') and not v['quotaDimensions'].get('model') for v in violations)
    ids=' '.join(str(v.get('quotaId',''))+' '+str(v.get('quotaMetric','')) for v in violations).lower()
    # Vendor error prose is read only to match an explicit quota unit; never exported.
    message = root.get('message','') if isinstance(root,dict) else ''
    text=ids+' '+(message[:4096].lower() if isinstance(message,str) else '')
    kinds=set()
    if 'tokensperminute' in text or 'tokens per minute' in text or re.search(r'\btpm\b',text):kinds.add('tokens_per_minute')
    if 'requestsperminute' in text or 'requests per minute' in text or re.search(r'\brpm\b',text):kinds.add('requests_per_minute')
    if 'perday' in text or 'per day' in text:kinds.add('daily_quota')
    kind=next(iter(kinds)) if len(kinds)==1 else 'unknown'
    request_id=next((headers[k] for k in ('x-request-id','request-id','x-goog-request-id') if k in headers),None)
    if request_id and not re.fullmatch(r'[A-Za-z0-9_.:-]{1,128}', request_id):request_id=None
    limits={};remaining={};resets={}
    for unit in ('requests','tokens'):
        for prefix,target in (('limit',limits),('remaining',remaining)):
            value=headers.get('x-ratelimit-'+prefix+'-'+unit)
            try:number=float(value)
            except (TypeError,ValueError):continue
            if math.isfinite(number) and number>=0:target[unit]=number
        reset=seconds(headers.get('x-ratelimit-reset-'+unit))
        if reset is not None:resets[unit]=reset
        if remaining.get(unit) == 0 and reset is not None:delay=max(delay or 0,reset)
    return RateLimitSignal(kind, 'model' if scoped else 'provider' if provider_scope else 'unknown', delay, 'daily_quota' in kinds, request_id, status, limits, remaining, resets)


def failure_from_exception(error):
    """Classify only structural evidence; inspect chained network causes safely."""
    from agenttree.providers.exceptions import (ProviderNetworkError,ProviderAuthenticationError,ProviderRateLimitError,ProviderTimeoutError,ProviderUnavailableError,ProviderInvalidRequestError,ProviderModelNotFoundError,MalformedProviderResponseError)
    current=error;visited=set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        diagnostic=getattr(current,"failure",None)
        if isinstance(diagnostic,ProviderFailure):return diagnostic
        if isinstance(current,ProviderRateLimitError):
            signal=getattr(current,'signal',None) or RateLimitSignal(scope=current.failure_scope,retry_after=current.retry_after,quota_exhausted=current.quota_exhausted)
            return ProviderFailure('quota_exhausted' if signal.quota_exhausted else 'rate_limit',not signal.quota_exhausted,signal.http_status,signal)
        if isinstance(current,ProviderAuthenticationError):return ProviderFailure('authentication',False)
        if isinstance(current,ProviderTimeoutError):return ProviderFailure('timeout',True)
        if isinstance(current,ProviderNetworkError):return ProviderFailure('network',current.stage!='tls',transport_stage=current.stage)
        if isinstance(current,ProviderUnavailableError):return ProviderFailure('capacity',True)
        if isinstance(current,ProviderModelNotFoundError):return ProviderFailure('model_unavailable',False)
        if isinstance(current,ProviderInvalidRequestError):return ProviderFailure('invalid_request',False)
        if isinstance(current,MalformedProviderResponseError):return ProviderFailure('malformed_response',False)
        name=type(current).__name__
        if name=='ExecutionCancelled':return ProviderFailure('cancelled',False)
        if name in ('gaierror',):return ProviderFailure('network',True,transport_stage='dns')
        if name in ('SSLError','SSLCertVerificationError'):return ProviderFailure('network',False,transport_stage='tls')
        if name=='ProxyError':return ProviderFailure('network',True,transport_stage='proxy')
        if name in ('ConnectError','ConnectionRefusedError','ConnectionError','URLError'):return ProviderFailure('network',True,transport_stage='connect')
        current=getattr(current,'__cause__',None) or getattr(current,'__context__',None)
    return ProviderFailure('unknown',False)
