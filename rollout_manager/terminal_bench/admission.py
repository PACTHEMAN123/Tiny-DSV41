"""Existing relay admission gates, without any episode or cohort scheduler."""
import os,stat,math,time,fcntl
from pathlib import Path
from contextlib import contextmanager
from . import relay_contract as episode
MAX_CONCURRENCY=64
MAX_MODEL_SEQUENCES=1024
LEGACY_MODEL_CAPACITY=4

def read_json(path, limit=2 * 1024 * 1024):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise episode.EpisodeError('Invalid or oversized JSON file: ' + str(path))
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise episode.EpisodeError('JSON file exceeded its size budget')
    return episode.strict_json(data)


def proc_identity(pid):
    data = Path('/proc/%d/stat' % pid).read_text()
    fields = data.rsplit(')', 1)[1].split()
    return int(fields[19]), fields[0]


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def check_readiness(root, model, concurrency, required_seconds, now=None, identity_reader=proc_identity,
                    required_generation_seconds=episode.DEFAULT_GENERATION_SECONDS,
                    serving_profile="vllm-v1", expected_policy=None):
    if type(concurrency) is not int or not 1 <= concurrency <= MAX_CONCURRENCY:
        raise episode.EpisodeError('Requested concurrency must be 1..%d' % MAX_CONCURRENCY)
    required_generation_seconds = episode.validate_generation_seconds(required_generation_seconds)
    now = time.time() if now is None else now
    root = Path(root)
    owner = episode.validate_private_directory(root)
    path = root / 'readiness.json'
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) & 0o077
            or (info.st_uid, info.st_gid) != owner):
        raise episode.EpisodeError('Relay readiness must be a private same-owner regular file')
    value = read_json(path, 65536)
    if value.get('schema_version') != 1 or value.get('protocol') != 2:
        raise episode.EpisodeError('Qualified relay protocol-2 readiness is unavailable')
    paired, until = value.get('paired_unix'), value.get('valid_until_unix')
    if (not finite_number(paired) or not finite_number(until) or paired > now + 1
            or until <= now or until > paired + 15.01):
        raise episode.EpisodeError('Relay readiness is stale or has an invalid TTL')
    endpoints = []
    for role in ('cpu', 'gpu'):
        endpoint = value.get(role)
        if not isinstance(endpoint, dict) or endpoint.get('model') != model:
            raise episode.EpisodeError('Readiness model differs from scheduled model')
        if not isinstance(endpoint.get('boot_id'), str) or not endpoint['boot_id']:
            raise episode.EpisodeError('Missing relay process boot identity')
        cap, expiry = endpoint.get('max_inflight'), endpoint.get('expires_unix')
        if type(cap) is not int or not 1 <= cap <= 64 or cap < concurrency:
            raise episode.EpisodeError('Actual paired relay capacity is too small')
        generation_seconds = episode.validate_generation_seconds(
            endpoint.get('generation_seconds', episode.DEFAULT_GENERATION_SECONDS))
        if generation_seconds < required_generation_seconds:
            raise episode.EpisodeError('Actual paired relay generation budget is too small')
        if not finite_number(expiry) or expiry - now < required_seconds:
            raise episode.EpisodeError('Relay lifetime cannot cover another bounded episode')
        endpoints.append(endpoint)
    cpu, gpu = endpoints
    from .serving_contracts import TRAINING_PROFILES, NATIVE_PROFILE, validate_training_controller
    if serving_profile in TRAINING_PROFILES:
        if any(endpoint.get("serving_profile") != serving_profile for endpoint in endpoints):
            raise episode.EpisodeError('Paired endpoints do not share the explicit native serving profile')
        try:
            key = "native_controller" if serving_profile == NATIVE_PROFILE else "training_controller"
            proof = validate_training_controller(gpu.get(key), model, serving_profile=serving_profile, now=now,
                required_seconds=required_seconds, expected_policy=expected_policy)
        except (ValueError, KeyError, TypeError) as error:
            raise episode.EpisodeError('Native controller readiness is unqualified: ' + str(error)) from error
        if concurrency > proof["generation_slots"] or any(endpoint["max_inflight"] != 1 for endpoint in endpoints):
            raise episode.EpisodeError('Native serving requires actual serial capacity and one resource wave')
        if "model_max_num_seqs" in gpu or "model_capacity_source" in gpu:
            raise episode.EpisodeError('Native controller must not claim vLLM capacity')
    elif serving_profile != "vllm-v1" or {"native_controller", "training_controller"} & set(gpu):
        raise episode.EpisodeError('Unsupported or mixed serving readiness profile')
    model_capacity = gpu.get('model_max_num_seqs')
    capacity_source = gpu.get('model_capacity_source')
    if serving_profile in TRAINING_PROFILES:
        pass
    elif 'model_max_num_seqs' not in gpu and 'model_capacity_source' not in gpu:
        if concurrency > LEGACY_MODEL_CAPACITY:
            raise episode.EpisodeError('Higher concurrency requires live vLLM model capacity attestation')
    elif (type(model_capacity) is not int or not concurrency <= model_capacity <= MAX_MODEL_SEQUENCES
          or capacity_source not in ('vllm_server_info', 'owned_vllm_process')):
        raise episode.EpisodeError('Actual vLLM model capacity is unqualified or too small')
    if type(gpu.get('uncertain')) is not int or gpu['uncertain'] != 0:
        raise episode.EpisodeError('GPU has unknown or uncertain generation reservations')
    if type(cpu.get('pid')) is not int or cpu['pid'] <= 0:
        raise episode.EpisodeError('Invalid CPU relay PID')
    observed_start, process_state = identity_reader(cpu['pid'])
    if (type(cpu.get('pid_start_ticks')) is not int or observed_start != cpu['pid_start_ticks']
            or process_state in ('Z', 'X')):
        raise episode.EpisodeError('CPU relay process identity is stale')
    return value


def check_empty_spool(root):
    root = Path(root)
    for name in ('inbox', 'inflight', 'outbox'):
        episode.validate_private_directory(root / name)
    pending = list((root / 'inbox').glob('*.json'))
    pending += [path for path in (root / 'inflight').glob('*.json')
                if not (root / 'outbox' / path.name).is_file()]
    if pending:
        raise episode.EpisodeError('Existing unfinished relay work; no automatic retry or journal deletion')


@contextmanager
def lock_experiment(root):
    root = Path(root)
    owner = episode.validate_private_directory(root)
    path = root / 'experiment.lock'
    created = False
    try:
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        created = True
    except FileExistsError:
        fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
    try:
        if created:
            episode.align_new_request_owner(fd, owner)
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) & 0o077
                or (info.st_uid, info.st_gid) != owner):
            raise episode.EpisodeError('Invalid existing experiment lock')
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise episode.EpisodeError('Another experiment owns this relay state') from error
        yield
    finally:
        os.close(fd)
