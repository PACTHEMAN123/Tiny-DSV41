"""Browser relay admission prerequisites extracted from reviewed v7 client."""
import os,stat,math,json
from pathlib import Path
DEFAULT_GENERATION_SECONDS=120
MAX_GENERATION_SECONDS=600

class EpisodeError(RuntimeError):
    pass


def validate_generation_seconds(value):
    if type(value) is not int or not 1 <= value <= MAX_GENERATION_SECONDS:
        raise EpisodeError('Generation seconds must be an integer from 1 through 600')
    return value


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise EpisodeError('Duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(
                          EpisodeError('Non-finite JSON number: ' + value)))


def validate_private_directory(path):
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) & 0o077:
        raise EpisodeError('Relay directory must be an existing private real directory: ' + str(path))
    return info.st_uid, info.st_gid


def align_new_request_owner(fd, owner):
    """A sudo harness publishes private jobs readable by the existing relay user."""
    info = os.fstat(fd)
    if (info.st_uid, info.st_gid) != owner:
        os.fchown(fd, *owner)
