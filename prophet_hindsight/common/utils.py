import logging
from typing import Any

import json_repair

logger = logging.getLogger(__name__)


def unified_json_loads(x: Any, expected_type: type | tuple, raise_on_unknown: bool = False) -> Any:
    """
    Load a JSON string into a Python object.
    """
    if isinstance(x, str):
        return json_repair.loads(x)
    elif isinstance(x, expected_type):
        return x
    else:
        msg = f"Invalid input type: {type(x)}, expected {expected_type}"
        if raise_on_unknown:
            raise ValueError(msg)
        else:
            logger.warning(msg)
            return x
