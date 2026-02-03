from functools import wraps

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError


def check_if_already_connected(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if getattr(self, "is_connected", False):
            raise DeviceAlreadyConnectedError(
                f"{self.__class__.__name__} is already connected. Run `.disconnect()` first."
            )
        return func(self, *args, **kwargs)

    return wrapper


def check_if_not_connected(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not getattr(self, "is_connected", False):
            raise DeviceNotConnectedError(
                f"{self.__class__.__name__} is not connected. Run `.connect()` first."
            )
        return func(self, *args, **kwargs)

    return wrapper
