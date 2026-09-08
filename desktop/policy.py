from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum


class PrivilegeLevel(IntEnum):
    OBSERVE = 0
    USER = 1
    ELEVATED = 2
    ADMINISTRATOR = 3
    SYSTEM = 4


class InternetLevel(IntEnum):
    OFFLINE = 0
    LOCAL = 1
    STANDARD = 2
    BROAD = 3
    UNRESTRICTED = 4


class ConfirmationMode(str, Enum):
    NEVER = "never"
    RISK_BASED = "risk_based"
    ALWAYS = "always"


@dataclass
class DesktopPolicy:
    privilege_level: PrivilegeLevel = PrivilegeLevel.USER
    internet_level: InternetLevel = InternetLevel.STANDARD
    confirmation_mode: ConfirmationMode = ConfirmationMode.RISK_BASED

    allow_screen_capture: bool = True
    allow_app_launch: bool = True
    allow_shortcut_creation: bool = True
    allow_background_tasks: bool = True
    allow_process_execution: bool = True

    def can_use_privilege(self, required: PrivilegeLevel) -> bool:
        return self.privilege_level >= required

    def can_use_internet(self, required: InternetLevel) -> bool:
        return self.internet_level >= required
