"""State-aware guard for actions sent to an interactive environment."""

from dataclasses import dataclass
from typing import Mapping


def _normalize(value):
  value = getattr(value, "value", value)
  return str(value).replace("_", "-")


@dataclass(frozen=True)
class ActionDecision:
  """Result of checking whether an action may be sent to a device."""

  requested: str
  state: str
  action: str
  allowed: bool
  reason: str


class ActionGuard:
  """Map recognized page states to safe, optionally confirmed actions.

  ``routes`` is intentionally supplied by the caller. This keeps screen
  detection and device-specific coordinates outside the interaction layer.
  For example, ``{"battle": {"home": "battle"}}`` allows the battle
  action only after the caller reports the ``home`` state.
  """

  def __init__(self, routes: Mapping[str, Mapping[str, str]], confirmation_required=True):
    self.routes = {
      _normalize(requested): {
        _normalize(state): action for state, action in states.items()
      }
      for requested, states in routes.items()
    }
    self.confirmation_required = confirmation_required

  def decide(self, requested, state, confirmed=False, force=False):
    """Return a decision without sending any input to the environment."""
    requested = _normalize(requested)
    state = _normalize(state)
    action = self.routes.get(requested, {}).get(state)

    if force:
      action = requested if action is None else action
      return ActionDecision(requested, state, action, True, "force enabled")

    if action is None:
      return ActionDecision(
        requested,
        state,
        "none",
        False,
        f"{requested} is not allowed in state {state}",
      )

    if self.confirmation_required and not confirmed:
      return ActionDecision(
        requested,
        state,
        action,
        False,
        f"{action} is suggested, but confirmation was not provided",
      )

    return ActionDecision(
      requested,
      state,
      action,
      True,
      f"{state} detected; {action} is allowed",
    )
