"""
viewmodels/observable.py
========================
Lightweight MVVM infrastructure: Observable properties and Commands.

Design principles
-----------------
* ``Observable`` — a descriptor / property factory that notifies registered
  listeners whenever a value changes.  Works on any class; no tkinter
  dependency.
* ``ObservableList`` — a list that fires ``on_change`` callbacks when mutated.
* ``Command`` — a callable wrapper with an optional ``can_execute`` predicate,
  matching the WPF / MVVM Command pattern.

The View layer (Tkinter) can subscribe to changes by registering callbacks,
rather than polling.  This keeps the ViewModel decoupled from the UI toolkit.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Generic, Iterable, Iterator, List, Optional, TypeVar

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Observable property
# ---------------------------------------------------------------------------


class ObservableProperty(Generic[T]):
    """
    A non-data descriptor that stores a value and notifies listeners when
    it changes.

    Usage (inside a ViewModel)::

        class MyVM:
            name = ObservableProperty("default_name")
            count = ObservableProperty(0)

        vm = MyVM()
        vm.subscribe("name", lambda old, new: print(f"name: {old} → {new}"))
        vm.name = "Alice"   # triggers callback

    Each ViewModel instance keeps its own value + listener dict under the
    ``_obs_values`` / ``_obs_listeners`` instance attributes, so multiple
    ViewModel instances are fully isolated.
    """

    def __init__(self, default: T = None) -> None:  # type: ignore[assignment]
        self._default = default
        self._attr: str = ""

    def __set_name__(self, owner, name: str) -> None:
        self._attr = name

    def __get__(self, obj, objtype=None) -> T:
        if obj is None:
            return self  # type: ignore[return-value]
        return obj._obs_values.get(self._attr, self._default)

    def __set__(self, obj, value: T) -> None:
        _ensure_obs_dicts(obj)
        old = obj._obs_values.get(self._attr, self._default)
        if old == value:
            return
        obj._obs_values[self._attr] = value
        for cb in list(obj._obs_listeners.get(self._attr, [])):
            try:
                cb(old, value)
            except Exception:
                pass


def _ensure_obs_dicts(obj: Any) -> None:
    if not hasattr(obj, "_obs_values"):
        object.__setattr__(obj, "_obs_values", {})
    if not hasattr(obj, "_obs_listeners"):
        object.__setattr__(obj, "_obs_listeners", {})


class ObservableMixin:
    """
    Mix this into any class that uses :class:`ObservableProperty` descriptors
    to get the ``subscribe`` / ``unsubscribe`` helpers.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

    def subscribe(
        self,
        prop_name: str,
        callback: Callable[[Any, Any], None],
    ) -> None:
        """Register *callback* to be invoked when *prop_name* changes."""
        _ensure_obs_dicts(self)
        self._obs_listeners.setdefault(prop_name, []).append(callback)  # type: ignore[attr-defined]

    def unsubscribe(
        self,
        prop_name: str,
        callback: Callable[[Any, Any], None],
    ) -> None:
        """Remove a previously registered *callback* for *prop_name*."""
        _ensure_obs_dicts(self)
        listeners = self._obs_listeners.get(prop_name, [])  # type: ignore[attr-defined]
        if callback in listeners:
            listeners.remove(callback)

    def notify(self, prop_name: str, old: Any, new: Any) -> None:
        """Manually fire all listeners for *prop_name*."""
        _ensure_obs_dicts(self)
        for cb in list(self._obs_listeners.get(prop_name, [])):  # type: ignore[attr-defined]
            try:
                cb(old, new)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Observable list
# ---------------------------------------------------------------------------


class ObservableList(List[T]):
    """
    A list subclass that fires ``on_change()`` whenever it is mutated.

    Assign ``on_change`` to any zero-argument callable to react to mutations.
    """

    def __init__(self, *args, **kwargs) -> None:  # type: ignore[override]
        super().__init__(*args, **kwargs)
        self.on_change: Callable[[], None] = lambda: None

    def _fire(self) -> None:
        try:
            self.on_change()
        except Exception:
            pass

    def append(self, item: T) -> None:  # type: ignore[override]
        super().append(item)
        self._fire()

    def remove(self, item: T) -> None:  # type: ignore[override]
        super().remove(item)
        self._fire()

    def clear(self) -> None:
        super().clear()
        self._fire()

    def __setitem__(self, index, value) -> None:  # type: ignore[override]
        super().__setitem__(index, value)
        self._fire()

    def __delitem__(self, index) -> None:  # type: ignore[override]
        super().__delitem__(index)
        self._fire()


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------


class Command:
    """
    Encapsulates an action (execute) with an optional guard (can_execute).

    Matches the MVVM Command pattern.  The View binds a button to a
    ``Command``; the button calls ``command()`` and can check
    ``command.can_execute()`` to enable/disable itself.
    """

    def __init__(
        self,
        execute: Callable[..., Any],
        can_execute: Optional[Callable[[], bool]] = None,
    ) -> None:
        self._execute = execute
        self._can_execute = can_execute or (lambda: True)
        self._listeners: List[Callable[[], None]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.can_execute():
            return self._execute(*args, **kwargs)

    def can_execute(self) -> bool:
        try:
            return bool(self._can_execute())
        except Exception:
            return False

    def raise_can_execute_changed(self) -> None:
        """Notify bound UI elements that ``can_execute`` may have changed."""
        for cb in list(self._listeners):
            try:
                cb()
            except Exception:
                pass

    def subscribe_can_execute_changed(self, callback: Callable[[], None]) -> None:
        self._listeners.append(callback)
