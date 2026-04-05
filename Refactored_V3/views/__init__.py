"""
views/ — Tkinter presentation layer.

Each module in this package is a self-contained View component that
binds to a ViewModel.  Views import tkinter and ttk freely, but must
NEVER contain business logic.  All user actions are delegated to the
ViewModel via Command objects.
"""
