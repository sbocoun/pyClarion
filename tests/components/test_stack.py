"""
Tests demonstrating how to create an Agent that uses a Stack.

A Stack models a last-in-first-out memory store. To embed one in an agent
you need to:

1. Define a Root that holds all required keyspace families and buses.
2. Register a StackOps sort (nil/stage/push/pop/flush) inside a DataFamily
   that belongs to the root, so the system can validate it.
3. Create the Agent context, then instantiate the Stack (and any Input
   processes that drive it) inside that context.
4. Wire the staging Input to stack.buffer.reader.input and the action Input
   to stack.controller.input.
5. Drive the stack by sending Chunk-style dimension-value pairs to the
   action Input using the StackOps atoms as values.

Stack operation semantics
-------------------------
push  – stages the current staging-input activations as a new chunk and
        adds it to the internal ChunkStore.
pop   – on first call: loads the top-ranked chunk (by base-level activation)
        into the buffer.  On a second consecutive call: removes the loaded
        chunk from the store and loads the next one.
flush – clears all chunks from the internal store.
stage – stages activations without committing them to the store.
"""

import unittest

from pyClarion import Agent, Stack, StackOps, Atom, Atoms, Input
from pyClarion.knowledge import Root, BusFamily, Buses, Bus, DataFamily


# ---------------------------------------------------------------------------
# Reusable keyspace definitions
# ---------------------------------------------------------------------------

class Color(Atoms):
    """Sample value sort: color terms."""
    red: Atom
    grn: Atom


class ActionBuses(Buses):
    """Buses used by the Stack.

    * action – carries control commands to the stack controller.
    * main   – carries staging data to the buffer's discriminal reader.
    * accum  – internal accumulation bus used by the buffer router.
    """
    action: Bus
    main: Bus
    accum: Bus


class MyBusFamily(BusFamily):
    buses: ActionBuses


class MyValues(DataFamily):
    """Value family.  StackOps will be registered here at runtime."""
    color: Color


class MyChunks(DataFamily):
    """Family that houses the stack's internal chunk sort."""


class MyParams(DataFamily):
    """Family for parameter and event sorts used by sub-components."""


class MyRoot(Root):
    b: MyBusFamily
    v: MyValues
    c: MyChunks
    p: MyParams


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _build_agent():
    """Return a fresh (agent, stack, staging, action, s, root) tuple."""
    root = MyRoot()
    s = StackOps()
    # StackOps must be registered inside a Family that belongs to root so
    # that system.check_root() can validate it.
    root.v["ops"] = s

    with Agent("agent", root) as agent:
        # Input that delivers feature activations to the buffer's reader.
        staging = Input("staging", (root.b.buses.main, root.v))
        # Input that delivers action commands to the stack controller.
        action = Input("action", (root.b.buses.action, s))

        stack = Stack(
            "stack",
            p=root.p,            # parameter family
            c=root.c,            # family for the stack's chunk sort
            a=root.b.buses.action,  # action bus → controller
            m=root.b.buses.main,    # staging bus  → buffer reader
            b=root.b.buses.accum,   # accumulation bus → buffer router
            v=root.v,            # value family
            s=s,                 # StackOps sort
        )
        # Connect external inputs to the internal buffer/controller sites.
        stack.buffer.reader.input = staging.main
        stack.controller.input = action.main

    return agent, stack, staging, action, s, root


def _non_nil_chunks(stack):
    """Return all non-nil chunks currently held in the stack store."""
    return [c for c in stack.chunks.c if str(c).split(":")[-1] != "nil"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class StackInitTestCase(unittest.TestCase):
    """Stack can be instantiated inside an Agent and starts idle."""

    def test_stack_initializes_with_nil_status(self):
        _, stack, _, _, s, _ = _build_agent()
        self.assertIs(stack.s, s)
        self.assertEqual(stack.current_status, ~s.nil)

    def test_stack_store_is_empty_on_creation(self):
        _, stack, _, _, _, _ = _build_agent()
        self.assertEqual(len(_non_nil_chunks(stack)), 0)

    def test_stack_is_accessible_from_top_level_import(self):
        # Stack and StackOps should be importable directly from pyClarion.
        from pyClarion import Stack as S, StackOps as SO  # noqa: F401


class StackPushTestCase(unittest.TestCase):
    """Push stages activations and adds a chunk to the store."""

    def setUp(self):
        (self.agent, self.stack, self.staging,
         self.action, self.s, self.root) = _build_agent()

    def _push(self, color_atom):
        s, root = self.s, self.root
        self.agent.system.schedule(
            self.staging.send(+ root.b.buses.main ** color_atom))
        self.agent.system.schedule(
            self.action.send(+ root.b.buses.action ** s.push))
        self.agent.run_all()

    def test_push_adds_one_chunk(self):
        self._push(self.root.v.color.red)
        self.assertEqual(len(_non_nil_chunks(self.stack)), 1)

    def test_push_returns_to_nil_status(self):
        self._push(self.root.v.color.red)
        self.assertEqual(self.stack.current_status, ~self.s.nil)

    def test_two_pushes_add_two_chunks(self):
        self._push(self.root.v.color.red)
        self._push(self.root.v.color.grn)
        self.assertEqual(len(_non_nil_chunks(self.stack)), 2)


class StackPopTestCase(unittest.TestCase):
    """Pop processes the action and always returns the stack to nil status."""

    def setUp(self):
        (self.agent, self.stack, self.staging,
         self.action, self.s, self.root) = _build_agent()
        # Pre-load two chunks; each push must complete before the next starts.
        for color in (self.root.v.color.red, self.root.v.color.grn):
            self.agent.system.schedule(
                self.staging.send(+ self.root.b.buses.main ** color))
            self.agent.system.schedule(
                self.action.send(+ self.root.b.buses.action ** self.s.push))
            self.agent.run_all()

    def _pop(self):
        self.agent.system.schedule(
            self.action.send(+ self.root.b.buses.action ** self.s.pop))
        self.agent.run_all()

    def test_pop_returns_to_nil_status(self):
        self._pop()
        self.assertEqual(self.stack.current_status, ~self.s.nil)

    def test_pop_does_not_add_chunks(self):
        count_before = len(_non_nil_chunks(self.stack))
        self._pop()
        self.assertLessEqual(len(_non_nil_chunks(self.stack)), count_before)

    def test_consecutive_pops_return_to_nil_status(self):
        self._pop()
        self._pop()
        self.assertEqual(self.stack.current_status, ~self.s.nil)


class StackFlushTestCase(unittest.TestCase):
    """Flush removes all chunks from the store."""

    def setUp(self):
        (self.agent, self.stack, self.staging,
         self.action, self.s, self.root) = _build_agent()
        for color in (self.root.v.color.red, self.root.v.color.grn):
            self.agent.system.schedule(
                self.staging.send(+ self.root.b.buses.main ** color))
            self.agent.system.schedule(
                self.action.send(+ self.root.b.buses.action ** self.s.push))
            self.agent.run_all()

    def test_flush_clears_all_chunks(self):
        self.agent.system.schedule(
            self.action.send(+ self.root.b.buses.action ** self.s.flush))
        self.agent.run_all()
        self.assertEqual(len(_non_nil_chunks(self.stack)), 0)

    def test_flush_returns_to_nil_status(self):
        self.agent.system.schedule(
            self.action.send(+ self.root.b.buses.action ** self.s.flush))
        self.agent.run_all()
        self.assertEqual(self.stack.current_status, ~self.s.nil)


if __name__ == "__main__":
    unittest.main()
