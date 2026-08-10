"""Restricted, timeout-bounded computation over the latest tool result."""

import ast
import contextlib
import io
import multiprocessing
import queue as queue_module
import traceback
from typing import Any

from .agent_state import AgentState
from .tool_result_schema import describe_tool_result


TIMEOUT_SECONDS = 15
MAX_CODE_LENGTH = 12_000
FORBIDDEN_NAMES = {
    "__import__", "open", "eval", "exec", "compile", "input", "globals",
    "locals", "vars", "getattr", "setattr", "delattr", "breakpoint",
    "help", "dir", "exit", "quit",
}
FORBIDDEN_ATTRIBUTES = {
    "system", "popen", "run", "call", "check_call", "check_output",
    "walk", "listdir", "remove", "unlink", "rename", "replace", "environ",
}


class PythonValidationError(ValueError):
    pass


def validate_python(code: str) -> ast.Module:
    """Reject syntax that can escape the computation-only sandbox."""
    if not isinstance(code, str) or not code.strip() or len(code) > MAX_CODE_LENGTH:
        raise PythonValidationError("Code is empty or exceeds the size limit.")

    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as error:
        raise PythonValidationError(f"Invalid Python syntax: {error.msg}") from error

    forbidden_nodes = (ast.Import, ast.ImportFrom, ast.With, ast.AsyncWith,
                       ast.While, ast.AsyncFor, ast.FunctionDef, ast.AsyncFunctionDef,
                       ast.ClassDef, ast.Lambda, ast.Try, ast.Raise, ast.Delete)
    for node in ast.walk(tree):
        if isinstance(node, forbidden_nodes):
            raise PythonValidationError(f"{type(node).__name__} is not allowed.")
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise PythonValidationError(f"{node.id} is not allowed.")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise PythonValidationError("Dunder attribute access is not allowed.")
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRIBUTES:
            raise PythonValidationError(f".{node.attr} is not allowed.")
    return tree


def _is_inline_safe(tree: ast.Module) -> bool:
    """Allow `exec` to run only small, non-iterative computations in-process.

    Windows process spawning can take several seconds in an API worker. Restricting
    this mode to a single `result = expression` keeps common arithmetic immediate;
    larger work remains isolated in the timeout-bounded subprocess mode.
    """
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign):
        return False
    assignment = tree.body[0]
    if len(assignment.targets) != 1 or not isinstance(assignment.targets[0], ast.Name):
        return False
    if assignment.targets[0].id != "result":
        return False
    disallowed = (ast.For, ast.AsyncFor, ast.ListComp, ast.SetComp, ast.DictComp,
                  ast.GeneratorExp, ast.NamedExpr, ast.Yield, ast.YieldFrom)
    return not any(isinstance(node, disallowed) for node in ast.walk(tree))


def _execute_code(code: str, last_tool_result: Any) -> dict[str, Any]:
    safe_builtins = {
        "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
        "enumerate": enumerate, "filter": filter, "float": float, "int": int,
        "len": len, "list": list, "max": max, "min": min, "range": range,
        "reversed": reversed, "round": round, "set": set, "sorted": sorted,
        "str": str, "sum": sum, "tuple": tuple, "zip": zip,
    }
    namespace = {"__builtins__": safe_builtins, "last_tool_result": last_tool_result}
    stdout, stderr = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exec(compile(code, "<node_python>", "exec"), namespace, namespace)
        return {
            "ok": True,
            "result": namespace.get("result"),
            "stdout": stdout.getvalue(),
            "stderr": stderr.getvalue(),
        }
    except Exception as error:  # Returned as a tool result, never raised into the graph.
        return {
            "ok": False,
            "error": str(error),
            "stdout": stdout.getvalue(),
            "stderr": stderr.getvalue(),
            "traceback": traceback.format_exc(limit=3),
        }


def _python_worker(code: str, last_tool_result: Any, output: multiprocessing.Queue) -> None:
    output.put(_execute_code(code, last_tool_result))


class NodePython:
    def search(self, state: AgentState):
        request = state.get("tool_request") or {}
        query = request.get("tool_query")
        code = request.get("tool_content", "")
        if query not in {"exec", "subprocess"}:
            result = {"ok": False, "error": "node_python only accepts exec or subprocess."}
            return self._result_update(result)

        try:
            tree = validate_python(code)
        except PythonValidationError as error:
            return self._result_update({"ok": False, "error": str(error)})

        if query == "exec" and _is_inline_safe(tree):
            return self._result_update(_execute_code(code, state.get("tool_results")))

        # `subprocess` and multi-statement exec requests are isolated so they
        # can be forcefully terminated if they exceed the execution deadline.
        queue: multiprocessing.Queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=_python_worker, args=(code, state.get("tool_results"), queue)
        )
        process.start()
        process.join(TIMEOUT_SECONDS)
        if process.is_alive():
            process.terminate()
            process.join()
            return self._result_update({"ok": False, "error": "Python execution timed out."})

        try:
            result = queue.get(timeout=1)
        except queue_module.Empty:
            result = {"ok": False, "error": "Python worker exited without a result."}
        return self._result_update(result)

    @staticmethod
    def _result_update(result: dict[str, Any]):
        return {
            "tool_results": result,
            "latest_tool_schema": describe_tool_result(result),
            "search_required": [False],
            "status_messages": ["generating"],
        }
