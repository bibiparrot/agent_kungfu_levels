"""Executable JSON-RPC teaching boundary, in-process transport (not A2A/MCP)."""
import json
from dataclasses import dataclass, field

from ..revenue_workspace import RevenueAnalysisWorkspace


@dataclass
class RevenueRpc:
    workspace: RevenueAnalysisWorkspace
    requested_n: int
    seen: set[str] = field(default_factory=set)
    exchanges: list[dict] = field(default_factory=list)

    def handle(self, wire: str) -> str:
        request_id = None
        code = -32700
        try:
            request = json.loads(wire)
            code = -32600
            if not isinstance(request, dict):
                raise ValueError("Request must be an object")
            request_id = request.get("id")
            if (request.get("jsonrpc") != "2.0" or not isinstance(request_id, str)
                    or not request_id or request_id in self.seen):
                raise ValueError("Invalid or replayed request id")
            self.seen.add(request_id)
            method, params = request.get("method"), request.get("params", {})
            code = -32601
            if method not in {"read_tables", "calculate_revenue"}:
                raise ValueError("Method not allowed")
            code = -32602
            expected = {} if method == "read_tables" else {"n": self.requested_n}
            if params != expected or (method == "calculate_revenue" and type(params.get("n")) is not int):
                raise ValueError("Arguments violate the task contract")
            code = -32000
            result = (self.workspace.load_tables() if method == "read_tables"
                      else self.workspace.calculate(self.requested_n))
            response = {"jsonrpc": "2.0", "id": request_id, "result": json.loads(result)}
        except (ValueError, TypeError, RuntimeError) as exc:
            response = {"jsonrpc": "2.0", "id": request_id,
                        "error": {"code": code, "message": str(exc)}}
        encoded = json.dumps(response, ensure_ascii=False)
        self.exchanges.append({"request": wire, "response": encoded})
        return encoded

    def call(self, method: str, n: int | None = None) -> str:
        request_id = str(len(self.exchanges) + 1)
        response = json.loads(self.handle(json.dumps({
            "jsonrpc": "2.0", "id": request_id, "method": method,
            "params": {} if n is None else {"n": n},
        })))
        if response["id"] != request_id or "error" in response:
            raise RuntimeError(f"RPC call rejected: {response}")
        return json.dumps(response["result"], ensure_ascii=False)
