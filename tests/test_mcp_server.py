"""MCP server over stdio, driven like a client would (subprocess, JSON-RPC lines)."""
import base64
import json
import subprocess
import sys

import pytest
import tifffile

from synth import PX_PER_MM, make_scan

PPI = PX_PER_MM * 25.4


class Client:
    def __init__(self):
        self.p = subprocess.Popen([sys.executable, "-m", "polaroid_processor.mcp_server"], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.next_id = 0

    def request(self, method, params=None):
        self.next_id += 1
        msg = {"jsonrpc": "2.0", "id": self.next_id, "method": method, "params": params or {}}
        self.p.stdin.write((json.dumps(msg) + "\n").encode())
        self.p.stdin.flush()
        notes = []
        while True:
            line = self.p.stdout.readline()
            assert line, self.p.stderr.read().decode(errors="replace")
            reply = json.loads(line)  # every stdout line must be protocol JSON
            if reply.get("id") == self.next_id:
                return reply, notes
            notes.append(reply)

    def notify(self, method):
        self.p.stdin.write((json.dumps({"jsonrpc": "2.0", "method": method}) + "\n").encode())
        self.p.stdin.flush()

    def call(self, name, **arguments):
        reply, notes = self.request("tools/call", {"name": name, "arguments": arguments,
                                                   "_meta": {"progressToken": "t1"}})
        return reply["result"], notes

    def close(self):
        self.p.stdin.close()
        self.p.wait(timeout=30)
        for s in (self.p.stdout, self.p.stderr):
            s.close()


@pytest.fixture(scope="module")
def client():
    c = Client()
    reply, _ = c.request("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                        "clientInfo": {"name": "test", "version": "0"}})
    assert reply["result"]["protocolVersion"] == "2025-06-18"
    assert "tools" in reply["result"]["capabilities"]
    c.notify("notifications/initialized")
    yield c
    c.close()


@pytest.fixture(scope="module")
def scans(tmp_path_factory):
    d = tmp_path_factory.mktemp("mcp_scans")
    for i, frame in enumerate(["white", "gold"]):
        rgb, _ = make_scan(frame, 0.3 * i, seed=30 + i)
        tifffile.imwrite(str(d / f"m_{frame}.tif"), rgb, photometric="rgb")
    return d


def test_tools_list(client):
    reply, _ = client.request("tools/list")
    names = {t["name"] for t in reply["result"]["tools"]}
    assert names == {"polaroid_process", "polaroid_detect", "polaroid_inspect", "polaroid_fix", "polaroid_status"}
    for t in reply["result"]["tools"]:
        assert t["inputSchema"]["type"] == "object"


def test_process_inspect_fix_status(client, scans, tmp_path):
    out = str(tmp_path / "out")
    res, notes = client.call("polaroid_process", inputs=[str(scans)], output=out, ppi=PPI,
                             derivatives=["web=400:jpg:80"], workers=1)
    assert not res["isError"], res
    data = res["structuredContent"]
    assert data["summary"]["scans"] == 2 and not data["summary"]["failed"]
    assert [n["params"]["progress"] for n in notes if n.get("method") == "notifications/progress"] == [1, 2]

    res, _ = client.call("polaroid_inspect", scan="m_gold", output=out)
    assert not res["isError"], res
    kinds = [c["type"] for c in res["content"]]
    assert kinds == ["text", "image"]
    jpg = base64.b64decode(res["content"][1]["data"])
    assert jpg[:2] == b"\xff\xd8"
    assert set(res["structuredContent"]["corners"]) == {"tl", "tr", "br", "bl"}

    res, _ = client.call("polaroid_fix", output=out, scan="m_gold", shift={"bottom": -3}, note="test")
    assert not res["isError"], res
    rec = res["structuredContent"]
    assert rec["quad_source"] == "override" and rec["exported"]
    assert rec["history"][-1]["author"] == "agent"

    res, _ = client.call("polaroid_status", output=out, needs_review=False)
    assert res["structuredContent"]["summary"]["scans"] == 2


def test_errors_are_tool_results(client, tmp_path):
    res, _ = client.call("polaroid_status", output=str(tmp_path / "nothing"))
    assert res["isError"] and "no manifest" in res["content"][0]["text"]
    res, _ = client.call("polaroid_fix", output=str(tmp_path), scan="x", shift={"middle": 3})
    assert res["isError"]
    reply, _ = client.request("tools/call", {"name": "nope", "arguments": {}})
    assert reply["error"]["code"] == -32602
    reply, _ = client.request("does/not/exist")
    assert reply["error"]["code"] == -32601
