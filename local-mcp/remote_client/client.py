"""Async HTTP client for one Remote CUDA Agent."""

from __future__ import annotations

import base64
import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx

from remote_client.models import BuildConfig, ExecutionConfig, ProfilerConfig, SyncManifest


class RemoteClientError(RuntimeError):
    pass


class RemoteAgentUnavailable(RemoteClientError):
    pass


class RemoteRequestRejected(RemoteClientError):
    pass


class RemoteClient:
    def __init__(
        self,
        *,
        server_id: str,
        endpoint: str,
        token: str | None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.server_id = server_id
        self.endpoint = endpoint.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self.endpoint,
            headers={
                "User-Agent": "remote-cuda-local-mcp/2.0",
                "Accept": "application/json",
            }
            | ({"Authorization": f"Bearer {token}"} if token is not None else {}),
            timeout=httpx.Timeout(timeout_seconds, connect=min(10.0, timeout_seconds)),
            limits=httpx.Limits(max_connections=8, max_keepalive_connections=4),
            # One connection-establishment retry; complete HTTP requests are
            # never replayed automatically, so POST duplication is avoided.
            transport=httpx.AsyncHTTPTransport(retries=1),
        )

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: Any | None = None,
        params: dict[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        expected_status: set[int] | None = None,
    ) -> Any:
        expected = expected_status or {200, 201, 202, 204}
        try:
            response = await self._client.request(method, path, json=json, params=params, headers=headers)
        except httpx.HTTPError as exc:
            raise RemoteAgentUnavailable(f"cannot reach {self.server_id}: {exc}") from exc
        if response.status_code not in expected:
            detail = response.text[:1000]
            raise RemoteRequestRejected(
                f"{self.server_id} returned HTTP {response.status_code}: {detail}"
            )
        if response.status_code == 204 or not response.content:
            return None
        try:
            return response.json()
        except ValueError as exc:
            raise RemoteClientError(f"{self.server_id} returned invalid JSON") from exc

    async def health(self) -> dict[str, Any]:
        return await self._request("GET", "/health")

    async def capabilities(self) -> dict[str, Any]:
        return await self._request("GET", "/v1/capabilities")

    async def list_gpus(self) -> list[dict[str, Any]]:
        payload = await self._request("GET", "/v1/gpus")
        return payload.get("gpus", payload) if isinstance(payload, dict) else payload

    async def gpu_info(self, device_id: int) -> dict[str, Any]:
        return await self._request("GET", f"/v1/gpus/{device_id}")

    async def gpu_processes(self, device_id: int) -> list[dict[str, Any]]:
        payload = await self._request("GET", f"/v1/gpus/{device_id}/processes")
        return payload.get("processes", []) if isinstance(payload, dict) else []

    async def create_workspace(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._request("POST", "/v1/workspaces", json=payload)

    async def list_workspaces(self) -> list[dict[str, Any]]:
        payload = await self._request("GET", "/v1/workspaces")
        return payload.get("workspaces", []) if isinstance(payload, dict) else payload

    async def workspace_info(self, workspace_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/v1/workspaces/{workspace_id}")

    async def delete_workspace(self, workspace_id: str) -> None:
        await self._request("DELETE", f"/v1/workspaces/{workspace_id}", expected_status={200, 202, 204})

    async def sync_manifest(self, manifest: SyncManifest) -> dict[str, Any]:
        return await self._request(
            "POST", f"/v1/workspaces/{manifest.workspace_id}/sync/manifest", json=manifest.model_dump()
        )

    async def upload_file(self, workspace_id: str, path: str, local_file: Path) -> dict[str, Any]:
        content = local_file.read_bytes()
        payload = {
            "workspace_id": workspace_id,
            "path": path,
            "content_base64": base64.b64encode(content).decode("ascii"),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        return await self._request("PUT", f"/v1/workspaces/{workspace_id}/sync/file", json=payload)

    async def build(
        self, workspace_id: str, config: BuildConfig, idempotency_key: str | None = None
    ) -> dict[str, Any]:
        headers = {"Idempotency-Key": idempotency_key} if idempotency_key else None
        return await self._request(
            "POST",
            f"/v1/workspaces/{workspace_id}/build",
            json=config.model_dump(),
            headers=headers,
        )

    async def clean_build(self, workspace_id: str) -> dict[str, Any]:
        return await self._request("DELETE", f"/v1/workspaces/{workspace_id}/build")

    async def run(
        self, workspace_id: str, config: ExecutionConfig, idempotency_key: str | None = None
    ) -> dict[str, Any]:
        headers = {"Idempotency-Key": idempotency_key} if idempotency_key else None
        return await self._request(
            "POST",
            f"/v1/workspaces/{workspace_id}/run",
            json=config.model_dump(),
            headers=headers,
        )

    async def benchmark(self, workspace_id: str, config: Any, idempotency_key: str | None = None) -> dict[str, Any]:
        headers = {"Idempotency-Key": idempotency_key} if idempotency_key else None
        return await self._request(
            "POST",
            f"/v1/workspaces/{workspace_id}/benchmark",
            json=config.model_dump(),
            headers=headers,
        )

    async def profile(
        self,
        workspace_id: str,
        config: ProfilerConfig,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        headers = {"Idempotency-Key": idempotency_key} if idempotency_key else None
        return await self._request(
            "POST",
            f"/v1/workspaces/{workspace_id}/profile",
            json=config.model_dump(),
            headers=headers,
        )

    async def job(self, job_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/v1/jobs/{job_id}")

    async def job_logs(self, job_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/v1/jobs/{job_id}/logs")

    async def benchmark_result(self, job_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/v1/jobs/{job_id}/benchmark-result")

    async def profiler_result(self, job_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/v1/jobs/{job_id}/profiler-result")

    async def cancel_job(self, job_id: str) -> dict[str, Any]:
        return await self._request("POST", f"/v1/jobs/{job_id}/cancel")

    async def artifacts(self, workspace_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/v1/workspaces/{workspace_id}/artifacts")

    async def artifact_bytes(self, workspace_id: str, artifact_path: str) -> bytes:
        try:
            response = await self._client.get(
                f"/v1/workspaces/{workspace_id}/artifacts/download",
                params={"path": artifact_path},
            )
        except httpx.HTTPError as exc:
            raise RemoteAgentUnavailable(f"cannot download from {self.server_id}: {exc}") from exc
        if response.status_code != 200:
            raise RemoteRequestRejected(
                f"{self.server_id} download failed HTTP {response.status_code}: {response.text[:500]}"
            )
        return response.content

    async def artifact_stream(self, workspace_id: str, artifact_path: str):
        """Stream an artifact without loading the whole response into memory."""

        request = self._client.build_request(
            "GET",
            f"/v1/workspaces/{workspace_id}/artifacts/download",
            params={"path": artifact_path},
        )
        try:
            response = await self._client.send(request, stream=True)
        except httpx.HTTPError as exc:
            raise RemoteAgentUnavailable(f"cannot download from {self.server_id}: {exc}") from exc
        if response.status_code != 200:
            detail = (await response.aread()).decode("utf-8", "replace")[:500]
            await response.aclose()
            raise RemoteRequestRejected(
                f"{self.server_id} download failed HTTP {response.status_code}: {detail}"
            )
        try:
            async for chunk in response.aiter_bytes():
                yield chunk
        finally:
            await response.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()
