#!/usr/bin/env python3
"""Build DisOps image and deploy to GMI Cloud Compute (containers).

Requires a **Cluster Engine** API key (console.gmicloud.ai → API Keys).
The inference-engine key (GMI_API_KEY for GPT-5.5 / TTS) is different — pass
both via environment or .env.

Usage:
  # List IDCs and templates (verify cluster API key)
  python deploy/gmi_compute.py list

  # Build + tag locally (needs Docker)
  python deploy/gmi_compute.py build --tag ghcr.io/chetas1208/disops:latest

  # Register image as a GMI template
  python deploy/gmi_compute.py template --image ghcr.io/chetas1208/disops:latest

  # Launch container (CPU product — adjust via --product)
  python deploy/gmi_compute.py deploy --template-id <uuid> --name disops-live

Full one-shot (after image is pushed to a registry):
  python deploy/gmi_compute.py deploy --image ghcr.io/chetas1208/disops:latest
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE = "https://console.gmicloud.ai/api"
DEFAULT_IDC = os.environ.get("GMI_COMPUTE_IDC", "us-denver-1")
DEFAULT_PRODUCT = os.environ.get("GMI_COMPUTE_PRODUCT", "container.cpu.x1")
DEFAULT_PORT = int(os.environ.get("PORT", "8000"))


def _load_dotenv() -> None:
    env_path = ROOT / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _cluster_key() -> str:
    key = (
        os.environ.get("GMI_CLUSTER_API_KEY", "").strip()
        or os.environ.get("GMI_COMPUTE_API_KEY", "").strip()
    )
    if not key:
        raise RuntimeError(
            "GMI_CLUSTER_API_KEY is not set.\n"
            "Create a Cluster Engine API key at https://console.gmicloud.ai "
            "(Profile → API Keys). The inference GMI_API_KEY cannot manage containers."
        )
    return key


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {_cluster_key()}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _base() -> str:
    return os.environ.get("GMI_COMPUTE_API_URL", DEFAULT_BASE).rstrip("/")


def _pretty(data: object, limit: int = 4000) -> str:
    try:
        text = json.dumps(data, indent=2, ensure_ascii=False)
    except TypeError:
        text = repr(data)
    return text if len(text) <= limit else text[:limit] + "\n..."


def _request(method: str, path: str, **kwargs) -> requests.Response:
    url = f"{_base()}{path}"
    resp = requests.request(method, url, headers=_headers(), timeout=60, **kwargs)
    return resp


def cmd_list(_: argparse.Namespace) -> int:
    print("=== IDCs (public) ===")
    r = requests.get(f"{_base()}/v1/idcs", timeout=30)
    print(_pretty(r.json() if r.ok else {"error": r.text}))

    try:
        key = _cluster_key()
    except RuntimeError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1

    print("\n=== Templates ===")
    r = requests.get(f"{_base()}/v1/templates", headers=_headers(), timeout=30)
    if r.status_code == 401:
        print("401 Unauthorized — use GMI_CLUSTER_API_KEY (not inference GMI_API_KEY)")
        return 1
    print(_pretty(r.json() if r.ok else {"error": r.text}))

    print("\n=== Containers ===")
    r = requests.get(f"{_base()}/v1/containers", headers=_headers(), timeout=30)
    print(_pretty(r.json() if r.ok else {"error": r.text}))
    return 0 if r.ok else 1


def cmd_build(args: argparse.Namespace) -> int:
    tag = args.tag
    docker = args.docker_bin
    if not shutil_which(docker):
        print(f"Docker not found ({docker}). Install Docker or pass --docker-bin.", file=sys.stderr)
        return 1
    print(f"Building {tag} ...")
    subprocess.run(
        [docker, "build", "-t", tag, str(ROOT)],
        check=True,
    )
    print(f"Built {tag}")
    if args.push:
        print(f"Pushing {tag} ...")
        subprocess.run([docker, "push", tag], check=True)
        print("Push complete.")
    return 0


def shutil_which(cmd: str) -> str | None:
    from shutil import which

    return which(cmd)


def cmd_template(args: argparse.Namespace) -> int:
    payload = {
        "name": args.name,
        "path": args.image,
        "description": args.description or "DisOps live evacuation guidance (GPT-5.5 + GMI TTS)",
        "status": "published",
    }
    if args.registry_user:
        payload["credential"] = {
            "username": args.registry_user,
            "secret": args.registry_secret or "",
        }
    r = _request("POST", "/v1/templates", json=payload)
    print(f"POST /v1/templates → {r.status_code}")
    body = r.json() if r.content else {}
    print(_pretty(body))
    if not r.ok:
        return 1
    template_id = body.get("id")
    if template_id:
        print(f"\nTemplate ID: {template_id}")
        print(f"Export: export GMI_TEMPLATE_ID={template_id}")
    return 0


def _runtime_envs() -> list[dict[str, str]]:
    """Env vars injected into the running container."""
    passthrough = (
        "GMI_API_KEY",
        "GMI_LLM_ENDPOINT_URL",
        "GMI_VISION_MODEL",
        "GMI_REQUEST_TIMEOUT_S",
        "GMI_REASONING_MODEL",
        "GMI_MAX_COMPLETION_TOKENS",
        "GMI_TTS_MODEL",
        "GMI_TTS_VOICE_ID",
        "GMI_VIDEO_ENDPOINT_URL",
        "VOICE_COOLDOWN_S",
        "VOICE_STABLE_FRAMES",
        "HOST",
        "PORT",
    )
    envs: list[dict[str, str]] = [
        {"name": "VOICE_PLAYBACK", "value": "0"},
        {"name": "HOST", "value": "0.0.0.0"},
        {"name": "PORT", "value": str(DEFAULT_PORT)},
    ]
    for key in passthrough:
        val = os.environ.get(key, "").strip()
        if val:
            envs.append({"name": key, "value": val})
    if not any(e["name"] == "GMI_API_KEY" for e in envs):
        raise RuntimeError("GMI_API_KEY must be set in .env for the container runtime.")
    return envs


def cmd_deploy(args: argparse.Namespace) -> int:
    template_id = args.template_id or os.environ.get("GMI_TEMPLATE_ID", "").strip()

    if not template_id and args.image:
        print("Creating template from image ...")
        t_args = argparse.Namespace(
            name=args.template_name,
            image=args.image,
            description="DisOps",
            registry_user=args.registry_user,
            registry_secret=args.registry_secret,
        )
        r = _request(
            "POST",
            "/v1/templates",
            json={
                "name": t_args.name,
                "path": t_args.image,
                "description": t_args.description,
                "status": "published",
            },
        )
        if not r.ok:
            print(_pretty(r.json() if r.content else {"error": r.text}))
            return 1
        template_id = r.json().get("id")
        print(f"Template ID: {template_id}")

    if not template_id:
        raise RuntimeError("Pass --template-id or --image to create a template.")

    payload = {
        "name": args.name,
        "templateId": template_id,
        "count": 1,
        "idc": args.idc,
        "product": args.product,
        "envs": _runtime_envs(),
        "ports": [
            {
                "containerPort": DEFAULT_PORT,
                "port": args.public_port,
                "protocol": "TCP",
            }
        ],
    }
    if args.command:
        payload["command"] = args.command

    print("Creating container ...")
    print(_pretty(payload))
    r = _request("POST", "/v1/containers", json=payload)
    print(f"POST /v1/containers → {r.status_code}")
    body = r.json() if r.content else {}
    print(_pretty(body))
    if not r.ok:
        return 1

    container_ids = [item.get("id") for item in body if isinstance(item, dict)]
    if not container_ids and isinstance(body, dict):
        container_ids = [body.get("id")]
    container_ids = [cid for cid in container_ids if cid]

    if args.wait and container_ids:
        cid = container_ids[0]
        print(f"\nWaiting for container {cid} ...")
        for _ in range(args.wait_timeout):
            s = _request("GET", f"/v1/containers/{cid}")
            if s.ok:
                info = s.json()
                status = str(info.get("status", info.get("phase", ""))).lower()
                print(f"  status: {status}")
                if status in {"running", "active", "ready", "success"}:
                    print("\nContainer is up. Open the public URL from the GMI console (Elastic IP / port mapping).")
                    print(f"Health check: http://<public-ip>:{args.public_port}/api/health")
                    return 0
            time.sleep(5)
        print("Timed out waiting for container.", file=sys.stderr)

    return 0


def main() -> int:
    _load_dotenv()
    parser = argparse.ArgumentParser(description="Deploy DisOps to GMI Cloud Compute")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="List IDCs, templates, containers")
    p_list.set_defaults(func=cmd_list)

    p_build = sub.add_parser("build", help="Docker build (and optional push)")
    p_build.add_argument("--tag", default=os.environ.get("DISOPS_IMAGE", "ghcr.io/chetas1208/disops:latest"))
    p_build.add_argument("--push", action="store_true")
    p_build.add_argument("--docker-bin", default="docker")
    p_build.set_defaults(func=cmd_build)

    p_tpl = sub.add_parser("template", help="Register a container image as GMI template")
    p_tpl.add_argument("--image", required=True, help="Docker image URI, e.g. ghcr.io/org/disops:latest")
    p_tpl.add_argument("--name", default="disops-gpt55")
    p_tpl.add_argument("--description", default="")
    p_tpl.add_argument("--registry-user", default=os.environ.get("REGISTRY_USERNAME", ""))
    p_tpl.add_argument("--registry-secret", default=os.environ.get("REGISTRY_PASSWORD", ""))
    p_tpl.set_defaults(func=cmd_template)

    p_dep = sub.add_parser("deploy", help="Create GMI compute container")
    p_dep.add_argument("--name", default="disops-live")
    p_dep.add_argument("--template-id", default="")
    p_dep.add_argument("--image", default="", help="Create template from image if no template-id")
    p_dep.add_argument("--template-name", default="disops-gpt55")
    p_dep.add_argument("--idc", default=DEFAULT_IDC)
    p_dep.add_argument("--product", default=DEFAULT_PRODUCT)
    p_dep.add_argument("--public-port", type=int, default=DEFAULT_PORT)
    p_dep.add_argument("--command", default="")
    p_dep.add_argument("--registry-user", default="")
    p_dep.add_argument("--registry-secret", default="")
    p_dep.add_argument("--wait", action="store_true")
    p_dep.add_argument("--wait-timeout", type=int, default=60, help="Poll attempts (5s each)")
    p_dep.set_defaults(func=cmd_deploy)

    args = parser.parse_args()
    try:
        return args.func(args)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
