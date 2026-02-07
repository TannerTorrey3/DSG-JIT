from __future__ import annotations

import json
import threading
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict, List, Tuple

import numpy as np

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>DSG-JIT SceneGraph Viewer</title>
  <script type="importmap">
  {
    "imports": {
      "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
      "three/examples/jsm/controls/OrbitControls.js":
        "https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js"
    }
  }
  </script>
  <style>
    body { margin: 0; overflow: hidden; background-color: #111; color: #eee; }
    #info {
      position: absolute;
      top: 10px;
      left: 10px;
      z-index: 10;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px;
      padding: 4px 8px;
      background: rgba(0, 0, 0, 0.45);
      border-radius: 4px;
    }
  </style>
</head>
<body>
  <div id="info">
    DSG-JIT SceneGraph (Three.js viewer)<br/>
    Mouse: rotate / pan / zoom &nbsp;|&nbsp;
    Keys: W/S forward-back, A/D strafe, Q/E up-down
  </div>
  <script type="module">
    import * as THREE from "https://unpkg.com/three@0.160.0/build/three.module.js";
    import { OrbitControls } from "https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js";

    const GRAPH_DATA = __GRAPH_JSON__;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a1a);

    const camera = new THREE.PerspectiveCamera(
      60,
      window.innerWidth / window.innerHeight,
      0.01,
      2000
    );
    camera.position.set(5, 5, 5);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0, 0);
    controls.update();

    // Brighter, more legible lighting setup
    scene.add(new THREE.AmbientLight(0x888888));
    const hemiLight = new THREE.HemisphereLight(0xffffff, 0x222222, 0.6);
    hemiLight.position.set(0, 20, 0);
    scene.add(hemiLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 0.9);
    dirLight.position.set(10, 15, 10);
    scene.add(dirLight);

    const nodes = GRAPH_DATA.nodes || [];
    const edges = GRAPH_DATA.edges || [];

    let moveStep = 0.5;

    const nodeMeshes = new Map();

    function createNodeMesh(node) {
      const pos = node.position || [0, 0, 0];
      const x = pos[0] || 0;
      const y = pos[1] || 0;
      const z = pos[2] || 0;

      let geom, color;
      switch (node.type) {
        case "room":
          geom = new THREE.BoxGeometry(0.8, 0.4, 0.8);
          color = 0x4caf50;
          break;
        case "place":
          // Larger, semitransparent sphere so poses/objects can sit inside.
          geom = new THREE.SphereGeometry(0.3, 24, 24);
          color = 0xffc107;
          break;
        case "object":
        case "voxel":
          geom = new THREE.SphereGeometry(0.1, 14, 14);
          color = 0xff5722;
          break;
        case "pose":
          geom = new THREE.SphereGeometry(0.12, 18, 18);
          color = 0x2196f3;
          break;
        default:
          geom = new THREE.SphereGeometry(0.1, 12, 12);
          color = 0x9e9e9e;
      }

      const matOptions = { color };
      if (node.type === "place") {
        matOptions.transparent = true;
        matOptions.opacity = 0.35;
      }

      const mat = new THREE.MeshStandardMaterial(matOptions);
      const mesh = new THREE.Mesh(geom, mat);
      mesh.position.set(x, y, z);
      mesh.userData = node;
      scene.add(mesh);
      return mesh;
    }

    nodes.forEach((node) => {
      const mesh = createNodeMesh(node);
      nodeMeshes.set(node.id, mesh);
    });

    edges.forEach((edge) => {
      const src = nodeMeshes.get(edge.source);
      const tgt = nodeMeshes.get(edge.target);
      if (!src || !tgt) return;

      let color = 0xaaaaaa;
      const rel = edge.relation || "";

      if (rel === "room-place") {
        color = 0x4caf50; // room–place hierarchy
      } else if (rel === "place-object") {
        color = 0xffc107; // place–object hierarchy
      } else if (rel === "pose-place") {
        color = 0x2196f3; // pose attachment
      } else if (rel.startsWith("factor:")) {
        color = 0xffffff; // factor-graph edges (e.g., odometry)
      }

      const edgeMaterial = new THREE.LineBasicMaterial({ color: color, linewidth: 1 });

      const points = [src.position.clone(), tgt.position.clone()];
      const geometry = new THREE.BufferGeometry().setFromPoints(points);
      const line = new THREE.Line(geometry, edgeMaterial);
      line.userData = edge;
      scene.add(line);
    });

    if (nodes.length > 0) {
      const box = new THREE.Box3();
      nodes.forEach((n) => {
        const pos = n.position || [0, 0, 0];
        box.expandByPoint(new THREE.Vector3(pos[0] || 0, pos[1] || 0, pos[2] || 0));
      });
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3()).length() || 1.0;
      moveStep = Math.max(size * 0.05, 0.1);
      camera.position.copy(center.clone().add(new THREE.Vector3(size, size, size)));
      controls.target.copy(center);
      controls.update();
    }

    function moveCamera(delta) {
      camera.position.add(delta);
      controls.target.add(delta);
      controls.update();
    }

    window.addEventListener("keydown", (event) => {
      const key = event.key.toLowerCase();
      const step = moveStep;
      const delta = new THREE.Vector3();

      // Forward / backward (W/S)
      if (key === "w" || key === "s") {
        const dir = new THREE.Vector3();
        camera.getWorldDirection(dir);
        dir.normalize();
        if (key === "w") {
          delta.add(dir.multiplyScalar(step));
        } else {
          delta.add(dir.multiplyScalar(-step));
        }
      }

      // Strafe left / right (A/D)
      if (key === "a" || key === "d") {
        const forward = new THREE.Vector3();
        camera.getWorldDirection(forward);
        forward.normalize();
        const right = new THREE.Vector3();
        right.crossVectors(forward, camera.up).normalize();
        if (key === "d") {
          delta.add(right.multiplyScalar(step));
        } else if (key === "a") {
          delta.add(right.multiplyScalar(-step));
        }
      }

      // Up / down (Q/E)
      if (key === "q") {
        delta.add(new THREE.Vector3(0, step, 0));
      } else if (key === "e") {
        delta.add(new THREE.Vector3(0, -step, 0));
      }

      if (delta.lengthSq() > 0) {
        event.preventDefault();
        moveCamera(delta);
      }
    });

    window.addEventListener("resize", () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    });

    function animate() {
      requestAnimationFrame(animate);
      renderer.render(scene, camera);
    }
    animate();
  </script>
</body>
</html>
"""

def export_scenegraph_to_threejs(sg: Any) -> Dict[str, Any]:
    """
    Convert a SceneGraphWorld into a JSON-serializable dict suitable
    for the Three.js web viewer.

    The returned dict has the form:
        {"nodes": [...], "edges": [...]}

    Nodes come from the SceneGraph memory layer (sg._memory), and edges
    from semantic relations (place_parents, object_parents, place_attachments).
    """
    mem = getattr(sg, "_memory", {}) or {}
    nodes: List[Dict[str, Any]] = []

    # Helper to map var_type to a coarse visualization type.
    def classify_type(var_type: str) -> str:
        vt = var_type or ""
        if vt.startswith("pose"):
            return "pose"
        if vt.startswith("room"):
            return "room"
        if vt.startswith("place"):
            return "place"
        if vt.startswith("object"):
            return "object"
        if vt.startswith("voxel"):
            return "voxel"
        return "other"

    # Iterate memory entries; assume each is a dataclass-like object
    # with node_id, var_type, value, and optional name.
    values_iter = getattr(mem, "values", lambda: [])()
    for state in values_iter:
        node_id = getattr(state, "node_id", None)
        if node_id is None:
            continue
        var_type = getattr(state, "var_type", "") or ""
        viz_type = classify_type(var_type)

        raw_val = getattr(state, "value", None)
        if raw_val is None:
            pos = [0.0, 0.0, 0.0]
        else:
            arr = np.asarray(raw_val).reshape(-1)
            if arr.size >= 3:
                pos = [float(arr[0]), float(arr[1]), float(arr[2])]
            elif arr.size == 2:
                pos = [float(arr[0]), float(arr[1]), 0.0]
            elif arr.size == 1:
                pos = [float(arr[0]), 0.0, 0.0]
            else:
                pos = [0.0, 0.0, 0.0]

        # Prefer a human-readable name if available; otherwise derive one.
        name = getattr(state, "name", None)
        if not name:
            name = f"{viz_type}_{node_id}"

        nodes.append(
            {
                "id": int(node_id),
                "type": viz_type,
                "label": str(name),
                "position": pos,
            }
        )

    edges: List[Dict[str, Any]] = []

    # 1) place_parents: place -> room
    place_parents = getattr(sg, "place_parents", {}) or {}
    for place_id, room_id in place_parents.items():
        edges.append(
            {
                "source": int(place_id),
                "target": int(room_id),
                "relation": "room-place",
            }
        )

    # 2) object_parents: object -> place
    object_parents = getattr(sg, "object_parents", {}) or {}
    for obj_id, place_id in object_parents.items():
        edges.append(
            {
                "source": int(obj_id),
                "target": int(place_id),
                "relation": "place-object",
            }
        )

    # 3) place_attachments: pose -> place
    attachments = getattr(sg, "place_attachments", []) or []
    for attachment in attachments:
        if not isinstance(attachment, (tuple, list)) or len(attachment) != 2:
            continue
        pose_id, place_id = attachment
        edges.append(
            {
                "source": int(pose_id),
                "target": int(place_id),
                "relation": "pose-place",
            }
        )

    # 4) Factor-graph edges from SceneGraph's persistent factor memory.
    factor_mem = getattr(sg, "_factor_memory", {}) or {}
    if factor_mem:
        for rec in factor_mem.values():
            # Skip inactive or degenerate records if those fields exist.
            if not getattr(rec, "active", True):
                continue
            var_ids = getattr(rec, "var_ids", None)
            if var_ids is None or len(var_ids) < 2:
                # Unary factors (e.g., priors) do not produce a visible edge here.
                continue
            src_id = int(var_ids[0])
            tgt_id = int(var_ids[1])
            rel = getattr(rec, "relation", None) or f"factor:{getattr(rec, 'f_type', 'unknown')}"
            edges.append(
                {
                    "source": src_id,
                    "target": tgt_id,
                    "relation": rel,
                }
            )

    return {"nodes": nodes, "edges": edges}

def run_scenegraph_web_viewer(
    sg: Any,
    host: str = "127.0.0.1",
    port: int = 8000,
    open_browser: bool = True,
) -> None:
    """
    Launch a simple local HTTP server that serves a Three.js-based 3D
    visualization of the given SceneGraphWorld.

    This is intended for interactive inspection and debugging; the server
    runs until interrupted (Ctrl+C).
    """
    graph_data = export_scenegraph_to_threejs(sg)
    print(
    "[DSG-JIT] export_scenegraph_to_threejs:",
    "nodes =", len(graph_data.get("nodes", [])),
    "edges =", len(graph_data.get("edges", [])),
    )
    graph_json = json.dumps(graph_data)

    html = HTML_TEMPLATE.replace("__GRAPH_JSON__", graph_json)

    class SceneGraphRequestHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # type: ignore[override]
            if self.path in ("/", "/index.html"):
                content = html.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format: str, *args: Any) -> None:  # type: ignore[override]
            # Keep the server quiet; override to suppress default logging to stderr.
            return

    server_address = (host, port)
    httpd = HTTPServer(server_address, SceneGraphRequestHandler)

    url = f"http://{host}:{port}/"

    if open_browser:
        # Open browser in a background thread so we don't block the server.
        threading.Thread(target=webbrowser.open, args=(url,), daemon=True).start()

    print(f"[DSG-JIT] SceneGraph web viewer running at {url} (Ctrl+C to stop)")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[DSG-JIT] SceneGraph web viewer stopped.")
    finally:
        httpd.server_close()
