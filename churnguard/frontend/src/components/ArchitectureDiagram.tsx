import Box from "@cloudscape-design/components/box";
import type { ArchitectureResponse } from "../api/types";

interface Props {
  architecture: ArchitectureResponse;
}

const TYPE_COLOR: Record<string, string> = {
  data: "#e9f2ff",
  pipeline: "#f1e9ff",
  registry: "#fff3e0",
  events: "#ffe9ec",
  endpoint: "#e6fff0",
  batch: "#e6fff0",
  deploy: "#fff8e1",
};

// Renders the nodes/edges returned by GET /api/architecture as an embedded SVG.
// Node positions are a static layout keyed by node id; all labels/types come
// from the backend response.
const POSITIONS: Record<string, { x: number; y: number }> = {
  dataset: { x: 40, y: 180 },
  pipeline: { x: 230, y: 60 },
  registry: { x: 440, y: 60 },
  events: { x: 650, y: 60 },
  bluegreen: { x: 650, y: 180 },
  realtime: { x: 850, y: 180 },
  serverless: { x: 230, y: 300 },
  async: { x: 440, y: 300 },
  batch: { x: 650, y: 300 },
  mme: { x: 850, y: 300 },
};

const NODE_W = 150;
const NODE_H = 50;

export default function ArchitectureDiagram({ architecture }: Props) {
  const center = (id: string) => {
    const p = POSITIONS[id];
    if (!p) return null;
    return { x: p.x + NODE_W / 2, y: p.y + NODE_H / 2 };
  };

  return (
    <Box>
      <svg
        viewBox="0 0 1030 380"
        role="img"
        aria-label="ChurnGuard architecture diagram"
        style={{ width: "100%", height: "auto", maxWidth: 1030 }}
      >
        {architecture.edges.map((edge, i) => {
          const a = center(edge.from);
          const b = center(edge.to);
          if (!a || !b) return null;
          return (
            <line
              key={`edge-${i}`}
              x1={a.x}
              y1={a.y}
              x2={b.x}
              y2={b.y}
              stroke="#879596"
              strokeWidth={1.5}
              markerEnd="url(#arrow)"
            />
          );
        })}
        <defs>
          <marker
            id="arrow"
            markerWidth="10"
            markerHeight="10"
            refX="8"
            refY="3"
            orient="auto"
            markerUnits="strokeWidth"
          >
            <path d="M0,0 L0,6 L9,3 z" fill="#879596" />
          </marker>
        </defs>
        {architecture.nodes.map((node) => {
          const p = POSITIONS[node.id];
          if (!p) return null;
          return (
            <g key={node.id}>
              <rect
                x={p.x}
                y={p.y}
                width={NODE_W}
                height={NODE_H}
                rx={8}
                fill={TYPE_COLOR[node.type] ?? "#f2f3f3"}
                stroke="#5f6b7a"
                strokeWidth={1}
              />
              <text
                x={p.x + NODE_W / 2}
                y={p.y + NODE_H / 2}
                textAnchor="middle"
                dominantBaseline="middle"
                fontSize={11}
                fill="#16191f"
              >
                {node.label.length > 22 ? node.label.slice(0, 21) + "…" : node.label}
              </text>
            </g>
          );
        })}
      </svg>
    </Box>
  );
}
