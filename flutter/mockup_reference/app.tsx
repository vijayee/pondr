import { useState, useRef, useEffect, type ReactNode } from "react"
import { motion, AnimatePresence } from "motion/react"
import {
  Plus, Search, Trash2, Menu, X, Send, Paperclip,
  LogOut, Eye, EyeOff, MessageCircle, FileText,
  ImageIcon, File, User as UserIcon, Sparkles, Brain,
  Settings, ArrowLeft, Bell, Shield, Palette, Info,
  ChevronRight, ChevronDown, Check, Cpu, Globe, KeyRound, Pencil,
  ToggleLeft, ToggleRight, AlertCircle, Bot,
} from "lucide-react"
import pondrLogoRaw from "@/imports/pondr_irridescent-2.svg?raw"
import thoughtSparkRaw from "@/imports/thoughtspark_irridescent.svg?raw"

// Inline SVG helpers — normalise Inkscape's absolute mm dimensions to fill container
function normalizeSvg(raw: string): string {
  return raw
    .replace(/\swidth="[^"]*"/, ' width="100%"')
    .replace(/\sheight="[^"]*"/, ' height="100%"')
}

function PondrLogo({ className, style }: { className?: string; style?: React.CSSProperties }) {
  return (
    <div
      className={className}
      style={style}
      dangerouslySetInnerHTML={{ __html: normalizeSvg(pondrLogoRaw) }}
    />
  )
}

function ThoughtSpark({ className, style }: { className?: string; style?: React.CSSProperties }) {
  return (
    <div
      className={className}
      style={style}
      dangerouslySetInnerHTML={{ __html: normalizeSvg(thoughtSparkRaw) }}
    />
  )
}

// ─── Types ───────────────────────────────────────────────

type View = "login" | "register" | "chat" | "settings" | "subconscious"

interface AttachedFile {
  id: string
  name: string
  type: string
  size: number
}

interface Message {
  id: string
  role: "user" | "assistant"
  content: string
  timestamp: Date
  files?: AttachedFile[]
}

interface ProviderModel {
  id: string
  modelId: string
  label: string
  enabled: boolean
}

interface Provider {
  id: string
  name: string
  baseUrl: string
  apiKey: string
  enabled: boolean
  models: ProviderModel[]
}

interface ChatSession {
  id: string
  name: string
  updatedAt: Date
  messages: Message[]
}

// ─── Helpers ─────────────────────────────────────────────

function formatTime(date: Date) {
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
}

function formatFileSize(bytes: number) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function getFileIcon(type: string) {
  if (type.startsWith("image/")) return ImageIcon
  if (type.includes("pdf") || type.includes("document") || type.includes("text")) return FileText
  return File
}

function groupSessions(sessions: ChatSession[]) {
  const now = new Date()
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate())
  const yesterday = new Date(today.getTime() - 864e5)
  const weekAgo = new Date(today.getTime() - 864e5 * 7)

  const groups: [string, ChatSession[]][] = [
    ["Today", []],
    ["Yesterday", []],
    ["This Week", []],
    ["Earlier", []],
  ]

  for (const s of sessions) {
    const d = new Date(s.updatedAt.getFullYear(), s.updatedAt.getMonth(), s.updatedAt.getDate())
    if (d >= today) groups[0][1].push(s)
    else if (d >= yesterday) groups[1][1].push(s)
    else if (d >= weekAgo) groups[2][1].push(s)
    else groups[3][1].push(s)
  }

  return groups.filter(([, arr]) => arr.length > 0)
}

// ─── AI response pool ────────────────────────────────────

const AI_POOL = [
  `What a rich question to sit with. Let me think through this carefully.

At its core, the tension here is between two competing intuitions that both feel true: the desire for clear, decisive answers and the reality that most interesting questions resist them. The most productive approach isn't to resolve this tension but to work *within* it.

**The key insight:** Complexity often emerges from surprisingly simple underlying patterns. When you trace a phenomenon back to first principles, you frequently find that what seemed like a tangled knot is actually a few threads pulled tight. The challenge is resisting the urge to cut it.

What aspect would you like to explore further?`,

  `I love how this question opens up into a landscape of interconnected ideas. There are two dimensions worth examining:

**The immediate dimension** — what this means practically, and how it changes how we act or think day-to-day.

**The structural dimension** — *why* this pattern exists at all, and what it reveals about the deeper system it's embedded in.

Most analyses stop at the first. The second is where the real leverage is. Once you see the structural reason something is the way it is, you gain the ability to anticipate where it will hold and where it will break down.

Does that framing resonate with what you're exploring?`,

  `The short answer is nuanced, but the long answer is illuminating.

Conventional wisdom here is often incomplete — not wrong exactly, but missing the piece that explains the exceptions. Most people approach this from the surface level, which gives you a working model about 80% of the time. The remaining 20% is where the real understanding lives.

The key is to ask not just *what* is true but *under what conditions* it's true. That shift from categorical claims to conditional ones is one of the most powerful upgrades you can make to your thinking.

What's the context you're working in? That would help me give you a sharper answer.`,

  `This is one of those questions that looks simple from a distance and reveals incredible depth up close.

The conventional framing treats this as a binary — either X or Y. But that's a false dichotomy. What actually happens is more interesting: X and Y exist on a spectrum, and most real-world cases sit somewhere between the poles, often shifting position depending on factors we don't always control.

What makes this practically useful is understanding which factors move things toward which end, and when that movement matters. The map is simpler than it looks once you have the right coordinate system.`,
]

// ─── Initial data ─────────────────────────────────────────

const SEED_SESSIONS: ChatSession[] = [
  {
    id: "s1",
    name: "Quantum Superposition & Qubits",
    updatedAt: new Date(Date.now() - 1000 * 60 * 9),
    messages: [
      {
        id: "m1a", role: "user",
        content: "Can you explain quantum superposition in simple terms?",
        timestamp: new Date(Date.now() - 1000 * 60 * 12),
      },
      {
        id: "m1b", role: "assistant",
        content: `Quantum superposition is one of the most counterintuitive—and beautiful—principles in physics.

Imagine a coin spinning in the air. It's neither heads nor tails until it lands. A quantum particle behaves similarly: it exists in *multiple states simultaneously* until it's observed or measured. This isn't a limitation of our knowledge—the particle genuinely occupies all states at once.

**Why does this matter?**
This is what gives quantum computers their extraordinary potential. Classical bits are either 0 or 1. Quantum bits (qubits) can be both at the same time—enabling a kind of massively parallel computation that classical machines fundamentally cannot replicate.

The collapse of superposition upon measurement is still one of the deepest mysteries in physics. What causes it? That question is still open.`,
        timestamp: new Date(Date.now() - 1000 * 60 * 11),
      },
      {
        id: "m1c", role: "user",
        content: "How does entanglement connect to this?",
        timestamp: new Date(Date.now() - 1000 * 60 * 9),
      },
      {
        id: "m1d", role: "assistant",
        content: `Entanglement is superposition's stranger sibling.

When two particles become entangled, their quantum states are linked—no matter the distance between them. Measure one, and you instantly know something about the other. Einstein famously called this "spooky action at a distance" and found it deeply troubling.

**The key point:** This isn't information traveling faster than light (that would violate relativity). It's more subtle—the correlated outcomes are built into the particles' shared quantum state from the start. You don't learn what the distant particle is; you learn what it *was always going to be*, given what its partner turned out to be.

This makes entanglement a resource for quantum communication and certain quantum algorithms—but it's not a way to send signals faster than light.`,
        timestamp: new Date(Date.now() - 1000 * 60 * 8),
      },
    ],
  },
  {
    id: "s2",
    name: "The Hard Problem of Consciousness",
    updatedAt: new Date(Date.now() - 1000 * 60 * 60 * 4),
    messages: [
      {
        id: "m2a", role: "user",
        content: "What is the hard problem of consciousness, and why is it hard?",
        timestamp: new Date(Date.now() - 1000 * 60 * 60 * 4),
      },
      {
        id: "m2b", role: "assistant",
        content: `Coined by philosopher David Chalmers in 1994, the hard problem asks perhaps the deepest question we have: *why is there subjective experience at all?*

**The easy problems** (still enormously complex): How does the brain integrate information? Process sensory data? Control behavior? These are hard in the engineering sense—but they're tractable. We can imagine a future neuroscience that answers them fully.

**The hard problem**: Even after you've explained all of that perfectly, something seems left out. Why does 700nm wavelength *look* red, rather than just triggering a signal? Why does pain *hurt*, rather than merely being a neutral alarm? Why is there an "inside" to experience at all?

This is what philosophers call the **explanatory gap**—between objective neural description and subjective phenomenal experience. The qualia (the redness of red, the sharpness of pain) seem to resist being fully captured in functional terms.

Whether this gap represents a genuine mystery about reality, or a failure of our conceptual vocabulary, remains unsettled—and may be the most important question of the 21st century.`,
        timestamp: new Date(Date.now() - 1000 * 60 * 60 * 4 + 60000),
      },
    ],
  },
  {
    id: "s3",
    name: "Supervised vs Unsupervised Learning",
    updatedAt: new Date(Date.now() - 864e5 * 1.5),
    messages: [
      {
        id: "m3a", role: "user",
        content: "What's the practical difference between supervised and unsupervised ML?",
        timestamp: new Date(Date.now() - 864e5 * 1.5),
      },
      {
        id: "m3b", role: "assistant",
        content: `The distinction is fundamental—and it shapes everything from data requirements to what questions you can even ask.

**Supervised learning** trains on labeled examples: input → known output. The model learns to map new inputs to outputs it hasn't seen. You need labeled data, which is often expensive to produce. But you get targeted, measurable predictions.
*Examples: spam filtering, image classification, fraud detection, price prediction.*

**Unsupervised learning** gets no labels. It has to find structure in raw data on its own—discovering clusters, patterns, and anomalies without being told what to look for. You can use far more data (labels aren't required), but the outputs are harder to evaluate.
*Examples: customer segmentation, topic modeling, anomaly detection, dimensionality reduction.*

**The mental model that clicks for most people:**
- Supervised: *"Here are the answers. Learn the rules that produce them."*
- Unsupervised: *"Here's the data. Find whatever's interesting."*

In practice, the most powerful systems often combine both—using unsupervised methods to build representations, then fine-tuning with supervision for specific tasks. That's roughly how modern LLMs work.`,
        timestamp: new Date(Date.now() - 864e5 * 1.5 + 90000),
      },
    ],
  },
  {
    id: "s4",
    name: "The Fermi Paradox",
    updatedAt: new Date(Date.now() - 864e5 * 4),
    messages: [
      {
        id: "m4a", role: "user",
        content: "If the universe is so vast and old, why haven't we found any signs of alien civilizations?",
        timestamp: new Date(Date.now() - 864e5 * 4),
      },
      {
        id: "m4b", role: "assistant",
        content: `This is the Fermi Paradox—and it's one of the most haunting open questions in science.

The universe is ~13.8 billion years old. Our galaxy contains ~400 billion stars, many with planets in habitable zones. The math suggests we should be surrounded by evidence of other civilizations. Yet: silence.

**The leading hypotheses—each with different implications:**

**The Great Filter** — Something vanishingly rare stands between simple chemistry and spacefaring civilizations. If the filter is *behind* us (complex life is incredibly rare), we might be alone. If it's *ahead* of us (civilizations tend to self-destruct), that's far more troubling.

**The Dark Forest** — Any sufficiently advanced civilization hides. In a universe of unknown others, broadcasting your location is suicidal. Silence is strategy. (Liu Cixin explores this beautifully in his trilogy.)

**We're early** — Cosmically speaking, we emerged young. Most stars that will ever form haven't yet. We may be among the first.

**The Zoo hypothesis** — They're watching, not interfering. (Uncomfortable for obvious reasons.)

**It's just hard** — Interstellar distances are staggering. The window of overlapping technological civilizations might be vanishingly narrow. We may simply have missed each other.

The answer matters enormously—each resolution carries profound implications for our own future.`,
        timestamp: new Date(Date.now() - 864e5 * 4 + 120000),
      },
    ],
  },
  {
    id: "s5",
    name: "What Makes Arguments Good?",
    updatedAt: new Date(Date.now() - 864e5 * 8),
    messages: [
      {
        id: "m5a", role: "user",
        content: "What actually makes an argument good or bad?",
        timestamp: new Date(Date.now() - 864e5 * 8),
      },
    ],
  },
]

// ─── Subconscious graph ───────────────────────────────────

const CLUSTER_COLORS: Record<string, string> = {
  Physics: "#888ddf",
  Philosophy: "#c3acda",
  AI: "#e2e6ff",
  Astronomy: "#e8d7bd",
  Logic: "#e0efe4",
}

interface SimNode {
  id: string; label: string; cluster: string; color: string
  x: number; y: number; vx: number; vy: number; r: number
  description: string
}
interface SimEdge { source: string; target: string }

const BASE_NODES: Omit<SimNode, "x"|"y"|"vx"|"vy">[] = [
  { id: "quantum-mechanics",   label: "Quantum Mechanics",      cluster: "Physics",    color: CLUSTER_COLORS.Physics,    r: 18, description: "The branch of physics describing subatomic particles and their probabilistic behavior." },
  { id: "superposition",       label: "Superposition",          cluster: "Physics",    color: CLUSTER_COLORS.Physics,    r: 14, description: "A particle exists in multiple states simultaneously until the moment it is measured." },
  { id: "qubits",              label: "Qubits",                 cluster: "Physics",    color: CLUSTER_COLORS.Physics,    r: 12, description: "Quantum bits that exploit superposition to enable massively parallel computation." },
  { id: "entanglement",        label: "Entanglement",           cluster: "Physics",    color: CLUSTER_COLORS.Physics,    r: 15, description: "Two particles share a correlated quantum state regardless of the distance between them." },
  { id: "wave-collapse",       label: "Wave Collapse",          cluster: "Physics",    color: CLUSTER_COLORS.Physics,    r: 11, description: "The reduction of a superposed wave function to a single definite state upon observation." },
  { id: "consciousness",       label: "Consciousness",          cluster: "Philosophy", color: CLUSTER_COLORS.Philosophy, r: 20, description: "Subjective experience and the inner phenomenal life of a mind." },
  { id: "qualia",              label: "Qualia",                 cluster: "Philosophy", color: CLUSTER_COLORS.Philosophy, r: 14, description: "The irreducible subjective character of experience — the redness of red, the sharpness of pain." },
  { id: "hard-problem",        label: "Hard Problem",           cluster: "Philosophy", color: CLUSTER_COLORS.Philosophy, r: 16, description: "Why physical brain processes give rise to subjective experience at all (Chalmers, 1994)." },
  { id: "explanatory-gap",     label: "Explanatory Gap",        cluster: "Philosophy", color: CLUSTER_COLORS.Philosophy, r: 12, description: "The conceptual gulf between complete neural description and phenomenal experience." },
  { id: "phenomenology",       label: "Phenomenology",          cluster: "Philosophy", color: CLUSTER_COLORS.Philosophy, r: 11, description: "Philosophical study of the structure of first-person experience (Husserl, Heidegger, Merleau-Ponty)." },
  { id: "machine-learning",    label: "Machine Learning",       cluster: "AI",         color: CLUSTER_COLORS.AI,         r: 18, description: "Systems that learn statistical patterns from data without explicit programming." },
  { id: "supervised",          label: "Supervised Learning",    cluster: "AI",         color: CLUSTER_COLORS.AI,         r: 13, description: "Training on labeled input-output pairs to generalise predictions to new examples." },
  { id: "unsupervised",        label: "Unsupervised Learning",  cluster: "AI",         color: CLUSTER_COLORS.AI,         r: 13, description: "Finding latent structure — clusters, manifolds, representations — in unlabeled data." },
  { id: "transformers",        label: "Transformers",           cluster: "AI",         color: CLUSTER_COLORS.AI,         r: 14, description: "Attention-based neural architecture that underpins modern large language models." },
  { id: "llms",                label: "Large Language Models",  cluster: "AI",         color: CLUSTER_COLORS.AI,         r: 17, description: "Neural networks trained on vast text corpora capable of reasoning, generation, and retrieval." },
  { id: "fermi-paradox",       label: "Fermi Paradox",          cluster: "Astronomy",  color: CLUSTER_COLORS.Astronomy,  r: 18, description: "The contradiction between high estimates for alien civilisations and the total absence of evidence." },
  { id: "great-filter",        label: "Great Filter",           cluster: "Astronomy",  color: CLUSTER_COLORS.Astronomy,  r: 14, description: "A hypothetical barrier — past or future — that prevents life becoming interstellar." },
  { id: "dark-forest",         label: "Dark Forest Theory",     cluster: "Astronomy",  color: CLUSTER_COLORS.Astronomy,  r: 14, description: "Any sufficiently advanced civilisation stays silent because broadcasting location invites destruction." },
  { id: "drake-equation",      label: "Drake Equation",         cluster: "Astronomy",  color: CLUSTER_COLORS.Astronomy,  r: 11, description: "A probabilistic formula estimating the number of detectable communicating civilisations in the galaxy." },
  { id: "kardashev",           label: "Kardashev Scale",        cluster: "Astronomy",  color: CLUSTER_COLORS.Astronomy,  r: 11, description: "Classification of civilisations by total energy consumption across planetary, stellar, and galactic scales." },
  { id: "argumentation",       label: "Argumentation",          cluster: "Logic",      color: CLUSTER_COLORS.Logic,      r: 15, description: "The structured process of forming claims, supplying reasons, and anticipating counterarguments." },
  { id: "first-principles",    label: "First Principles",       cluster: "Logic",      color: CLUSTER_COLORS.Logic,      r: 13, description: "Reasoning up from self-evident foundational truths rather than by analogy or convention." },
  { id: "epistemology",        label: "Epistemology",           cluster: "Logic",      color: CLUSTER_COLORS.Logic,      r: 14, description: "The philosophical study of knowledge, justified belief, and the limits of what can be known." },
  { id: "quantum-consciousness", label: "Quantum Consciousness", cluster: "Philosophy", color: "#b8a4dc",                r: 12, description: "Penrose-Hameroff hypothesis: quantum processes in neural microtubules give rise to consciousness." },
  { id: "ai-consciousness",    label: "AI Consciousness",       cluster: "AI",         color: "#aab3e8",                 r: 13, description: "The open question of whether artificial systems can possess genuine phenomenal experience." },
]

const BASE_EDGES: SimEdge[] = [
  { source: "quantum-mechanics", target: "superposition" },
  { source: "quantum-mechanics", target: "entanglement" },
  { source: "quantum-mechanics", target: "wave-collapse" },
  { source: "superposition",     target: "qubits" },
  { source: "entanglement",      target: "wave-collapse" },
  { source: "consciousness",     target: "qualia" },
  { source: "consciousness",     target: "hard-problem" },
  { source: "hard-problem",      target: "qualia" },
  { source: "hard-problem",      target: "explanatory-gap" },
  { source: "hard-problem",      target: "phenomenology" },
  { source: "explanatory-gap",   target: "qualia" },
  { source: "machine-learning",  target: "supervised" },
  { source: "machine-learning",  target: "unsupervised" },
  { source: "machine-learning",  target: "transformers" },
  { source: "transformers",      target: "llms" },
  { source: "fermi-paradox",     target: "great-filter" },
  { source: "fermi-paradox",     target: "dark-forest" },
  { source: "fermi-paradox",     target: "drake-equation" },
  { source: "great-filter",      target: "kardashev" },
  { source: "argumentation",     target: "first-principles" },
  { source: "argumentation",     target: "epistemology" },
  { source: "first-principles",  target: "epistemology" },
  { source: "quantum-mechanics", target: "quantum-consciousness" },
  { source: "consciousness",     target: "quantum-consciousness" },
  { source: "consciousness",     target: "ai-consciousness" },
  { source: "llms",              target: "ai-consciousness" },
  { source: "machine-learning",  target: "ai-consciousness" },
  { source: "epistemology",      target: "consciousness" },
  { source: "epistemology",      target: "hard-problem" },
]

function SubconsciousView({ onClose }: { onClose: () => void }) {
  const initNodes = (): SimNode[] =>
    BASE_NODES.map((n, i) => {
      const angle = (i / BASE_NODES.length) * Math.PI * 2
      const rad = 180 + Math.random() * 120
      return { ...n, x: Math.cos(angle) * rad, y: Math.sin(angle) * rad, vx: 0, vy: 0 }
    })

  const nodesRef = useRef<SimNode[]>(initNodes())
  const alphaRef = useRef(1.0)
  const rafRef = useRef<number>()
  const [, forceRender] = useState(0)
  const [selectedNode, setSelectedNode] = useState<SimNode | null>(null)

  // Precompute edge index pairs once
  const edgeIndices = useRef(
    BASE_EDGES.map(e => ({
      si: BASE_NODES.findIndex(n => n.id === e.source),
      ti: BASE_NODES.findIndex(n => n.id === e.target),
    })).filter(e => e.si >= 0 && e.ti >= 0)
  )

  useEffect(() => {
    const CHARGE = 2400
    const SPRING = 0.07
    const REST = 160
    const GRAVITY = 0.025
    const DAMPING = 0.82

    function tick() {
      const nodes = nodesRef.current
      const alpha = alphaRef.current
      if (alpha < 0.002) { forceRender(k => k + 1); return }
      alphaRef.current *= 0.988

      // Repulsion
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const dx = nodes[i].x - nodes[j].x
          const dy = nodes[i].y - nodes[j].y
          const d2 = dx * dx + dy * dy || 1
          const d = Math.sqrt(d2)
          const f = CHARGE / d2
          const fx = f * dx / d; const fy = f * dy / d
          nodes[i].vx += fx; nodes[i].vy += fy
          nodes[j].vx -= fx; nodes[j].vy -= fy
        }
      }

      // Spring edges
      for (const { si, ti } of edgeIndices.current) {
        const a = nodes[si]; const b = nodes[ti]
        const dx = b.x - a.x; const dy = b.y - a.y
        const d = Math.sqrt(dx * dx + dy * dy) || 1
        const f = SPRING * (d - REST)
        const fx = f * dx / d; const fy = f * dy / d
        a.vx += fx; a.vy += fy; b.vx -= fx; b.vy -= fy
      }

      // Center gravity + integrate
      for (const n of nodes) {
        n.vx -= GRAVITY * n.x; n.vy -= GRAVITY * n.y
        n.x += n.vx * alpha; n.y += n.vy * alpha
        n.vx *= DAMPING; n.vy *= DAMPING
      }

      forceRender(k => k + 1)
      rafRef.current = requestAnimationFrame(tick)
    }

    rafRef.current = requestAnimationFrame(tick)
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current) }
  }, [])

  // Close on Escape
  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === "Escape") onClose() }
    window.addEventListener("keydown", handler)
    return () => window.removeEventListener("keydown", handler)
  }, [onClose])

  // Pan / zoom
  const panRef = useRef({ x: 0, y: 0 })
  const [pan, setPan] = useState({ x: 0, y: 0 })
  const [scale, setScale] = useState(1)
  const scaleRef = useRef(1)
  const dragging = useRef<{ active: boolean; sx: number; sy: number; spx: number; spy: number }>({ active: false, sx: 0, sy: 0, spx: 0, spy: 0 })

  function onWheel(e: React.WheelEvent) {
    e.preventDefault()
    const next = Math.max(0.25, Math.min(4, scaleRef.current * (e.deltaY < 0 ? 1.12 : 0.89)))
    scaleRef.current = next
    setScale(next)
  }

  function onMouseDown(e: React.MouseEvent) {
    if (e.button !== 0) return
    dragging.current = { active: true, sx: e.clientX, sy: e.clientY, spx: panRef.current.x, spy: panRef.current.y }
  }

  function onMouseMove(e: React.MouseEvent) {
    if (!dragging.current.active) return
    const nx = dragging.current.spx + (e.clientX - dragging.current.sx)
    const ny = dragging.current.spy + (e.clientY - dragging.current.sy)
    panRef.current = { x: nx, y: ny }
    setPan({ x: nx, y: ny })
  }

  function onMouseUp() { dragging.current.active = false }

  const nodes = nodesRef.current
  const W = typeof window !== "undefined" ? window.innerWidth : 1200
  const H = typeof window !== "undefined" ? window.innerHeight : 800
  const cx = W / 2 + pan.x
  const cy = H / 2 + pan.y

  const connectedIds = selectedNode
    ? new Set(BASE_EDGES.filter(e => e.source === selectedNode.id || e.target === selectedNode.id).flatMap(e => [e.source, e.target]))
    : null

  return (
    <div className="flex flex-col h-full overflow-hidden select-none"
      style={{ background: "#07061a" }}>

      {/* Ambient glows */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden">
        <div className="absolute rounded-full blur-[160px]" style={{ width: 600, height: 600, top: "10%", left: "15%", background: "rgba(136,141,223,0.07)" }} />
        <div className="absolute rounded-full blur-[180px]" style={{ width: 500, height: 500, bottom: "10%", right: "20%", background: "rgba(195,172,218,0.06)" }} />
        <div className="absolute rounded-full blur-[120px]" style={{ width: 300, height: 300, top: "40%", right: "35%", background: "rgba(224,239,228,0.04)" }} />
      </div>

      {/* Header */}
      <div className="relative z-10 flex items-center justify-between px-6 py-3.5 flex-shrink-0"
        style={{ background: "rgba(7,6,26,0.85)", backdropFilter: "blur(16px)", borderBottom: "1px solid rgba(136,141,223,0.14)" }}>
        <div className="flex items-center gap-3">
          <ThoughtSpark className="w-9 h-9 flex-shrink-0" />
          <div>
            <h2 className="font-nunito font-bold text-white text-base leading-tight tracking-wide">The Subconscious</h2>
            <p className="text-[11px]" style={{ color: "#888ddf" }}>
              {nodes.length} concepts · {BASE_EDGES.length} connections · {Object.keys(CLUSTER_COLORS).length} domains
            </p>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <p className="text-xs hidden sm:block" style={{ color: "rgba(136,141,223,0.55)" }}>
            Scroll to zoom · Drag to pan · Click a node to explore
          </p>
          <button onClick={onClose}
            className="w-8 h-8 rounded-lg flex items-center justify-center transition-all hover:bg-white/10"
            style={{ color: "#888ddf" }}>
            <X size={16} />
          </button>
        </div>
      </div>

      {/* Legend */}
      <div className="absolute top-16 right-4 z-10 flex flex-col gap-1.5 p-3 rounded-xl"
        style={{ background: "rgba(7,6,26,0.8)", backdropFilter: "blur(10px)", border: "1px solid rgba(136,141,223,0.16)" }}>
        {Object.entries(CLUSTER_COLORS).map(([name, color]) => (
          <div key={name} className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: color, boxShadow: `0 0 5px ${color}` }} />
            <span className="text-xs" style={{ color: "rgba(255,255,255,0.55)" }}>{name}</span>
          </div>
        ))}
      </div>

      {/* Graph canvas */}
      <div className="flex-1 relative overflow-hidden cursor-grab active:cursor-grabbing"
        onWheel={onWheel} onMouseDown={onMouseDown} onMouseMove={onMouseMove}
        onMouseUp={onMouseUp} onMouseLeave={onMouseUp}
        onClick={() => setSelectedNode(null)}>
        <svg width="100%" height="100%">
          <defs>
            <filter id="sc-glow-sm" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="3" result="b" />
              <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
            </filter>
            <filter id="sc-glow-lg" x="-100%" y="-100%" width="300%" height="300%">
              <feGaussianBlur stdDeviation="8" result="b" />
              <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
            </filter>
            {nodes.map(n => (
              <radialGradient key={n.id} id={`sc-g-${n.id}`} cx="40%" cy="35%" r="65%">
                <stop offset="0%" stopColor={n.color} stopOpacity="1" />
                <stop offset="100%" stopColor={n.color} stopOpacity="0.35" />
              </radialGradient>
            ))}
          </defs>

          <g transform={`translate(${cx},${cy}) scale(${scale})`}>
            {/* Edges */}
            {BASE_EDGES.map((e, i) => {
              const a = nodes[BASE_NODES.findIndex(n => n.id === e.source)]
              const b = nodes[BASE_NODES.findIndex(n => n.id === e.target)]
              if (!a || !b) return null
              const lit = selectedNode && (e.source === selectedNode.id || e.target === selectedNode.id)
              return (
                <line key={i} x1={a.x} y1={a.y} x2={b.x} y2={b.y}
                  stroke={lit ? a.color : "rgba(136,141,223,0.13)"}
                  strokeWidth={lit ? 1.5 : 0.75}
                  strokeOpacity={lit ? 0.8 : 1}
                />
              )
            })}

            {/* Nodes */}
            {nodes.map(n => {
              const sel = selectedNode?.id === n.id
              const dim = selectedNode && !sel && connectedIds && !connectedIds.has(n.id)
              return (
                <g key={n.id} transform={`translate(${n.x},${n.y})`} style={{ cursor: "pointer" }}
                  onClick={ev => { ev.stopPropagation(); setSelectedNode(sel ? null : n) }}>
                  {sel && (
                    <circle r={n.r + 10} fill="none" stroke={n.color} strokeWidth={1.5}
                      opacity={0.45} filter="url(#sc-glow-lg)" />
                  )}
                  <circle r={n.r} fill={`url(#sc-g-${n.id})`}
                    stroke={n.color} strokeWidth={sel ? 2 : 0.8}
                    opacity={dim ? 0.15 : 1}
                    filter={sel ? "url(#sc-glow-lg)" : "url(#sc-glow-sm)"} />
                  <text y={n.r + 13} textAnchor="middle" fontSize={9.5}
                    fill={dim ? "rgba(255,255,255,0.12)" : sel ? "#fff" : "rgba(255,255,255,0.65)"}
                    fontFamily="Inter, sans-serif" fontWeight={sel ? "600" : "400"}
                    style={{ pointerEvents: "none" }}>
                    {n.label}
                  </text>
                </g>
              )
            })}
          </g>
        </svg>
      </div>

      {/* Node info panel */}
      <AnimatePresence>
        {selectedNode && (
          <motion.div
            key={selectedNode.id}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 8 }}
            transition={{ duration: 0.18 }}
            className="absolute bottom-5 left-5 max-w-[300px] rounded-2xl p-4 z-20"
            style={{ background: "rgba(14,12,32,0.96)", backdropFilter: "blur(20px)", border: `1px solid ${selectedNode.color}55`, boxShadow: `0 8px 32px rgba(0,0,0,0.5), 0 0 0 1px ${selectedNode.color}22` }}
          >
            <div className="flex items-start gap-2.5 mb-2.5">
              <div className="w-3 h-3 rounded-full mt-0.5 flex-shrink-0"
                style={{ background: selectedNode.color, boxShadow: `0 0 10px ${selectedNode.color}` }} />
              <div>
                <p className="text-sm font-semibold text-white leading-tight">{selectedNode.label}</p>
                <p className="text-[10px] font-bold uppercase tracking-widest mt-0.5" style={{ color: selectedNode.color }}>{selectedNode.cluster}</p>
              </div>
            </div>
            <p className="text-xs leading-relaxed" style={{ color: "rgba(236,233,255,0.7)" }}>{selectedNode.description}</p>
            <div className="mt-3 pt-2.5 flex items-center gap-4 border-t" style={{ borderColor: "rgba(255,255,255,0.07)" }}>
              {(() => {
                const conns = BASE_EDGES.filter(e => e.source === selectedNode.id || e.target === selectedNode.id)
                return (
                  <p className="text-[10px]" style={{ color: "rgba(136,141,223,0.55)" }}>
                    {conns.length} connection{conns.length !== 1 ? "s" : ""}
                  </p>
                )
              })()}
              <div className="flex gap-1 flex-wrap">
                {BASE_EDGES
                  .filter(e => e.source === selectedNode.id || e.target === selectedNode.id)
                  .map(e => {
                    const peerId = e.source === selectedNode.id ? e.target : e.source
                    const peer = BASE_NODES.find(n => n.id === peerId)
                    if (!peer) return null
                    return (
                      <span key={peerId} className="text-[9px] px-1.5 py-0.5 rounded-full"
                        style={{ background: `${peer.color}20`, color: peer.color, border: `1px solid ${peer.color}40` }}>
                        {peer.label}
                      </span>
                    )
                  })}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

// ─── Markdown renderer (lite) ─────────────────────────────

function renderMarkdown(text: string): ReactNode {
  const lines = text.split("\n")
  const result: ReactNode[] = []

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]

    if (line.startsWith("**") && line.endsWith("**") && line.length > 4) {
      result.push(<strong key={i} className="block font-semibold mt-3 mb-0.5">{line.slice(2, -2)}</strong>)
      continue
    }

    const parts = line.split(/(\*\*[^*]+\*\*|\*[^*]+\*)/g)
    const inline: ReactNode[] = parts.map((part, j) => {
      if (part.startsWith("**") && part.endsWith("**")) return <strong key={j}>{part.slice(2, -2)}</strong>
      if (part.startsWith("*") && part.endsWith("*") && part.length > 2) return <em key={j}>{part.slice(1, -1)}</em>
      return part
    })

    result.push(
      <span key={i}>
        {inline}
        {i < lines.length - 1 && line !== "" && "\n"}
        {line === "" && i < lines.length - 1 && "\n"}
      </span>
    )
  }

  return result
}

// ─── Typing indicator ─────────────────────────────────────

function TypingIndicator() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0 }}
      className="flex gap-3 max-w-4xl"
    >
      <div className="w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5 overflow-hidden"
        style={{ background: "rgba(195,172,218,0.2)" }}>
        <ThoughtSpark className="w-5 h-5" />
      </div>
      <div className="rounded-2xl px-4 py-3.5 flex items-center gap-1.5"
        style={{ background: "rgba(136,141,223,0.12)", border: "1px solid rgba(136,141,223,0.3)", borderRadius: "0.25rem 1rem 1rem 1rem" }}>
        {[0, 1, 2].map(i => (
          <motion.span
            key={i}
            className="block w-1.5 h-1.5 rounded-full bg-primary/70"
            animate={{ scale: [1, 1.5, 1], opacity: [0.4, 1, 0.4] }}
            transition={{ duration: 1.1, repeat: Infinity, delay: i * 0.18 }}
          />
        ))}
      </div>
    </motion.div>
  )
}

// ─── Message bubble ───────────────────────────────────────

function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === "user"

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.22, ease: "easeOut" }}
      className={`flex gap-3 ${isUser ? "flex-row-reverse ml-auto" : ""} max-w-[90%] md:max-w-3xl`}
    >
      <div
        className={`w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5 ${
          isUser ? "bg-primary/20" : ""
        }`}
        style={!isUser ? { background: "rgba(195,172,218,0.18)" } : undefined}
      >
        {isUser
          ? <UserIcon size={13} className="text-primary" />
          : <ThoughtSpark className="w-5 h-5" />
        }
      </div>

      <div className={`flex flex-col gap-1 ${isUser ? "items-end" : "items-start"} min-w-0`}>
        <div
          className="px-4 py-3 rounded-2xl text-sm leading-relaxed whitespace-pre-wrap break-words text-white"
          style={isUser
            ? { background: "rgb(136,141,223)", borderRadius: "1rem 0.25rem 1rem 1rem", boxShadow: "0 4px 16px rgba(136,141,223,0.35)" }
            : { background: "rgba(136,141,223,0.12)", border: "1px solid rgba(136,141,223,0.3)", borderRadius: "0.25rem 1rem 1rem 1rem" }
          }
        >
          {message.files && message.files.length > 0 && (
            <div className="mb-2.5 space-y-1.5">
              {message.files.map(file => {
                const Icon = getFileIcon(file.type)
                return (
                  <div key={file.id}
                    className={`flex items-center gap-2 rounded-lg px-2.5 py-1.5 ${
                      isUser ? "bg-white/15" : "bg-white/10 border border-white/15"
                    }`}
                  >
                    <Icon size={12} className="flex-shrink-0 opacity-70" />
                    <span className="text-xs truncate max-w-[180px]">{file.name}</span>
                    <span className="text-xs opacity-50 flex-shrink-0">{formatFileSize(file.size)}</span>
                  </div>
                )
              })}
            </div>
          )}
          {message.content && (
            <div>{renderMarkdown(message.content)}</div>
          )}
        </div>
        <span className="text-[11px] text-white/40 px-1">{formatTime(message.timestamp)}</span>
      </div>
    </motion.div>
  )
}

// ─── Prompt suggestions ────────────────────────────────────

const SUGGESTIONS = [
  "Explain the Fermi paradox",
  "What is the hard problem of consciousness?",
  "How do transformer models work?",
  "What makes an argument convincing?",
]

// ─── Main App ─────────────────────────────────────────────

export default function App() {
  const [view, setView] = useState<View>("login")
  const [sessions, setSessions] = useState<ChatSession[]>(SEED_SESSIONS)
  const [activeId, setActiveId] = useState("s1")
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(true)
  const [isTyping, setIsTyping] = useState(false)
  const [searchQuery, setSearchQuery] = useState("")
  const [showSearch, setShowSearch] = useState(false)

  // Auth
  const [loginUsername, setLoginUsername] = useState("")
  const [loginPw, setLoginPw] = useState("")
  const [showLoginPw, setShowLoginPw] = useState(false)
  const [regName, setRegName] = useState("")
  const [regUsername, setRegUsername] = useState("")
  const [regPw, setRegPw] = useState("")
  const [regConfirm, setRegConfirm] = useState("")
  const [showRegPw, setShowRegPw] = useState(false)

  // Settings
  const [settingsSection, setSettingsSection] = useState("profile")
  const [displayName, setDisplayName] = useState("Ada Lovelace")
  const [username, setUsername] = useState("ada_lovelace")
  const [bio, setBio] = useState("")
  const [notifMessages, setNotifMessages] = useState(true)
  const [notifSounds, setNotifSounds] = useState(false)
  const [accentColor, setAccentColor] = useState("#888ddf")
  const [providers, setProviders] = useState<Provider[]>([])
  const [expandedProviderId, setExpandedProviderId] = useState<string | null>(null)
  const [showProviderForm, setShowProviderForm] = useState(false)
  const [editingProviderId, setEditingProviderId] = useState<string | null>(null)
  const [providerForm, setProviderForm] = useState({ name: "", baseUrl: "", apiKey: "" })
  const [showProviderKey, setShowProviderKey] = useState(false)
  const [addingModelToId, setAddingModelToId] = useState<string | null>(null)
  const [editingModelId, setEditingModelId] = useState<string | null>(null)
  const [modelForm, setModelForm] = useState({ modelId: "", label: "" })
  const [providerSaved, setProviderSaved] = useState(false)
  const [selectedModelKey, setSelectedModelKey] = useState<string | null>(null)
  const [modelPickerOpen, setModelPickerOpen] = useState(false)

  // Chat
  const [inputText, setInputText] = useState("")
  const [attached, setAttached] = useState<AttachedFile[]>([])
  const fileRef = useRef<HTMLInputElement>(null)
  const bottomRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const activeSession = sessions.find(s => s.id === activeId)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [activeSession?.messages.length, isTyping])

  function handleLogin(e: React.FormEvent) {
    e.preventDefault()
    setView("chat")
  }

  function handleRegister(e: React.FormEvent) {
    e.preventDefault()
    setView("chat")
  }

  function handleNewChat() {
    const id = `s${Date.now()}`
    setSessions(prev => [{
      id,
      name: "New conversation",
      updatedAt: new Date(),
      messages: [],
    }, ...prev])
    setActiveId(id)
    setSidebarOpen(false)
    setInputText("")
  }

  function handleDeleteSession(id: string, e: React.MouseEvent) {
    e.stopPropagation()
    setSessions(prev => {
      const next = prev.filter(s => s.id !== id)
      if (activeId === id && next.length > 0) setActiveId(next[0].id)
      return next
    })
  }

  function handleFiles(e: React.ChangeEvent<HTMLInputElement>) {
    const files = Array.from(e.target.files ?? [])
    setAttached(prev => [
      ...prev,
      ...files.map(f => ({ id: `f${Date.now()}-${Math.random()}`, name: f.name, type: f.type, size: f.size })),
    ])
    if (fileRef.current) fileRef.current.value = ""
  }

  function handleSend() {
    if (!inputText.trim() && attached.length === 0) return

    const userMsg: Message = {
      id: `m${Date.now()}`,
      role: "user",
      content: inputText.trim(),
      timestamp: new Date(),
      files: attached.length > 0 ? [...attached] : undefined,
    }

    setSessions(prev => prev.map(s => {
      if (s.id !== activeId) return s
      const name = (s.name === "New conversation" && inputText.trim())
        ? inputText.trim().slice(0, 45) + (inputText.length > 45 ? "…" : "")
        : s.name
      return { ...s, name, messages: [...s.messages, userMsg], updatedAt: new Date() }
    }))

    setInputText("")
    setAttached([])
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto"
    }
    setIsTyping(true)

    setTimeout(() => {
      const reply = AI_POOL[Math.floor(Math.random() * AI_POOL.length)]
      setSessions(prev => prev.map(s =>
        s.id === activeId
          ? { ...s, messages: [...s.messages, { id: `m${Date.now()}`, role: "assistant", content: reply, timestamp: new Date() }], updatedAt: new Date() }
          : s
      ))
      setIsTyping(false)
    }, 1100 + Math.random() * 700)
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const filteredSessions = sessions.filter(s =>
    s.name.toLowerCase().includes(searchQuery.toLowerCase())
  )
  const grouped = groupSessions(filteredSessions)

  // ──────────────────── AUTH ────────────────────────────

  const inputCls = "w-full rounded-xl px-4 py-3 text-white placeholder:text-white/35 focus:outline-none focus:ring-2 focus:ring-primary/60 transition-all font-medium"
  const inputStyle = { background: "rgba(136,141,223,0.12)", border: "1px solid rgba(136,141,223,0.45)" }
  const labelCls = "text-xs font-bold text-white/80 tracking-widest uppercase mb-1.5 block"

  if (view === "login" || view === "register") {
    return (
      <div className="min-h-screen flex items-center justify-center p-4 relative overflow-hidden"
        style={{ background: "rgba(255,255,255,0.1)" }}>
        {/* Ambient glows */}
        <div className="absolute inset-0 pointer-events-none overflow-hidden">
          <div className="absolute top-1/4 left-1/3 w-[500px] h-[500px] rounded-full blur-[120px]"
            style={{ background: "rgba(136,141,223,0.22)" }} />
          <div className="absolute bottom-1/4 right-1/4 w-80 h-80 rounded-full blur-[100px]"
            style={{ background: "rgba(195,172,218,0.16)" }} />
          <div className="absolute inset-0 opacity-[0.15]"
            style={{
              backgroundImage: "radial-gradient(circle, rgba(200,195,255,0.5) 1px, transparent 1px)",
              backgroundSize: "48px 48px",
            }}
          />
        </div>

        <AnimatePresence mode="wait">
          {view === "login" ? (
            <motion.div key="login"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -16 }}
              transition={{ duration: 0.3, ease: "easeOut" }}
              className="w-full max-w-md relative z-10"
            >
              <div className="rounded-2xl p-8"
                style={{
                  background: "rgba(26,25,41,0.96)",
                }}
              >
                {/* Brand */}
                <div className="flex flex-col items-center mb-8">
                  <PondrLogo className="w-full max-w-sm h-[104px] overflow-hidden" />
                  <p className="text-white/75 text-[14px] tracking-[0.25em] uppercase font-nunito font-bold mt-2">
                    The Ponder Engine
                  </p>
                </div>

                <p className="text-center text-white font-nunito font-bold text-xl mb-6">
                  Welcome back
                </p>

                <form onSubmit={handleLogin} className="space-y-4">
                  <div>
                    <label className={labelCls}>Username</label>
                    <input
                      type="text"
                      value={loginUsername}
                      onChange={e => setLoginUsername(e.target.value)}
                      placeholder="your_username"
                      autoComplete="username"
                      className={inputCls}
                      style={inputStyle}
                    />
                  </div>
                  <div>
                    <label className={labelCls}>Password</label>
                    <div className="relative">
                      <input
                        type={showLoginPw ? "text" : "password"}
                        value={loginPw}
                        onChange={e => setLoginPw(e.target.value)}
                        placeholder="••••••••"
                        autoComplete="current-password"
                        className={inputCls + " pr-11 no-password-reveal"}
                        style={inputStyle}
                      />
                      <button type="button" onClick={() => setShowLoginPw(!showLoginPw)}
                        className="absolute right-3.5 top-1/2 -translate-y-1/2 text-white/50 hover:text-white/90 transition-colors">
                        {showLoginPw ? <EyeOff size={15} /> : <Eye size={15} />}
                      </button>
                    </div>
                  </div>
                  <button
                    type="submit"
                    className="w-full bg-primary hover:bg-primary/85 text-white font-nunito font-bold py-3 rounded-xl transition-all active:scale-[0.98] mt-1"
                    style={{ boxShadow: "0 4px 24px rgba(136,141,223,0.45)" }}
                  >
                    Sign In
                  </button>
                </form>

                <p className="text-center text-sm text-white/55 mt-6">
                  {"Don't have an account? "}
                  <button onClick={() => setView("register")}
                    className="text-primary hover:text-primary/80 font-semibold transition-colors">
                    Create one
                  </button>
                </p>
              </div>
            </motion.div>
          ) : (
            <motion.div key="register"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -16 }}
              transition={{ duration: 0.3, ease: "easeOut" }}
              className="w-full max-w-md relative z-10"
            >
              <div className="rounded-2xl p-8"
                style={{
                  background: "rgba(26,25,41,0.96)",
                }}
              >
                {/* Brand */}
                <div className="flex flex-col items-center mb-6">
                  <PondrLogo className="w-full max-w-sm h-[104px] overflow-hidden" />
                  <p className="text-white/75 text-[14px] tracking-[0.25em] uppercase font-nunito font-bold mt-2">
                    The Ponder Engine
                  </p>
                </div>

                <p className="text-center text-white font-nunito font-bold text-xl mb-5">
                  Create your account
                </p>

                <form onSubmit={handleRegister} className="space-y-3.5">
                  <div>
                    <label className={labelCls}>Full Name</label>
                    <input
                      type="text"
                      value={regName}
                      onChange={e => setRegName(e.target.value)}
                      placeholder="Ada Lovelace"
                      className={inputCls}
                      style={inputStyle}
                    />
                  </div>
                  <div>
                    <label className={labelCls}>Username</label>
                    <input
                      type="text"
                      value={regUsername}
                      onChange={e => setRegUsername(e.target.value)}
                      placeholder="ada_lovelace"
                      autoComplete="username"
                      className={inputCls}
                      style={inputStyle}
                    />
                  </div>
                  <div>
                    <label className={labelCls}>Password</label>
                    <div className="relative">
                      <input
                        type={showRegPw ? "text" : "password"}
                        value={regPw}
                        onChange={e => setRegPw(e.target.value)}
                        placeholder="••••••••"
                        autoComplete="new-password"
                        className={inputCls + " pr-11 no-password-reveal"}
                        style={inputStyle}
                      />
                      <button type="button" onClick={() => setShowRegPw(!showRegPw)}
                        className="absolute right-3.5 top-1/2 -translate-y-1/2 text-white/50 hover:text-white/90 transition-colors">
                        {showRegPw ? <EyeOff size={15} /> : <Eye size={15} />}
                      </button>
                    </div>
                  </div>
                  <div>
                    <label className={labelCls}>Confirm Password</label>
                    <input
                      type="password"
                      value={regConfirm}
                      onChange={e => setRegConfirm(e.target.value)}
                      placeholder="••••••••"
                      autoComplete="new-password"
                      className={inputCls + " no-password-reveal"}
                      style={inputStyle}
                    />
                  </div>
                  <button
                    type="submit"
                    className="w-full bg-primary hover:bg-primary/85 text-white font-nunito font-bold py-3 rounded-xl transition-all active:scale-[0.98] mt-1"
                    style={{ boxShadow: "0 4px 24px rgba(136,141,223,0.45)" }}
                  >
                    Create Account
                  </button>
                </form>

                <p className="text-center text-sm text-white/55 mt-5">
                  Already have an account?{" "}
                  <button onClick={() => setView("login")}
                    className="text-primary hover:text-primary/80 font-semibold transition-colors">
                    Sign in
                  </button>
                </p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    )
  }

  // ──────────────────── SETTINGS ───────────────────────────

  const ACCENT_OPTIONS = [
    { label: "Periwinkle", value: "#888ddf" },
    { label: "Lilac", value: "#c3acda" },
    { label: "Sage", value: "#e0efe4" },
    { label: "Champagne", value: "#e8d7bd" },
    { label: "Ice Blue", value: "#e2e6ff" },
  ]

  const SETTINGS_NAV = [
    { id: "profile", label: "Profile", icon: UserIcon },
    { id: "appearance", label: "Appearance", icon: Palette },
    { id: "providers", label: "Providers", icon: Cpu },
    { id: "notifications", label: "Notifications", icon: Bell },
    { id: "security", label: "Security", icon: Shield },
    { id: "about", label: "About", icon: Info },
  ]

  function openNewProviderForm() {
    setEditingProviderId(null)
    setProviderForm({ name: "", baseUrl: "", apiKey: "" })
    setShowProviderKey(false)
    setAddingModelToId(null)
    setEditingModelId(null)
    setShowProviderForm(true)
  }

  function openEditProviderForm(p: Provider) {
    setEditingProviderId(p.id)
    setProviderForm({ name: p.name, baseUrl: p.baseUrl, apiKey: p.apiKey })
    setShowProviderKey(false)
    setShowProviderForm(true)
  }

  function saveProvider() {
    if (!providerForm.name.trim() || !providerForm.baseUrl.trim()) return
    if (editingProviderId) {
      setProviders(prev => prev.map(p => p.id === editingProviderId ? { ...p, ...providerForm } : p))
    } else {
      const id = `p${Date.now()}`
      setProviders(prev => [...prev, { id, ...providerForm, enabled: true, models: [] }])
      setExpandedProviderId(id)
    }
    setShowProviderForm(false)
    setEditingProviderId(null)
    setProviderSaved(true)
    setTimeout(() => setProviderSaved(false), 2200)
  }

  function deleteProvider(id: string) {
    setProviders(prev => prev.filter(p => p.id !== id))
    if (expandedProviderId === id) setExpandedProviderId(null)
    if (editingProviderId === id) setShowProviderForm(false)
  }

  function toggleProvider(id: string) {
    setProviders(prev => prev.map(p => p.id === id ? { ...p, enabled: !p.enabled } : p))
  }

  function openAddModelForm(providerId: string) {
    setAddingModelToId(providerId)
    setEditingModelId(null)
    setModelForm({ modelId: "", label: "" })
  }

  function openEditModelForm(providerId: string, m: ProviderModel) {
    setAddingModelToId(providerId)
    setEditingModelId(m.id)
    setModelForm({ modelId: m.modelId, label: m.label })
  }

  function saveModel(providerId: string) {
    if (!modelForm.modelId.trim()) return
    setProviders(prev => prev.map(p => {
      if (p.id !== providerId) return p
      if (editingModelId) {
        return { ...p, models: p.models.map(m => m.id === editingModelId ? { ...m, ...modelForm } : m) }
      }
      return { ...p, models: [...p.models, { id: `m${Date.now()}`, ...modelForm, enabled: true }] }
    }))
    setAddingModelToId(null)
    setEditingModelId(null)
    setProviderSaved(true)
    setTimeout(() => setProviderSaved(false), 2200)
  }

  function deleteModel(providerId: string, modelId: string) {
    setProviders(prev => prev.map(p =>
      p.id === providerId ? { ...p, models: p.models.filter(m => m.id !== modelId) } : p
    ))
  }

  function toggleModel(providerId: string, modelId: string) {
    setProviders(prev => prev.map(p =>
      p.id === providerId
        ? { ...p, models: p.models.map(m => m.id === modelId ? { ...m, enabled: !m.enabled } : m) }
        : p
    ))
  }

  if (view === "settings") {
    return (
      <div className="flex h-screen overflow-hidden" style={{ background: "#0c0b1a" }}>
        {/* Settings sidebar */}
        <aside className="w-64 flex-shrink-0 flex flex-col border-r" style={{ background: "#1a1929", borderColor: "rgba(136,141,223,0.18)" }}>
          <div className="px-4 py-4 border-b flex items-center gap-3" style={{ borderColor: "rgba(136,141,223,0.18)" }}>
            <button
              onClick={() => setView("chat")}
              className="w-8 h-8 rounded-lg flex items-center justify-center transition-all hover:scale-105 flex-shrink-0"
              style={{ background: "rgba(136,141,223,0.15)", color: "#888ddf" }}
              title="Back to chat"
            >
              <ArrowLeft size={15} />
            </button>
            <span className="font-nunito font-bold text-white text-base">Settings</span>
          </div>

          {/* User summary */}
          <div className="px-4 py-4 border-b" style={{ borderColor: "rgba(136,141,223,0.18)" }}>
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0"
                style={{ background: "rgba(136,141,223,0.22)" }}>
                <UserIcon size={16} className="text-primary" />
              </div>
              <div className="min-w-0">
                <p className="text-sm font-semibold text-white truncate">{displayName || "Ada Lovelace"}</p>
                <p className="text-xs truncate" style={{ color: "#888ddf" }}>@{username || "ada_lovelace"}</p>
              </div>
            </div>
          </div>

          {/* Nav */}
          <nav className="flex-1 px-3 py-3 space-y-0.5">
            {SETTINGS_NAV.map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                onClick={() => setSettingsSection(id)}
                className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all text-left ${
                  settingsSection === id ? "text-white" : "text-white/55 hover:text-white/80 hover:bg-white/5"
                }`}
                style={settingsSection === id ? { background: "rgba(136,141,223,0.18)", color: "#fff" } : undefined}
              >
                <Icon size={14} className="flex-shrink-0" style={settingsSection === id ? { color: "#888ddf" } : undefined} />
                {label}
                {settingsSection === id && <ChevronRight size={12} className="ml-auto opacity-60" />}
              </button>
            ))}
          </nav>

          {/* Sign out */}
          <div className="px-3 py-3 border-t" style={{ borderColor: "rgba(136,141,223,0.18)" }}>
            <button
              onClick={() => setView("login")}
              className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium text-white/50 hover:text-red-400 hover:bg-red-500/10 transition-all"
            >
              <LogOut size={14} className="flex-shrink-0" />
              Sign out
            </button>
          </div>
        </aside>

        {/* Settings content */}
        <main className="flex-1 overflow-y-auto px-8 py-8 scrollbar-hide">
          <AnimatePresence mode="wait">
            <motion.div
              key={settingsSection}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -6 }}
              transition={{ duration: 0.18 }}
              className="max-w-xl"
            >
              {/* PROFILE */}
              {settingsSection === "profile" && (
                <div className="space-y-6">
                  <div>
                    <h2 className="text-white font-nunito font-bold text-xl mb-1">Profile</h2>
                    <p className="text-sm" style={{ color: "#9b96c8" }}>Manage how you appear in Pondr.</p>
                  </div>

                  {/* Avatar */}
                  <div className="flex items-center gap-4">
                    <div className="w-16 h-16 rounded-full flex items-center justify-center flex-shrink-0"
                      style={{ background: "rgba(136,141,223,0.22)", border: "2px solid rgba(136,141,223,0.35)" }}>
                      <UserIcon size={24} style={{ color: "#888ddf" }} />
                    </div>
                    <div>
                      <p className="text-sm font-semibold text-white mb-1">Profile picture</p>
                      <p className="text-xs mb-2" style={{ color: "#9b96c8" }}>JPG, PNG or GIF, max 2 MB</p>
                      <button className="text-xs font-semibold px-3 py-1.5 rounded-lg transition-all"
                        style={{ background: "rgba(136,141,223,0.15)", color: "#888ddf", border: "1px solid rgba(136,141,223,0.3)" }}>
                        Upload photo
                      </button>
                    </div>
                  </div>

                  <div className="space-y-4">
                    {[
                      { label: "Display Name", value: displayName, setter: setDisplayName, placeholder: "Ada Lovelace" },
                      { label: "Username", value: username, setter: setUsername, placeholder: "ada_lovelace" },
                    ].map(({ label, value, setter, placeholder }) => (
                      <div key={label}>
                        <label className="text-xs font-bold tracking-widest uppercase block mb-1.5" style={{ color: "rgba(236,233,255,0.7)" }}>{label}</label>
                        <input
                          type="text"
                          value={value}
                          onChange={e => setter(e.target.value)}
                          placeholder={placeholder}
                          className="w-full rounded-xl px-4 py-3 text-white placeholder:text-white/30 focus:outline-none focus:ring-2 transition-all text-sm"
                          style={{ background: "rgba(136,141,223,0.1)", border: "1px solid rgba(136,141,223,0.35)", focusRingColor: "#888ddf" } as React.CSSProperties}
                        />
                      </div>
                    ))}
                    <div>
                      <label className="text-xs font-bold tracking-widest uppercase block mb-1.5" style={{ color: "rgba(236,233,255,0.7)" }}>Bio</label>
                      <textarea
                        value={bio}
                        onChange={e => setBio(e.target.value)}
                        placeholder="Tell us a little about yourself…"
                        rows={3}
                        className="w-full rounded-xl px-4 py-3 text-white placeholder:text-white/30 focus:outline-none focus:ring-2 transition-all text-sm resize-none"
                        style={{ background: "rgba(136,141,223,0.1)", border: "1px solid rgba(136,141,223,0.35)" }}
                      />
                    </div>
                  </div>

                  <button className="px-5 py-2.5 rounded-xl text-sm font-nunito font-bold transition-all active:scale-[0.98]"
                    style={{ background: "#888ddf", color: "#0c0b1a", boxShadow: "0 4px 16px rgba(136,141,223,0.35)" }}>
                    Save changes
                  </button>
                </div>
              )}

              {/* APPEARANCE */}
              {settingsSection === "appearance" && (
                <div className="space-y-6">
                  <div>
                    <h2 className="text-white font-nunito font-bold text-xl mb-1">Appearance</h2>
                    <p className="text-sm" style={{ color: "#9b96c8" }}>Personalise the look of Pondr.</p>
                  </div>

                  <div>
                    <label className="text-xs font-bold tracking-widest uppercase block mb-3" style={{ color: "rgba(236,233,255,0.7)" }}>Accent color</label>
                    <div className="flex flex-wrap gap-3">
                      {ACCENT_OPTIONS.map(opt => (
                        <button
                          key={opt.value}
                          onClick={() => setAccentColor(opt.value)}
                          className="flex items-center gap-2.5 px-4 py-2.5 rounded-xl text-sm font-medium transition-all"
                          style={{
                            background: accentColor === opt.value ? `${opt.value}22` : "rgba(255,255,255,0.05)",
                            border: `1px solid ${accentColor === opt.value ? opt.value : "rgba(255,255,255,0.1)"}`,
                            color: accentColor === opt.value ? opt.value : "rgba(255,255,255,0.6)",
                          }}
                        >
                          <span className="w-3.5 h-3.5 rounded-full flex-shrink-0 flex items-center justify-center"
                            style={{ background: opt.value }}>
                            {accentColor === opt.value && <Check size={9} style={{ color: "#0c0b1a" }} />}
                          </span>
                          {opt.label}
                        </button>
                      ))}
                    </div>
                  </div>

                  <div>
                    <label className="text-xs font-bold tracking-widest uppercase block mb-3" style={{ color: "rgba(236,233,255,0.7)" }}>Theme</label>
                    <div className="flex gap-3">
                      {["Dark", "System"].map(t => (
                        <button key={t}
                          className="flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-medium transition-all"
                          style={{
                            background: t === "Dark" ? "rgba(136,141,223,0.18)" : "rgba(255,255,255,0.05)",
                            border: `1px solid ${t === "Dark" ? "rgba(136,141,223,0.45)" : "rgba(255,255,255,0.1)"}`,
                            color: t === "Dark" ? "#fff" : "rgba(255,255,255,0.5)",
                          }}>
                          {t === "Dark" && <Check size={12} style={{ color: "#888ddf" }} />}
                          {t}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {/* PROVIDERS */}
              {settingsSection === "providers" && (
                <div className="space-y-5">
                  <div>
                    <h2 className="text-white font-nunito font-bold text-xl mb-1">Providers</h2>
                    <p className="text-sm" style={{ color: "#9b96c8" }}>Connect OpenAI-compatible endpoints and configure their models.</p>
                  </div>

                  {/* Accordion list */}
                  {providers.length > 0 && (
                    <div className="space-y-2">
                      {providers.map(p => {
                        const isOpen = expandedProviderId === p.id
                        const isEditingThis = showProviderForm && editingProviderId === p.id
                        return (
                          <div key={p.id} className="rounded-xl overflow-hidden transition-all"
                            style={{ border: `1px solid ${isOpen ? "rgba(136,141,223,0.4)" : "rgba(136,141,223,0.18)"}`, background: "rgba(136,141,223,0.06)" }}>

                            {/* Accordion header */}
                            <div
                              className="flex items-center gap-3 px-4 py-3 cursor-pointer select-none hover:bg-white/5 transition-colors"
                              onClick={() => setExpandedProviderId(isOpen ? null : p.id)}
                            >
                              <motion.div animate={{ rotate: isOpen ? 90 : 0 }} transition={{ duration: 0.18 }}>
                                <ChevronRight size={14} style={{ color: "#888ddf" }} />
                              </motion.div>
                              <div className="flex-1 min-w-0">
                                <div className="flex items-center gap-2">
                                  <p className="text-sm font-semibold text-white truncate">{p.name}</p>
                                  <span className="text-[10px] font-bold px-1.5 py-0.5 rounded-full flex-shrink-0"
                                    style={p.enabled
                                      ? { background: "rgba(224,239,228,0.15)", color: "#e0efe4" }
                                      : { background: "rgba(255,255,255,0.06)", color: "rgba(255,255,255,0.35)" }}>
                                    {p.enabled ? "Active" : "Disabled"}
                                  </span>
                                </div>
                                <p className="text-xs truncate mt-0.5" style={{ color: "#9b96c8" }}>
                                  {p.baseUrl.replace(/https?:\/\//, "")}
                                  {p.models.length > 0 && <span style={{ color: "#888ddf" }}> · {p.models.length} model{p.models.length !== 1 ? "s" : ""}</span>}
                                </p>
                              </div>
                              <div className="flex items-center gap-1 flex-shrink-0" onClick={e => e.stopPropagation()}>
                                <button onClick={() => toggleProvider(p.id)} className="p-1.5 rounded-lg hover:bg-white/10 transition-colors" title={p.enabled ? "Disable" : "Enable"}>
                                  {p.enabled ? <ToggleRight size={16} style={{ color: "#888ddf" }} /> : <ToggleLeft size={16} className="text-white/30" />}
                                </button>
                                <button onClick={() => { openEditProviderForm(p); setExpandedProviderId(p.id) }} className="p-1.5 rounded-lg text-white/35 hover:text-primary hover:bg-white/10 transition-colors" title="Edit provider">
                                  <Pencil size={12} />
                                </button>
                                <button onClick={() => deleteProvider(p.id)} className="p-1.5 rounded-lg text-white/35 hover:text-red-400 hover:bg-red-500/10 transition-colors" title="Remove provider">
                                  <Trash2 size={12} />
                                </button>
                              </div>
                            </div>

                            {/* Accordion body */}
                            <AnimatePresence initial={false}>
                              {isOpen && (
                                <motion.div
                                  initial={{ height: 0, opacity: 0 }}
                                  animate={{ height: "auto", opacity: 1 }}
                                  exit={{ height: 0, opacity: 0 }}
                                  transition={{ duration: 0.22, ease: "easeOut" }}
                                  style={{ overflow: "hidden" }}
                                >
                                  <div className="px-4 pb-4 space-y-2 border-t" style={{ borderColor: "rgba(136,141,223,0.15)" }}>

                                    {/* Edit provider form (inline) */}
                                    <AnimatePresence>
                                      {isEditingThis && (
                                        <motion.div initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                                          className="mt-3 rounded-xl p-4 space-y-3"
                                          style={{ background: "rgba(136,141,223,0.1)", border: "1px solid rgba(136,141,223,0.3)" }}>
                                          <p className="text-xs font-nunito font-bold text-white/80 uppercase tracking-widest">Edit provider</p>
                                          {[
                                            { label: "Provider name", key: "name", type: "text", placeholder: "e.g. OpenAI, Groq, Ollama" },
                                            { label: "Base URL", key: "baseUrl", type: "url", placeholder: "https://api.openai.com/v1" },
                                          ].map(({ label, key, type, placeholder }) => (
                                            <div key={key}>
                                              <label className="text-xs font-bold tracking-widest uppercase block mb-1" style={{ color: "rgba(236,233,255,0.6)" }}>{label}</label>
                                              <input type={type} value={(providerForm as any)[key]} onChange={e => setProviderForm(f => ({ ...f, [key]: e.target.value }))}
                                                placeholder={placeholder}
                                                className="w-full rounded-lg px-3 py-2.5 text-white placeholder:text-white/25 focus:outline-none text-sm transition-all"
                                                style={{ background: "rgba(255,255,255,0.07)", border: "1px solid rgba(136,141,223,0.3)" }} />
                                            </div>
                                          ))}
                                          <div>
                                            <label className="text-xs font-bold tracking-widest uppercase block mb-1" style={{ color: "rgba(236,233,255,0.6)" }}>API key</label>
                                            <div className="relative">
                                              <KeyRound size={13} className="absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none" style={{ color: "#888ddf" }} />
                                              <input type={showProviderKey ? "text" : "password"} value={providerForm.apiKey}
                                                onChange={e => setProviderForm(f => ({ ...f, apiKey: e.target.value }))}
                                                placeholder="sk-••••••••••••••••"
                                                className="no-password-reveal w-full rounded-lg pl-8 pr-10 py-2.5 text-white placeholder:text-white/25 focus:outline-none text-sm transition-all"
                                                style={{ background: "rgba(255,255,255,0.07)", border: "1px solid rgba(136,141,223,0.3)" }} />
                                              <button type="button" onClick={() => setShowProviderKey(v => !v)}
                                                className="absolute right-3 top-1/2 -translate-y-1/2" style={{ color: "rgba(255,255,255,0.35)" }}>
                                                {showProviderKey ? <EyeOff size={13} /> : <Eye size={13} />}
                                              </button>
                                            </div>
                                          </div>
                                          <div className="flex gap-2 pt-1">
                                            <button onClick={saveProvider} disabled={!providerForm.name.trim() || !providerForm.baseUrl.trim()}
                                              className="px-4 py-2 rounded-lg text-sm font-nunito font-bold transition-all active:scale-[0.97] disabled:opacity-40"
                                              style={{ background: "#888ddf", color: "#0c0b1a" }}>Save</button>
                                            <button onClick={() => { setShowProviderForm(false); setEditingProviderId(null) }}
                                              className="px-4 py-2 rounded-lg text-sm text-white/55 hover:text-white transition-colors"
                                              style={{ background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.1)" }}>Cancel</button>
                                          </div>
                                        </motion.div>
                                      )}
                                    </AnimatePresence>

                                    {/* Models list */}
                                    {p.models.length > 0 && (
                                      <div className="mt-3 space-y-1.5">
                                        <p className="text-[10px] font-bold uppercase tracking-widest px-1 mb-2" style={{ color: "rgba(136,141,223,0.7)" }}>Models</p>
                                        {p.models.map(m => (
                                          <div key={m.id}>
                                            <div className="flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all"
                                              style={{ background: "rgba(255,255,255,0.04)", border: "1px solid rgba(136,141,223,0.14)" }}>
                                              <div className="flex-1 min-w-0">
                                                <p className="text-sm font-medium text-white truncate">{m.label || m.modelId}</p>
                                                {m.label && <p className="text-xs truncate" style={{ color: "#888ddf" }}>{m.modelId}</p>}
                                              </div>
                                              <div className="flex items-center gap-1 flex-shrink-0">
                                                <button onClick={() => toggleModel(p.id, m.id)} className="p-1 rounded hover:bg-white/10 transition-colors" title={m.enabled ? "Disable" : "Enable"}>
                                                  {m.enabled ? <ToggleRight size={15} style={{ color: "#888ddf" }} /> : <ToggleLeft size={15} className="text-white/25" />}
                                                </button>
                                                <button onClick={() => openEditModelForm(p.id, m)} className="p-1 rounded text-white/30 hover:text-primary hover:bg-white/10 transition-colors">
                                                  <Pencil size={11} />
                                                </button>
                                                <button onClick={() => deleteModel(p.id, m.id)} className="p-1 rounded text-white/30 hover:text-red-400 hover:bg-red-500/10 transition-colors">
                                                  <Trash2 size={11} />
                                                </button>
                                              </div>
                                            </div>

                                            {/* Inline edit model form */}
                                            <AnimatePresence>
                                              {addingModelToId === p.id && editingModelId === m.id && (
                                                <motion.div initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                                                  className="mt-1.5 rounded-lg p-3 space-y-2.5"
                                                  style={{ background: "rgba(136,141,223,0.1)", border: "1px solid rgba(136,141,223,0.28)" }}>
                                                  <div className="grid grid-cols-2 gap-2">
                                                    <div>
                                                      <label className="text-[10px] font-bold uppercase tracking-widest block mb-1" style={{ color: "rgba(236,233,255,0.6)" }}>Model ID</label>
                                                      <input type="text" value={modelForm.modelId} onChange={e => setModelForm(f => ({ ...f, modelId: e.target.value }))}
                                                        placeholder="gpt-4o" className="w-full rounded-lg px-3 py-2 text-white placeholder:text-white/25 focus:outline-none text-sm"
                                                        style={{ background: "rgba(255,255,255,0.07)", border: "1px solid rgba(136,141,223,0.28)" }} />
                                                    </div>
                                                    <div>
                                                      <label className="text-[10px] font-bold uppercase tracking-widest block mb-1" style={{ color: "rgba(236,233,255,0.6)" }}>Display label</label>
                                                      <input type="text" value={modelForm.label} onChange={e => setModelForm(f => ({ ...f, label: e.target.value }))}
                                                        placeholder="GPT-4o" className="w-full rounded-lg px-3 py-2 text-white placeholder:text-white/25 focus:outline-none text-sm"
                                                        style={{ background: "rgba(255,255,255,0.07)", border: "1px solid rgba(136,141,223,0.28)" }} />
                                                    </div>
                                                  </div>
                                                  <div className="flex gap-2">
                                                    <button onClick={() => saveModel(p.id)} disabled={!modelForm.modelId.trim()}
                                                      className="px-3 py-1.5 rounded-lg text-xs font-bold transition-all disabled:opacity-40"
                                                      style={{ background: "#888ddf", color: "#0c0b1a" }}>Save</button>
                                                    <button onClick={() => { setAddingModelToId(null); setEditingModelId(null) }}
                                                      className="px-3 py-1.5 rounded-lg text-xs text-white/50 hover:text-white transition-colors"
                                                      style={{ background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.1)" }}>Cancel</button>
                                                  </div>
                                                </motion.div>
                                              )}
                                            </AnimatePresence>
                                          </div>
                                        ))}
                                      </div>
                                    )}

                                    {/* Empty models state */}
                                    {p.models.length === 0 && addingModelToId !== p.id && (
                                      <p className="text-xs text-center py-3" style={{ color: "#9b96c8" }}>No models added yet.</p>
                                    )}

                                    {/* Add model form */}
                                    <AnimatePresence>
                                      {addingModelToId === p.id && editingModelId === null && (
                                        <motion.div initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                                          className="mt-2 rounded-lg p-3 space-y-2.5"
                                          style={{ background: "rgba(136,141,223,0.1)", border: "1px solid rgba(136,141,223,0.28)" }}>
                                          <p className="text-[10px] font-bold uppercase tracking-widest" style={{ color: "rgba(136,141,223,0.8)" }}>Add model</p>
                                          <div className="grid grid-cols-2 gap-2">
                                            <div>
                                              <label className="text-[10px] font-bold uppercase tracking-widest block mb-1" style={{ color: "rgba(236,233,255,0.6)" }}>Model ID</label>
                                              <input type="text" value={modelForm.modelId} onChange={e => setModelForm(f => ({ ...f, modelId: e.target.value }))}
                                                placeholder="gpt-4o" autoFocus
                                                className="w-full rounded-lg px-3 py-2 text-white placeholder:text-white/25 focus:outline-none text-sm"
                                                style={{ background: "rgba(255,255,255,0.07)", border: "1px solid rgba(136,141,223,0.28)" }} />
                                            </div>
                                            <div>
                                              <label className="text-[10px] font-bold uppercase tracking-widest block mb-1" style={{ color: "rgba(236,233,255,0.6)" }}>Display label</label>
                                              <input type="text" value={modelForm.label} onChange={e => setModelForm(f => ({ ...f, label: e.target.value }))}
                                                placeholder="GPT-4o (optional)"
                                                className="w-full rounded-lg px-3 py-2 text-white placeholder:text-white/25 focus:outline-none text-sm"
                                                style={{ background: "rgba(255,255,255,0.07)", border: "1px solid rgba(136,141,223,0.28)" }} />
                                            </div>
                                          </div>
                                          <div className="flex gap-2">
                                            <button onClick={() => saveModel(p.id)} disabled={!modelForm.modelId.trim()}
                                              className="px-3 py-1.5 rounded-lg text-xs font-bold transition-all disabled:opacity-40 active:scale-[0.97]"
                                              style={{ background: "#888ddf", color: "#0c0b1a" }}>Add model</button>
                                            <button onClick={() => setAddingModelToId(null)}
                                              className="px-3 py-1.5 rounded-lg text-xs text-white/50 hover:text-white transition-colors"
                                              style={{ background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.1)" }}>Cancel</button>
                                          </div>
                                        </motion.div>
                                      )}
                                    </AnimatePresence>

                                    {/* Add model button */}
                                    {addingModelToId !== p.id && (
                                      <button onClick={() => openAddModelForm(p.id)}
                                        className="mt-1 flex items-center gap-1.5 text-xs font-semibold px-3 py-2 rounded-lg transition-all"
                                        style={{ color: "#888ddf", background: "rgba(136,141,223,0.1)", border: "1px solid rgba(136,141,223,0.22)" }}>
                                        <Plus size={12} />
                                        Add model
                                      </button>
                                    )}
                                  </div>
                                </motion.div>
                              )}
                            </AnimatePresence>
                          </div>
                        )
                      })}
                    </div>
                  )}

                  {/* Empty state */}
                  {providers.length === 0 && !showProviderForm && (
                    <div className="rounded-xl px-6 py-10 text-center"
                      style={{ background: "rgba(136,141,223,0.06)", border: "1px dashed rgba(136,141,223,0.25)" }}>
                      <Cpu size={28} className="mx-auto mb-3 opacity-30" style={{ color: "#888ddf" }} />
                      <p className="text-sm font-semibold text-white/60 mb-1">No providers yet</p>
                      <p className="text-xs" style={{ color: "#9b96c8" }}>Add an OpenAI-compatible service to get started.</p>
                    </div>
                  )}

                  {/* Add new provider form */}
                  <AnimatePresence>
                    {showProviderForm && !editingProviderId && (
                      <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 6 }} transition={{ duration: 0.18 }}
                        className="rounded-xl p-5 space-y-4"
                        style={{ background: "rgba(136,141,223,0.08)", border: "1px solid rgba(136,141,223,0.35)" }}>
                        <p className="text-sm font-nunito font-bold text-white">New provider</p>
                        {[
                          { label: "Provider name", key: "name", type: "text", placeholder: "e.g. OpenAI, Groq, Ollama" },
                          { label: "Base URL", key: "baseUrl", type: "url", placeholder: "https://api.openai.com/v1" },
                        ].map(({ label, key, type, placeholder }) => (
                          <div key={key}>
                            <label className="text-xs font-bold tracking-widest uppercase block mb-1.5" style={{ color: "rgba(236,233,255,0.7)" }}>{label}</label>
                            <input type={type} value={(providerForm as any)[key]} onChange={e => setProviderForm(f => ({ ...f, [key]: e.target.value }))}
                              placeholder={placeholder}
                              className="w-full rounded-xl px-4 py-3 text-white placeholder:text-white/30 focus:outline-none transition-all text-sm"
                              style={{ background: "rgba(255,255,255,0.07)", border: "1px solid rgba(136,141,223,0.35)" }} />
                          </div>
                        ))}
                        <div>
                          <label className="text-xs font-bold tracking-widest uppercase block mb-1.5" style={{ color: "rgba(236,233,255,0.7)" }}>API key</label>
                          <div className="relative">
                            <KeyRound size={14} className="absolute left-3.5 top-1/2 -translate-y-1/2 pointer-events-none" style={{ color: "#888ddf" }} />
                            <input type={showProviderKey ? "text" : "password"} value={providerForm.apiKey}
                              onChange={e => setProviderForm(f => ({ ...f, apiKey: e.target.value }))}
                              placeholder="sk-••••••••••••••••"
                              className="no-password-reveal w-full rounded-xl pl-9 pr-11 py-3 text-white placeholder:text-white/30 focus:outline-none transition-all text-sm"
                              style={{ background: "rgba(255,255,255,0.07)", border: "1px solid rgba(136,141,223,0.35)" }} />
                            <button type="button" onClick={() => setShowProviderKey(v => !v)}
                              className="absolute right-3.5 top-1/2 -translate-y-1/2 transition-colors" style={{ color: "rgba(255,255,255,0.4)" }}>
                              {showProviderKey ? <EyeOff size={14} /> : <Eye size={14} />}
                            </button>
                          </div>
                          <p className="text-xs mt-1.5 flex items-center gap-1.5" style={{ color: "#9b96c8" }}>
                            <AlertCircle size={10} className="flex-shrink-0" />
                            Stored locally in your browser only.
                          </p>
                        </div>
                        <div className="flex gap-2 pt-1">
                          <button onClick={saveProvider} disabled={!providerForm.name.trim() || !providerForm.baseUrl.trim()}
                            className="px-5 py-2.5 rounded-xl text-sm font-nunito font-bold transition-all active:scale-[0.97] disabled:opacity-40 disabled:cursor-not-allowed"
                            style={{ background: "#888ddf", color: "#0c0b1a", boxShadow: "0 4px 14px rgba(136,141,223,0.35)" }}>
                            Add provider
                          </button>
                          <button onClick={() => { setShowProviderForm(false) }}
                            className="px-5 py-2.5 rounded-xl text-sm font-medium text-white/60 hover:text-white transition-colors"
                            style={{ background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.1)" }}>
                            Cancel
                          </button>
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>

                  {/* Saved toast */}
                  <AnimatePresence>
                    {providerSaved && (
                      <motion.div initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                        className="flex items-center gap-2 text-sm px-4 py-2.5 rounded-xl"
                        style={{ background: "rgba(224,239,228,0.12)", border: "1px solid rgba(224,239,228,0.3)", color: "#e0efe4" }}>
                        <Check size={13} />
                        Saved successfully.
                      </motion.div>
                    )}
                  </AnimatePresence>

                  {!showProviderForm && (
                    <button onClick={openNewProviderForm}
                      className="flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-semibold transition-all active:scale-[0.97]"
                      style={{ background: "rgba(136,141,223,0.15)", border: "1px solid rgba(136,141,223,0.35)", color: "#888ddf" }}>
                      <Plus size={14} />
                      Add provider
                    </button>
                  )}
                </div>
              )}

              {/* NOTIFICATIONS */}
              {settingsSection === "notifications" && (
                <div className="space-y-6">
                  <div>
                    <h2 className="text-white font-nunito font-bold text-xl mb-1">Notifications</h2>
                    <p className="text-sm" style={{ color: "#9b96c8" }}>Control when and how Pondr alerts you.</p>
                  </div>

                  <div className="space-y-3">
                    {[
                      { label: "Message notifications", desc: "Get notified when Pondr finishes a response", value: notifMessages, setter: setNotifMessages },
                      { label: "Sound effects", desc: "Play a soft chime when responses arrive", value: notifSounds, setter: setNotifSounds },
                    ].map(({ label, desc, value, setter }) => (
                      <div key={label}
                        className="flex items-center justify-between px-4 py-4 rounded-xl"
                        style={{ background: "rgba(136,141,223,0.08)", border: "1px solid rgba(136,141,223,0.2)" }}>
                        <div>
                          <p className="text-sm font-semibold text-white">{label}</p>
                          <p className="text-xs mt-0.5" style={{ color: "#9b96c8" }}>{desc}</p>
                        </div>
                        <button
                          onClick={() => setter(!value)}
                          className="relative w-11 h-6 rounded-full flex-shrink-0 transition-all"
                          style={{ background: value ? "#888ddf" : "rgba(255,255,255,0.12)" }}
                        >
                          <span
                            className="absolute top-0.5 w-5 h-5 rounded-full transition-all"
                            style={{
                              background: "#fff",
                              left: value ? "calc(100% - 1.375rem)" : "0.125rem",
                              boxShadow: "0 1px 4px rgba(0,0,0,0.3)",
                            }}
                          />
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* SECURITY */}
              {settingsSection === "security" && (
                <div className="space-y-6">
                  <div>
                    <h2 className="text-white font-nunito font-bold text-xl mb-1">Security</h2>
                    <p className="text-sm" style={{ color: "#9b96c8" }}>Keep your account safe.</p>
                  </div>

                  <div className="space-y-4">
                    <div>
                      <label className="text-xs font-bold tracking-widest uppercase block mb-1.5" style={{ color: "rgba(236,233,255,0.7)" }}>Current password</label>
                      <input type="password" placeholder="••••••••" className="no-password-reveal w-full rounded-xl px-4 py-3 text-white placeholder:text-white/30 focus:outline-none transition-all text-sm"
                        style={{ background: "rgba(136,141,223,0.1)", border: "1px solid rgba(136,141,223,0.35)" }} />
                    </div>
                    <div>
                      <label className="text-xs font-bold tracking-widest uppercase block mb-1.5" style={{ color: "rgba(236,233,255,0.7)" }}>New password</label>
                      <input type="password" placeholder="••••••••" className="no-password-reveal w-full rounded-xl px-4 py-3 text-white placeholder:text-white/30 focus:outline-none transition-all text-sm"
                        style={{ background: "rgba(136,141,223,0.1)", border: "1px solid rgba(136,141,223,0.35)" }} />
                    </div>
                    <div>
                      <label className="text-xs font-bold tracking-widest uppercase block mb-1.5" style={{ color: "rgba(236,233,255,0.7)" }}>Confirm new password</label>
                      <input type="password" placeholder="••••••••" className="no-password-reveal w-full rounded-xl px-4 py-3 text-white placeholder:text-white/30 focus:outline-none transition-all text-sm"
                        style={{ background: "rgba(136,141,223,0.1)", border: "1px solid rgba(136,141,223,0.35)" }} />
                    </div>
                  </div>

                  <button className="px-5 py-2.5 rounded-xl text-sm font-nunito font-bold transition-all active:scale-[0.98]"
                    style={{ background: "#888ddf", color: "#0c0b1a", boxShadow: "0 4px 16px rgba(136,141,223,0.35)" }}>
                    Update password
                  </button>

                  <div className="pt-4 border-t" style={{ borderColor: "rgba(136,141,223,0.15)" }}>
                    <p className="text-xs font-bold tracking-widest uppercase mb-3" style={{ color: "rgba(236,233,255,0.5)" }}>Danger zone</p>
                    <button className="px-5 py-2.5 rounded-xl text-sm font-medium transition-all"
                      style={{ background: "rgba(212,24,61,0.12)", border: "1px solid rgba(212,24,61,0.3)", color: "#f87171" }}>
                      Delete account
                    </button>
                  </div>
                </div>
              )}

              {/* ABOUT */}
              {settingsSection === "about" && (
                <div className="space-y-6">
                  <div>
                    <h2 className="text-white font-nunito font-bold text-xl mb-1">About</h2>
                    <p className="text-sm" style={{ color: "#9b96c8" }}>Pondr version and legal information.</p>
                  </div>

                  <div className="flex items-center gap-4 px-4 py-4 rounded-xl" style={{ background: "rgba(136,141,223,0.08)", border: "1px solid rgba(136,141,223,0.2)" }}>
                    <div className="w-12 h-12 flex-shrink-0">
                      <ThoughtSpark className="w-full h-full" />
                    </div>
                    <div>
                      <p className="text-white font-nunito font-bold">Pondr</p>
                      <p className="text-xs mt-0.5" style={{ color: "#888ddf" }}>The Ponder Engine</p>
                      <p className="text-xs mt-0.5" style={{ color: "#9b96c8" }}>Version 1.0.0</p>
                    </div>
                  </div>

                  <div className="space-y-2">
                    {["Privacy Policy", "Terms of Service", "Open Source Licenses"].map(item => (
                      <button key={item}
                        className="w-full flex items-center justify-between px-4 py-3 rounded-xl text-sm transition-all text-white/70 hover:text-white"
                        style={{ background: "rgba(255,255,255,0.04)", border: "1px solid rgba(136,141,223,0.15)" }}>
                        {item}
                        <ChevronRight size={14} className="opacity-40" />
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </motion.div>
          </AnimatePresence>
        </main>
      </div>
    )
  }

  // ──────────────────── CHAT LAYOUT ────────────────────────

  return (
    <div className="flex h-screen bg-background overflow-hidden">

      {/* Mobile sidebar overlay */}
      <AnimatePresence>
        {sidebarOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="fixed inset-0 bg-black/60 z-20 lg:hidden backdrop-blur-sm"
            onClick={() => setSidebarOpen(false)}
          />
        )}
      </AnimatePresence>

      {/* Sidebar */}
      <motion.aside
        onMouseEnter={() => setSidebarCollapsed(false)}
        onMouseLeave={() => setSidebarCollapsed(true)}
        animate={{ width: sidebarCollapsed ? 64 : 288 }}
        transition={{ duration: 0.25, ease: "easeOut" }}
        className={`
          fixed lg:static inset-y-0 left-0 z-30 lg:z-auto
          flex-shrink-0 flex flex-col
          border-r border-sidebar-border
          transition-transform duration-300 ease-out
          lg:translate-x-0
          ${sidebarOpen ? "translate-x-0" : "-translate-x-full"}
        `}
        style={{ background: "var(--sidebar)" }}
      >
        {/* Sidebar top */}
        <div className="flex items-center justify-between px-4 py-4 border-b border-sidebar-border flex-shrink-0">
          <div className="flex items-center gap-2.5 min-w-0">
            <div className="relative w-11 h-11 flex-shrink-0">
              <div className="absolute inset-0 rounded-full blur-sm" style={{ background: "rgba(136,141,223,0.4)" }} />
              <ThoughtSpark className="w-full h-full relative z-10" />
            </div>
            <motion.div
              initial={false}
              animate={{ opacity: sidebarCollapsed ? 0 : 1, width: sidebarCollapsed ? 0 : "auto" }}
              transition={{ duration: 0.2 }}
              className="overflow-hidden"
            >
              <PondrLogo className="h-[40px] w-[174px] flex-shrink-0" />
            </motion.div>
          </div>
          <button onClick={() => setSidebarOpen(false)}
            className="lg:hidden text-white/50 hover:text-white transition-colors p-1 flex-shrink-0">
            <X size={17} />
          </button>
        </div>

        {/* New chat + search */}
        <div className="px-3 py-3 space-y-2 flex-shrink-0">
          <button
            onClick={handleNewChat}
            className={`w-full flex items-center justify-center text-white px-4 py-2.5 rounded-xl font-semibold text-sm transition-all active:scale-[0.97] ${
              sidebarCollapsed ? "" : "gap-2.5"
            }`}
            style={{ backgroundColor: "rgb(136,141,223)", boxShadow: "0 2px 16px rgba(136,141,223,0.3)" }}
            title={sidebarCollapsed ? "New Chat" : undefined}
          >
            <Plus size={15} className="flex-shrink-0" />
            <motion.span
              initial={false}
              animate={{ opacity: sidebarCollapsed ? 0 : 1, width: sidebarCollapsed ? 0 : "auto" }}
              transition={{ duration: 0.2 }}
              className="overflow-hidden whitespace-nowrap"
            >
              New Chat
            </motion.span>
          </button>

          {!sidebarCollapsed && (showSearch ? (
            <div className="relative">
              <Search size={13} className="absolute left-3 top-1/2 -translate-y-1/2 text-white/40 pointer-events-none" />
              <input
                type="text"
                value={searchQuery}
                onChange={e => setSearchQuery(e.target.value)}
                placeholder="Search conversations…"
                autoFocus
                className="w-full border border-border rounded-xl pl-8 pr-8 py-2 text-sm text-white placeholder:text-white/40 focus:outline-none focus:ring-1 focus:ring-primary/30 transition-all"
                style={{ background: "rgba(255,255,255,0.1)" }}
              />
              <button onClick={() => { setShowSearch(false); setSearchQuery("") }}
                className="absolute right-2.5 top-1/2 -translate-y-1/2 text-white/50 hover:text-white">
                <X size={12} />
              </button>
            </div>
          ) : (
            <button onClick={() => setShowSearch(true)}
              className="w-full flex items-center gap-2.5 text-white/50 hover:text-white/70 px-3 py-2 rounded-xl text-sm transition-colors hover:bg-sidebar-accent">
              <Search size={13} />
              Search conversations…
            </button>
          ))}

          {sidebarCollapsed && (
            <button
              onClick={() => setShowSearch(true)}
              className="w-full flex items-center justify-center text-white/50 hover:text-white/70 px-3 py-2 rounded-xl text-sm transition-colors hover:bg-sidebar-accent"
              title="Search conversations"
            >
              <Search size={13} />
            </button>
          )}

          {/* View Subconscious */}
          {!sidebarCollapsed ? (
            <motion.button
              onClick={() => setView("subconscious")}
              whileHover={{ scale: 1.01 }}
              whileTap={{ scale: 0.97 }}
              className="w-full flex items-center gap-2 px-3 py-2 rounded-xl text-sm font-medium transition-all"
              style={view === "subconscious"
                ? { background: "rgba(195,172,218,0.2)", border: "1px solid rgba(195,172,218,0.45)", color: "#e2e6ff" }
                : { background: "rgba(195,172,218,0.1)", border: "1px solid rgba(195,172,218,0.22)", color: "#c3acda" }}
            >
              <Brain size={12} style={{ color: "#e2e6ff" }} className="flex-shrink-0" />
              <span className="flex-1 text-left">View Subconscious</span>
              <span className="text-[9px] font-bold px-1.5 py-0.5 rounded-full flex-shrink-0"
                style={{ background: "rgba(136,141,223,0.18)", color: "#888ddf" }}>
                {BASE_NODES.length}
              </span>
            </motion.button>
          ) : (
            <button
              onClick={() => setShowSubconscious(true)}
              className="w-full flex items-center justify-center py-2 rounded-xl transition-colors hover:bg-sidebar-accent"
              style={{ color: "#c3acda" }}
              title="View Subconscious"
            >
              <Brain size={13} style={view === "subconscious" ? { color: "#e2e6ff" } : undefined} />
            </button>
          )}
        </div>

        {/* Sessions list */}
        <div className="flex-1 overflow-y-auto px-3 pb-3 scrollbar-hide">
          {grouped.length === 0 ? (
            !sidebarCollapsed && <p className="text-center text-xs text-white/40 py-8">No conversations yet</p>
          ) : (
            grouped.map(([label, group]) => (
              <div key={label} className="mb-4">
                {!sidebarCollapsed && (
                  <p className="text-[10px] font-bold text-white/40 uppercase tracking-widest px-2 mb-1.5">
                    {label}
                  </p>
                )}
                <div className="space-y-0.5">
                  {group.map(session => (
                    <div
                      key={session.id}
                      onClick={() => { setActiveId(session.id); setSidebarOpen(false) }}
                      className={`
                        group flex items-center rounded-xl cursor-pointer transition-all
                        ${sidebarCollapsed ? "justify-center px-3 py-2.5" : "gap-2.5 px-3 py-2.5"}
                        ${activeId === session.id
                          ? "text-white"
                          : "text-white/65 hover:text-white/90 hover:bg-sidebar-accent"
                        }
                      `}
                      style={activeId === session.id ? { background: "rgba(136,141,223,0.18)" } : undefined}
                      title={sidebarCollapsed ? session.name : undefined}
                    >
                      <MessageCircle size={13} className="flex-shrink-0 opacity-50" />
                      {!sidebarCollapsed && (
                        <>
                          <span className="text-sm flex-1 truncate">{session.name}</span>
                          <button
                            onClick={e => handleDeleteSession(session.id, e)}
                            className="opacity-0 group-hover:opacity-100 text-white/40 hover:text-destructive transition-all flex-shrink-0 p-0.5"
                          >
                            <Trash2 size={11} />
                          </button>
                        </>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            ))
          )}
        </div>

        {/* User footer */}
        <div className="border-t border-sidebar-border px-3 py-3 flex-shrink-0">
          <div
            onClick={() => !sidebarCollapsed && setView("settings")}
            className={`group rounded-xl hover:bg-sidebar-accent transition-colors cursor-pointer ${
              sidebarCollapsed ? "flex items-center justify-center px-2 py-2" : "flex items-center gap-3 px-2 py-2"
            }`}
            title={sidebarCollapsed ? "Settings" : undefined}
          >
            <div className="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0"
              style={{ background: "rgba(136,141,223,0.2)" }}>
              <UserIcon size={13} className="text-primary" />
            </div>
            {!sidebarCollapsed && (
              <>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-semibold text-white truncate">{displayName || "Ada Lovelace"}</p>
                  <p className="text-xs text-white/50 truncate">@{username || "ada_lovelace"}</p>
                </div>
                <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-all flex-shrink-0">
                  <button
                    onClick={e => { e.stopPropagation(); setView("settings") }}
                    className="text-white/50 hover:text-primary transition-colors p-0.5"
                    title="Settings">
                    <Settings size={13} />
                  </button>
                  <button
                    onClick={e => { e.stopPropagation(); setView("login") }}
                    className="text-white/50 hover:text-white transition-colors p-0.5"
                    title="Sign out">
                    <LogOut size={13} />
                  </button>
                </div>
              </>
            )}
          </div>
          {sidebarCollapsed && (
            <button
              onClick={() => setView("settings")}
              className="w-full flex items-center justify-center text-white/40 hover:text-primary px-2 py-2 rounded-xl transition-colors hover:bg-sidebar-accent mt-1"
              title="Settings"
            >
              <Settings size={13} />
            </button>
          )}
        </div>
      </motion.aside>

      {/* Main area */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {view === "subconscious" && (
          <SubconsciousView onClose={() => setView("chat")} />
        )}
        {view !== "subconscious" && <>

        {/* Header */}
        <header className="flex items-center gap-3 px-4 py-3.5 border-b border-border flex-shrink-0"
          style={{ background: "#1a1929", backdropFilter: "blur(12px)" }}>
          <button onClick={() => {
            setSidebarOpen(true)
            setSidebarCollapsed(false)
          }}
            className="lg:hidden text-white/70 hover:text-white transition-colors flex-shrink-0">
            <div className="relative w-11 h-11">
              <div className="absolute inset-0 rounded-full blur-sm" style={{ background: "rgba(136,141,223,0.4)" }} />
              <ThoughtSpark className="w-full h-full relative z-10" />
            </div>
          </button>
          <div className="flex-1 min-w-0">
            <h1 className="font-nunito font-bold text-white truncate text-base leading-tight">
              {activeSession?.name ?? "New Conversation"}
            </h1>
            <p className="text-xs text-white/50 leading-tight">
              {activeSession
                ? `${activeSession.messages.length} message${activeSession.messages.length !== 1 ? "s" : ""}`
                : "Start a new conversation"}
            </p>
          </div>
          <div className="flex items-center gap-1.5 flex-shrink-0">
            <span className="w-2 h-2 rounded-full bg-emerald-500 shadow-sm shadow-emerald-500/50" />
            <span className="text-xs text-white/55">Pondr AI</span>
          </div>
        </header>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-4 py-6 scrollbar-hide"
          style={{ background: "rgba(255,255,255,0.1)" }}>
          {!activeSession || activeSession.messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-center py-12">
              <div className="relative w-[261px] h-[261px] mb-6">
                <div className="absolute inset-0 rounded-full blur-3xl"
                  style={{ background: "rgba(136,141,223,0.25)" }} />
                <ThoughtSpark className="w-full h-full relative z-10 opacity-90" />
              </div>
              <PondrLogo className="h-9 w-44 overflow-hidden mb-3" />
              <p className="text-white/65 text-sm max-w-sm leading-relaxed mb-8">
                What would you like to ponder today? Ask anything — explanations, analysis, ideas, or creative exploration.
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 w-full max-w-lg">
                {SUGGESTIONS.map(s => (
                  <button key={s}
                    onClick={() => setInputText(s)}
                    className="text-left text-sm text-white/70 hover:text-white px-4 py-3 rounded-xl transition-all flex items-start gap-2.5"
                    style={{ background: "rgba(136,141,223,0.1)", border: "1px solid rgba(136,141,223,0.3)" }}
                  >
                    <Sparkles size={13} className="flex-shrink-0 mt-0.5" style={{ color: "#e2e6ff" }} />
                    {s}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="space-y-6 max-w-4xl mx-auto w-full">
              {activeSession.messages.map(msg => (
                <MessageBubble key={msg.id} message={msg} />
              ))}
              <AnimatePresence>
                {isTyping && <TypingIndicator />}
              </AnimatePresence>
              <div ref={bottomRef} />
            </div>
          )}
        </div>

        {/* Input area */}
        <div className="px-4 pb-4 pt-2 flex-shrink-0"
          style={{ background: "rgba(26,25,41,0.96)" }}>
          {/* Attached files preview */}
          {attached.length > 0 && (
            <div className="flex flex-wrap gap-2 mb-2 px-1">
              {attached.map(file => {
                const Icon = getFileIcon(file.type)
                return (
                  <div key={file.id}
                    className="flex items-center gap-2 rounded-lg px-3 py-1.5 text-sm"
                    style={{ background: "rgba(136,141,223,0.12)", border: "1px solid rgba(136,141,223,0.35)" }}>
                    <Icon size={12} className="text-primary flex-shrink-0" />
                    <span className="text-white/85 max-w-[140px] truncate text-xs">{file.name}</span>
                    <span className="text-white/45 text-xs flex-shrink-0">{formatFileSize(file.size)}</span>
                    <button onClick={() => setAttached(prev => prev.filter(f => f.id !== file.id))}
                      className="text-white/45 hover:text-red-400 transition-colors ml-0.5">
                      <X size={11} />
                    </button>
                  </div>
                )
              })}
            </div>
          )}

          {/* Input box */}
          <div
            className="flex items-end gap-2 border rounded-2xl px-4 py-3 transition-all focus-within:border-primary/40"
            style={{ background: "rgba(136,141,223,0.1)", border: "1px solid rgba(136,141,223,0.35)" }}
          >
            <button onClick={() => fileRef.current?.click()}
              className="text-white/50 hover:text-primary transition-colors flex-shrink-0 mb-0.5"
              title="Attach documents or images">
              <Paperclip size={17} />
            </button>
            <input ref={fileRef} type="file" multiple className="hidden" onChange={handleFiles}
              accept=".pdf,.doc,.docx,.txt,.md,.csv,.xlsx,.png,.jpg,.jpeg,.gif,.webp" />

            <textarea
              ref={textareaRef}
              value={inputText}
              onChange={e => {
                setInputText(e.target.value)
                e.target.style.height = "auto"
                e.target.style.height = Math.min(e.target.scrollHeight, 160) + "px"
              }}
              onKeyDown={handleKeyDown}
              placeholder="Ask Pondr anything… (Shift+Enter for new line)"
              rows={1}
              className="flex-1 bg-transparent text-white placeholder:text-white/35 resize-none focus:outline-none text-sm leading-6 scrollbar-hide"
              style={{ minHeight: "24px", maxHeight: "160px" }}
            />

            <button
              onClick={handleSend}
              disabled={!inputText.trim() && attached.length === 0}
              className={`flex-shrink-0 w-8 h-8 rounded-lg flex items-center justify-center transition-all mb-0.5 ${
                inputText.trim() || attached.length > 0
                  ? "active:scale-90"
                  : "cursor-not-allowed opacity-40"
              }`}
              style={{
                backgroundColor: "#e0efe4",
                boxShadow: (inputText.trim() || attached.length > 0) ? "0 2px 14px rgba(224,239,228,0.5)" : "none",
                color: "#0c0b1a"
              }}
            >
              <Send size={13} />
            </button>
          </div>

          {/* Bottom bar: model picker + disclaimer */}
          {(() => {
            const availableModels = providers.flatMap(p =>
              p.enabled ? p.models.filter(m => m.enabled).map(m => ({ key: `${p.id}::${m.id}`, providerName: p.name, modelId: m.modelId, label: m.label || m.modelId })) : []
            )
            const selected = availableModels.find(m => m.key === selectedModelKey) ?? availableModels[0] ?? null

            return (
              <div className="flex items-center justify-between mt-2 px-1">
                {/* Model picker */}
                <div className="relative">
                  <button
                    onClick={() => availableModels.length > 0 && setModelPickerOpen(o => !o)}
                    className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg transition-all group"
                    style={selected
                      ? { background: "rgba(136,141,223,0.12)", border: "1px solid rgba(136,141,223,0.28)" }
                      : { background: "transparent", border: "1px solid transparent" }
                    }
                    title={availableModels.length === 0 ? "Add a provider in Settings → Providers" : undefined}
                  >
                    <Bot size={11} style={{ color: selected ? "#888ddf" : "rgba(255,255,255,0.25)" }} />
                    <span className="text-[11px] font-medium max-w-[160px] truncate"
                      style={{ color: selected ? "#c3acda" : "rgba(255,255,255,0.25)" }}>
                      {selected ? `${selected.providerName} · ${selected.label}` : "No model selected"}
                    </span>
                    {availableModels.length > 1 && (
                      <ChevronDown size={10} style={{ color: "rgba(136,141,223,0.6)" }} />
                    )}
                  </button>

                  {/* Dropdown */}
                  <AnimatePresence>
                    {modelPickerOpen && availableModels.length > 0 && (
                      <>
                        <div className="fixed inset-0 z-10" onClick={() => setModelPickerOpen(false)} />
                        <motion.div
                          initial={{ opacity: 0, y: 6, scale: 0.97 }}
                          animate={{ opacity: 1, y: 0, scale: 1 }}
                          exit={{ opacity: 0, y: 4, scale: 0.97 }}
                          transition={{ duration: 0.15 }}
                          className="absolute bottom-full mb-2 left-0 z-20 min-w-[220px] rounded-xl overflow-hidden py-1"
                          style={{ background: "#1a1929", border: "1px solid rgba(136,141,223,0.3)", boxShadow: "0 8px 32px rgba(0,0,0,0.5)" }}
                        >
                          {providers.filter(p => p.enabled && p.models.some(m => m.enabled)).map(p => (
                            <div key={p.id}>
                              <p className="text-[10px] font-bold uppercase tracking-widest px-3 pt-2.5 pb-1" style={{ color: "rgba(136,141,223,0.7)" }}>
                                {p.name}
                              </p>
                              {p.models.filter(m => m.enabled).map(m => {
                                const key = `${p.id}::${m.id}`
                                const isActive = (selected?.key === key) || (!selectedModelKey && availableModels[0]?.key === key)
                                return (
                                  <button
                                    key={m.id}
                                    onClick={() => { setSelectedModelKey(key); setModelPickerOpen(false) }}
                                    className="w-full flex items-center gap-2.5 px-3 py-2 text-left transition-colors hover:bg-white/6"
                                    style={{ background: isActive ? "rgba(136,141,223,0.12)" : undefined }}
                                  >
                                    <span className="flex-1 text-sm truncate" style={{ color: isActive ? "#fff" : "rgba(255,255,255,0.65)" }}>
                                      {m.label || m.modelId}
                                    </span>
                                    <span className="text-[10px] truncate" style={{ color: "rgba(136,141,223,0.6)" }}>{m.modelId}</span>
                                    {isActive && <Check size={11} style={{ color: "#888ddf", flexShrink: 0 }} />}
                                  </button>
                                )
                              })}
                            </div>
                          ))}
                        </motion.div>
                      </>
                    )}
                  </AnimatePresence>
                </div>

                <p className="text-[11px] text-white/30">
                  Pondr may make mistakes — verify important information.
                </p>
              </div>
            )
          })()}
        </div>
      </>}
      </div>
    </div>
  )
}