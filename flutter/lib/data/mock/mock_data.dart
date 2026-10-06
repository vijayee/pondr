/// The export's seed data — VERBATIM from `mockup_reference/app.tsx`
/// (the `AI_POOL` block at lines 127-159, `SEED_SESSIONS` at 163-303,
/// `SUGGESTIONS` at 792-797, `BASE_NODES`/`BASE_EDGES` at 322-380).
/// The message texts are the chat view's fixtures; do not paraphrase them.
///
/// The export builds `SEED_SESSIONS` at module load with `Date.now()`
/// offsets; `seedSessions()` is that expression (fresh per call). Dart
/// naming is camelCase (`constant_identifier_names`); the export's names
/// appear in the doc comments.
library;

import '../models.dart';

/// The export's `AI_POOL` (its whole reply bank) — `MockChatService` picks
/// uniformly at random per send.
const List<String> aiPool = <String>[
  '''What a rich question to sit with. Let me think through this carefully.

At its core, the tension here is between two competing intuitions that both feel true: the desire for clear, decisive answers and the reality that most interesting questions resist them. The most productive approach isn't to resolve this tension but to work *within* it.

**The key insight:** Complexity often emerges from surprisingly simple underlying patterns. When you trace a phenomenon back to first principles, you frequently find that what seemed like a tangled knot is actually a few threads pulled tight. The challenge is resisting the urge to cut it.

What aspect would you like to explore further?''',

  '''I love how this question opens up into a landscape of interconnected ideas. There are two dimensions worth examining:

**The immediate dimension** — what this means practically, and how it changes how we act or think day-to-day.

**The structural dimension** — *why* this pattern exists at all, and what it reveals about the deeper system it's embedded in.

Most analyses stop at the first. The second is where the real leverage is. Once you see the structural reason something is the way it is, you gain the ability to anticipate where it will hold and where it will break down.

Does that framing resonate with what you're exploring?''',

  '''The short answer is nuanced, but the long answer is illuminating.

Conventional wisdom here is often incomplete — not wrong exactly, but missing the piece that explains the exceptions. Most people approach this from the surface level, which gives you a working model about 80% of the time. The remaining 20% is where the real understanding lives.

The key is to ask not just *what* is true but *under what conditions* it's true. That shift from categorical claims to conditional ones is one of the most powerful upgrades you can make to your thinking.

What's the context you're working in? That would help me give you a sharper answer.''',

  '''This is one of those questions that looks simple from a distance and reveals incredible depth up close.

The conventional framing treats this as a binary — either X or Y. But that's a false dichotomy. What actually happens is more interesting: X and Y exist on a spectrum, and most real-world cases sit somewhere between the poles, often shifting position depending on factors we don't always control.

What makes this practically useful is understanding which factors move things toward which end, and when that movement matters. The map is simpler than it looks once you have the right coordinate system.''',
];

/// The export's `SUGGESTIONS` — the four chips shown on an empty session.
const List<String> suggestions = <String>[
  'Explain the Fermi paradox',
  'What is the hard problem of consciousness?',
  'How do transformer models work?',
  'What makes an argument convincing?',
];

/// The export's `BASE_NODES` — 25 nodes, radii and descriptions verbatim;
/// two nodes carry their own colour instead of their cluster's.
const List<SimNode> baseNodes = <SimNode>[
  // The export's `CLUSTER_COLORS.Physics` is '#888ddf'.
  SimNode(id: 'quantum-mechanics', label: 'Quantum Mechanics', cluster: 'Physics', color: '#888ddf', r: 18, description: 'The branch of physics describing subatomic particles and their probabilistic behavior.'),
  SimNode(id: 'superposition', label: 'Superposition', cluster: 'Physics', color: '#888ddf', r: 14, description: 'A particle exists in multiple states simultaneously until the moment it is measured.'),
  SimNode(id: 'qubits', label: 'Qubits', cluster: 'Physics', color: '#888ddf', r: 12, description: 'Quantum bits that exploit superposition to enable massively parallel computation.'),
  SimNode(id: 'entanglement', label: 'Entanglement', cluster: 'Physics', color: '#888ddf', r: 15, description: 'Two particles share a correlated quantum state regardless of the distance between them.'),
  SimNode(id: 'wave-collapse', label: 'Wave Collapse', cluster: 'Physics', color: '#888ddf', r: 11, description: 'The reduction of a superposed wave function to a single definite state upon observation.'),
  SimNode(id: 'consciousness', label: 'Consciousness', cluster: 'Philosophy', color: '#c3acda', r: 20, description: 'Subjective experience and the inner phenomenal life of a mind.'),
  SimNode(id: 'qualia', label: 'Qualia', cluster: 'Philosophy', color: '#c3acda', r: 14, description: 'The irreducible subjective character of experience — the redness of red, the sharpness of pain.'),
  SimNode(id: 'hard-problem', label: 'Hard Problem', cluster: 'Philosophy', color: '#c3acda', r: 16, description: 'Why physical brain processes give rise to subjective experience at all (Chalmers, 1994).'),
  SimNode(id: 'explanatory-gap', label: 'Explanatory Gap', cluster: 'Philosophy', color: '#c3acda', r: 12, description: 'The conceptual gulf between complete neural description and phenomenal experience.'),
  SimNode(id: 'phenomenology', label: 'Phenomenology', cluster: 'Philosophy', color: '#c3acda', r: 11, description: 'Philosophical study of the structure of first-person experience (Husserl, Heidegger, Merleau-Ponty).'),
  SimNode(id: 'machine-learning', label: 'Machine Learning', cluster: 'AI', color: '#e2e6ff', r: 18, description: 'Systems that learn statistical patterns from data without explicit programming.'),
  SimNode(id: 'supervised', label: 'Supervised Learning', cluster: 'AI', color: '#e2e6ff', r: 13, description: 'Training on labeled input-output pairs to generalise predictions to new examples.'),
  SimNode(id: 'unsupervised', label: 'Unsupervised Learning', cluster: 'AI', color: '#e2e6ff', r: 13, description: 'Finding latent structure — clusters, manifolds, representations — in unlabeled data.'),
  SimNode(id: 'transformers', label: 'Transformers', cluster: 'AI', color: '#e2e6ff', r: 14, description: 'Attention-based neural architecture that underpins modern large language models.'),
  SimNode(id: 'llms', label: 'Large Language Models', cluster: 'AI', color: '#e2e6ff', r: 17, description: 'Neural networks trained on vast text corpora capable of reasoning, generation, and retrieval.'),
  SimNode(id: 'fermi-paradox', label: 'Fermi Paradox', cluster: 'Astronomy', color: '#e8d7bd', r: 18, description: 'The contradiction between high estimates for alien civilisations and the total absence of evidence.'),
  SimNode(id: 'great-filter', label: 'Great Filter', cluster: 'Astronomy', color: '#e8d7bd', r: 14, description: 'A hypothetical barrier — past or future — that prevents life becoming interstellar.'),
  SimNode(id: 'dark-forest', label: 'Dark Forest Theory', cluster: 'Astronomy', color: '#e8d7bd', r: 14, description: 'Any sufficiently advanced civilisation stays silent because broadcasting location invites destruction.'),
  SimNode(id: 'drake-equation', label: 'Drake Equation', cluster: 'Astronomy', color: '#e8d7bd', r: 11, description: 'A probabilistic formula estimating the number of detectable communicating civilisations in the galaxy.'),
  SimNode(id: 'kardashev', label: 'Kardashev Scale', cluster: 'Astronomy', color: '#e8d7bd', r: 11, description: 'Classification of civilisations by total energy consumption across planetary, stellar, and galactic scales.'),
  SimNode(id: 'argumentation', label: 'Argumentation', cluster: 'Logic', color: '#e0efe4', r: 15, description: 'The structured process of forming claims, supplying reasons, and anticipating counterarguments.'),
  SimNode(id: 'first-principles', label: 'First Principles', cluster: 'Logic', color: '#e0efe4', r: 13, description: 'Reasoning up from self-evident foundational truths rather than by analogy or convention.'),
  SimNode(id: 'epistemology', label: 'Epistemology', cluster: 'Logic', color: '#e0efe4', r: 14, description: 'The philosophical study of knowledge, justified belief, and the limits of what can be known.'),
  SimNode(id: 'quantum-consciousness', label: 'Quantum Consciousness', cluster: 'Philosophy', color: '#b8a4dc', r: 12, description: 'Penrose-Hameroff hypothesis: quantum processes in neural microtubules give rise to consciousness.'),
  SimNode(id: 'ai-consciousness', label: 'AI Consciousness', cluster: 'AI', color: '#aab3e8', r: 13, description: 'The open question of whether artificial systems can possess genuine phenomenal experience.'),
];

/// The export's `BASE_EDGES` — all 29, verbatim.
const List<SimEdge> baseEdges = <SimEdge>[
  SimEdge(source: 'quantum-mechanics', target: 'superposition'),
  SimEdge(source: 'quantum-mechanics', target: 'entanglement'),
  SimEdge(source: 'quantum-mechanics', target: 'wave-collapse'),
  SimEdge(source: 'superposition', target: 'qubits'),
  SimEdge(source: 'entanglement', target: 'wave-collapse'),
  SimEdge(source: 'consciousness', target: 'qualia'),
  SimEdge(source: 'consciousness', target: 'hard-problem'),
  SimEdge(source: 'hard-problem', target: 'qualia'),
  SimEdge(source: 'hard-problem', target: 'explanatory-gap'),
  SimEdge(source: 'hard-problem', target: 'phenomenology'),
  SimEdge(source: 'explanatory-gap', target: 'qualia'),
  SimEdge(source: 'machine-learning', target: 'supervised'),
  SimEdge(source: 'machine-learning', target: 'unsupervised'),
  SimEdge(source: 'machine-learning', target: 'transformers'),
  SimEdge(source: 'transformers', target: 'llms'),
  SimEdge(source: 'fermi-paradox', target: 'great-filter'),
  SimEdge(source: 'fermi-paradox', target: 'dark-forest'),
  SimEdge(source: 'fermi-paradox', target: 'drake-equation'),
  SimEdge(source: 'great-filter', target: 'kardashev'),
  SimEdge(source: 'argumentation', target: 'first-principles'),
  SimEdge(source: 'argumentation', target: 'epistemology'),
  SimEdge(source: 'first-principles', target: 'epistemology'),
  SimEdge(source: 'quantum-mechanics', target: 'quantum-consciousness'),
  SimEdge(source: 'consciousness', target: 'quantum-consciousness'),
  SimEdge(source: 'consciousness', target: 'ai-consciousness'),
  SimEdge(source: 'llms', target: 'ai-consciousness'),
  SimEdge(source: 'machine-learning', target: 'ai-consciousness'),
  SimEdge(source: 'epistemology', target: 'consciousness'),
  SimEdge(source: 'epistemology', target: 'hard-problem'),
];

/// The export's `SEED_SESSIONS` — 5 sessions (the export has five, not the
/// three the plan's summary guessed; the reference is truth), message texts
/// byte-for-byte, timestamps the export's `Date.now()` offsets.
List<ChatSession> seedSessions() {
  final now = DateTime.now();
  return <ChatSession>[
    ChatSession(
      id: 's1',
      name: 'Quantum Superposition & Qubits',
      updatedAt: now.subtract(const Duration(minutes: 9)),
      messages: <Message>[
        Message(
          id: 'm1a',
          role: MessageRole.user,
          content: 'Can you explain quantum superposition in simple terms?',
          timestamp: now.subtract(const Duration(minutes: 12)),
        ),
        Message(
          id: 'm1b',
          role: MessageRole.assistant,
          content: '''Quantum superposition is one of the most counterintuitive—and beautiful—principles in physics.

Imagine a coin spinning in the air. It's neither heads nor tails until it lands. A quantum particle behaves similarly: it exists in *multiple states simultaneously* until it's observed or measured. This isn't a limitation of our knowledge—the particle genuinely occupies all states at once.

**Why does this matter?**
This is what gives quantum computers their extraordinary potential. Classical bits are either 0 or 1. Quantum bits (qubits) can be both at the same time—enabling a kind of massively parallel computation that classical machines fundamentally cannot replicate.

The collapse of superposition upon measurement is still one of the deepest mysteries in physics. What causes it? That question is still open.''',
          timestamp: now.subtract(const Duration(minutes: 11)),
        ),
        Message(
          id: 'm1c',
          role: MessageRole.user,
          content: 'How does entanglement connect to this?',
          timestamp: now.subtract(const Duration(minutes: 9)),
        ),
        Message(
          id: 'm1d',
          role: MessageRole.assistant,
          content: '''Entanglement is superposition's stranger sibling.

When two particles become entangled, their quantum states are linked—no matter the distance between them. Measure one, and you instantly know something about the other. Einstein famously called this "spooky action at a distance" and found it deeply troubling.

**The key point:** This isn't information traveling faster than light (that would violate relativity). It's more subtle—the correlated outcomes are built into the particles' shared quantum state from the start. You don't learn what the distant particle is; you learn what it *was always going to be*, given what its partner turned out to be.

This makes entanglement a resource for quantum communication and certain quantum algorithms—but it's not a way to send signals faster than light.''',
          timestamp: now.subtract(const Duration(minutes: 8)),
        ),
      ],
    ),
    ChatSession(
      id: 's2',
      name: 'The Hard Problem of Consciousness',
      updatedAt: now.subtract(const Duration(hours: 4)),
      messages: <Message>[
        Message(
          id: 'm2a',
          role: MessageRole.user,
          content: 'What is the hard problem of consciousness, and why is it hard?',
          timestamp: now.subtract(const Duration(hours: 4)),
        ),
        Message(
          id: 'm2b',
          role: MessageRole.assistant,
          content: '''Coined by philosopher David Chalmers in 1994, the hard problem asks perhaps the deepest question we have: *why is there subjective experience at all?*

**The easy problems** (still enormously complex): How does the brain integrate information? Process sensory data? Control behavior? These are hard in the engineering sense—but they're tractable. We can imagine a future neuroscience that answers them fully.

**The hard problem**: Even after you've explained all of that perfectly, something seems left out. Why does 700nm wavelength *look* red, rather than just triggering a signal? Why does pain *hurt*, rather than merely being a neutral alarm? Why is there an "inside" to experience at all?

This is what philosophers call the **explanatory gap**—between objective neural description and subjective phenomenal experience. The qualia (the redness of red, the sharpness of pain) seem to resist being fully captured in functional terms.

Whether this gap represents a genuine mystery about reality, or a failure of our conceptual vocabulary, remains unsettled—and may be the most important question of the 21st century.''',
          timestamp: now
              .subtract(const Duration(hours: 4))
              .add(const Duration(seconds: 60)),
        ),
      ],
    ),
    ChatSession(
      id: 's3',
      name: 'Supervised vs Unsupervised Learning',
      updatedAt: now.subtract(const Duration(days: 1, hours: 12)),
      messages: <Message>[
        Message(
          id: 'm3a',
          role: MessageRole.user,
          content: "What's the practical difference between supervised and unsupervised ML?",
          timestamp: now.subtract(const Duration(days: 1, hours: 12)),
        ),
        Message(
          id: 'm3b',
          role: MessageRole.assistant,
          content: '''The distinction is fundamental—and it shapes everything from data requirements to what questions you can even ask.

**Supervised learning** trains on labeled examples: input → known output. The model learns to map new inputs to outputs it hasn't seen. You need labeled data, which is often expensive to produce. But you get targeted, measurable predictions.
*Examples: spam filtering, image classification, fraud detection, price prediction.*

**Unsupervised learning** gets no labels. It has to find structure in raw data on its own—discovering clusters, patterns, and anomalies without being told what to look for. You can use far more data (labels aren't required), but the outputs are harder to evaluate.
*Examples: customer segmentation, topic modeling, anomaly detection, dimensionality reduction.*

**The mental model that clicks for most people:**
- Supervised: *"Here are the answers. Learn the rules that produce them."*
- Unsupervised: *"Here's the data. Find whatever's interesting."*

In practice, the most powerful systems often combine both—using unsupervised methods to build representations, then fine-tuning with supervision for specific tasks. That's roughly how modern LLMs work.''',
          timestamp: now
              .subtract(const Duration(days: 1, hours: 12))
              .add(const Duration(seconds: 90)),
        ),
      ],
    ),
    ChatSession(
      id: 's4',
      name: 'The Fermi Paradox',
      updatedAt: now.subtract(const Duration(days: 4)),
      messages: <Message>[
        Message(
          id: 'm4a',
          role: MessageRole.user,
          content:
              "If the universe is so vast and old, why haven't we found any signs of alien civilizations?",
          timestamp: now.subtract(const Duration(days: 4)),
        ),
        Message(
          id: 'm4b',
          role: MessageRole.assistant,
          content: '''This is the Fermi Paradox—and it's one of the most haunting open questions in science.

The universe is ~13.8 billion years old. Our galaxy contains ~400 billion stars, many with planets in habitable zones. The math suggests we should be surrounded by evidence of other civilizations. Yet: silence.

**The leading hypotheses—each with different implications:**

**The Great Filter** — Something vanishingly rare stands between simple chemistry and spacefaring civilizations. If the filter is *behind* us (complex life is incredibly rare), we might be alone. If it's *ahead* of us (civilizations tend to self-destruct), that's far more troubling.

**The Dark Forest** — Any sufficiently advanced civilization hides. In a universe of unknown others, broadcasting your location is suicidal. Silence is strategy. (Liu Cixin explores this beautifully in his trilogy.)

**We're early** — Cosmically speaking, we emerged young. Most stars that will ever form haven't yet. We may be among the first.

**The Zoo hypothesis** — They're watching, not interfering. (Uncomfortable for obvious reasons.)

**It's just hard** — Interstellar distances are staggering. The window of overlapping technological civilizations might be vanishingly narrow. We may simply have missed each other.

The answer matters enormously—each resolution carries profound implications for our own future.''',
          timestamp: now
              .subtract(const Duration(days: 4))
              .add(const Duration(seconds: 120)),
        ),
      ],
    ),
    ChatSession(
      id: 's5',
      name: 'What Makes Arguments Good?',
      updatedAt: now.subtract(const Duration(days: 8)),
      messages: <Message>[
        Message(
          id: 'm5a',
          role: MessageRole.user,
          content: 'What actually makes an argument good or bad?',
          timestamp: now.subtract(const Duration(days: 8)),
        ),
      ],
    ),
  ];
}