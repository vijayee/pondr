# Hippocampal Memory

A brain-inspired memory architecture for AI agents, built on [WaveDB](https://github.com/vijayee/WaveDB).
The WaveDB **Graph layer** is the hippocampal index (sparse pointers — who/what/when/tone/
decisions/relations); the WaveDB **HBTrie** is the neocortical store (content — summary,
full text, timestamp).

Phase 1a is the encoding pipeline: it consumes raw conversation turns, runs GLiNER-Decoder
(open discovery) + GLiNER2 (stable extraction) + Bonsai (a local 8B ternary relation model)
to extract entities / topics / tones / decisions / relations, and stores the result as an
**Episode** — content in HBTrie, structure in the Graph layer.

## Layout

```
src/
├── config.py                 # central configuration
├── memory/
│   ├── episode.py            # Episode data model
│   ├── ontology.py           # seed ontology (conversation + code)
│   └── store.py              # WaveDB wrapper (HBTrie + Graph)
└── encoding/
    ├── gliner_extractor.py   # GLiNER-Decoder + GLiNER2
    ├── bonsai_relations.py   # local 8B ternary Bonsai
    └── encoder.py            # orchestrator
```

## Running

The heavy compute (GLiNER models + Bonsai) runs on a RunPod GPU pod — see
`infra/runpod/`. The local machine only edits code and reads results. See
`docs/Phase 1a.md` for the full plan and checkpoint criteria.