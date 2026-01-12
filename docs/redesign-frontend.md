# Frontend Redesign: Functional Grouping

## Goal
Restructure the `frontend/src/pages` directory to group pages by function (Detection, Safety, Evaluation), aligning with the new backend architecture.

## Proposed Structure

```text
frontend/src/
├── pages/
│   ├── Dashboard.tsx        # (Remains at root of pages)
│   │
│   ├── detection/
│   │   ├── BiasBatch.tsx
│   │   ├── HallucinationDetection.tsx
│   │   ├── RealtimeAnalysis.tsx
│   │   └── ToxicityBatch.tsx
│   │
│   ├── safety/
│   │   └── Guardrails.tsx
│   │
│   └── evaluation/
│       ├── AdversarialTesting.tsx
│       ├── Benchmarks.tsx
│       ├── PrivacyTesting.tsx
│       └── ReliabilityTesting.tsx
│
├── components/              # Shared components
│   ├── Navbar.tsx
│   └── Sidebar.tsx
│
├── api/                     # Shared API client
└── hooks/                   # Shared hooks
```

## Migration Plan

1.  Create new directory structure in `frontend/src/pages/`.
2.  Move existing page files to their new locations.
3.  Update imports within the moved pages (adjust relative paths by adding `../`).
4.  Update imports in `App.tsx` to point to the new locations.
5.  (Optional) Update `Sidebar.tsx` to reflect the logical grouping if desired (e.g. moving Hallucination to Detection section).
