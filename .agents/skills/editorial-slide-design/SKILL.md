---
name: editorial-slide-design
description: >-
  Designs and builds publication-grade, human-crafted academic and executive
  PowerPoint (.pptx) presentations ready for PowerPoint, Canva, and Google Slides.
  Use this skill whenever the user asks to create, redesign, or polish presentation
  slides, capstone defense decks, or technical pitch decks without generic AI-slop
  aesthetics.
---

# Editorial & Academic Slide Design Skill (Anti-AI Aesthetic)

Follow this workflow whenever creating or redesigning `.pptx` presentation decks so every slide looks crafted by a senior human editorial designer (McKinsey / MIT / IEEE CVPR / University Defense standard) and imports cleanly into Canva or PowerPoint.

## 1. Strict Anti-AI-Slop Bans (NEVER Violate)

1. **NO Neon Dark-Mode Dashboard Clichés:** Never combine dark navy backgrounds with multiple glowing neon accents (cyan + neon green + yellow + pink + purple). Never draw thick colored borders around rounded dark cards.
2. **NO Pill-Badge Soup:** Do not put rounded pill badges above every heading and inside every card. Use clean, tracked-out uppercase typographic kickers (`8.5–9.5 pt` bold muted/brand ink, no border) instead.
3. **NO Repetitive 3-Box Templates:** Never repeat the same "header + 3 equal rounded rectangles" layout across consecutive slides. Every slide in a 5-slide sequence must use a distinct spatial composition tailored to its content.
4. **NO Decorative Title Underlines:** Never place short colored bars or glowing lines directly under slide titles. Use deliberate whitespace and clean `0.5 pt` hairline dividers only where structural separation is needed.
5. **NO Stock AI Faces When Project Assets Exist:** Always extract and showcase real photographs, dataset crops, experimental plots, and architectural diagrams from the user's actual codebase/thesis.

## 2. The 65–25–10 Editorial Color Architecture

- **65% Dominant Breathable Canvas:** Warm Paper Alabaster (`#F7F5F0`) and Crisp Card White (`#FFFFFF`), with subtle Warm Stone (`#E4E0D8`) `0.5–0.75 pt` hairlines. Ensures effortless projector legibility in lit defense rooms.
- **25% Structural Deep Ink:** Oxford Navy / Charcoal Ink (`#111C2D` for primary display text, architectural split panels on cover/closing slides, and native table headers; `#475569` for secondary body copy).
- **10% Institutional Signature Accent:** One sharp focal accent tied to the institution or topic (e.g. FPT Burnt Terracotta `#E05A1E`), plus muted editorial pigments (`#1B6B4A` Forest, `#B42338` Crimson) strictly for empirical Real-vs-Fake data or error comparisons.

## 3. Typography & Hierarchy Pairing (Canva & Vietnamese Safe)

- **Display Numerals, Formulas & English Academic Subtitles:** `Georgia` (Bold for oversized editorial index numbers `01`–`04`, Italic for mathematical equations and English thesis subtitles).
- **Headings, UI Labels & Vietnamese Body Copy:** `Arial` (100% Vietnamese Unicode coverage and 1:1 native font mapping upon Canva `.pptx` upload without diacritic Substitution errors).
- **Action Titles:** Write informative, thesis-driven slide titles (*"ArcFace Đạt >99% Định Danh Nhưng Mù Tính Sống Vật Lý"*) rather than vague labels (*"Đặt Vấn Đề"*).
- **Scale Contrast:** Contrast `24–36 pt` hero metrics and `20–22 pt` action titles against `10–11 pt` airy body copy (`line_spacing = 1.28–1.35`).

## 4. Spatial Variety Blueprint (5-Slide Arc)

1. **Slide 1 (Architectural Split Cover):** `60%` Warm Ivory editorial typography + institutional logo on left; `40%` full-bleed Deep Ink `#111C2D` showcase panel on right featuring real project captures and understated typographic benchmarks.
2. **Slide 2 (Editorial Thesis Column + Minimalist Flow + Hairline Rows):** Left column states the core tension with an oversized serif numeral; right workspace shows a clean minimalist pipeline above 3 borderless horizontal hairline rows.
3. **Slide 3 (Native Academic Comparison Table + Swiss Metric Grid):** Use a real `slide.shapes.add_table()` publication table paired with giant typographic KPIs and left-ruled (`2 pt` vertical bar, borderless background) contribution blocks.
4. **Slide 4 (Authentic Dataset Visuals + Scientific White-Paper Diagram):** Pair genuine dataset/experiment crops with crisp publication-style scientific diagrams on white/ivory surfaces.
5. **Slide 5 (Mathematical Callouts + Tech Stack Matrix + Deep-Ink Handover Band):** Set key equations in `Georgia` serif accompanied by a compact empirical bar chart, a 4-stage tabular stack, and a `#111C2D` bottom anchor band transitioning to the next speaker.

## 5. Mandatory Visual Verification Step

After generating any `.pptx`, render every slide to `1920×1080` PNG using [render_pptx_previews.ps1](./scripts/render_pptx_previews.ps1) and inspect every PNG with `read_file` to verify:
- Zero text overlapping or touching container edges (at least `0.14 in` internal padding).
- Zero awkward 1-word orphan wraps on headings or formula lines.
- Balanced whitespace (`>= 25%` open canvas area per slide).
